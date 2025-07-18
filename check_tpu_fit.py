#!/usr/bin/env python3
"""TPU Memory and Compilation Checker

This script validates whether a given configuration will fit on the available TPU hardware
by checking:
1. Model parameter memory requirements
2. Activation memory requirements  
3. Gradient and optimizer state memory
4. Device mesh compatibility
5. Actual JAX compilation

Usage:
    python check_tpu_fit.py --config=configs/data_parallel.py
    python check_tpu_fit.py --config=configs/default.py --batch_size=64
"""

import jax
import jax.numpy as jnp
import dataclasses
from flax import nnx
from absl import app, flags, logging
from ml_collections import config_flags
import importlib.util
import sys
import traceback

# Import local modules
import utils
import model
import modules
import input_pipeline

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    'config',
    'configs/data_parallel.py',
    'Path to the configuration file to check.',
    lock_config=True,
)
flags.DEFINE_integer('batch_size', None, 'Override per-device batch size for testing')
flags.DEFINE_boolean('verbose', True, 'Print detailed memory breakdown')
flags.DEFINE_boolean('dry_run', True, 'Perform actual compilation test')


def format_bytes(num_bytes):
    """Format bytes in human readable format."""
    if num_bytes >= 1e12:
        return f"{num_bytes/1e12:.2f} TB"
    elif num_bytes >= 1e9:
        return f"{num_bytes/1e9:.2f} GB"
    elif num_bytes >= 1e6:
        return f"{num_bytes/1e6:.2f} MB"
    elif num_bytes >= 1e3:
        return f"{num_bytes/1e3:.2f} KB"
    else:
        return f"{num_bytes} B"


def format_number(num):
    """Format large numbers in human readable format."""
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return str(num)


def estimate_bytes_per_param(use_bfloat16=True):
    """Estimate bytes per parameter based on precision."""
    if use_bfloat16:
        return 2  # bfloat16 = 2 bytes
    else:
        return 4  # float32 = 4 bytes


def estimate_model_memory(config, total_params, batch_size, seq_len):
    """Estimate total memory requirements for model training."""
    
    bytes_per_param = estimate_bytes_per_param(config.use_bfloat16)
    
    # 1. Model parameters
    param_memory = total_params * bytes_per_param
    
    # 2. Gradients (same size as parameters)
    grad_memory = total_params * bytes_per_param
    
    # 3. Optimizer state (AdamW: momentum + variance)
    # AdamW stores momentum and variance in float32 regardless of model precision
    optimizer_memory = total_params * 4 * 2  # 4 bytes * 2 states
    
    # 4. Activations (rough estimate)
    # Transformer activations scale with: batch_size * seq_len * hidden_size * num_layers
    activation_size_per_token = config.emb_dim * config.num_layers * 4  # rough multiplier
    activation_memory = batch_size * seq_len * activation_size_per_token * bytes_per_param
    
    # 5. Attention cache (for sequence length scaling)
    # K, V caches: batch_size * num_heads * seq_len * head_dim * num_layers
    head_dim = config.qkv_dim // config.num_heads
    kv_cache_memory = batch_size * config.num_heads * seq_len * head_dim * config.num_layers * bytes_per_param * 2
    
    total_memory = param_memory + grad_memory + optimizer_memory + activation_memory + kv_cache_memory
    
    return {
        'param_memory': param_memory,
        'grad_memory': grad_memory, 
        'optimizer_memory': optimizer_memory,
        'activation_memory': activation_memory,
        'kv_cache_memory': kv_cache_memory,
        'total_memory': total_memory,
        'bytes_per_param': bytes_per_param
    }


def get_tpu_info():
    """Get information about available TPU devices."""
    devices = jax.devices()
    
    tpu_info = {
        'num_devices': len(devices),
        'device_kind': devices[0].device_kind if devices else 'unknown',
        'platform': devices[0].platform if devices else 'unknown',
        'num_slices': 1,
        'devices_per_slice': len(devices)
    }
    
    # Try to get slice information
    try:
        if hasattr(devices[0], 'slice_index'):
            slice_indices = [d.slice_index for d in devices]
            tpu_info['num_slices'] = len(set(slice_indices))
            tpu_info['devices_per_slice'] = len(devices) // tpu_info['num_slices']
    except:
        pass
    
    # Estimate memory per device based on known TPU types
    memory_per_device = estimate_tpu_memory_per_device(tpu_info['device_kind'])
    tpu_info['memory_per_device'] = memory_per_device
    tpu_info['total_memory'] = memory_per_device * tpu_info['num_devices']
    
    return tpu_info


def estimate_tpu_memory_per_device(device_kind):
    """Estimate TPU memory per device based on device kind."""
    # Common TPU memory sizes (approximate)
    memory_map = {
        'TPU v2': 8 * 1024**3,   # 8 GB
        'TPU v3': 16 * 1024**3,  # 16 GB  
        'TPU v4': 32 * 1024**3,  # 32 GB
        'TPU v5': 64 * 1024**3,  # 64 GB (estimate)
    }
    
    # Default fallback
    default_memory = 16 * 1024**3  # 16 GB
    
    for tpu_type, memory in memory_map.items():
        if tpu_type.lower() in device_kind.lower():
            return memory
    
    return default_memory


def check_mesh_compatibility(config, tpu_info):
    """Check if the mesh configuration is compatible with available devices."""
    try:
        devices_array = utils.create_device_mesh(config)
        mesh_shape = devices_array.shape
        
        expected_devices = 1
        for dim in mesh_shape:
            expected_devices *= dim
            
        if expected_devices != tpu_info['num_devices']:
            return False, f"Mesh requires {expected_devices} devices but only {tpu_info['num_devices']} available"
            
        return True, f"Mesh shape {mesh_shape} compatible with {tpu_info['num_devices']} devices"
        
    except Exception as e:
        return False, f"Mesh creation failed: {str(e)}"


def test_compilation(config, batch_size=None):
    """Test actual JAX compilation with the configuration."""
    try:
        logging.info("🔧 Testing JAX compilation...")
        
        # Override batch size if provided
        if batch_size is not None:
            config = config.replace(per_device_batch_size=batch_size)
        
        # Create device mesh
        devices_array = utils.create_device_mesh(config)
        mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)
        
        # Create model config
        model_config = modules.TransformerConfig(
            vocab_size=config.vocab_size,
            output_vocab_size=config.vocab_size,
            logits_via_embedding=config.logits_via_embedding,
            dtype=jnp.bfloat16 if config.use_bfloat16 else jnp.float32,
            emb_dim=config.emb_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            qkv_dim=config.qkv_dim,
            mlp_dim=config.mlp_dim,
            max_len=config.max_target_length,
            dropout_rate=config.dropout_rate,
            attention_dropout_rate=config.attention_dropout_rate,
            kernel_init=nnx.initializers.xavier_uniform(),
            bias_init=nnx.initializers.normal(stddev=1e-6),
            axis_rules=config.axis_rules,
        )
        
        # Initialize model
        rng = jax.random.PRNGKey(config.seed)
        rng, init_rng = jax.random.split(rng)
        
        def constructor(config: modules.TransformerConfig, key: jax.Array):
            return model.TransformerLM(config, rngs=nnx.Rngs(params=key))
        
        # Create dummy optimizer
        import optax
        learning_rate_fn = lambda step: config.learning_rate
        optimizer = optax.adamw(learning_rate_fn, weight_decay=config.weight_decay)
        
        # Setup initial state
        with mesh:
            state, state_sharding = utils.setup_initial_state(
                constructor, optimizer, model_config, init_rng, mesh
            )
        
        # Create dummy batch
        batch_size = config.per_device_batch_size * len(jax.devices())
        dummy_batch = {
            'inputs': jnp.ones((batch_size, config.max_target_length), dtype=jnp.int32)
        }
        
        # Test forward pass compilation
        @jax.jit
        def dummy_forward(state, batch):
            module = nnx.merge(state.graphdef, state.params, state.non_diff_state)
            return module(batch['inputs'], rngs=nnx.Rngs(jax.random.PRNGKey(0)))
        
        # Compile and run once
        _ = dummy_forward(state, dummy_batch)
        
        return True, "Compilation successful"
        
    except Exception as e:
        return False, f"Compilation failed: {str(e)}\n{traceback.format_exc()}"


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    
    config = FLAGS.config
    
    # Override batch size if specified
    if FLAGS.batch_size is not None:
        config = config.replace(per_device_batch_size=FLAGS.batch_size)
    
    print("🚀 TPU Configuration Compatibility Check")
    print("=" * 60)
    
    # Get TPU information
    print("🔍 Analyzing TPU Hardware...")
    tpu_info = get_tpu_info()
    print(f"  Platform: {tpu_info['platform']}")
    print(f"  Device kind: {tpu_info['device_kind']}")
    print(f"  Number of devices: {tpu_info['num_devices']}")
    print(f"  Number of slices: {tpu_info['num_slices']}")
    print(f"  Devices per slice: {tpu_info['devices_per_slice']}")
    print(f"  Estimated memory per device: {format_bytes(tpu_info['memory_per_device'])}")
    print(f"  Total estimated memory: {format_bytes(tpu_info['total_memory'])}")
    
    # Check mesh compatibility
    print("\n⚙️  Checking Device Mesh Compatibility...")
    mesh_compatible, mesh_message = check_mesh_compatibility(config, tpu_info)
    if mesh_compatible:
        print(f"  ✅ {mesh_message}")
    else:
        print(f"  ❌ {mesh_message}")
        return
    
    # Create model and count parameters
    print("\n🧠 Analyzing Model Configuration...")
    print(f"  Vocabulary size: {config.vocab_size:,}")
    print(f"  Embedding dimension: {config.emb_dim}")
    print(f"  Number of layers: {config.num_layers}")
    print(f"  Number of heads: {config.num_heads}")
    print(f"  MLP dimension: {config.mlp_dim}")
    print(f"  Max sequence length: {config.max_target_length}")
    print(f"  Per-device batch size: {config.per_device_batch_size}")
    print(f"  Use bfloat16: {config.use_bfloat16}")
    
    # Create temporary model to count parameters
    transformer_config = modules.TransformerConfig(
        vocab_size=config.vocab_size,
        output_vocab_size=config.vocab_size,
        emb_dim=config.emb_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        qkv_dim=config.qkv_dim,
        mlp_dim=config.mlp_dim,
        max_len=config.max_target_length,
        dropout_rate=config.dropout_rate,
        attention_dropout_rate=config.attention_dropout_rate,
        logits_via_embedding=config.logits_via_embedding,
    )
    
    rngs = nnx.Rngs(params=jax.random.key(0), dropout=jax.random.key(1))
    test_model = model.TransformerLM(transformer_config, decode=False, rngs=rngs)
    params = nnx.state(test_model, nnx.Param)
    total_params, _ = utils.count_parameters(params)
    
    print(f"  Total parameters: {format_number(total_params)} ({total_params:,})")
    
    # Memory analysis
    print("\n💾 Memory Requirements Analysis...")
    batch_size = config.per_device_batch_size * tpu_info['num_devices']
    memory_est = estimate_model_memory(config, total_params, batch_size, config.max_target_length)
    
    print(f"  Effective batch size: {batch_size}")
    print(f"  Bytes per parameter: {memory_est['bytes_per_param']}")
    print(f"  Model parameters: {format_bytes(memory_est['param_memory'])}")
    print(f"  Gradients: {format_bytes(memory_est['grad_memory'])}")
    print(f"  Optimizer state: {format_bytes(memory_est['optimizer_memory'])}")
    print(f"  Activations: {format_bytes(memory_est['activation_memory'])}")
    print(f"  KV cache: {format_bytes(memory_est['kv_cache_memory'])}")
    print(f"  TOTAL ESTIMATED: {format_bytes(memory_est['total_memory'])}")
    
    # Memory fit check
    memory_per_device = memory_est['total_memory'] // tpu_info['num_devices']
    print(f"\n📊 Memory Per Device: {format_bytes(memory_per_device)}")
    print(f"  Available per device: {format_bytes(tpu_info['memory_per_device'])}")
    
    memory_utilization = memory_per_device / tpu_info['memory_per_device']
    print(f"  Memory utilization: {memory_utilization:.1%}")
    
    if memory_utilization < 0.8:
        print("  ✅ Memory usage looks good (< 80%)")
    elif memory_utilization < 0.95:
        print("  ⚠️  High memory usage (80-95%) - monitor closely")
    else:
        print("  ❌ Memory usage too high (> 95%) - likely to OOM")
    
    # Compilation test
    if FLAGS.dry_run:
        print("\n🔧 Testing JAX Compilation...")
        compile_success, compile_message = test_compilation(config, FLAGS.batch_size)
        if compile_success:
            print(f"  ✅ {compile_message}")
        else:
            print(f"  ❌ {compile_message}")
    
    # Summary
    print("\n📋 Summary:")
    overall_compatible = mesh_compatible and memory_utilization < 0.95
    if FLAGS.dry_run:
        overall_compatible = overall_compatible and compile_success
    
    if overall_compatible:
        print("  ✅ Configuration should work on this TPU setup!")
    else:
        print("  ❌ Configuration may have issues on this TPU setup")
        
    print("\n💡 Recommendations:")
    if memory_utilization > 0.8:
        print("  • Consider reducing batch size")
        print("  • Consider using gradient checkpointing")
        print("  • Consider FSDP for larger models")
    if not mesh_compatible:
        print("  • Adjust parallelism settings to match device count")
    
    print(f"\n🔧 For different batch sizes, run:")
    print("  python check_tpu_fit.py --config=<config_name> --batch_size=32")


if __name__ == '__main__':
    jax.config.config_with_absl()
    app.run(main) 