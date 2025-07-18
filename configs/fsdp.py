"""Data parallel configuration with 4x batch size - standalone config."""

import dataclasses
from configs.default import MeshRules


@dataclasses.dataclass(unsafe_hash=True)
class DataParallelConfig:
  """Configuration for data parallel training with larger batch size."""
  
  # Dataset and tokenization
  vocab_path: str | None = None
  vocab_size: int = 30_000
  max_corpus_chars: int = 10**7
  dataset_name: str = 'lm1b'
  eval_dataset_name: str = 'lm1b'
  eval_split: str = 'test'
  
  # Batch sizes - 4x larger than default
  per_device_batch_size: int = 128  # Was 32, now 128 (4x)
  eval_per_device_batch_size: int = 128  # Was 32, now 128 (4x)
  
  # Sampling parameters
  sampling_temperature: float = 0.6
  sampling_top_k: int = 20
  
  # Training parameters
  num_train_steps: int = 500_000
  num_eval_steps: int = 2_000
  num_predict_steps: int = -1
  
  # Learning parameters - scaled for larger batch size
  learning_rate: float = 0.0064  # Was 0.0016, scaled 4x for 4x batch size
  warmup_steps: int = 1000
  label_smoothing: float = 0.0
  weight_decay: float = 0.1
  
  # Sequence lengths
  max_target_length: int = 128
  max_eval_target_length: int = 512
  max_predict_length: int = 50
  
  # Model architecture
  logits_via_embedding: bool = False
  num_layers: int = 6
  qkv_dim: int = 512
  emb_dim: int = 512
  mlp_dim: int = 2048
  num_heads: int = 8
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  
  # Checkpointing and evaluation - adjusted for larger batches
  save_checkpoints: bool = True
  restore_checkpoints: bool = True
  checkpoint_every_steps: int = 20_000  # Was 10_000, less frequent
  eval_every_steps: int = 2_000  # Was 1_000, less frequent
  
  # Precision and optimization
  use_bfloat16: bool = True
  seed: int = 0
  
  # Generation
  prompts: str = 'I love to '
  
  # Parallelism configuration - DATA PARALLEL SETUP
  mesh_axes: tuple[str, ...] = ('data', 'fsdp', 'tensor')
  axis_rules: MeshRules = MeshRules(
    embed='fsdp',
    mlp='tensor', 
    kv='tensor',
    vocab='tensor',
  )
  data_sharding: tuple[str, ...] = ('data',)
  
  # DCN parallelism - Data parallel across slices
  dcn_data_parallelism: int = 1  # No data parallelism
  dcn_fsdp_parallelism: int = -1   # Auto-scale FSDP across slices
  dcn_tensor_parallelism: int = 1 # No tensor parallelism across slices
  
  # ICI parallelism - Data parallel within slice
  ici_data_parallelism: int = 1  # No data parallelism
  ici_fsdp_parallelism: int = -1   # Auto-scale FSDP within slice  
  ici_tensor_parallelism: int = 1 # No tensor parallelism within slice

  def replace(self, **kwargs):
    return dataclasses.replace(self, **kwargs)


def get_config():
  """Get the data parallel configuration."""
  return DataParallelConfig()


def print_config_summary():
  """Print a summary of the data parallel configuration."""
  config = get_config()
  
  print("🚀 Data Parallel Configuration Summary")
  print("=" * 50)
  
  print("📦 Batch Configuration:")
  print(f"  Per-device batch size: {config.per_device_batch_size}")
  print(f"  Eval batch size: {config.eval_per_device_batch_size}")
  
  print("\n🧠 Model Architecture:")
  print(f"  Vocabulary size: {config.vocab_size:,}")
  print(f"  Embedding dimension: {config.emb_dim}")
  print(f"  Number of layers: {config.num_layers}")
  print(f"  Number of heads: {config.num_heads}")
  print(f"  MLP dimension: {config.mlp_dim}")
  print(f"  Max sequence length: {config.max_target_length}")
  
  print("\n⚡ Parallelism Strategy:")
  print(f"  DCN data parallelism: {config.dcn_data_parallelism} (auto-scale)")
  print(f"  DCN FSDP parallelism: {config.dcn_fsdp_parallelism} (disabled)")
  print(f"  ICI data parallelism: {config.ici_data_parallelism} (auto-scale)")  
  print(f"  ICI FSDP parallelism: {config.ici_fsdp_parallelism} (disabled)")
  print("  → Pure data parallelism across all devices")
  
  print("\n🎯 Training Configuration:")
  print(f"  Learning rate: {config.learning_rate}")
  print(f"  Warmup steps: {config.warmup_steps:,}")
  print(f"  Total train steps: {config.num_train_steps:,}")
  print(f"  Eval every: {config.eval_every_steps:,} steps")
  print(f"  Checkpoint every: {config.checkpoint_every_steps:,} steps")
  
  print("\n💾 Memory & Performance:")
  print("  Benefits:")
  print("    • 4x larger effective batch size")
  print("    • Simpler gradient-only communication")
  print("    • Better throughput for medium-sized models")
  print("    • More predictable memory usage")
  
  print("  Considerations:")
  print("    • Higher per-device memory usage")
  print("    • Full model copy on each device")
  
  # Calculate effective batch sizes for different device counts
  print("\n📊 Effective Batch Sizes:")
  device_counts = [1, 2, 4, 8]
  for n_devices in device_counts:
    effective_batch = config.per_device_batch_size * n_devices
    print(f"    {n_devices} devices: {effective_batch}")


if __name__ == "__main__":
    print_config_summary() 