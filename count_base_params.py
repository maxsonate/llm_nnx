import jax
from flax import nnx
from model import TransformerLM
from modules import TransformerConfig
from configs import default
from utils import count_parameters


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


def main():
    """Count parameters in the base config model."""
    # Get the default config
    config = default.get_config()
    
    print("🔧 Base Configuration:")
    print(f"  Vocabulary size: {config.vocab_size:,}")
    print(f"  Embedding dim: {config.emb_dim}")
    print(f"  Number of layers: {config.num_layers}")
    print(f"  Number of heads: {config.num_heads}")
    print(f"  QKV dim: {config.qkv_dim}")
    print(f"  MLP dim: {config.mlp_dim}")
    print(f"  Max sequence length: {config.max_target_length}")
    
    # Create transformer config from default config
    transformer_config = TransformerConfig(
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
    
    print("\n🏗️  Creating model...")
    
    # Initialize model
    rngs = nnx.Rngs(params=jax.random.key(0), dropout=jax.random.key(1))
    model = TransformerLM(transformer_config, decode=False, rngs=rngs)
    
    # Get parameters
    params = nnx.state(model, nnx.Param)
    
    # Count parameters
    total_params, param_breakdown = count_parameters(params)
    
    print(f"\n🔢 Total Parameters: {format_number(total_params)} ({total_params:,})")
    
    print("\n📊 Parameter Breakdown by Component:")
    print("-" * 60)
    
    # Group parameters by major components
    component_totals = {}
    for path, count in param_breakdown.items():
        # Extract the main component name from the path
        if isinstance(path, tuple) and len(path) > 0:
            component = str(path[0])  # First element of the path tuple
        else:
            component = str(path).split('.')[0] if '.' in str(path) else str(path)
        
        component_totals[component] = component_totals.get(component, 0) + count
    
    # Sort by parameter count (descending)
    for component, count in sorted(component_totals.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_params) * 100
        print(f"{component:<20}: {format_number(count):>8} ({percentage:5.1f}%)")
    
    # Calculate some theoretical values for comparison
    print(f"\n🧮 Theoretical Calculations:")
    embedding_params = config.vocab_size * config.emb_dim
    if config.logits_via_embedding:
        output_params = 0  # Shared with embedding
    else:
        output_params = config.emb_dim * config.vocab_size
    
    print(f"  Embedding parameters: ~{format_number(embedding_params)}")
    print(f"  Output head parameters: ~{format_number(output_params)}")
    
    # Per-layer approximation
    # Attention: 4 * (emb_dim * qkv_dim) for Q, K, V, O projections
    attention_params_per_layer = 4 * (config.emb_dim * config.qkv_dim)
    # MLP: 2 * (emb_dim * mlp_dim) for up and down projections  
    mlp_params_per_layer = 2 * (config.emb_dim * config.mlp_dim)
    # LayerNorm: 2 * emb_dim (scale and bias) * 2 (two layer norms per block)
    layernorm_params_per_layer = 4 * config.emb_dim
    
    total_per_layer = attention_params_per_layer + mlp_params_per_layer + layernorm_params_per_layer
    all_layers_params = total_per_layer * config.num_layers
    
    print(f"  ~{format_number(attention_params_per_layer)} attention params per layer")
    print(f"  ~{format_number(mlp_params_per_layer)} MLP params per layer") 
    print(f"  ~{format_number(layernorm_params_per_layer)} LayerNorm params per layer")
    print(f"  ~{format_number(all_layers_params)} total transformer layers")
    
    theoretical_total = embedding_params + output_params + all_layers_params
    print(f"  Theoretical total: ~{format_number(theoretical_total)}")


if __name__ == "__main__":
    main() 