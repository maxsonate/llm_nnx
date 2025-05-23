from flax import nnx
import jax
import jax.numpy as jnp
from modules import TransformerConfig, MlpBlock


class EncoderDecoder1DBlock(nnx.Module):
  """Transformer encoder-decoder layer.

  Args:
    config: TransformerConfig dataclass containing hyperparameters.
  """

  def __init__(self, config: TransformerConfig, *, decode: bool = False, rngs: nnx.Rngs):
    self.config = config
    self.decode = decode # Store decode flag
    
    # If decoding, sub-modules should operate deterministically
    _deterministic = config.deterministic or decode

    self.ln1 = nnx.LayerNorm(
      num_features=config.emb_dim,
      dtype=config.dtype,
      bias_init=nnx.with_partitioning(
        nnx.initializers.zeros_init(),
        config.axis_rules('embed'),
      ),
      scale_init=nnx.with_partitioning(
        nnx.initializers.ones_init(),
        config.axis_rules('embed'),
      ),
      rngs=rngs,
    )
    self.ln2 = nnx.LayerNorm(
      num_features=config.emb_dim,
      dtype=config.dtype,
      bias_init=nnx.with_partitioning(
        nnx.initializers.zeros_init(),
        config.axis_rules('embed'),
      ),
      scale_init=nnx.with_partitioning(
        nnx.initializers.ones_init(),
        config.axis_rules('embed'),
      ),
      rngs=rngs,
    )
    self.attention = nnx.MultiHeadAttention(
      num_heads=config.num_heads,
      in_features=config.emb_dim,
      qkv_features=config.qkv_dim,
      dtype=config.dtype,
      kernel_init=nnx.with_partitioning(
        config.kernel_init, config.axis_rules('embed', 'kv')
      ),
      bias_init=nnx.with_partitioning(
        config.bias_init, config.axis_rules('embed') # Assuming bias, if used, aligns with 'embed'
      ),
      use_bias=False, # As per original snippet for this block
      broadcast_dropout=False, # As per original snippet
      dropout_rate=config.attention_dropout_rate,
      deterministic=_deterministic, # Set deterministic for MHA's internal dropout
      decode=decode, # Set decode flag for attention
      rngs=rngs,
    )
    # Pass updated config to MlpBlock for its deterministic state
    mlp_config = config.replace(deterministic=_deterministic)
    self.mlp = MlpBlock(config=mlp_config, rngs=rngs)
    self.dropout = nnx.Dropout(rate=config.dropout_rate, deterministic=_deterministic)

  def init_cache(self, input_shape):
    """Initialize cache for autoregressive decoding.
    
    Args:
      input_shape: Shape of the input tensor (batch_size, seq_len, features)
    """
    if self.decode:
      self.attention.init_cache(input_shape)

  def __call__(
    self,
    inputs: jax.Array,
    *,
    decoder_mask: jax.Array | None = None,
    rngs: nnx.Rngs | None = None,
  ):
    """Applies EncoderDecoder1DBlock module.

    Args:
      inputs: input data for decoder
      decoder_mask: decoder self-attention mask.

    Returns:
      output after transformer encoder-decoder block.
    """
    # Decoder block.
    assert inputs.ndim == 3
    x = self.ln1(inputs)
    x = self.attention(x, mask=decoder_mask, decode=self.decode, rngs=rngs) # Pass self.decode here
    x = self.dropout(x, rngs=rngs)
    x = x + inputs
    # MLP block.
    z = self.ln2(x)
    z = self.mlp(z, rngs=rngs)
    return x + z
