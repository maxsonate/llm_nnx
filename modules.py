import dataclasses
import functools
from typing import Any, Callable, Sequence

import flax.linen as nn
from flax import nnx, struct
from flax.linen import initializers
from flax.typing import (
  PromoteDtypeFn,
  Initializer,
  PrecisionLike,
  DotGeneralT,
)
import jax
import jax.lax as lax
import jax.numpy as jnp
from jax import Array
import numpy as np
from flax.nnx.nn.normalization import LayerNorm
from flax.nnx.nn.attention import dot_product_attention

from configs import default

Shape = Sequence[int]
Dtype = Any


@dataclasses.dataclass(unsafe_hash=True)
class TransformerConfig:
  """Global hyperparameters used to minimize obnoxious kwarg plumbing."""

  vocab_size: int
  output_vocab_size: int
  logits_via_embedding: bool = False
  dtype: Any = jnp.float32
  emb_dim: int = 512
  num_heads: int = 8
  num_layers: int = 6
  qkv_dim: int = 512
  mlp_dim: int = 2048
  max_len: int = 2048
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  kernel_init: nnx.Initializer = nnx.initializers.xavier_uniform()
  bias_init: nnx.Initializer = nnx.initializers.normal(stddev=1e-6)
  posemb_init: nnx.Initializer | None = None
  deterministic: bool = False
  axis_rules: default.MeshRules = dataclasses.field(
    default_factory=default.MeshRules
  )

  def replace(self, **kwargs):
    return dataclasses.replace(self, **kwargs)


def shift_right(x, axis=1):
  """Shift the input to the right by padding and slicing on axis."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[axis] = (1, 0)
  padded = jnp.pad(
      x, pad_widths, mode='constant', constant_values=x.dtype.type(0)
  )
  return lax.dynamic_slice_in_dim(padded, 0, padded.shape[axis] - 1, axis)


def shift_inputs(x, segment_ids=None, axis=1):
  """Shift inputs and replace EOS by 0 for packed inputs."""
  shifted = shift_right(x, axis=axis)
  # For packed targets, the first shifted token of a new sequence is made
  # 0, rather than being the EOS token for the last sequence.
  if segment_ids is not None:
    shifted *= segment_ids == shift_right(segment_ids, axis=axis)
  return shifted


def sinusoidal_init(max_len=2048, min_scale=1.0, max_scale=10000.0):
  """1D Sinusoidal Position Embedding Initializer.

  Args:
      max_len: maximum possible length for the input.
      min_scale: float: minimum frequency-scale in sine grating.
      max_scale: float: maximum frequency-scale in sine grating.

  Returns:
      output: init function returning `(1, max_len, d_feature)`
  """

  def init(key, shape, dtype=np.float32):
    """Sinusoidal init."""
    del key, dtype
    d_feature = shape[-1]
    
    # Validate minimum d_feature size to prevent division by zero
    if d_feature < 2:
      raise ValueError(f"d_feature must be at least 2, got {d_feature}")
    
    pe = np.zeros((max_len, d_feature), dtype=np.float32)
    position = np.arange(0, max_len)[:, np.newaxis]
    
    # Handle special case when d_feature = 2 to avoid division by zero
    if d_feature == 2:
      # For d_feature=2, we have only one frequency component
      div_term = np.array([min_scale])
    else:
      scale_factor = -np.log(max_scale / min_scale) / (d_feature // 2 - 1)
      div_term = min_scale * np.exp(np.arange(0, d_feature // 2) * scale_factor)
    
    pe[:, : d_feature // 2] = np.sin(position * div_term)
    pe[:, d_feature // 2 : 2 * (d_feature // 2)] = np.cos(position * div_term)
    pe = pe[np.newaxis, :, :]  # [1, max_len, d_feature]
    return jnp.array(pe)

  return init



class AddPositionEmbs(nnx.Module):
  """Adds (optionally learned) positional embeddings to the inputs.

  Args:
    config: TransformerConfig dataclass containing hyperparameters.
  """

  def __init__(
    self,
    config: TransformerConfig,
    *,
    decode: bool = False,
    rngs: nnx.Rngs,
  ):
    self.config = config
    self.decode = decode
    self.pos_emb_shape = (1, config.max_len, config.emb_dim)

    if config.posemb_init is not None:
      self.pos_embedding = nnx.Param(
        config.posemb_init(rngs.params(), self.pos_emb_shape)
      )
    else:
      self.pos_embedding = None

  def __call__(self, inputs: jax.Array, inputs_positions=None):
    """Applies AddPositionEmbs module.

    By default this layer uses a fixed sinusoidal embedding table. If a
    learned position embedding is desired, pass an initializer to
    posemb_init in the configuration.

    Args:
      inputs: input data.
      inputs_positions: input position indices for packed sequences.

    Returns:
      output: `(bs, timesteps, in_dim)`
    """
    config = self.config
    # inputs.shape is (batch_size, seq_len, emb_dim)
    assert inputs.ndim == 3, (
      'Number of dimensions should be 3, but it is: %d' % inputs.ndim
    )
    length = inputs.shape[1]

    if self.pos_embedding is None:
      # Use a fixed (non-learned) sinusoidal position embedding.
      pos_embedding = sinusoidal_init(max_len=config.max_len)(
        None, self.pos_emb_shape
      )
    else:
      pos_embedding = self.pos_embedding.value

    # We use a cache position index for tracking decoding position.
    if self.decode:
      # Ensure cache is initialized when in decode mode
      if not hasattr(self, 'cache_index'):
        raise RuntimeError("Cache not initialized. Call init_cache() before using decode mode.")
      
      _, _, df = pos_embedding.shape
      # equivalent to pos_embedding[:, i:i+1] but traceable
      pos_embedding = lax.dynamic_slice(
        pos_embedding, jnp.array((0, self.cache_index.value, 0)), (1, 1, df)
      )
      self.cache_index.value += 1
    else:
      pos_embedding = pos_embedding[:, :length, :]

    if inputs_positions is None:
      # normal unpacked case:
      return inputs + pos_embedding
    else:
      # for packed data we need to use known position indices:
      return inputs + jnp.take(pos_embedding[0], inputs_positions, axis=0)

  def init_cache(self, input_shape: Shape, dtype: Dtype = jnp.float32):
    self.cache_index = nnx.Cache(jnp.array(0, dtype=jnp.uint32))



class MlpBlock(nnx.Module):
  """Transformer MLP / feed-forward block.

  Args:
    config: TransformerConfig dataclass containing hyperparameters.
    out_dim: optionally specify out dimension.
  """

  def __init__(self, config: TransformerConfig, *, rngs: nnx.Rngs):
    self.config = config

    self.linear1 = nnx.Linear(
      config.emb_dim,
      config.mlp_dim,
      dtype=config.dtype,
      kernel_init=nnx.with_partitioning(
        config.kernel_init,
        config.axis_rules('embed', 'mlp'),
      ),
      bias_init=nnx.with_partitioning(
        config.bias_init,
        config.axis_rules('mlp'),
      ),
      rngs=rngs,
    )
    self.linear2 = nnx.Linear(
      config.mlp_dim,
      config.emb_dim,
      dtype=config.dtype,
      kernel_init=nnx.with_partitioning(
        config.kernel_init,
        config.axis_rules('mlp', 'embed'),
      ),
      bias_init=nnx.with_partitioning(
        config.bias_init,
        config.axis_rules('embed'),
      ),
      rngs=rngs,
    )
    self.dropout = nnx.Dropout(rate=config.dropout_rate, deterministic=config.deterministic)

  def __call__(self, inputs: jax.Array, *, rngs: nnx.Rngs | None = None):
    """Applies Transformer MlpBlock module."""
    x = self.linear1(inputs)
    x = nnx.relu(x)
    x = self.dropout(x, rngs=rngs)
    output = self.linear2(x)
    output = self.dropout(output, rngs=rngs)
    return output


class MultiHeadAttention(nnx.Module):
  """Custom Multi-Head Attention implementation.
  
  Args:
    num_heads: Number of attention heads.
    in_features: Input feature dimension.
    qkv_features: Query/key/value feature dimension.
    dtype: Data type for computations.
    kernel_init: Kernel weight initializer.
    bias_init: Bias initializer.
    use_bias: Whether to use bias terms.
    broadcast_dropout: Whether to broadcast dropout across dimensions.
    dropout_rate: Dropout rate for attention weights.
    deterministic: Whether to use deterministic behavior.
    decode: Whether to enable autoregressive decoding.
    rngs: Random number generators.
  """

  def __init__(
    self,
    num_heads: int,
    in_features: int,
    qkv_features: int | None = None,
    out_features: int | None = None,
    in_kv_features: int | None = None,
    *,
    dtype: Dtype | None = None,
    param_dtype: Dtype = jnp.float32,
    broadcast_dropout: bool = True,
    dropout_rate: float = 0.0,
    deterministic: bool | None = None,
    precision: PrecisionLike = None,
    kernel_init: Initializer = nnx.nn.linear.default_kernel_init,
    out_kernel_init: Initializer | None = None,
    bias_init: Initializer = initializers.zeros_init(),
    out_bias_init: Initializer | None = None,
    use_bias: bool = True,
    attention_fn: Callable[..., Array] = dot_product_attention,
    decode: bool | None = None,
    normalize_qk: bool = False,
    rngs: nnx.Rngs,
  ):
    self.num_heads = num_heads
    self.in_features = in_features
    self.qkv_features = (
      qkv_features if qkv_features is not None else in_features
    )
    self.out_features = (
      out_features if out_features is not None else in_features
    )
    self.in_kv_features = (
      in_kv_features if in_kv_features is not None else in_features
    )
    self.dtype = dtype
    self.param_dtype = param_dtype
    self.broadcast_dropout = broadcast_dropout
    self.dropout_rate = dropout_rate
    self.deterministic = deterministic
    self.precision = precision
    self.kernel_init = kernel_init
    self.out_kernel_init = out_kernel_init
    self.bias_init = bias_init
    self.out_bias_init = out_bias_init
    self.use_bias = use_bias
    self.attention_fn = attention_fn
    self.decode = decode
    self.normalize_qk = normalize_qk

    
    # Combined QKV projection (more efficient than separate projections)
    assert in_features % num_heads == 0, "in_features must be divisible by num_heads"
    head_dim = in_features // num_heads
    
    linear_general = functools.partial(nnx.LinearGeneral,
                                       in_features=in_features,
      out_features=(num_heads,head_dim), 
      dtype=dtype,
      kernel_init=kernel_init,
      bias_init=bias_init,
      use_bias=use_bias,
      rngs=rngs,
    )
    self.query = linear_general(self.in_features)
    self.key = linear_general(self.in_kv_features)
    self.value = linear_general(self.in_kv_features)


    
    self.out_proj = nnx.LinearGeneral(
      in_features=(num_heads, head_dim),
      out_features=self.out_features,
      dtype=dtype,
      kernel_init=kernel_init,
      bias_init=bias_init,
      use_bias=use_bias,
      rngs=rngs,
    )

    self.query_ln = LayerNorm(in_features=head_dim,
                              dtype=dtype,
                              param_dtype=param_dtype,
                              use_bias=False,
                              rngs=rngs,
                              )
    self.key_ln = LayerNorm(in_features=head_dim,
                            dtype=dtype,
                            use_bias=False,
                            param_dtype=param_dtype,
                            rngs=rngs,
                            )
    

    self.rngs = rngs if dropout_rate > 0 else None

    self.dropout = nnx.Dropout(rate=dropout_rate, deterministic=deterministic)

    self.cache_key: nnx.Cache[Array] | None = None
    self.cache_value: nnx.Cache[Array] | None = None
    self.cache_index: nnx.Cache[Array] | None = None


  def init_cache(self, input_shape: Shape):
    """Initialize cache for autoregressive decoding.
    
    Args:
      input_shape: Shape of input tensor (batch_size, seq_len, features).
      
    TODO: Implement cache initialization for keys and values.
    """
    pass

  def __call__(
    self,
    inputs: jax.Array,
    *,
    mask: jax.Array | None = None,
    decode: bool | None = None,
    rngs: nnx.Rngs | None = None,
  ) -> jax.Array:
    """Apply multi-head attention.
    
    Args:
      inputs: Input tensor of shape (batch_size, seq_len, features).
      mask: Attention mask of shape (batch_size, num_heads, seq_len, seq_len).
      decode: Whether to use autoregressive decoding.
      rngs: Random number generators for dropout.
      
    Returns:
      Output tensor of shape (batch_size, seq_len, features).
      
    TODO: Implement attention computation including:
    - Scaled dot-product attention
    - Multi-head concatenation
    - Output projection
    - Residual connections and dropout
    """
    # Combined QKV projection and split

    
    # TODO: Implement scaled dot-product attention
    # TODO: Apply mask if provided
    # TODO: Handle decode mode with caching
    # TODO: Apply dropout
    # TODO: Concatenate heads and apply output projection
    
    # Placeholder: return inputs unchanged for now
    return inputs 