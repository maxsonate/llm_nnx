import dataclasses
import functools
import math
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
from flax.nnx.nn import dtypes
from flax.nnx.nn.normalization import LayerNorm
from flax.nnx.module import first_from
from einops import rearrange, repeat, reduce, pack, unpack


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


def combine_masks(
    *masks: Array | None,
    dtype: Dtype = jnp.bool_,
) -> Array | None:
  """Combine multiple attention masks into a single mask.
  
  Args:
    *masks: Variable number of attention masks to combine. None values are ignored.
    dtype: Data type for the combined mask. Defaults to jnp.bool_.
    
  Returns:
    Combined mask using logical AND operation, or None if no valid masks provided.
  """
  # Filter out None masks
  mask_list = [mask for mask in masks if mask is not None]
  if not mask_list:
    return None
  
  # Validate all masks have the same number of dimensions
  first_mask_ndim = mask_list[0].ndim
  mask_ndims = [mask.ndim for mask in mask_list]
  
  assert all(ndim == first_mask_ndim for ndim in mask_ndims), (
    f'All masks should have the same rank. '
    f'Expected: {first_mask_ndim}, got: {mask_ndims}'
  )

  # Combine masks using logical AND
  mask, *rest = mask_list
  for next_mask in rest:
    mask = jnp.logical_and(mask, next_mask)

  return mask.astype(dtype)




def dot_product_attention_weights(
  query: Array,
  key: Array,
  bias: Array | None = None,
  mask: Array | None = None,
  broadcast_dropout: bool = True,
  dropout_rng: Array | None = None,
  dropout_rate: float = 0.0,
  deterministic: bool = False,
  dtype: Dtype | None = None,
  precision: PrecisionLike = None,
  module: nnx.Module | None = None,
  promote_dtype: PromoteDtypeFn = dtypes.promote_dtype,
):
  """Computes dot-product attention weights.

  Args:
    query: query tensor.
    key: key tensor.
    bias: bias for the attention weights. This should be broadcastable to the
      shape `[..., num_heads, query_length, key_length]`.
    mask: boolean mask for the attention weights. This should be broadcastable
      to the shape `[..., num_heads, query_length, key_length]`.
    broadcast_dropout: bool: use a broadcasted dropout along batch dims.
    dropout_rng: JAX PRNGKey: to be used for dropout
    dropout_rate: dropout rate
    deterministic: bool, deterministic or not (to apply dropout)
    dtype: the dtype of the computation (default: infer from inputs)
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    module: the Module that will sow the attention weights.
    promote_dtype: function to promote dtypes.

  Returns:
    Output of shape `[..., num_heads, query_length, key_length]`.
  """
  
  query, key = promote_dtype((query, key), dtype=dtype)

  dtype = query.dtype

  assert query.ndim == key.ndim, 'q, k must have same rank.'
  assert query.shape[:-3] == key.shape[:-3], 'q, k batch dims must match.'
  assert query.shape[-2] == key.shape[-2], 'q, k num_heads must match.'
  assert query.shape[-1] == key.shape[-1], 'q, k depths must match.'  


    # calculate attention matrix  
  depth = query.shape[-1]

  query = query / jnp.sqrt(depth).astype(dtype)

  attn_weights = jnp.einsum('...qhd,...khd->...hqk', query, key, precision=precision)

  if bias is not None:
    attn_weights = attn_weights + bias

  if mask is not None:
    neg_inf = jnp.finfo(dtype).min
    attn_weights = jnp.where(mask, attn_weights, neg_inf)

  
  attn_weights = jax.nn.softmax(attn_weights, axis=-1)
  if module is not None:
    module.sow(nnx.Intermediate, 'attn_weights', attn_weights
          )
    
  if not deterministic and dropout_rate > 0:
    keep_prob = 1 - dropout_rate
  
    if broadcast_dropout:
      dropout_shape = (1,) * (attn_weights.ndim - 2) + attn_weights.shape[-2:]
      keep = jax.random.bernoulli(dropout_rng, keep_prob, dropout_shape)
    else:
      keep = jax.random.bernoulli(dropout_rng, keep_prob, attn_weights.shape)
    
    multiplier = keep.astype(dtype) / jnp.asarray(keep_prob, dtype)

    attn_weights = attn_weights * multiplier

  return attn_weights


def dot_product_attention(
  query: Array,
  key: Array,
  value: Array,
  bias: Array | None = None,
  mask: Array | None = None,
  broadcast_dropout: bool = True,
  dropout_rng: Array | None = None,
  dropout_rate: float = 0.0,
  deterministic: bool = False,
  dtype: Dtype | None = None,
  precision: PrecisionLike = None,
  module: nnx.Module | None = None,
  promote_dtype: PromoteDtypeFn = dtypes.promote_dtype,
):
  """Computes dot-product attention on the given ``query``, ``key`` and
  ``value`` tensors and returns the attention output.

  This is a convenience wrapper around :pyfunc:`dot_product_attention_weights`
  that first computes the attention weights and then applies them to the
  ``value`` tensor via an einsum.  It mirrors the semantics of
  :pyfunc:`jax.nn.dot_product_attention` while providing additional
  functionality such as:

    • Optional dropout with support for broadcasting across batch/head dims.
    • Boolean masking of invalid key positions.
    • Automatic dtype promotion for mixed-precision training.
    • Optional recording (``sow``-ing) of the intermediate attention
      weights for later inspection.

  Args:
    query: Array of shape ``(..., query_length, num_heads, head_dim)`` — the
      queries.
    key: Array of shape ``(..., key_length, num_heads, head_dim)`` — the
      keys.
    value: Array of shape ``(..., key_length, num_heads, head_dim)`` — the
      values.
    bias: Optional array broadcastable to
      ``(..., num_heads, query_length, key_length)`` providing an additive
      bias term (e.g. relative positional bias).
    mask: Optional boolean array broadcastable to the same shape as ``bias``.
      ``True`` entries are kept while ``False`` entries are masked out.
    broadcast_dropout: Whether dropout masks are broadcast across the batch
      and head dimensions.
    dropout_rng: PRNGKey used for stochastic dropout when
      ``deterministic`` is ``False``.
    dropout_rate: Probability of dropping an attention weight.  A value of
      ``0.0`` disables dropout.
    deterministic: Disables dropout when set to ``True`` regardless of
      ``dropout_rate``.
    dtype: Optional computation dtype.  If ``None``, the dtype is inferred
      from the inputs.
    precision: XLA computation precision for the underlying einsum.
    module: Optional ``nnx.Module`` used to record (``sow``) the attention
      weights.
    promote_dtype: Callable that promotes the dtypes of the inputs prior to
      the computation.

  Returns:
    Array of shape ``(..., query_length, num_heads, head_dim)`` containing
    the attended values.
  """

  query, key, value = promote_dtype((query, key, value), dtype=dtype)

  dtype = query.dtype

  assert query.ndim == key.ndim == value.ndim, 'q, k, v must have same rank.'
  assert query.shape[:-3] == key.shape[:-3], 'q, k batch dims must match.'
  assert query.shape[-2] == key.shape[-2], 'q, k num_heads must match.'
  assert query.shape[-1] == key.shape[-1], 'q, k depths must match.'
  assert query.shape[-1] == value.shape[-1], 'q, v depths must match.'

  if dropout_rate == 0 and module is None:
    query_shape = query.shape
    
    if len(query_shape) > 4:
      def reshape_4d(x):
        return jnp.reshape(x, (math.prod(query_shape[:-3]), *query_shape[-3:]))
    
      query, key, value, bias, mask = jax.tree.map(reshape_4d, (query, key, value, bias, mask))

    if mask is not None:
      mask = mask.astype(jnp.bool_)
    
    out = jax.nn.dot_product_attention(query, key, value, bias, mask)

    if len(query_shape) > 4:
      out = jnp.reshape(out, query_shape)

    return out

  attn_weights = dot_product_attention_weights(
    query, key, bias, mask, broadcast_dropout, dropout_rng, dropout_rate, deterministic, dtype, precision, module, promote_dtype
  )

  return jnp.einsum('...hqk,...khd->...qhd', attn_weights, value, precision=precision)


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
    adaptive_layer_norm: Whether to use adaptive layer normalization for query/key.
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
    adaptive_layer_norm: bool = False,
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
    self.adaptive_layer_norm = adaptive_layer_norm
    
    # Combined QKV projection (more efficient than separate projections)
    assert in_features % num_heads == 0, "in_features must be divisible by num_heads"
    head_dim = in_features // num_heads
    
    linear_general = functools.partial(nnx.LinearGeneral,
                                       out_features=(num_heads, head_dim), 
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
      axis=(-2, -1),  # Contract over the last two dimensions (num_heads, head_dim)
      dtype=dtype,
      kernel_init=kernel_init,
      bias_init=bias_init,
      use_bias=use_bias,
      rngs=rngs,
    )

    if adaptive_layer_norm:
      layernorm_cls = AdaptiveLayerNorm
    else:
      layernorm_cls = nnx.LayerNorm

    if normalize_qk:
      self.query_ln = layernorm_cls(head_dim,
                                  dtype=dtype,
                                  param_dtype=param_dtype,
                                  use_bias=False,
                                  rngs=rngs,
                                  )
      self.key_ln = layernorm_cls(head_dim,
                                dtype=dtype,
                                use_bias=False,
                                param_dtype=param_dtype,
                                rngs=rngs,
                                )
    else:
      self.query_ln = None
      self.key_ln = None
    

    self.rngs = rngs if dropout_rate > 0 else None

    self.dropout = nnx.Dropout(rate=dropout_rate, deterministic=deterministic)

    self.cached_key: nnx.Cache[Array] | None = None
    self.cached_value: nnx.Cache[Array] | None = None
    self.cache_index: nnx.Cache[Array] | None = None


  def init_cache(self, input_shape: Shape):
    """Initialize cache for autoregressive decoding.
    
    Args:
      input_shape: Shape of input tensor (batch_size, seq_len, features).
      
    TODO: Implement cache initialization for keys and values.
    """
    self.cached_key = nnx.Cache(jnp.zeros(input_shape[:-1] + (self.num_heads, self.in_features // self.num_heads), dtype=self.dtype))
    self.cached_value = nnx.Cache(jnp.zeros(input_shape[:-1] + (self.num_heads, self.in_features // self.num_heads), dtype=self.dtype))
    self.cache_index = nnx.Cache(jnp.array(0, dtype=jnp.uint32))

  def __call__(
    self,
    inputs_q: Array,
    inputs_k: Array | None = None,
    inputs_v: Array | None = None,
    *,
    mask: Array | None = None,
    deterministic: bool | None = None,
    rngs: nnx.Rngs | None = None,
    sow_weights: bool = False,
    decode: bool | None = None,
    condition: Array | None = None,
  ):
    """Apply multi-head attention.
    
    Args:
      inputs_q: Query input tensor of shape (batch_size, seq_len, features).
      inputs_k: Key input tensor of shape (batch_size, seq_len, features). 
                If None, uses inputs_q.
      inputs_v: Value input tensor of shape (batch_size, seq_len, features). 
                If None, uses inputs_k.
      mask: Attention mask of shape (batch_size, num_heads, seq_len, seq_len).
      deterministic: Whether to use deterministic behavior (disables dropout).
      rngs: Random number generators for dropout.
      sow_weights: Whether to store attention weights for debugging.
      decode: Whether to use autoregressive decoding.
      condition: Conditioning input for adaptive layer normalization (if enabled).
                 If None and adaptive_layer_norm=True, uses global sequence context 
                 (mean pooling of inputs_q).
      
    Returns:
      Output tensor of shape (batch_size, seq_len, features).
      
    TODO: Implement attention computation including:
    - Scaled dot-product attention
    - Multi-head concatenation
    - Output projection
    - Residual connections and dropout
    """

    if rngs is None:
      rngs = self.rngs

    if inputs_k is None:
      if inputs_v is not None:
        raise ValueError(
          '`inputs_k` cannot be None if `inputs_v` is not None. '
          'To have both `inputs_k` and `inputs_v` be the same value, pass in the '
          'value to `inputs_k` and leave `inputs_v` as None.'
        )
      inputs_k = inputs_q
    if inputs_v is None:
      inputs_v = inputs_k
    
    if inputs_q.shape[-1] != self.in_features:
      raise ValueError(f"inputs_q.shape[-1] ({inputs_q.shape[-1]}) != in_features ({self.in_features})")
    if inputs_k.shape[-1] != self.in_kv_features:
      raise ValueError(f"inputs_k.shape[-1] ({inputs_k.shape[-1]}) != in_kv_features ({self.in_kv_features})")
    if inputs_v.shape[-1] != self.in_kv_features:
      raise ValueError(f"inputs_v.shape[-1] ({inputs_v.shape[-1]}) != in_kv_features ({self.in_kv_features})")
    
  
    query = self.query(inputs_q)
    key = self.key(inputs_k)
    value = self.value(inputs_v)

    if self.normalize_qk:
      # Apply layer normalization to query and key after projection
      if self.adaptive_layer_norm:
        # Auto-generate condition from global sequence context if not provided
        if condition is None:
          condition = jnp.mean(query, axis=1, keepdims=True)  # (batch, 1, features)

        query = self.query_ln(query, condition=condition)
        key = self.key_ln(key, condition=condition)
      else:
        query = self.query_ln(query)
        key = self.key_ln(key)


    decode = first_from(decode,
                        self.decode,
                        error_msg='no decode argument provided either in __call__ or in init')
    
    # TBD: Add decoding, if decode is True.
    if decode:
      if (
        self.cached_key is None
        or self.cached_value is None
        or self.cache_index is None
      ):
        raise ValueError(
          'Autoregressive cache not initialized, call ``init_cache`` first.'
        )
      (
        *batch_dims,
        max_length,
        num_heads,
        depth_per_head,
      ) = self.cached_key.value.shape
      # shape check of cached keys against query input
      expected_shape = tuple(batch_dims) + (1, num_heads, depth_per_head)
      if expected_shape != query.shape:
        raise ValueError(
          'Autoregressive cache shape error, '
          'expected query shape %s instead got %s.'
          % (expected_shape, query.shape)
        )
      # update key, value caches with our new 1d spatial slices
      cur_index = self.cache_index.value
      zero = jnp.array(0, dtype=lax.dtype(cur_index.dtype))
      indices = (zero,) * len(batch_dims) + (cur_index, zero, zero)
      key = lax.dynamic_update_slice(self.cached_key.value, key, indices)
      value = lax.dynamic_update_slice(self.cached_value.value, value, indices)
      self.cached_key.value = key
      self.cached_value.value = value
      self.cache_index.value += 1
      # causal mask for cached decoder self-attention:
      # our single query position should only attend to those key
      # positions that have already been generated and cached,
      # not the remaining zero elements.
      mask = combine_masks(
        mask,
        jnp.broadcast_to(
          jnp.arange(max_length) <= cur_index,
          tuple(batch_dims) + (1, 1, max_length),
        ),
      )


    if self.dropout_rate > 0:
      deterministic = first_from(deterministic,
                                self.deterministic,
                                error_msg='no deterministic argument provided either in __call__ or in init')
      
      # Only generate dropout RNGs when actually needed (non-deterministic mode)
      # This avoids unnecessary RNG operations during inference/evaluation
      if not deterministic:
        # Ensure RNGs are provided when dropout is active - dropout requires randomness
        if rngs is None:
          raise ValueError('rngs must be provided if dropout_rate > 0 and deterministic is False')
        dropout_rngs = rngs.dropout()
      else:
        # In deterministic mode, no dropout RNGs needed as dropout is disabled
        dropout_rngs = None
      
    else:
      # When dropout_rate is 0, bypass all dropout logic for efficiency
      deterministic = False  # Value doesn't matter since dropout_rate is 0
      dropout_rngs = None
    
    attn = self.attention_fn(query, key, value,       
                             mask=mask,
                            dropout_rng=dropout_rngs,
                            dropout_rate=self.dropout_rate,
                            broadcast_dropout=self.broadcast_dropout,
                            deterministic=deterministic,
                            dtype=self.dtype,
                            precision=self.precision,
                            module=self if sow_weights else None,)

    

    output = self.out_proj(attn)

    return output


def default(value, x):
  if value is None:
    return x() if callable(x) else x
  return value

class AdaptiveLayerNorm(nnx.Module):
  """Adaptive Layer Normalization.
  
  Applies layer normalization with learnable scaling conditioned on external input.
  
  Args:
    dim: Dimension of the input features to be normalized.
    dim_condition: Dimension of the conditioning input. If None, defaults to dim.
    epsilon: Small value to avoid division by zero.
    dtype: Computation dtype.
    param_dtype: Parameter dtype.
    use_bias: Whether to use bias in the underlying LayerNorm.
    use_scale: Whether to use scale in the underlying LayerNorm.
    bias_init: Bias initializer for LayerNorm.
    scale_init: Scale initializer for LayerNorm.
    reduction_axes: Axes for normalization statistics.
    feature_axes: Feature axes for learned parameters.
    axis_name: Axis name for parallel computation.
    axis_index_groups: Groups for parallel computation.
    use_fast_variance: Whether to use fast variance computation.
    rngs: Random number generators for parameter initialization.
  """
  def __init__(self, 
               dim, 
               dim_condition=None, 
               *,
               epsilon: float = 1e-6,
               dtype=None,
               param_dtype=jnp.float32,
               use_bias: bool = False,
               use_scale: bool = False,
               bias_init=nnx.initializers.zeros_init(),
               scale_init=nnx.initializers.ones_init(),
               reduction_axes: int = -1,
               feature_axes: int = -1,
               axis_name=None,
               axis_index_groups=None,
               use_fast_variance: bool = True,
               rngs: nnx.Rngs):
    dim_condition = default(dim_condition, dim)

    self.ln = nnx.LayerNorm(
      num_features=dim,
      epsilon=epsilon,
      dtype=dtype,
      param_dtype=param_dtype,
      use_bias=use_bias,
      use_scale=use_scale,
      bias_init=bias_init,
      scale_init=scale_init,
      reduction_axes=reduction_axes,
      feature_axes=feature_axes,
      axis_name=axis_name,
      axis_index_groups=axis_index_groups,
      use_fast_variance=use_fast_variance,
      rngs=rngs
    )
    self.to_gamma = nnx.Linear(
      dim_condition, 
      dim, 
      use_bias=False, 
      kernel_init=nnx.initializers.xavier_uniform(), 
      rngs=rngs
    )

  def __call__(self, x, *, condition):
    if condition.ndim == 2:
      condition = rearrange(condition, 'b d -> b 1 d')

    normed = self.ln(x)
    gamma = self.to_gamma(condition)

    return normed * (gamma + 1.)

# TODO: Train a model with adaptive layer normalization