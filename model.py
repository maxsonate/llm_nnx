from typing import Any, Callable, Sequence

import flax.linen as nn
from flax import nnx
import jax.lax as lax
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
import dataclasses
from llm_nnx.configs import default
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
    pe = np.zeros((max_len, d_feature), dtype=np.float32)
    position = np.arange(0, max_len)[:, np.newaxis]
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

