from flax import nnx
import jax
import jax.numpy as jnp
from modules import TransformerConfig, MlpBlock, MultiHeadAttention, AddPositionEmbs, shift_inputs


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
    self.attention = MultiHeadAttention(
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
      adaptive_layer_norm=config.adaptive_layer_norm,
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


class Decoder(nnx.Module):
  """Transformer Model Decoder for sequence to sequence translation.

  Args:
    config: TransformerConfig dataclass containing hyperparameters.
    shared_embedding: a shared embedding layer to use.
  """

  def __init__(
    self,
    config: TransformerConfig,
    shared_embedding: nnx.Embed | None = None,
    *,
    decode: bool = False,
    rngs: nnx.Rngs,
  ):
    self.config = config
    self.decode = decode
    self.shared_embedding = shared_embedding

    # Target Embedding
    if self.shared_embedding is None:
      self.output_embed = nnx.Embed(
        num_embeddings=config.output_vocab_size,
        features=config.emb_dim,
        embedding_init=nnx.with_partitioning(
          nnx.initializers.normal(stddev=1.0),
          config.axis_rules('vocab', 'embed'),
        ),
        rngs=rngs,
      )
    else:
      self.output_embed = self.shared_embedding

    self.posembed_output = AddPositionEmbs(config=config, rngs=rngs)
    self.dropout = nnx.Dropout(rate=config.dropout_rate)
    for idx in range(config.num_layers):
      layer = EncoderDecoder1DBlock(
        config=config, decode=decode, rngs=rngs
      )
      setattr(self, f'encoderdecoderblock_{idx}', layer)

    self.encoderdecoder_norm = nnx.LayerNorm(
      num_features=config.emb_dim,
      dtype=config.dtype,
      bias_init=nnx.with_partitioning(
        nnx.initializers.zeros_init(), config.axis_rules('embed')
      ),
      scale_init=nnx.with_partitioning(
        nnx.initializers.ones_init(), config.axis_rules('embed')
      ),
      rngs=rngs,
    )
    if not config.logits_via_embedding:
      self.logitdense = nnx.Linear(
        in_features=config.emb_dim,
        out_features=config.output_vocab_size,
        dtype=config.dtype,
        kernel_init=nnx.with_partitioning(
          config.kernel_init, config.axis_rules('embed', 'vocab')
        ),
        bias_init=nnx.with_partitioning(
          config.bias_init, config.axis_rules('vocab')
        ),
        rngs=rngs,
      )
    else:
      self.logitdense = None

  def init_cache(self, batch_size):
    """Initialize cache for autoregressive decoding."""
    if self.decode:
      # The shape of the tensor before it enters the attention layers.
      input_shape = (batch_size, self.config.max_len, self.config.emb_dim)
      for idx in range(self.config.num_layers):
        layer = getattr(self, f'encoderdecoderblock_{idx}')
        layer.init_cache(input_shape)

  def __call__(
    self,
    inputs,
    *,
    inputs_positions=None,
    inputs_segmentation=None,
    decoder_mask=None,
    rngs: nnx.Rngs | None = None,
  ):
    """Applies Transformer model on the inputs.

    Args:
      inputs: input data.
      inputs_positions: input subsequence positions for packed examples.
      inputs_segmentation: input segmentation info for packed examples.
      decoder_mask: decoder self-attention mask.

    Returns:
      output of a transformer decoder.
    """
    config = self.config
    assert inputs.ndim == 2  # (batch, len)

    y = inputs.astype('int32')
    if not self.decode:
      y = shift_inputs(y, segment_ids=inputs_segmentation)
    y = self.output_embed(y)
    y = self.posembed_output(y, inputs_positions=inputs_positions)
    y = self.dropout(y, rngs=rngs)

    y = y.astype(config.dtype)

    # Target-Input Decoder
    for idx in range(config.num_layers):
      # TODO(cgarciae): use a list of layers instead of getattr
      layer: EncoderDecoder1DBlock = getattr(self, f'encoderdecoderblock_{idx}')
      y = layer(
        y,
        decoder_mask=decoder_mask,
        rngs=rngs,
      )
    y = self.encoderdecoder_norm(y)

    # Decoded Logits
    if self.logitdense:
      logits = self.logitdense(y)
    else:
      # Use the transpose of embedding matrix for logit transform.
      logits = self.output_embed.attend(y.astype(jnp.float32))
      # Correctly normalize pre-softmax logits for this shared case.
      logits = logits / jnp.sqrt(y.shape[-1])
    return logits


class TransformerLM(nnx.Module):
  """Transformer pure decoder stack for language modelling.

  Args:
    config: TransformerConfig dataclass containing hyperparameters.
  """

  def __init__(
    self, config: TransformerConfig, *, decode: bool = False, rngs: nnx.Rngs
  ):
    self.config = config
    self.decode = decode
    self.decoder = Decoder(config=config, shared_embedding=None, decode=decode, rngs=rngs)

  def __call__(
    self,
    inputs,
    *,
    inputs_positions=None,
    inputs_segmentation=None,
    rngs: nnx.Rngs | None = None,
  ):
    """Applies TransformerLM on the inputs.

    Args:
      inputs: target data.
      inputs_positions: input subsequence positions for packed examples.
      inputs_segmentation: input segmentation info for packed examples.

    Returns:
      logits array from transformer decoder.
    """
    config = self.config

    # Make padding attention masks.
    if self.decode:
      # for fast autoregressive decoding we use no decoder mask
      decoder_mask = None
    else:
      decoder_mask = nnx.combine_masks(
        nnx.make_attention_mask(inputs > 0, inputs > 0, dtype=config.dtype), # This is the padding mask
        nnx.make_causal_mask(inputs, dtype=config.dtype),
      )

    # Add segmentation block-diagonal attention masks if using segmented data.
    if inputs_segmentation is not None:
      decoder_mask = nnx.combine_masks(
        decoder_mask,
        nnx.make_attention_mask(
          inputs_segmentation,
          inputs_segmentation,
          jnp.equal,
          dtype=config.dtype,
        ),
      )

    logits = self.decoder(
      inputs,
      inputs_positions=inputs_positions,
      inputs_segmentation=inputs_segmentation,
      decoder_mask=decoder_mask,
      rngs=rngs,
    )
    return logits.astype(self.config.dtype)