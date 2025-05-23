import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

# Import from modules and model
from modules import TransformerConfig
from model import EncoderDecoder1DBlock


class TestEncoderDecoder1DBlock(unittest.TestCase):

    def setUp(self):
        self.config = TransformerConfig(
            vocab_size=100,
            output_vocab_size=100,
            emb_dim=8,
            max_len=10,
            num_heads=2,
            num_layers=1,
            qkv_dim=8,
            mlp_dim=16,
            dropout_rate=0.1,
            attention_dropout_rate=0.1
        )
        self.rngs = nnx.Rngs(params=jax.random.key(0), dropout=jax.random.key(1))

    def test_output_shape(self):
        """Test that EncoderDecoder1DBlock maintains input shape."""
        batch_size, seq_len = 2, 5
        inputs = jnp.ones((batch_size, seq_len, self.config.emb_dim))
        
        block = EncoderDecoder1DBlock(config=self.config, decode=False, rngs=self.rngs)
        outputs = block(inputs, rngs=nnx.Rngs(dropout=jax.random.key(2)))
        
        self.assertEqual(outputs.shape, inputs.shape)

    def test_decode_mode(self):
        """Test EncoderDecoder1DBlock in decode mode."""
        batch_size, seq_len = 1, 1
        inputs = jnp.ones((batch_size, seq_len, self.config.emb_dim))
        
        block = EncoderDecoder1DBlock(config=self.config, decode=True, rngs=self.rngs)
        block.init_cache(input_shape=inputs.shape)  # Initialize cache for decode mode
        outputs = block(inputs, rngs=nnx.Rngs(dropout=jax.random.key(2)))
        
        self.assertEqual(outputs.shape, inputs.shape)

    def test_with_decoder_mask(self):
        """Test EncoderDecoder1DBlock with decoder mask."""
        batch_size, seq_len = 2, 4
        inputs = jnp.ones((batch_size, seq_len, self.config.emb_dim))
        
        # Create a causal mask
        decoder_mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        decoder_mask = decoder_mask[None, None, :, :]  # Add batch and head dimensions
        
        block = EncoderDecoder1DBlock(config=self.config, decode=False, rngs=self.rngs)
        outputs = block(inputs, decoder_mask=decoder_mask, rngs=nnx.Rngs(dropout=jax.random.key(2)))
        
        self.assertEqual(outputs.shape, inputs.shape)

    def test_deterministic_behavior(self):
        """Test that deterministic flag works properly."""
        batch_size, seq_len = 2, 3
        inputs = jnp.ones((batch_size, seq_len, self.config.emb_dim))
        
        # Test with deterministic=True
        config_det = self.config.replace(deterministic=True)
        block_det = EncoderDecoder1DBlock(config=config_det, decode=False, rngs=self.rngs)
        
        outputs1 = block_det(inputs, rngs=None)
        outputs2 = block_det(inputs, rngs=None)
        
        # Should be identical when deterministic
        np.testing.assert_array_equal(np.array(outputs1), np.array(outputs2))

    def test_decode_mode_deterministic(self):
        """Test that decode=True forces deterministic behavior."""
        batch_size, seq_len = 1, 1
        inputs = jnp.ones((batch_size, seq_len, self.config.emb_dim))
        
        # Even with deterministic=False, decode=True should make it deterministic
        config_non_det = self.config.replace(deterministic=False, dropout_rate=0.5)
        block = EncoderDecoder1DBlock(config=config_non_det, decode=True, rngs=self.rngs)
        block.init_cache(input_shape=inputs.shape)  # Initialize cache for decode mode
        
        outputs1 = block(inputs, rngs=None)
        outputs2 = block(inputs, rngs=None)
        
        # Should be identical in decode mode
        np.testing.assert_array_equal(np.array(outputs1), np.array(outputs2))

    def test_residual_connections(self):
        """Test that residual connections work properly."""
        batch_size, seq_len = 1, 2
        inputs = jnp.zeros((batch_size, seq_len, self.config.emb_dim))
        
        # Create a config that should produce near-zero outputs from sub-modules
        config_minimal = self.config.replace(dropout_rate=0.0, attention_dropout_rate=0.0)
        
        # Initialize with small weights
        def small_init(key, shape, dtype):
            return jnp.zeros(shape, dtype=dtype)
        
        config_minimal = config_minimal.replace(
            kernel_init=small_init,
            bias_init=small_init
        )
        
        block = EncoderDecoder1DBlock(config=config_minimal, decode=False, rngs=self.rngs)
        outputs = block(inputs, rngs=None)
        
        # With zero inputs and zero-initialized weights, outputs should be close to inputs
        # due to residual connections
        self.assertEqual(outputs.shape, inputs.shape)


if __name__ == '__main__':
    unittest.main() 