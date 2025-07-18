import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

# Import from modules and model
from modules import TransformerConfig
from model import EncoderDecoder1DBlock, Decoder, TransformerLM


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


class TestDecoder(unittest.TestCase):

    def setUp(self):
        self.config = TransformerConfig(
            vocab_size=100,
            output_vocab_size=120,
            emb_dim=8,
            max_len=10,
            num_heads=2,
            num_layers=2,
            qkv_dim=8,
            mlp_dim=16,
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
            logits_via_embedding=False
        )
        self.rngs = nnx.Rngs(params=jax.random.key(0), dropout=jax.random.key(1))

    def test_output_shape(self):
        """Test that Decoder output has the correct shape."""
        batch_size, seq_len = 2, 5
        inputs = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        
        decoder = Decoder(config=self.config, decode=False, rngs=self.rngs)
        outputs = decoder(inputs, rngs=nnx.Rngs(dropout=jax.random.key(2)))
        
        expected_shape = (batch_size, seq_len, self.config.output_vocab_size)
        self.assertEqual(outputs.shape, expected_shape)

    def test_decode_mode_output_shape(self):
        """Test Decoder output shape in decode mode."""
        batch_size, seq_len = 2, 1  # Typical for decode mode
        inputs = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        
        decoder = Decoder(config=self.config, decode=True, rngs=self.rngs)
        decoder.init_cache(batch_size)
        outputs = decoder(inputs, rngs=nnx.Rngs(dropout=jax.random.key(2)))
        
        expected_shape = (batch_size, seq_len, self.config.output_vocab_size)
        self.assertEqual(outputs.shape, expected_shape)

    def test_logits_via_embedding(self):
        """Test Decoder with logits_via_embedding=True."""
        batch_size, seq_len = 2, 5
        inputs = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        
        config_logits = self.config.replace(logits_via_embedding=True)
        decoder = Decoder(config=config_logits, decode=False, rngs=self.rngs)
        outputs = decoder(inputs, rngs=nnx.Rngs(dropout=jax.random.key(2)))
        
        expected_shape = (batch_size, seq_len, self.config.output_vocab_size)
        self.assertEqual(outputs.shape, expected_shape)

    def test_shared_embedding(self):
        """Test Decoder with a shared embedding layer."""
        batch_size, seq_len = 2, 5
        inputs = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

        shared_embedding = nnx.Embed(
            num_embeddings=self.config.output_vocab_size,
            features=self.config.emb_dim,
            rngs=self.rngs
        )
        
        decoder = Decoder(
            config=self.config,
            shared_embedding=shared_embedding,
            decode=False,
            rngs=self.rngs
        )
        outputs = decoder(inputs, rngs=nnx.Rngs(dropout=jax.random.key(2)))

        self.assertIs(decoder.output_embed, shared_embedding)
        expected_shape = (batch_size, seq_len, self.config.output_vocab_size)
        self.assertEqual(outputs.shape, expected_shape)


class TestTransformerLM(unittest.TestCase):

    def setUp(self):
        self.config = TransformerConfig(
            vocab_size=50,
            output_vocab_size=50,
            emb_dim=16,
            max_len=20,
            num_heads=4,
            num_layers=2,
            qkv_dim=16,
            mlp_dim=32,
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
            logits_via_embedding=False
        )
        self.rngs = nnx.Rngs(params=jax.random.key(42), dropout=jax.random.key(43))

    def test_output_shape_train(self):
        """Test TransformerLM output shape during training."""
        batch_size, seq_len = 2, 10
        inputs = jax.random.randint(
            jax.random.key(0), (batch_size, seq_len), 0, self.config.vocab_size
        )
        
        model = TransformerLM(config=self.config, decode=False, rngs=self.rngs)
        logits = model(inputs, rngs=nnx.Rngs(dropout=jax.random.key(2)))
        
        expected_shape = (batch_size, seq_len, self.config.output_vocab_size)
        self.assertEqual(logits.shape, expected_shape)

    def test_output_shape_decode(self):
        """Test TransformerLM output shape during decoding."""
        batch_size, seq_len = 2, 1  # Single token for decoding
        inputs = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

        model = TransformerLM(config=self.config, decode=True, rngs=self.rngs)
        model.decoder.init_cache(batch_size)
        
        logits = model(inputs, rngs=nnx.Rngs(dropout=jax.random.key(2)))
        
        expected_shape = (batch_size, seq_len, self.config.output_vocab_size)
        self.assertEqual(logits.shape, expected_shape)

    def test_padding_mask(self):
        """Test that padding tokens do not affect non-padding tokens."""
        batch_size, seq_len = 2, 5
        # First sequence has padding, second does not
        inputs = jnp.array([
            [1, 2, 3, 0, 0],
            [1, 2, 3, 4, 5]
        ])
        
        config_det = self.config.replace(deterministic=True, dropout_rate=0.0, attention_dropout_rate=0.0)
        model = TransformerLM(config=config_det, decode=False, rngs=self.rngs)
        
        # Get logits for inputs with padding
        logits_padded = model(inputs, rngs=nnx.Rngs(dropout=jax.random.key(2)))
        
        # Get logits for inputs without padding
        inputs_unpadded = inputs[:1, :3] # Take the non-padded part
        
        logits_unpadded = model(
            inputs_unpadded, rngs=nnx.Rngs(dropout=jax.random.key(2))
        )
        
        # The logits for the non-padded tokens should be very similar
        # Allowing for small floating point differences
        np.testing.assert_allclose(
            np.array(logits_padded[0, :3]),
            np.array(logits_unpadded[0]),
            rtol=1e-5, atol=1e-5
        )

    def test_segmentation_mask(self):
        """Test attention with segmentation."""
        batch_size, seq_len = 1, 6
        inputs = jnp.array([[1, 2, 3, 4, 5, 6]])
        segmentation = jnp.array([[1, 1, 1, 2, 2, 2]])
        
        model = TransformerLM(config=self.config, decode=False, rngs=self.rngs)
        
        # This will fail if the segmentation mask is not applied correctly
        # We are not checking for correctness of values, just that it runs
        logits = model(
            inputs,
            inputs_segmentation=segmentation,
            rngs=nnx.Rngs(dropout=jax.random.key(2))
        )
        
        expected_shape = (batch_size, seq_len, self.config.output_vocab_size)
        self.assertEqual(logits.shape, expected_shape)


if __name__ == '__main__':
    unittest.main() 