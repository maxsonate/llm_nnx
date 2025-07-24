import unittest
import jax # Added for jax.random.key
import jax.numpy as jnp
import numpy as np # Kept for existing tests
from flax import nnx # For new tests

# Import from modules.py
from modules import (
    shift_right, shift_inputs, sinusoidal_init, 
    TransformerConfig, AddPositionEmbs, MlpBlock, MultiHeadAttention, Shape,
    dot_product_attention_weights, dot_product_attention, combine_masks,
    AdaptiveLayerNorm,
)

class TestModelFunctions(unittest.TestCase):

    def test_shift_right(self):
        x = jnp.array([[1, 2, 3], [4, 5, 6]])
        expected_output = jnp.array([[0, 1, 2], [0, 4, 5]])
        output = shift_right(x, axis=1)
        np.testing.assert_array_equal(np.array(output), np.array(expected_output))

        x_3d = jnp.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]) # (2, 2, 2)
        # Shift along axis 1
        expected_output_axis1 = jnp.array([[[0, 0], [1, 2]], [[0, 0], [5, 6]]])
        output_axis1 = shift_right(x_3d, axis=1)
        np.testing.assert_array_equal(np.array(output_axis1), np.array(expected_output_axis1))
        
        # Shift along axis 2
        expected_output_axis2 = jnp.array([[[0, 1], [0, 3]], [[0, 5], [0, 7]]])
        output_axis2 = shift_right(x_3d, axis=2)
        np.testing.assert_array_equal(np.array(output_axis2), np.array(expected_output_axis2))

    def test_shift_inputs_no_segment_ids(self):
        x = jnp.array([[1, 2, 3], [4, 5, 6]])
        expected_output = jnp.array([[0, 1, 2], [0, 4, 5]]) # Same as shift_right
        output = shift_inputs(x, segment_ids=None, axis=1)
        np.testing.assert_array_equal(np.array(output), np.array(expected_output))

    def test_shift_inputs_with_segment_ids(self):
        x = jnp.array([[1, 2, 3, 4, 5]]) # Single sequence for simplicity
        segment_ids = jnp.array([[1, 1, 1, 2, 2]]) # Two segments
        
        # Expected: shift_right applied, then zero out where segment changes
        # shifted_x = [0, 1, 2, 3, 4]
        # shifted_segment_ids = [0, 1, 1, 1, 2]
        # segment_ids == shifted_segment_ids --> [F, T, T, F, T]
        # output = shifted_x * (segment_ids == shifted_segment_ids)
        # output = [0*F, 1*T, 2*T, 3*F, 4*T] = [0, 1, 2, 0, 4]
        expected_output = jnp.array([[0, 1, 2, 0, 4]])
        output = shift_inputs(x, segment_ids=segment_ids, axis=1)
        np.testing.assert_array_equal(np.array(output), np.array(expected_output))

        # Test with axis=0 if it were a valid use case (it's typically axis=1 for sequences)
        x_t = x.T # (5,1)
        segment_ids_t = segment_ids.T # (5,1)
        expected_output_t = jnp.array([[0],[1],[2],[0],[4]])
        output_t = shift_inputs(x_t, segment_ids=segment_ids_t, axis=0)
        np.testing.assert_array_equal(np.array(output_t), np.array(expected_output_t))

    def test_sinusoidal_init(self):
        max_len = 50
        d_feature = 128
        key = jax.random.key(0) # JAX needs a key for random ops, though not strictly used by this init

        # Get the initializer function
        init_fn = sinusoidal_init(max_len=max_len)
        
        # Call the init function
        shape = (1, max_len, d_feature) # Typical shape for positional embeddings
        pos_emb = init_fn(key, shape)
        
        self.assertEqual(pos_emb.shape, shape)
        
        # Check that it's not all zeros (a basic sanity check)
        self.assertFalse(jnp.all(pos_emb == 0))
        
        # Check some properties of sinusoidal embeddings if possible
        # For example, first row should be different from the second if max_len > 1
        if max_len > 1:
            self.assertFalse(jnp.allclose(pos_emb[0, 0, :], pos_emb[0, 1, :]))

        # Test with different d_feature (odd/even handling for div_term)
        d_feature_odd = 129
        shape_odd = (1, max_len, d_feature_odd)
        pos_emb_odd = init_fn(key, shape_odd) # init_fn is for a fixed max_len, but d_feature comes from shape
        self.assertEqual(pos_emb_odd.shape, shape_odd)
        self.assertFalse(jnp.all(pos_emb_odd == 0))

    def test_sinusoidal_init_edge_cases(self):
        """Test edge cases for sinusoidal_init validation."""
        key = jax.random.key(0)
        
        # Test that d_feature < 2 raises ValueError
        init_fn = sinusoidal_init(max_len=10)
        with self.assertRaises(ValueError):
            init_fn(key, (1, 10, 1))  # d_feature = 1 should raise error
        
        # Test minimum valid d_feature works
        shape_min = (1, 10, 2)
        pos_emb_min = init_fn(key, shape_min)
        self.assertEqual(pos_emb_min.shape, shape_min)
        self.assertFalse(jnp.all(pos_emb_min == 0))

class TestAddPositionEmbs(unittest.TestCase):

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
            dropout_rate=0.0,
            attention_dropout_rate=0.0
        )
        self.rngs = nnx.Rngs(params=jax.random.key(0), dropout=jax.random.key(1))

    def test_unpacked_sinusoidal(self):
        batch_size, seq_len = 2, 5
        inputs = jnp.ones((batch_size, seq_len, self.config.emb_dim))
        module = AddPositionEmbs(config=self.config, decode=False, rngs=self.rngs)
        outputs = module(inputs)
        self.assertEqual(outputs.shape, inputs.shape)
        self.assertFalse(jnp.array_equal(outputs, inputs))
        expected_sin_emb = sinusoidal_init(max_len=self.config.max_len)(None, (1, self.config.max_len, self.config.emb_dim))[:, :seq_len, :]
        self.assertTrue(jnp.allclose(outputs, inputs + expected_sin_emb, atol=1e-6))

    def test_unpacked_learned(self):
        batch_size, seq_len = 2, 5
        inputs = jnp.ones((batch_size, seq_len, self.config.emb_dim))
        init_fn = lambda rng, shape: jax.random.normal(rng, shape) * 0.1
        config_learned = self.config.replace(posemb_init=init_fn)
        module = AddPositionEmbs(config=config_learned, decode=False, rngs=self.rngs)
        outputs = module(inputs)
        self.assertEqual(outputs.shape, inputs.shape)
        learned_emb = module.pos_embedding.value[:, :seq_len, :]
        self.assertTrue(jnp.allclose(outputs, inputs + learned_emb, atol=1e-6))

    def test_packed_learned(self):
        batch_size, packed_len, emb_dim = 1, 7, self.config.emb_dim
        inputs = jnp.arange(batch_size * packed_len * emb_dim, dtype=jnp.float32).reshape((batch_size, packed_len, emb_dim))
        positions = jnp.array([[0,1,2,0,1,2,3]])
        init_fn = lambda rng, shape: (jax.random.normal(rng, shape) * 0.01) + jnp.arange(shape[1], dtype=jnp.float32).reshape(1, -1, 1)
        config_packed = self.config.replace(posemb_init=init_fn)
        module = AddPositionEmbs(config=config_packed, decode=False, rngs=self.rngs)
        outputs = module(inputs, inputs_positions=positions)
        self.assertEqual(outputs.shape, inputs.shape)
        gathered_emb = jnp.take(module.pos_embedding.value[0], positions[0], axis=0)[None, ...]
        self.assertTrue(jnp.allclose(outputs, inputs + gathered_emb, atol=1e-5))

    def test_decode_mode_learned(self):
        batch_size, seq_len, emb_dim = 1, 1, self.config.emb_dim
        inputs = jnp.ones((batch_size, seq_len, emb_dim))
        init_fn = lambda rng, shape: jax.random.normal(rng, shape) * 0.1
        config_decode = self.config.replace(posemb_init=init_fn)
        module = AddPositionEmbs(config=config_decode, decode=True, rngs=self.rngs)
        module.init_cache(input_shape=inputs.shape)
        
        for i in range(2):
            current_idx = module.cache_index.value
            outputs = module(inputs * (i + 1))
            self.assertEqual(module.cache_index.value, current_idx + 1)
            expected_emb = module.pos_embedding.value[:, current_idx:current_idx+1, :]
            self.assertTrue(jnp.allclose(outputs, inputs * (i+1) + expected_emb, atol=1e-6))

    def test_init_cache(self):
        module = AddPositionEmbs(config=self.config, decode=True, rngs=self.rngs)
        module.init_cache(input_shape=(1,1,self.config.emb_dim))
        self.assertTrue(hasattr(module, 'cache_index'))
        self.assertEqual(module.cache_index.value, 0)
        self.assertEqual(module.cache_index.value.dtype, jnp.uint32)

    def test_decode_mode_without_cache_init(self):
        """Test that decode mode raises error when cache is not initialized."""
        batch_size, seq_len, emb_dim = 1, 1, self.config.emb_dim
        inputs = jnp.ones((batch_size, seq_len, emb_dim))
        module = AddPositionEmbs(config=self.config, decode=True, rngs=self.rngs)
        
        # Should raise RuntimeError when trying to use decode mode without cache init
        with self.assertRaises(RuntimeError):
            module(inputs)

class TestMlpBlock(unittest.TestCase):

    def setUp(self):
        self.config = TransformerConfig(
            vocab_size=100,
            output_vocab_size=100,
            emb_dim=8,
            max_len=10, # Though not directly used by MlpBlock, keep for consistency
            num_heads=2, # Not used by MlpBlock
            num_layers=1, # Not used by MlpBlock
            qkv_dim=8, # Not used by MlpBlock
            mlp_dim=16,
            dropout_rate=0.1,
            attention_dropout_rate=0.0 # Not used by MlpBlock
        )
        self.rngs_params = nnx.Rngs(params=jax.random.key(0))
        self.rngs_dropout_active = nnx.Rngs(dropout=jax.random.key(1))

    def test_output_shape(self):
        batch_size, seq_len = 2, 5
        inputs = jnp.ones((batch_size, seq_len, self.config.emb_dim))
        mlp_block = MlpBlock(config=self.config, rngs=self.rngs_params)
        outputs = mlp_block(inputs, rngs=self.rngs_dropout_active)
        self.assertEqual(outputs.shape, inputs.shape)

    def test_dropout_deterministic_behavior(self):
        batch_size, seq_len = 2, 5
        inputs = jnp.ones((batch_size, seq_len, self.config.emb_dim))
        
        config_with_dropout = self.config.replace(dropout_rate=0.5, deterministic=True)
        rngs_init = nnx.Rngs(params=jax.random.key(10))
        mlp_block = MlpBlock(config=config_with_dropout, rngs=rngs_init)
        
        outputs1 = mlp_block(inputs, rngs=None)
        outputs2 = mlp_block(inputs, rngs=None)
        np.testing.assert_array_equal(np.array(outputs1), np.array(outputs2))

    def test_dropout_non_deterministic_behavior(self):
        batch_size, seq_len = 2, 5
        inputs = jnp.ones((batch_size, seq_len, self.config.emb_dim))
        
        config_with_dropout = self.config.replace(dropout_rate=0.5)
        rngs_init = nnx.Rngs(params=jax.random.key(20))
        mlp_block = MlpBlock(config=config_with_dropout, rngs=rngs_init)
        
        outputs1 = mlp_block(inputs, rngs=nnx.Rngs(dropout=jax.random.key(31)))
        outputs2 = mlp_block(inputs, rngs=nnx.Rngs(dropout=jax.random.key(32)))
        
        self.assertFalse(jnp.allclose(outputs1, outputs2), "Outputs should differ when dropout is active and rngs differ.")

    def test_mlp_block_values(self):
        batch_size, seq_len, emb_dim, mlp_dim = 1, 1, 2, 3
        inputs = jnp.array([[[1., 2.]]])

        config_value_test = TransformerConfig(
            emb_dim=emb_dim,
            mlp_dim=mlp_dim,
            dropout_rate=0.0,
            vocab_size=10, output_vocab_size=10,
        )
        
        def K_init_fn(key, shape, dtype):
            if shape == (emb_dim, mlp_dim):
                return jnp.array([[1.,0.,1.], [0.,1.,1.]]).astype(dtype)
            elif shape == (mlp_dim, emb_dim):
                return jnp.array([[1.,0.], [0.,1.], [1.,1.]]).astype(dtype)
            return nnx.initializers.xavier_uniform()(key, shape, dtype)

        def B_init_fn(key, shape, dtype):
             return jnp.zeros(shape, dtype=dtype)

        config_value_test = config_value_test.replace(kernel_init=K_init_fn, bias_init=B_init_fn)
        
        mlp_rngs_init = nnx.Rngs(params=jax.random.key(40))
        mlp_block = MlpBlock(config=config_value_test, rngs=mlp_rngs_init)
        
        outputs = mlp_block(inputs, rngs=None)
        
        expected_output = jnp.array([[[4., 5.]]])
        np.testing.assert_allclose(np.array(outputs), np.array(expected_output), rtol=1e-6)

class TestMultiHeadAttention(unittest.TestCase):

    def setUp(self):
        self.config = TransformerConfig(
            vocab_size=100,
            output_vocab_size=100,
            emb_dim=16,
            max_len=10,
            num_heads=4,
            num_layers=1,
            qkv_dim=16,
            mlp_dim=32,
            dropout_rate=0.1,
            attention_dropout_rate=0.1
        )
        self.rngs_params = nnx.Rngs(params=jax.random.key(0))
        self.rngs_dropout = nnx.Rngs(dropout=jax.random.key(1))

    def test_basic_self_attention_shape(self):
        """Test that self-attention preserves input shape."""
        batch_size, seq_len, features = 2, 8, 16
        inputs = jax.random.normal(jax.random.key(42), (batch_size, seq_len, features))
        
        attention = MultiHeadAttention(
            num_heads=4,
            in_features=features,
            qkv_features=features,
            dropout_rate=0.0,
            deterministic=True,
            decode=False,
            rngs=self.rngs_params
        )
        
        outputs = attention(inputs, rngs=None)
        self.assertEqual(outputs.shape, inputs.shape)

    def test_cross_attention_shape(self):
        """Test cross-attention with different key/value inputs."""
        batch_size, seq_len_q, seq_len_kv, features = 2, 6, 8, 16
        inputs_q = jax.random.normal(jax.random.key(1), (batch_size, seq_len_q, features))
        inputs_kv = jax.random.normal(jax.random.key(2), (batch_size, seq_len_kv, features))
        
        attention = MultiHeadAttention(
            num_heads=4,
            in_features=features,
            qkv_features=features,
            dropout_rate=0.0,
            deterministic=True,
            decode=False,
            rngs=self.rngs_params
        )
        
        outputs = attention(inputs_q, inputs_k=inputs_kv, inputs_v=inputs_kv, rngs=None)
        self.assertEqual(outputs.shape, inputs_q.shape)

    def test_different_feature_dimensions(self):
        """Test attention with different input/output feature dimensions."""
        batch_size, seq_len = 2, 6
        in_features, qkv_features, out_features = 12, 16, 20
        inputs = jax.random.normal(jax.random.key(3), (batch_size, seq_len, in_features))
        
        attention = MultiHeadAttention(
            num_heads=4,
            in_features=in_features,
            qkv_features=qkv_features,
            out_features=out_features,
            dropout_rate=0.0,
            deterministic=True,
            decode=False,
            rngs=self.rngs_params
        )
        
        outputs = attention(inputs, rngs=None)
        self.assertEqual(outputs.shape, (batch_size, seq_len, out_features))

    def test_mask_application(self):
        """Test that attention mask is properly applied."""
        batch_size, seq_len, features, num_heads = 1, 4, 8, 2
        # Use varied inputs instead of all ones to make masking effects more visible
        inputs = jax.random.normal(jax.random.key(42), (batch_size, seq_len, features))
        
        # Create a causal mask that blocks attention to future positions
        mask = jnp.tril(jnp.ones((seq_len, seq_len)))  # Lower triangular mask
        mask = mask[None, None, :, :]  # Add batch and head dimensions
        mask = jnp.broadcast_to(mask, (batch_size, num_heads, seq_len, seq_len))
        
        attention = MultiHeadAttention(
            num_heads=num_heads,
            in_features=features,
            dropout_rate=0.0,
            deterministic=True,
            decode=False,
            rngs=self.rngs_params
        )
        
        outputs_with_mask = attention(inputs, mask=mask, rngs=None)
        outputs_without_mask = attention(inputs, rngs=None)
        
        # Outputs should be different when mask is applied
        # Use a looser tolerance since the differences might be subtle
        self.assertFalse(jnp.allclose(outputs_with_mask, outputs_without_mask, atol=1e-6))

    def test_init_cache(self):
        """Test cache initialization for autoregressive decoding."""
        batch_size, seq_len, features = 2, 10, 16
        input_shape = (batch_size, seq_len, features)
        
        attention = MultiHeadAttention(
            num_heads=4,
            in_features=features,
            decode=True,
            dropout_rate=0.0,
            deterministic=True,
            rngs=self.rngs_params
        )
        
        attention.init_cache(input_shape)
        
        # Check that cache attributes are initialized
        self.assertIsNotNone(attention.cached_key)
        self.assertIsNotNone(attention.cached_value)
        self.assertIsNotNone(attention.cache_index)
        
        # Check cache shapes
        expected_cache_shape = (batch_size, seq_len, 4, features // 4)  # (batch, seq, heads, head_dim)
        self.assertEqual(attention.cached_key.value.shape, expected_cache_shape)
        self.assertEqual(attention.cached_value.value.shape, expected_cache_shape)
        self.assertEqual(attention.cache_index.value, 0)

    def test_decode_mode_without_cache_init(self):
        """Test that decode mode raises error when cache is not initialized."""
        batch_size, seq_len, features = 1, 1, 8
        inputs = jnp.ones((batch_size, seq_len, features))
        
        attention = MultiHeadAttention(
            num_heads=2,
            in_features=features,
            decode=True,
            rngs=self.rngs_params
        )
        
        # Should raise ValueError when trying to use decode mode without cache init
        with self.assertRaises(ValueError):
            attention(inputs, rngs=None)

    def test_decode_mode_functionality(self):
        """Test autoregressive decoding functionality."""
        batch_size, max_seq_len, features = 1, 5, 8
        
        attention = MultiHeadAttention(
            num_heads=2,
            in_features=features,
            decode=True,
            dropout_rate=0.0,
            deterministic=True,
            rngs=self.rngs_params
        )
        
        attention.init_cache((batch_size, max_seq_len, features))
        
        # Process tokens one by one
        outputs = []
        for i in range(3):
            token_input = jax.random.normal(jax.random.key(i + 10), (batch_size, 1, features))
            output = attention(token_input, decode=True, rngs=None)
            outputs.append(output)
            self.assertEqual(output.shape, (batch_size, 1, features))
        
        # Cache index should have incremented
        self.assertEqual(attention.cache_index.value, 3)

    def test_dropout_deterministic_vs_non_deterministic(self):
        """Test dropout behavior in deterministic vs non-deterministic modes."""
        batch_size, seq_len, features = 1, 4, 8
        inputs = jnp.ones((batch_size, seq_len, features))
        
        # Test deterministic mode
        attention_det = MultiHeadAttention(
            num_heads=2,
            in_features=features,
            dropout_rate=0.5,
            deterministic=True,
            decode=False,
            rngs=self.rngs_params
        )
        
        outputs1 = attention_det(inputs, rngs=None)
        outputs2 = attention_det(inputs, rngs=None)
        np.testing.assert_array_equal(np.array(outputs1), np.array(outputs2))
        
        # Test non-deterministic mode
        attention_nondet = MultiHeadAttention(
            num_heads=2,
            in_features=features,
            dropout_rate=0.5,
            deterministic=False,
            decode=False,
            rngs=self.rngs_params
        )
        
        outputs3 = attention_nondet(inputs, rngs=nnx.Rngs(dropout=jax.random.key(100)))
        outputs4 = attention_nondet(inputs, rngs=nnx.Rngs(dropout=jax.random.key(101)))
        
        # Outputs should be different with different dropout keys
        self.assertFalse(jnp.allclose(outputs3, outputs4, atol=1e-6))

    def test_no_dropout_efficiency(self):
        """Test that no RNGs are needed when dropout_rate is 0."""
        batch_size, seq_len, features = 1, 4, 8
        inputs = jnp.ones((batch_size, seq_len, features))
        
        attention = MultiHeadAttention(
            num_heads=2,
            in_features=features,
            dropout_rate=0.0,  # No dropout
            decode=False,
            rngs=self.rngs_params
        )
        
        # Should work without providing dropout RNGs
        outputs = attention(inputs, rngs=None)
        self.assertEqual(outputs.shape, inputs.shape)

    def test_input_validation(self):
        """Test input dimension validation."""
        attention = MultiHeadAttention(
            num_heads=2,
            in_features=8,
            in_kv_features=6,
            rngs=self.rngs_params
        )
        
        # Test incorrect query dimension
        wrong_q = jnp.ones((1, 4, 10))  # Should be 8
        correct_kv = jnp.ones((1, 4, 6))
        
        with self.assertRaises(ValueError):
            attention(wrong_q, inputs_k=correct_kv, inputs_v=correct_kv, rngs=None)
        
        # Test incorrect key/value dimensions
        correct_q = jnp.ones((1, 4, 8))
        wrong_kv = jnp.ones((1, 4, 10))  # Should be 6
        
        with self.assertRaises(ValueError):
            attention(correct_q, inputs_k=wrong_kv, inputs_v=correct_kv, rngs=None)

    def test_normalize_qk_feature(self):
        """Test query-key normalization feature."""
        batch_size, seq_len, features = 1, 4, 8
        inputs = jax.random.normal(jax.random.key(20), (batch_size, seq_len, features))
        
        attention_normalized = MultiHeadAttention(
            num_heads=2,
            in_features=features,
            normalize_qk=True,
            dropout_rate=0.0,
            deterministic=True,
            decode=False,
            rngs=self.rngs_params
        )
        
        attention_regular = MultiHeadAttention(
            num_heads=2,
            in_features=features,
            normalize_qk=False,
            dropout_rate=0.0,
            deterministic=True,
            decode=False,
            rngs=nnx.Rngs(params=jax.random.key(0))  # Same initialization
        )
        
        outputs_normalized = attention_normalized(inputs, rngs=None)
        outputs_regular = attention_regular(inputs, rngs=None)
        
        # Outputs should be different when normalization is applied
        self.assertFalse(jnp.allclose(outputs_normalized, outputs_regular, atol=1e-6))

    def test_heads_divisibility_assertion(self):
        """Test that in_features must be divisible by num_heads."""
        with self.assertRaises(AssertionError):
            MultiHeadAttention(
                num_heads=3,
                in_features=8,  # 8 is not divisible by 3
                rngs=self.rngs_params
            )

class TestDotProductAttentionWeights(unittest.TestCase):

    def setUp(self):
        self.key = jax.random.key(0)
        self.batch_size = 2
        self.num_heads = 2
        self.seq_len = 4
        self.head_dim = 8

        q_key, k_key, self.dropout_key = jax.random.split(self.key, 3)
        self.query = jax.random.normal(q_key, (self.batch_size, self.seq_len, self.num_heads, self.head_dim))
        self.key = jax.random.normal(k_key, (self.batch_size, self.seq_len, self.num_heads, self.head_dim))
        self.dtype = jnp.float32

    def test_output_shape(self):
        """Test that the output shape is correct."""
        attn_weights = dot_product_attention_weights(
            self.query, self.key, deterministic=True, dtype=self.dtype
        )
        expected_shape = (self.batch_size, self.num_heads, self.seq_len, self.seq_len)
        self.assertEqual(attn_weights.shape, expected_shape)

    def test_softmax_application(self):
        """Test that weights for each query position sum to 1 (softmax)."""
        attn_weights = dot_product_attention_weights(
            self.query, self.key, deterministic=True, dtype=self.dtype
        )
        sums = jnp.sum(attn_weights, axis=-1)
        np.testing.assert_allclose(np.array(sums), np.ones(sums.shape), rtol=1e-6)

    def test_masking(self):
        """Test that masking works as expected."""
        mask = jnp.ones((self.batch_size, self.num_heads, self.seq_len, self.seq_len), dtype=jnp.bool_)
        mask = mask.at[:, :, :, -1].set(False)
        
        attn_weights = dot_product_attention_weights(
            self.query, self.key, mask=mask, deterministic=True, dtype=self.dtype
        )
        
        weights_for_last_key = attn_weights[:, :, :, -1]
        np.testing.assert_allclose(np.array(weights_for_last_key), np.zeros_like(weights_for_last_key), atol=1e-7)
        
        sums = jnp.sum(attn_weights, axis=-1)
        np.testing.assert_allclose(np.array(sums), np.ones(sums.shape), rtol=1e-6)

    def test_bias(self):
        """Test the application of an attention bias."""
        bias_val = 100.0
        bias = jnp.zeros((self.batch_size, self.num_heads, self.seq_len, self.seq_len))
        bias = bias.at[:, :, :, 0].set(bias_val)
        
        attn_weights = dot_product_attention_weights(
            self.query, self.key, bias=bias, deterministic=True, dtype=self.dtype
        )
        
        weights_for_first_key = attn_weights[:, :, :, 0]
        np.testing.assert_allclose(np.array(weights_for_first_key), np.ones_like(weights_for_first_key), rtol=1e-6)

    def test_dropout_non_deterministic(self):
        """Test that dropout is applied in non-deterministic mode."""
        dropout_rate = 0.5
        key1, key2 = jax.random.split(self.dropout_key)
        
        weights1 = dot_product_attention_weights(
            self.query, self.key, dropout_rng=key1, dropout_rate=dropout_rate, deterministic=False
        )
        weights2 = dot_product_attention_weights(
            self.query, self.key, dropout_rng=key2, dropout_rate=dropout_rate, deterministic=False
        )
        
        self.assertFalse(jnp.allclose(weights1, weights2))

    def test_dropout_deterministic(self):
        """Test that dropout is NOT applied in deterministic mode."""
        dropout_rate = 0.5
        key1, key2 = jax.random.split(self.dropout_key)
        
        weights1 = dot_product_attention_weights(
            self.query, self.key, dropout_rng=key1, dropout_rate=dropout_rate, deterministic=True
        )
        weights2 = dot_product_attention_weights(
            self.query, self.key, dropout_rng=key2, dropout_rate=dropout_rate, deterministic=True
        )
        
        np.testing.assert_allclose(np.array(weights1), np.array(weights2))

    def test_broadcast_dropout(self):
        """Test broadcast dropout behavior."""
        query = jnp.ones((self.batch_size, self.num_heads, self.seq_len, self.head_dim))
        key = jnp.ones((self.batch_size, self.num_heads, self.seq_len, self.head_dim))
        
        weights = dot_product_attention_weights(
            query, key,
            dropout_rng=self.dropout_key,
            dropout_rate=0.5,
            deterministic=False,
            broadcast_dropout=True
        )
        
        first_matrix = weights[0, 0]
        for b in range(self.batch_size):
            for h in range(self.num_heads):
                np.testing.assert_allclose(np.array(weights[b, h]), np.array(first_matrix))

    def test_no_broadcast_dropout(self):
        """Test non-broadcast dropout behavior."""
        query = jnp.ones((self.batch_size, self.num_heads, self.seq_len, self.head_dim))
        key = jnp.ones((self.batch_size, self.num_heads, self.seq_len, self.head_dim))
        
        weights = dot_product_attention_weights(
            query, key,
            dropout_rng=self.dropout_key,
            dropout_rate=0.5,
            deterministic=False,
            broadcast_dropout=False
        )
        
        self.assertFalse(jnp.allclose(weights[0, 0], weights[1, 1]))

    def test_sow_weights(self):
        """Test if attention weights are sown correctly."""
        class TestModule(nnx.Module):
            def __init__(self, *, rngs):
                pass
            def __call__(self, query, key):
                return dot_product_attention_weights(
                    query, key, module=self, deterministic=True
                )

        module = TestModule(rngs=nnx.Rngs(0))
        weights = module(self.query, self.key)
        
        state = nnx.state(module, nnx.Intermediate)
        
        self.assertIn('attn_weights', state)
        
        sown_weights_list = state['attn_weights'].value
        self.assertIsInstance(sown_weights_list, tuple)
        self.assertEqual(len(sown_weights_list), 1)
        sown_weights = sown_weights_list[0]
        
        self.assertEqual(sown_weights.shape, weights.shape)
        np.testing.assert_allclose(np.array(sown_weights), np.array(weights))

class TestCombineMasks(unittest.TestCase):
    """Test the combine_masks function."""

    def setUp(self):
        self.batch_size = 2
        self.num_heads = 2 
        self.seq_len = 4
        self.mask_shape = (self.batch_size, self.num_heads, self.seq_len, self.seq_len)

    def test_no_masks(self):
        """Test that combine_masks returns None when no masks are provided."""
        result = combine_masks()
        self.assertIsNone(result)

    def test_single_mask(self):
        """Test that combine_masks returns the single mask when only one is provided."""
        mask = jnp.ones(self.mask_shape, dtype=jnp.bool_)
        result = combine_masks(mask)
        np.testing.assert_array_equal(np.array(result), np.array(mask))
        self.assertEqual(result.dtype, jnp.bool_)

    def test_single_mask_with_nones(self):
        """Test that combine_masks ignores None values."""
        mask = jnp.ones(self.mask_shape, dtype=jnp.bool_)
        result = combine_masks(None, mask, None)
        np.testing.assert_array_equal(np.array(result), np.array(mask))

    def test_two_masks_logical_and(self):
        """Test that combine_masks performs logical AND on two masks."""
        mask1 = jnp.ones(self.mask_shape, dtype=jnp.bool_)
        mask2 = jnp.zeros(self.mask_shape, dtype=jnp.bool_)
        mask2 = mask2.at[:, :, 0, :].set(True)  # First row is True
        
        result = combine_masks(mask1, mask2)
        expected = jnp.logical_and(mask1, mask2)
        
        np.testing.assert_array_equal(np.array(result), np.array(expected))
        self.assertEqual(result.dtype, jnp.bool_)

    def test_multiple_masks(self):
        """Test that combine_masks works with multiple masks."""
        mask1 = jnp.ones(self.mask_shape, dtype=jnp.bool_)
        mask2 = jnp.ones(self.mask_shape, dtype=jnp.bool_)
        mask2 = mask2.at[:, :, -1, :].set(False)  # Last row is False
        mask3 = jnp.ones(self.mask_shape, dtype=jnp.bool_)
        mask3 = mask3.at[:, :, :, -1].set(False)  # Last column is False
        
        result = combine_masks(mask1, mask2, mask3)
        expected = jnp.logical_and(jnp.logical_and(mask1, mask2), mask3)
        
        np.testing.assert_array_equal(np.array(result), np.array(expected))

    def test_different_dtypes(self):
        """Test that combine_masks converts to the specified dtype."""
        mask1 = jnp.ones(self.mask_shape, dtype=jnp.bool_)
        mask2 = jnp.ones(self.mask_shape, dtype=jnp.bool_)
        
        result_float = combine_masks(mask1, mask2, dtype=jnp.float32)
        result_int = combine_masks(mask1, mask2, dtype=jnp.int32)
        
        self.assertEqual(result_float.dtype, jnp.float32)
        self.assertEqual(result_int.dtype, jnp.int32)
        np.testing.assert_array_equal(np.array(result_float), np.ones(self.mask_shape, dtype=jnp.float32))
        np.testing.assert_array_equal(np.array(result_int), np.ones(self.mask_shape, dtype=jnp.int32))

    def test_mismatched_dimensions(self):
        """Test that combine_masks raises error for masks with different dimensions."""
        mask1 = jnp.ones((2, 2, 4, 4), dtype=jnp.bool_)  # 4D
        mask2 = jnp.ones((2, 4, 4), dtype=jnp.bool_)     # 3D
        
        with self.assertRaises(AssertionError):
            combine_masks(mask1, mask2)

    def test_broadcastable_shapes(self):
        """Test that combine_masks works with broadcastable shapes."""
        mask1 = jnp.ones((2, 2, 4, 4), dtype=jnp.bool_)
        mask2 = jnp.ones((1, 1, 4, 4), dtype=jnp.bool_)  # Broadcastable
        mask2 = mask2.at[:, :, 0, :].set(False)
        
        result = combine_masks(mask1, mask2)
        expected = jnp.logical_and(mask1, mask2)
        
        np.testing.assert_array_equal(np.array(result), np.array(expected))


class TestDotProductAttention(unittest.TestCase):
    """Test the dot_product_attention function."""

    def setUp(self):
        self.key = jax.random.key(42)
        self.batch_size = 2
        self.num_heads = 2
        self.seq_len = 4
        self.head_dim = 8

        q_key, k_key, v_key, self.dropout_key = jax.random.split(self.key, 4)
        self.query = jax.random.normal(q_key, (self.batch_size, self.seq_len, self.num_heads, self.head_dim))
        self.key_tensor = jax.random.normal(k_key, (self.batch_size, self.seq_len, self.num_heads, self.head_dim))
        self.value = jax.random.normal(v_key, (self.batch_size, self.seq_len, self.num_heads, self.head_dim))
        self.dtype = jnp.float32

    def test_basic_output_shape(self):
        """Test that dot_product_attention returns correct output shape."""
        output = dot_product_attention(
            self.query, self.key_tensor, self.value, 
            deterministic=True, dtype=self.dtype
        )
        expected_shape = (self.batch_size, self.seq_len, self.num_heads, self.head_dim)
        self.assertEqual(output.shape, expected_shape)

    def test_self_attention(self):
        """Test self-attention (query, key, value are the same)."""
        inputs = self.query
        output = dot_product_attention(
            inputs, inputs, inputs, 
            deterministic=True, dtype=self.dtype
        )
        self.assertEqual(output.shape, inputs.shape)

    def test_cross_attention(self):
        """Test cross-attention with different query and key/value."""
        # Different sequence length for key/value
        kv_seq_len = 6
        k_key, v_key = jax.random.split(jax.random.key(123), 2)
        key_cross = jax.random.normal(k_key, (self.batch_size, kv_seq_len, self.num_heads, self.head_dim))
        value_cross = jax.random.normal(v_key, (self.batch_size, kv_seq_len, self.num_heads, self.head_dim))
        
        output = dot_product_attention(
            self.query, key_cross, value_cross,
            deterministic=True, dtype=self.dtype
        )
        expected_shape = (self.batch_size, self.seq_len, self.num_heads, self.head_dim)  # Query length
        self.assertEqual(output.shape, expected_shape)

    def test_masking_effect(self):
        """Test that masking affects the output."""
        # Create a mask that blocks the last key position
        mask = jnp.ones((self.batch_size, self.num_heads, self.seq_len, self.seq_len), dtype=jnp.bool_)
        mask = mask.at[:, :, :, -1].set(False)
        
        output_masked = dot_product_attention(
            self.query, self.key_tensor, self.value,
            mask=mask, deterministic=True, dtype=self.dtype
        )
        
        output_unmasked = dot_product_attention(
            self.query, self.key_tensor, self.value,
            deterministic=True, dtype=self.dtype
        )
        
        # Outputs should be different when masking is applied
        self.assertFalse(jnp.allclose(output_masked, output_unmasked, atol=1e-6))

    def test_bias_application(self):
        """Test that bias affects the attention computation."""
        bias = jnp.zeros((self.batch_size, self.num_heads, self.seq_len, self.seq_len))
        bias = bias.at[:, :, :, 0].set(100.0)  # Strong bias toward first position
        
        output_biased = dot_product_attention(
            self.query, self.key_tensor, self.value,
            bias=bias, deterministic=True, dtype=self.dtype
        )
        
        output_unbiased = dot_product_attention(
            self.query, self.key_tensor, self.value,
            deterministic=True, dtype=self.dtype
        )
        
        # Outputs should be different when bias is applied
        self.assertFalse(jnp.allclose(output_biased, output_unbiased, atol=1e-6))

    def test_dropout_deterministic_vs_non_deterministic(self):
        """Test dropout behavior in deterministic vs non-deterministic modes."""
        dropout_rate = 0.5
        
        # Deterministic mode should give same results
        output1_det = dot_product_attention(
            self.query, self.key_tensor, self.value,
            dropout_rng=jax.random.key(1), dropout_rate=dropout_rate,
            deterministic=True, dtype=self.dtype
        )
        output2_det = dot_product_attention(
            self.query, self.key_tensor, self.value,
            dropout_rng=jax.random.key(2), dropout_rate=dropout_rate,
            deterministic=True, dtype=self.dtype
        )
        np.testing.assert_allclose(np.array(output1_det), np.array(output2_det))
        
        # Non-deterministic mode should give different results
        output1_nondet = dot_product_attention(
            self.query, self.key_tensor, self.value,
            dropout_rng=jax.random.key(1), dropout_rate=dropout_rate,
            deterministic=False, dtype=self.dtype
        )
        output2_nondet = dot_product_attention(
            self.query, self.key_tensor, self.value,
            dropout_rng=jax.random.key(2), dropout_rate=dropout_rate,
            deterministic=False, dtype=self.dtype
        )
        self.assertFalse(jnp.allclose(output1_nondet, output2_nondet, atol=1e-6))

    def test_no_dropout_efficiency_path(self):
        """Test the efficiency path when dropout_rate=0 and module=None."""
        # This should use the optimized jax.nn.dot_product_attention path
        output_efficient = dot_product_attention(
            self.query, self.key_tensor, self.value,
            dropout_rate=0.0, deterministic=True, dtype=self.dtype
        )
        
        # Compare with explicit dropout path
        output_explicit = dot_product_attention(
            self.query, self.key_tensor, self.value,
            dropout_rate=0.001, deterministic=True, dtype=self.dtype  # Small dropout to avoid efficient path
        )
        
        # Should have same shape and similar values (though not exactly equal due to different code paths)
        self.assertEqual(output_efficient.shape, output_explicit.shape)

    def test_broadcast_dropout(self):
        """Test broadcast dropout functionality."""
        dropout_rate = 0.5
        
        # Use identical inputs to ensure dropout is the only source of variation
        query = jnp.ones((self.batch_size, self.seq_len, self.num_heads, self.head_dim))
        key = jnp.ones((self.batch_size, self.seq_len, self.num_heads, self.head_dim))
        value = jnp.ones((self.batch_size, self.seq_len, self.num_heads, self.head_dim))
        
        output_broadcast = dot_product_attention(
            query, key, value,
            dropout_rng=self.dropout_key, dropout_rate=dropout_rate,
            broadcast_dropout=True, deterministic=False, dtype=self.dtype
        )
        
        output_no_broadcast = dot_product_attention(
            query, key, value, 
            dropout_rng=self.dropout_key, dropout_rate=dropout_rate,
            broadcast_dropout=False, deterministic=False, dtype=self.dtype
        )
        
        # Both should have same shape
        self.assertEqual(output_broadcast.shape, output_no_broadcast.shape)

    def test_higher_dimensional_inputs(self):
        """Test attention with higher-dimensional inputs (>4D)."""
        # 5D inputs: (batch, time, head, seq, features)
        batch_dims = (2, 3)  # Extra batch dimensions
        query_5d = jax.random.normal(
            jax.random.key(1), 
            batch_dims + (self.seq_len, self.num_heads, self.head_dim)
        )
        key_5d = jax.random.normal(
            jax.random.key(2), 
            batch_dims + (self.seq_len, self.num_heads, self.head_dim)
        )
        value_5d = jax.random.normal(
            jax.random.key(3), 
            batch_dims + (self.seq_len, self.num_heads, self.head_dim)
        )
        
        output = dot_product_attention(
            query_5d, key_5d, value_5d,
            deterministic=True, dtype=self.dtype
        )
        
        expected_shape = batch_dims + (self.seq_len, self.num_heads, self.head_dim)
        self.assertEqual(output.shape, expected_shape)

    def test_dimension_validation(self):
        """Test that mismatched dimensions raise appropriate errors."""
        # Mismatched rank
        query_3d = jnp.ones((self.batch_size, self.seq_len, self.head_dim))  # Missing head dim
        
        with self.assertRaises(AssertionError):
            dot_product_attention(query_3d, self.key_tensor, self.value, deterministic=True)
        
        # Mismatched batch dimensions
        query_wrong_batch = jnp.ones((3, self.seq_len, self.num_heads, self.head_dim))  # Wrong batch size
        
        with self.assertRaises(AssertionError):
            dot_product_attention(query_wrong_batch, self.key_tensor, self.value, deterministic=True)
        
        # Mismatched head dimensions
        key_wrong_heads = jnp.ones((self.batch_size, self.seq_len, 3, self.head_dim))  # Wrong num_heads
        
        with self.assertRaises(AssertionError):
            dot_product_attention(self.query, key_wrong_heads, self.value, deterministic=True)
        
        # Mismatched feature dimensions
        value_wrong_features = jnp.ones((self.batch_size, self.seq_len, self.num_heads, 16))  # Wrong head_dim
        
        with self.assertRaises(AssertionError):
            dot_product_attention(self.query, self.key_tensor, value_wrong_features, deterministic=True)

    def test_attention_weights_recording(self):
        """Test that attention weights can be recorded with module.sow."""
        class TestModule(nnx.Module):
            def __init__(self, *, rngs):
                pass
                
            def __call__(self, query, key, value):
                return dot_product_attention(
                    query, key, value, module=self, deterministic=True
                )
        
        module = TestModule(rngs=nnx.Rngs(0))
        output = module(self.query, self.key_tensor, self.value)
        
        # Check that attention weights were sown
        state = nnx.state(module, nnx.Intermediate)
        self.assertIn('attn_weights', state)
        
        sown_weights_list = state['attn_weights'].value
        self.assertIsInstance(sown_weights_list, tuple)
        self.assertEqual(len(sown_weights_list), 1)
        
        sown_weights = sown_weights_list[0]
        expected_weights_shape = (self.batch_size, self.num_heads, self.seq_len, self.seq_len)
        self.assertEqual(sown_weights.shape, expected_weights_shape)

    def test_precision_parameter(self):
        """Test that precision parameter is accepted and doesn't break computation."""
        output_default = dot_product_attention(
            self.query, self.key_tensor, self.value, 
            deterministic=True, dtype=self.dtype
        )
        
        output_high_precision = dot_product_attention(
            self.query, self.key_tensor, self.value,
            precision=jax.lax.Precision.HIGHEST,
            deterministic=True, dtype=self.dtype
        )
        
        # Should have same shape and similar values
        self.assertEqual(output_default.shape, output_high_precision.shape)
        # Values should be close but might not be exactly equal due to precision differences
        self.assertTrue(jnp.allclose(output_default, output_high_precision, rtol=1e-5))


class TestAdaptiveLayerNorm(unittest.TestCase):
    """Test the AdaptiveLayerNorm module."""

    def setUp(self):
        self.dim = 8
        self.batch_size = 2
        self.seq_len = 4
        self.rngs = nnx.Rngs(params=jax.random.key(0))

    def test_basic_adaptive_normalization(self):
        """Test basic adaptive layer normalization functionality."""
        x = jax.random.normal(jax.random.key(1), (self.batch_size, self.seq_len, self.dim))
        condition = jax.random.normal(jax.random.key(2), (self.batch_size, self.seq_len, self.dim))
        
        aln = AdaptiveLayerNorm(self.dim, rngs=self.rngs)
        output = aln(x, condition=condition)
        
        self.assertEqual(output.shape, x.shape)
        self.assertFalse(jnp.allclose(output, x))  # Should be different due to normalization

    def test_condition_reshaping(self):
        """Test that 2D conditions are properly reshaped."""
        x = jax.random.normal(jax.random.key(1), (self.batch_size, self.seq_len, self.dim))
        condition_2d = jax.random.normal(jax.random.key(2), (self.batch_size, self.dim))
        condition_3d = condition_2d[:, None, :]  # Manual reshape
        
        aln = AdaptiveLayerNorm(self.dim, rngs=self.rngs)
        output_2d = aln(x, condition=condition_2d)
        output_3d = aln(x, condition=condition_3d)
        
        np.testing.assert_allclose(np.array(output_2d), np.array(output_3d), rtol=1e-6)


if __name__ == '__main__':
    unittest.main() 