import unittest
import jax # Added for jax.random.key
import jax.numpy as jnp
import numpy as np # Kept for existing tests
from flax import nnx # For new tests

# Consolidated and corrected model imports
from llm_nnx.model import (
    shift_right, shift_inputs, sinusoidal_init, 
    TransformerConfig, AddPositionEmbs, MlpBlock, Shape
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
        key = jax.random.PRNGKey(0) # JAX needs a key for random ops, though not strictly used by this init

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
        self.rngs_params_and_dropout = nnx.Rngs(params=jax.random.key(0), dropout=jax.random.key(1))

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

if __name__ == '__main__':
    unittest.main() 