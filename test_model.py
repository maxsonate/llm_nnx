import unittest
import numpy as np
import jax.numpy as jnp
from jax import random

# Assuming model.py is in the same directory or accessible in PYTHONPATH
from model import shift_right, shift_inputs, sinusoidal_init, TransformerConfig # Added TransformerConfig as it's used by sinusoidal_init indirectly via default posemb_init

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
        key = random.PRNGKey(0) # JAX needs a key for random ops, though not strictly used by this init

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

if __name__ == '__main__':
    unittest.main() 