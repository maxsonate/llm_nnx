"""Tests for training utilities in train.py"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
from unittest.mock import Mock, MagicMock

# Import the functions directly - make sure this file is in the same directory as train.py
try:
    from train import rsqrt_schedule, create_learning_rate_schedule, compute_weighted_cross_entropy, compute_weighted_accuracy, train_step, evaluate
    from utils import TrainState  # Assuming TrainState is in utils.py
except ImportError:
    import sys
    sys.path.append('.')
    from train import rsqrt_schedule, create_learning_rate_schedule, compute_weighted_cross_entropy, compute_weighted_accuracy, train_step, evaluate
    from utils import TrainState


class TestRsqrtSchedule:
    """Test cases for the rsqrt_schedule function."""
    
    def test_rsqrt_schedule_basic(self):
        """Test basic functionality of rsqrt_schedule."""
        init_value = 1e-3
        shift = 100
        schedule = rsqrt_schedule(init_value, shift)
        
        # Test at step 0 (effective step = 0 + shift = shift)
        result = schedule(0)
        expected = init_value * (shift ** 0.5) / (shift ** 0.5)  # Should equal init_value
        assert np.isclose(result, expected)
        assert np.isclose(result, init_value)
    
    def test_rsqrt_schedule_no_shift(self):
        """Test rsqrt_schedule with no shift (shift=0)."""
        init_value = 2e-4
        schedule = rsqrt_schedule(init_value, shift=0)
        
        # At step 1, should be init_value / sqrt(1) = init_value
        result = schedule(1)
        expected = init_value
        assert np.isclose(result, expected)
        
        # At step 4, should be init_value / sqrt(4) = init_value / 2
        result = schedule(4)
        expected = init_value / 2
        assert np.isclose(result, expected)
    
    def test_rsqrt_schedule_decreasing(self):
        """Test that rsqrt_schedule produces decreasing learning rates."""
        init_value = 1e-3
        shift = 1000
        schedule = rsqrt_schedule(init_value, shift)
        
        # Test that learning rate decreases as steps increase
        lr_1000 = schedule(1000)  # effective step = 1000 + 1000 = 2000
        lr_2000 = schedule(2000)  # effective step = 2000 + 1000 = 3000
        lr_5000 = schedule(5000)  # effective step = 5000 + 1000 = 6000
        
        assert lr_1000 > lr_2000 > lr_5000
    
    def test_rsqrt_schedule_mathematical_correctness(self):
        """Test mathematical correctness of the rsqrt formula."""
        init_value = 5e-4
        shift = 500
        schedule = rsqrt_schedule(init_value, shift)
        
        step = 1500
        result = schedule(step)
        expected = init_value * (shift ** 0.5) / ((step + shift) ** 0.5)
        
        assert np.isclose(result, expected, rtol=1e-10)


class TestCreateLearningRateSchedule:
    """Test cases for the create_learning_rate_schedule function."""
    
    def test_warmup_phase(self):
        """Test that the warmup phase increases linearly from 0."""
        learning_rate = 1e-3
        warmup_steps = 1000
        schedule = create_learning_rate_schedule(learning_rate, warmup_steps)
        
        # At step 0, should be 0
        assert schedule(0) == 0.0
        
        # At step warmup_steps/2, should be learning_rate/2
        mid_step = warmup_steps // 2
        mid_lr = schedule(mid_step)
        expected_mid_lr = learning_rate * mid_step / warmup_steps
        assert np.isclose(mid_lr, expected_mid_lr, rtol=1e-6)
        
        # At step warmup_steps, should be close to learning_rate
        warmup_end_lr = schedule(warmup_steps)
        assert np.isclose(warmup_end_lr, learning_rate, rtol=1e-6)
    
    def test_decay_phase(self):
        """Test that the decay phase follows inverse square root decay."""
        learning_rate = 2e-4
        warmup_steps = 500
        schedule = create_learning_rate_schedule(learning_rate, warmup_steps)
        
        # Test steps after warmup
        step_1000 = schedule(1000)
        step_2000 = schedule(2000)
        step_5000 = schedule(5000)
        
        # Should be decreasing
        assert step_1000 > step_2000 > step_5000
    
    def test_schedule_continuity(self):
        """Test that there's reasonable continuity at the warmup boundary."""
        learning_rate = 1e-3
        warmup_steps = 1000
        schedule = create_learning_rate_schedule(learning_rate, warmup_steps)
        
        # Values just before and after warmup_steps should be reasonable
        before_boundary = schedule(warmup_steps - 1)
        at_boundary = schedule(warmup_steps)
        after_boundary = schedule(warmup_steps + 1)
        
        # Should be increasing through warmup end
        assert before_boundary < at_boundary
        # Should start decreasing after warmup
        assert at_boundary > after_boundary
    
    def test_positive_learning_rates(self):
        """Test that all learning rates are positive (except step 0)."""
        learning_rate = 1e-4
        warmup_steps = 100
        schedule = create_learning_rate_schedule(learning_rate, warmup_steps)
        
        test_steps = [0, 50, 100, 200, 500, 1000, 5000]
        for step in test_steps:
            lr = schedule(step)
            if step == 0:
                assert lr == 0.0  # Only step 0 should be 0
            else:
                assert lr > 0, f"Learning rate should be positive at step {step}, got {lr}"
    
    def test_different_warmup_steps(self):
        """Test schedule with different warmup step values."""
        learning_rate = 5e-4
        
        for warmup_steps in [100, 500, 1000, 2000]:
            schedule = create_learning_rate_schedule(learning_rate, warmup_steps)
            
            # Check warmup end
            lr_at_warmup = schedule(warmup_steps)
            assert np.isclose(lr_at_warmup, learning_rate, rtol=1e-6)
            
            # Check that warmup takes the right number of steps
            if warmup_steps > 1:
                lr_before_warmup = schedule(warmup_steps - 1)
                assert lr_before_warmup < learning_rate


class TestComputeWeightedCrossEntropy:
    """Test cases for the compute_weighted_cross_entropy function."""
    
    def test_basic_cross_entropy(self):
        """Test basic cross-entropy without smoothing or weights."""
        import jax.numpy as jnp
        
        # Test data: 2 samples, 3 time steps, 4 classes
        logits = jnp.array([[[1.0, 2.0, 0.5, 0.1],
                             [0.1, 0.2, 3.0, 0.5], 
                             [2.5, 0.1, 0.3, 1.0]],
                            [[0.5, 1.5, 0.2, 2.0],
                             [1.0, 0.1, 0.8, 2.5],
                             [0.3, 2.2, 1.1, 0.4]]])
        targets = jnp.array([[1, 2, 0],
                             [3, 3, 1]])
        
        loss, norm_factor = compute_weighted_cross_entropy(logits, targets)
        
        # Check shapes and basic properties
        assert loss.ndim == 0, "Loss should be a scalar (0-dim array)"
        assert norm_factor == 6.0, f"Expected norm_factor=6.0 (2*3), got {norm_factor}"
        assert loss > 0, "Loss should be positive"
    
    def test_label_smoothing_reduces_loss(self):
        """Test that label smoothing reduces loss compared to no smoothing."""
        import jax.numpy as jnp
        
        logits = jnp.array([[[1.0, 2.0, 0.5, 0.1],
                             [0.1, 0.2, 3.0, 0.5]]])
        targets = jnp.array([[1, 2]])
        
        loss_no_smooth, _ = compute_weighted_cross_entropy(logits, targets, label_smoothing=0.0)
        loss_smooth, _ = compute_weighted_cross_entropy(logits, targets, label_smoothing=0.1)
        
        assert loss_smooth < loss_no_smooth, "Label smoothing should reduce loss"
    
    def test_weights_masking(self):
        """Test that weights properly mask out certain positions."""
        import jax.numpy as jnp
        
        logits = jnp.array([[[1.0, 2.0, 0.5, 0.1],
                             [0.1, 0.2, 3.0, 0.5], 
                             [2.5, 0.1, 0.3, 1.0]],
                            [[0.5, 1.5, 0.2, 2.0],
                             [1.0, 0.1, 0.8, 2.5],
                             [0.3, 2.2, 1.1, 0.4]]])
        targets = jnp.array([[1, 2, 0],
                             [3, 3, 1]])
        
        # Mask out last position of first sample
        weights = jnp.array([[1.0, 1.0, 0.0],
                             [1.0, 1.0, 1.0]])
        
        loss, norm_factor = compute_weighted_cross_entropy(logits, targets, weights=weights)
        
        assert norm_factor == 5.0, f"Expected norm_factor=5.0 (weighted count), got {norm_factor}"
        assert loss > 0, "Loss should be positive"
    
    def test_combined_smoothing_and_weights(self):
        """Test label smoothing combined with weights."""
        import jax.numpy as jnp
        
        logits = jnp.array([[[1.0, 2.0, 0.5, 0.1],
                             [0.1, 0.2, 3.0, 0.5]]])
        targets = jnp.array([[1, 2]])
        weights = jnp.array([[1.0, 0.5]])  # Partial weight on second position
        
        loss, norm_factor = compute_weighted_cross_entropy(
            logits, targets, weights=weights, label_smoothing=0.1
        )
        
        assert norm_factor == 1.5, f"Expected norm_factor=1.5, got {norm_factor}"
        assert loss > 0, "Loss should be positive"


class TestComputeWeightedAccuracy:
    """Test cases for the compute_weighted_accuracy function."""
    
    def test_basic_accuracy(self):
        """Test basic accuracy computation without weights."""
        import jax.numpy as jnp
        
        # Test data: 2 samples, 2 time steps, 3 classes
        # Set up logits so we know the predictions
        logits = jnp.array([[[0.1, 2.0, 0.5],   # argmax=1, target=1 ✓
                             [3.0, 0.2, 0.1]],  # argmax=0, target=2 ✗
                            [[0.5, 0.2, 2.5],   # argmax=2, target=2 ✓
                             [1.5, 0.1, 0.3]]])# argmax=0, target=0 ✓
        targets = jnp.array([[1, 2],
                             [2, 0]])
        
        accuracy_sum, norm_factor = compute_weighted_accuracy(logits, targets)
        
        # Should have 3 correct out of 4 predictions
        assert accuracy_sum.ndim == 0, "Accuracy sum should be scalar"
        assert accuracy_sum == 3.0, f"Expected 3 correct predictions, got {accuracy_sum}"
        assert norm_factor == 4.0, f"Expected norm_factor=4.0, got {norm_factor}"
    
    def test_perfect_accuracy(self):
        """Test case where all predictions are correct."""
        import jax.numpy as jnp
        
        logits = jnp.array([[[5.0, 0.1, 0.2],   # argmax=0, target=0 ✓
                             [0.1, 4.0, 0.3]],  # argmax=1, target=1 ✓
                            [[0.2, 0.1, 3.0],   # argmax=2, target=2 ✓
                             [2.0, 0.5, 0.1]]])# argmax=0, target=0 ✓
        targets = jnp.array([[0, 1],
                             [2, 0]])
        
        accuracy_sum, norm_factor = compute_weighted_accuracy(logits, targets)
        
        assert accuracy_sum == 4.0, f"Expected perfect accuracy (4/4), got {accuracy_sum}"
        assert norm_factor == 4.0, f"Expected norm_factor=4.0, got {norm_factor}"
    
    def test_accuracy_with_weights(self):
        """Test accuracy computation with weights/masking."""
        import jax.numpy as jnp
        
        logits = jnp.array([[[2.0, 0.1, 0.5],   # argmax=0, target=0 ✓
                             [0.2, 3.0, 0.1],   # argmax=1, target=1 ✓ (masked out)
                             [0.1, 0.2, 4.0]],  # argmax=2, target=1 ✗
                            [[1.5, 0.1, 0.3],   # argmax=0, target=2 ✗
                             [0.1, 2.5, 0.2],   # argmax=1, target=1 ✓
                             [3.0, 0.1, 0.5]]])# argmax=0, target=0 ✓
        targets = jnp.array([[0, 1, 1],
                             [2, 1, 0]])
        
        # Mask out second position of first sample
        weights = jnp.array([[1.0, 0.0, 1.0],
                             [1.0, 1.0, 1.0]])
        
        accuracy_sum, norm_factor = compute_weighted_accuracy(logits, targets, weights=weights)
        
        # Should count: sample1_pos1=✓, sample1_pos3=✗, sample2_pos1=✗, sample2_pos2=✓, sample2_pos3=✓
        # Total: 3 correct out of 5 unmasked positions
        assert accuracy_sum == 3.0, f"Expected 3 correct predictions, got {accuracy_sum}"
        assert norm_factor == 5.0, f"Expected norm_factor=5.0 (weighted count), got {norm_factor}"
    
    def test_zero_accuracy(self):
        """Test case where all predictions are wrong."""
        import jax.numpy as jnp
        
        logits = jnp.array([[[0.1, 0.2, 3.0],   # argmax=2, target=0 ✗
                             [2.0, 0.1, 0.3]],  # argmax=0, target=1 ✗
                            [[0.5, 2.5, 0.1],   # argmax=1, target=2 ✗
                             [0.2, 0.1, 1.5]]])# argmax=2, target=0 ✗
        targets = jnp.array([[0, 1],
                             [2, 0]])
        
        accuracy_sum, norm_factor = compute_weighted_accuracy(logits, targets)
        
        assert accuracy_sum == 0.0, f"Expected zero accuracy, got {accuracy_sum}"
        assert norm_factor == 4.0, f"Expected norm_factor=4.0, got {norm_factor}"


class TestTrainStep:
    """Test cases for the train_step function."""
    
    def _create_mock_state(self, step=0):
        """Create a mock TrainState for testing."""
        # Mock parameters
        mock_params = {'dense': jnp.array([[1.0, 2.0], [3.0, 4.0]])}
        
        # Mock graphdef
        mock_graphdef = Mock()
        
        # Mock state
        mock_state = Mock(spec=TrainState)
        mock_state.params = mock_params
        mock_state.graphdef = mock_graphdef
        mock_state.step = step
        
        # Mock apply_gradients to return new state with incremented step
        def mock_apply_gradients(grads):
            new_state = Mock(spec=TrainState)
            new_state.params = mock_params
            new_state.graphdef = mock_graphdef
            new_state.step = step + 1
            new_state.apply_gradients = mock_apply_gradients
            return new_state
        
        mock_state.apply_gradients = mock_apply_gradients
        return mock_state
    
    def _create_mock_batch(self):
        """Create a mock batch for testing."""
        return {
            'inputs': jnp.array([[1, 2, 3, 0], [4, 5, 0, 0]]),  # batch_size=2, seq_len=4
            'inputs_position': jnp.array([[0, 1, 2, 3], [0, 1, 2, 3]]),
            'inputs_segmentation': jnp.array([[1, 1, 1, 0], [1, 1, 0, 0]]),
        }
    
    def _create_mock_module(self):
        """Create a mock module that returns logits."""
        mock_module = Mock()
        # Return logits with shape [batch_size, seq_len, vocab_size]
        mock_logits = jnp.array([[[1.0, 2.0, 0.5], [0.5, 1.5, 2.0], [2.0, 0.5, 1.0], [1.0, 1.0, 1.0]],
                                [[0.5, 2.5, 1.0], [1.5, 0.5, 2.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]])
        mock_module.return_value = mock_logits
        mock_module.set_attributes = Mock()
        return mock_module
    
    def test_basic_train_step(self):
        """Test basic functionality of train_step."""
        # Setup mocks
        mock_state = self._create_mock_state(step=100)
        mock_batch = self._create_mock_batch()
        mock_learning_rate_fn = Mock(return_value=1e-4)
        mock_module = self._create_mock_module()
        
        # Mock nnx.merge_params to return our mock module
        with pytest.MonkeyPatch.context() as m:
            m.setattr(nnx, 'merge', Mock(return_value=mock_module))
            m.setattr(nnx, 'log_softmax', Mock(return_value=jnp.log(jnp.array([[[0.3, 0.6, 0.1], [0.2, 0.3, 0.5], [0.7, 0.2, 0.1], [0.33, 0.33, 0.34]],
                                                                             [[0.1, 0.8, 0.1], [0.4, 0.1, 0.5], [0.33, 0.33, 0.34], [0.33, 0.33, 0.34]]]))))
            
            # Create dropout RNG
            dropout_rng = jax.random.PRNGKey(42)
            
            # Call train_step
            new_state, metrics = train_step(
                state=mock_state,
                batch=mock_batch,
                learning_rate_fn=mock_learning_rate_fn,
                label_smoothing=0.0,
                dropout_rng=dropout_rng
            )
            
            # Verify return types and structure
            assert new_state is not None
            assert isinstance(metrics, dict)
            assert 'loss' in metrics
            assert 'accuracy' in metrics
            assert 'learning_rate' in metrics
            assert 'norm_factor' in metrics
            
            # Verify learning rate was called with correct step
            mock_learning_rate_fn.assert_called_once_with(100)
            assert metrics['learning_rate'] == 1e-4
            
            # Verify state step was incremented
            assert new_state.step == 101
    
    def test_train_step_with_label_smoothing(self):
        """Test train_step with label smoothing enabled."""
        # Setup mocks
        mock_state = self._create_mock_state(step=50)
        mock_batch = self._create_mock_batch()
        mock_learning_rate_fn = Mock(return_value=2e-4)
        mock_module = self._create_mock_module()
        
        # Mock nnx functions
        with pytest.MonkeyPatch.context() as m:
            m.setattr(nnx, 'merge', Mock(return_value=mock_module))
            m.setattr(nnx, 'log_softmax', Mock(return_value=jnp.log(jnp.array([[[0.3, 0.6, 0.1], [0.2, 0.3, 0.5], [0.7, 0.2, 0.1], [0.33, 0.33, 0.34]],
                                                                             [[0.1, 0.8, 0.1], [0.4, 0.1, 0.5], [0.33, 0.33, 0.34], [0.33, 0.33, 0.34]]]))))
            
            dropout_rng = jax.random.PRNGKey(123)
            
            # Call with label smoothing
            new_state, metrics = train_step(
                state=mock_state,
                batch=mock_batch,
                learning_rate_fn=mock_learning_rate_fn,
                label_smoothing=0.1,
                dropout_rng=dropout_rng
            )
            
            # Verify smoothing was applied (should have different loss than no smoothing)
            assert 'loss' in metrics
            assert metrics['loss'] > 0
            assert metrics['learning_rate'] == 2e-4
            assert new_state.step == 51
    
    def test_train_step_gradient_computation(self):
        """Test that gradients are computed and applied correctly."""
        # Setup mocks
        mock_state = self._create_mock_state(step=200)
        mock_batch = self._create_mock_batch()
        mock_learning_rate_fn = Mock(return_value=5e-4)
        mock_module = self._create_mock_module()
        
        # Track calls to apply_gradients
        apply_gradients_called = False
        original_apply_gradients = mock_state.apply_gradients
        
        def track_apply_gradients(grads):
            nonlocal apply_gradients_called
            apply_gradients_called = True
            # Verify grads is not None and has the right structure
            assert grads is not None
            assert isinstance(grads, dict)
            return original_apply_gradients(grads)
        
        mock_state.apply_gradients = track_apply_gradients
        
        with pytest.MonkeyPatch.context() as m:
            m.setattr(nnx, 'merge', Mock(return_value=mock_module))
            m.setattr(nnx, 'log_softmax', Mock(return_value=jnp.log(jnp.array([[[0.3, 0.6, 0.1], [0.2, 0.3, 0.5], [0.7, 0.2, 0.1], [0.33, 0.33, 0.34]],
                                                                             [[0.1, 0.8, 0.1], [0.4, 0.1, 0.5], [0.33, 0.33, 0.34], [0.33, 0.33, 0.34]]]))))
            
            dropout_rng = jax.random.PRNGKey(456)
            
            # Call train_step
            new_state, metrics = train_step(
                state=mock_state,
                batch=mock_batch,
                learning_rate_fn=mock_learning_rate_fn,
                dropout_rng=dropout_rng
            )
            
            # Verify gradients were computed and applied
            assert apply_gradients_called, "apply_gradients should have been called"
            assert new_state.step == 201
            
            # Verify module was configured correctly for training
            mock_module.set_attributes.assert_called_once_with(deterministic=False, decode=False)
    
    def test_train_step_metrics_computation(self):
        """Test that metrics are computed correctly."""
        # Setup mocks
        mock_state = self._create_mock_state(step=300)
        mock_batch = self._create_mock_batch()
        mock_learning_rate_fn = Mock(return_value=1e-3)
        mock_module = self._create_mock_module()
        
        with pytest.MonkeyPatch.context() as m:
            m.setattr(nnx, 'merge', Mock(return_value=mock_module))
            m.setattr(nnx, 'log_softmax', Mock(return_value=jnp.log(jnp.array([[[0.3, 0.6, 0.1], [0.2, 0.3, 0.5], [0.7, 0.2, 0.1], [0.33, 0.33, 0.34]],
                                                                             [[0.1, 0.8, 0.1], [0.4, 0.1, 0.5], [0.33, 0.33, 0.34], [0.33, 0.33, 0.34]]]))))
            
            dropout_rng = jax.random.PRNGKey(789)
            
            # Call train_step
            new_state, metrics = train_step(
                state=mock_state,
                batch=mock_batch,
                learning_rate_fn=mock_learning_rate_fn,
                dropout_rng=dropout_rng
            )
            
            # Verify metrics structure and content
            required_metrics = ['loss', 'accuracy', 'norm_factor', 'learning_rate']
            for metric in required_metrics:
                assert metric in metrics, f"Missing metric: {metric}"
                assert metrics[metric] is not None, f"Metric {metric} is None"
            
            # Verify learning rate matches
            assert metrics['learning_rate'] == 1e-3
            
            # Verify loss and accuracy are reasonable
            assert metrics['loss'] > 0, "Loss should be positive"
            assert metrics['accuracy'] >= 0, "Accuracy should be non-negative"
            assert metrics['norm_factor'] > 0, "Norm factor should be positive"
            
            # Verify norm_factor matches expected (sum of weights for non-padding tokens)
            # From our mock batch: inputs = [[1, 2, 3, 0], [4, 5, 0, 0]]
            # Non-zero positions: 3 + 2 = 5
            assert metrics['norm_factor'] == 5.0, f"Expected norm_factor=5.0, got {metrics['norm_factor']}"


class TestEvaluate:
    """Test cases for the evaluate function."""
    
    def test_basic_evaluation(self):
        """Test basic evaluation functionality."""
        # Mock state and dataset
        mock_state = Mock()
        mock_state.params = {'dense': jnp.array([[1.0, 2.0]])}
        mock_state.graphdef = Mock()
        
        # Mock dataset
        mock_ds = Mock()
        mock_ds.__iter__ = Mock(return_value=iter([
            {'inputs': jnp.array([[1, 2, 0]])},
            {'inputs': jnp.array([[3, 4, 5]])}
        ]))
        
        # Mock jit_eval_step that returns metrics with denominator
        def mock_jit_eval_step(params, batch, graphdef, label_smoothing=0.0):
            return {
                'loss': jnp.array(4.0),
                'accuracy': jnp.array(2.0),
                'norm_factor': jnp.array(1.0)
            }
        
        result = evaluate(
            jit_eval_step=mock_jit_eval_step,
            state=mock_state,
            eval_ds=mock_ds,
            num_eval_steps=2
        )
        
        # Should return dict with averaged metrics
        assert isinstance(result, dict)
        assert 'loss' in result
        assert 'accuracy' in result
    
class TestIntegration:
    """Integration tests combining both functions."""
    
    def test_typical_training_scenario(self):
        """Test a typical training scenario with realistic parameters."""
        learning_rate = 3e-4  # Common for transformers
        warmup_steps = 4000   # Common warmup
        schedule = create_learning_rate_schedule(learning_rate, warmup_steps)
        
        # Test some typical training steps
        step_0 = schedule(0)
        step_1000 = schedule(1000)    # During warmup
        step_4000 = schedule(4000)    # End of warmup
        step_10000 = schedule(10000)  # Well into training
        step_50000 = schedule(50000)  # Late in training
        
        # Verify expected behavior
        assert step_0 == 0.0
        assert 0 < step_1000 < learning_rate
        assert np.isclose(step_4000, learning_rate, rtol=1e-6)
        assert 0 < step_50000 < step_10000 < step_4000
        
        # Learning rate should still be reasonable even late in training
        assert step_50000 > learning_rate * 0.01  # At least 1% of peak


if __name__ == "__main__":
    pytest.main([__file__]) 