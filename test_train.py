"""Tests for training utilities in train.py"""

import pytest
import numpy as np

# Import the functions directly - make sure this file is in the same directory as train.py
try:
    from train import rsqrt_schedule, create_learning_rate_schedule
except ImportError:
    import sys
    sys.path.append('.')
    from train import rsqrt_schedule, create_learning_rate_schedule


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