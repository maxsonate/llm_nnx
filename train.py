"""Training utilities for learning rate scheduling.

This module provides functions for creating learning rate schedules commonly used
in transformer training, including warmup and inverse square root decay schedules.
"""
from typing import Callable
import optax


def rsqrt_schedule(init_value: float, shift: int = 0) -> Callable[[int], float]:
    """Create an inverse square root learning rate schedule.
    
    This schedule implements the inverse square root decay commonly used in 
    transformer training, where the learning rate follows the pattern:
    lr = init_value * sqrt(shift) / sqrt(step + shift)
    
    Args:
        init_value: The initial learning rate value.
        shift: Number of steps to shift the schedule. This is typically set to
            the number of warmup steps to ensure smooth transition from warmup.
            When shift=0, uses max(step, 1) to avoid division by zero at step 0.
            
    Returns:
        A callable that takes a step number and returns the learning rate for that step.
    """
    def schedule(step: int) -> float:
        if shift == 0:
            # Handle edge case where shift is 0 to avoid division by zero at step 0
            effective_step = max(step, 1)
            return init_value / (effective_step ** 0.5)
        else:
            # Standard rsqrt schedule: lr = init_value * sqrt(shift) / sqrt(step + shift)
            # Handle JAX evaluation where both branches of jnp.where get called
            effective_step = max(step + shift, 1)
            return init_value * (shift ** 0.5) / (effective_step ** 0.5)

    return schedule


def create_learning_rate_schedule(learning_rate: float, warmup_steps: int) -> Callable[[int], float]:
    """Create a complete learning rate schedule with warmup and inverse square root decay.
    
    This function creates a two-phase learning rate schedule:
    1. Linear warmup from 0 to the target learning rate over warmup_steps
    2. Inverse square root decay starting from the target learning rate
    
    This schedule is commonly used in transformer training and helps with training
    stability and convergence.
    
    Args:
        learning_rate: The peak learning rate to reach after warmup.
        warmup_steps: Number of steps for the linear warmup phase.
        
    Returns:
        A callable that takes a step number and returns the learning rate for that step.
    """
    return optax.join_schedules(
        [
            optax.linear_schedule(
                init_value=0,
                end_value=learning_rate,
                transition_steps=warmup_steps,
            ),
            rsqrt_schedule(init_value=learning_rate, shift=warmup_steps),
        ],
        boundaries=[warmup_steps],
    )

