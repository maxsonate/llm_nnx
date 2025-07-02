"""Training utilities for learning rate scheduling.

This module provides functions for creating learning rate schedules commonly used
in transformer training, including warmup and inverse square root decay schedules.
"""
from typing import Callable
import jax.numpy as jnp
import optax
from flax.training import common_utils
from flax import nnx
import numpy as np

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



def compute_weighted_cross_entropy(
  logits, targets, weights=None, label_smoothing=0.0
):
  """Compute weighted cross entropy and entropy for log probs and targets.

  Args:
   logits: [batch, length, num_classes] float array.
   targets: categorical targets [batch, length] int array.
   weights: None or array of shape [batch, length].
   label_smoothing: label smoothing constant, used to determine the on and off
     values.

  Returns:
    Tuple of scalar loss and batch normalizing factor.
  """
  if logits.ndim != targets.ndim + 1:
    raise ValueError(
      'Incorrect shapes. Got shape %s logits and %s targets'
      % (str(logits.shape), str(targets.shape))
    )
  vocab_size = logits.shape[-1]
  # Label smoothing: correct class gets 'confidence' prob, others get 'low_confidence'  
  confidence = 1.0 - label_smoothing  # e.g., 0.9 for smoothing=0.1
  low_confidence = (1.0 - confidence) / (vocab_size - 1)  # remaining prob split among wrong classes
  
  # Normalizing constant accounts for entropy added by label smoothing
  normalizing_constant = -(
    confidence * jnp.log(confidence)
    + (vocab_size - 1) * low_confidence * jnp.log(low_confidence + 1e-20)
  )
  
  # Create soft targets: [0.03, 0.9, 0.03, 0.03] instead of [0, 1, 0, 0]
  soft_targets = common_utils.onehot(
    targets, vocab_size, on_value=confidence, off_value=low_confidence
  )

  loss = -jnp.sum(soft_targets * nnx.log_softmax(logits), axis=-1)
  loss = loss - normalizing_constant

  normalizing_factor = np.prod(targets.shape)
  if weights is not None:
    loss = loss * weights
    normalizing_factor = weights.sum()

  return loss.sum(), normalizing_factor


def compute_weighted_accuracy(logits, targets, weights=None):
  """Compute weighted accuracy for log probs and targets.

  Args:
   logits: [batch, length, num_classes] float array.
   targets: categorical targets [batch, length] int array.
   weights: None or array of shape [batch, length]

  Returns:
    Tuple of scalar loss and batch normalizing factor.
  """

  if logits.ndim != targets.ndim + 1:
    raise ValueError(
      'Incorrect shapes. Got shape %s logits and %s targets'
      % (str(logits.shape), str(targets.shape))
    )
  

  # Compute accuracy: 1 for correct predictions, 0 for incorrect
  accuracy = logits.argmax(axis=-1) == targets

  normalizing_factor = np.prod(targets.shape)
  if weights is not None:
    accuracy = accuracy * weights
    normalizing_factor = weights.sum()

  return accuracy.sum(), normalizing_factor

def compute_metrics(logits, labels, weights, label_smoothing=0.0):
  """Compute summary metrics."""

  loss, norm_factor = compute_weighted_cross_entropy(logits, labels, weights, label_smoothing)
  accuracy_sum, _ = compute_weighted_accuracy(logits, labels, weights)

  return {
    'loss': loss,
    'accuracy': accuracy_sum,
    'norm_factor': norm_factor,
  }