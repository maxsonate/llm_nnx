"""Training utilities for learning rate scheduling.

This module provides functions for creating learning rate schedules commonly used
in transformer training, including warmup and inverse square root decay schedules.
"""
from typing import Callable
import logging
import jax.numpy as jnp
import jax
import optax
import model
import modules
from flax.training import common_utils
from flax import nnx
import numpy as np
from utils import TrainState
from configs import default
import input_pipeline
import os
import temperature_sampler
import utils
from jax.sharding import Mesh, NamedSharding


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


def train_step(
  state: TrainState,
  batch,
  learning_rate_fn,
  label_smoothing=0.0,
  dropout_rng=None,
):
  """Perform a single training step with gradient computation and state update.
  
  Args:
    state: Current training state containing model parameters and optimizer state.
    batch: Training batch containing input sequences and metadata.
    learning_rate_fn: Function that returns learning rate for given step.
    label_smoothing: Label smoothing factor for cross-entropy loss (0.0 = no smoothing).
    dropout_rng: Random number generator key for dropout operations.
    
  Returns:
    Tuple of (updated_state, metrics) where metrics contains loss, accuracy, and lr.
  """
  # Extract input sequences and position/segmentation info from batch
  train_keys = ['inputs', 'inputs_position', 'inputs_segmentation']
  inputs, input_positions, input_segmentation = (batch.get(k, None) for k in train_keys)

  # Create attention mask: 1.0 for real tokens, 0.0 for padding tokens
  weights = jnp.where(inputs > 0, 1.0, 0.0).astype(jnp.float32)

  # Generate unique dropout RNG for this training step
  dropout_rngs = jax.random.fold_in(dropout_rng, state.step)

  def loss_fn(params):
    """Compute loss and logits for given parameters."""
    # Reconstruct model with current parameters
    module = nnx.merge(state.graphdef, params)
    module.set_attributes(deterministic=False, decode=False)

    # Forward pass through model
    logits = module(inputs, input_positions, input_segmentation, nnx.Rngs(dropout_rngs))

    # Compute weighted cross-entropy loss
    loss, weight_sum = compute_weighted_cross_entropy(logits, inputs, weights, label_smoothing)
    mean_loss = loss / weight_sum
    return mean_loss, logits
  
  # Get current learning rate for this step
  step = state.step
  lr = learning_rate_fn(step)
  
  # Compute gradients and loss simultaneously
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, logits), grads = grad_fn(state.params)

  # Apply gradients to update model parameters
  new_state = state.apply_gradients(grads=grads)
  
  # Compute training metrics (loss, accuracy, etc.)
  metrics = compute_metrics(logits, inputs, weights, label_smoothing)
  metrics['learning_rate'] = lr

  return new_state, metrics

def eval_step(
  params: nnx.State,
  batch,
  graphdef: nnx.GraphDef[model.TransformerLM],
  label_smoothing=0.0,
):
   
   inputs = batch['inputs']
   weights = jnp.where(inputs > 0, 1.0, 0.0).astype(jnp.float32)

   module = nnx.merge(graphdef, params)
   module.set_attributes(deterministic=True, decode=False)

   logits = module(inputs)
   # This is assuming that inputs are not packed.
   return compute_metrics(logits, inputs, weights, label_smoothing)
  

def evaluate(
    *,
    jit_eval_step,
    state: TrainState,
    eval_ds: "tf.data.Dataset",
    num_eval_steps: int,
    label_smoothing: float = 0.0,
):
    """Evaluate the model on the given dataset.
    
    Args:
        jit_eval_step: JIT-compiled evaluation step function.
        state: Current training state with model parameters.
        eval_ds: Evaluation dataset.
        num_eval_steps: Number of evaluation steps to run.
        label_smoothing: Label smoothing factor for evaluation.
        
    Returns:
        Dictionary of averaged evaluation metrics.
    """
    logging.info(f"Starting evaluation for {num_eval_steps} steps...")
    
    # Collect metrics from all evaluation batches
    all_metrics = []
    
    for step, batch in enumerate(eval_ds):
        if step >= num_eval_steps:
            break
            
        # Convert batch to numpy if needed (handle both tensor types gracefully)
        try:
            if hasattr(batch, 'numpy'):
                batch = jax.tree.map(lambda x: x.numpy(), batch)
        except AttributeError:
            # Batch is already in the correct format
            pass
            
        # Evaluate single batch
        batch_metrics = jit_eval_step(
            state.params, batch, state.graphdef, label_smoothing=label_smoothing
        )
        all_metrics.append(batch_metrics)
    
    # Aggregate metrics across all batches
    return _aggregate_metrics(all_metrics)


def _aggregate_metrics(metrics_list):
    """Aggregate metrics from multiple evaluation batches."""
    if not metrics_list:
        raise ValueError("No metrics to aggregate")
    
    # Stack and sum metrics across batches
    stacked_metrics = common_utils.stack_forest(metrics_list)
    summed_metrics = jax.tree.map(jnp.sum, stacked_metrics)
    
    # Extract normalization factor and compute averages
    total_norm_factor = summed_metrics.pop('norm_factor')
    averaged_metrics = jax.tree.map(
        lambda x: x / total_norm_factor, summed_metrics
    )
    
    logging.info(f"Evaluation completed. Total samples: {total_norm_factor}")
    return averaged_metrics


def train_and_evaluate(config: default.Config, workdir: str):
  """Runs a training and evaluation loop.

  """

  workdir = os.path.abspath(workdir)
  os.makedirs(workdir, exist_ok=True)

  vocab_path = config.vocab_path
  if vocab_path is None:
    vocab_path = os.path.join(workdir, 'sentencepiece_model')
    config.vocab_path = vocab_path
  os.makedirs(os.path.split(vocab_path)[0], exist_ok=True)

  # Load Dataset
  # ---------------------------------------------------------------------------
  logging.info('Initializing dataset.')
  train_ds, eval_ds, _, encoder = input_pipeline.get_datasets(
    n_devices=jax.local_device_count(), config=config, vocab_path=vocab_path
  )

  train_iter = iter(train_ds)
  vocab_size = int(encoder.vocab_size())

  eos_id = temperature_sampler.EOS_ID  # Default Sentencepiece EOS token.

  def decode_tokens(toks):
    valid_toks = toks[: np.argmax(toks == eos_id) + 1].astype(np.int32)
    return encoder.detokenize(valid_toks).numpy().decode('utf-8')

  def encode_strings(strs, max_len):
    tokenized_batch = np.zeros((len(strs), max_len), np.int32)
    for i, s in enumerate(strs):
      toks = encoder.tokenize(s).numpy()
      # Remove EOS token in prompt.
      tokenized_batch[i, : toks.shape[0] - 1] = toks[:-1]
    return tokenized_batch

  tokenized_prompts = encode_strings(
    [config.prompts], config.max_predict_length
  )

  logging.info('Initializing model, optimizer, and step functions.')
  # Build Model and Optimizer
 # ---------------------------------------------------------------------------
  model_config = modules.TransformerConfig(
    vocab_size=vocab_size,
    output_vocab_size=vocab_size,
    logits_via_embedding=config.logits_via_embedding,
    dtype=jnp.bfloat16 if config.use_bfloat16 else jnp.float32,
    emb_dim=config.emb_dim,
    num_heads=config.num_heads,
    num_layers=config.num_layers,
    qkv_dim=config.qkv_dim,
    mlp_dim=config.mlp_dim,
    max_len=max(config.max_target_length, config.max_eval_target_length),
    dropout_rate=config.dropout_rate,
    attention_dropout_rate=config.attention_dropout_rate,
    kernel_init=nnx.initializers.xavier_uniform(),
    bias_init=nnx.initializers.normal(stddev=1e-6),
    axis_rules=config.axis_rules,
  )


# Mesh definition
  devices_array = utils.create_device_mesh(config)
  mesh = Mesh(devices_array, config.mesh_axes)

  start_step = 0
  rng = jax.random.PRNGKey(config.seed)
  rng, init_rng = jax.random.split(rng)
  rng, inference_rng = jax.random.split(rng)

  def constructor(config: modules.TransformerConfig, key: jax.Array):
    return model.TransformerLM(config, rngs=nnx.Rngs(params=key))
  
  # TODO: add create mesh device to utils.
  