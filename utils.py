from flax.training import train_state
from flax import nnx
import logging
import jax
import numpy as np
from configs import default
from jax.experimental import mesh_utils
from jax import numpy as jnp
from typing import Callable
from model import TransformerLM, TransformerConfig

class TrainState(train_state.TrainState):
    graphdef: nnx.GraphDef



def create_device_mesh(config: default.Config):
  """Creates a device mesh with each slice in its own data parallel group. If there is only one slice, uses two replicas."""
  devices = jax.devices()
  num_devices = len(devices)
  try:
    num_slices = 1 + max([d.slice_index for d in devices])
  except:
    num_slices = 1

  num_devices_per_slice = num_devices // num_slices
  logging.info(f'Devices: {devices}')
  logging.info(f'Number of devices: {num_devices}')

  multi_slice_env = hasattr(jax.devices()[0], 'slice_index')

  dcn_parallelism = [
    config.dcn_data_parallelism,
    config.dcn_fsdp_parallelism,
    config.dcn_tensor_parallelism,
  ]
  ici_parallelism = [
    config.ici_data_parallelism,
    config.ici_fsdp_parallelism,
    config.ici_tensor_parallelism,
  ]

  # Find possible unspecified parallelisms
  dcn_parallelism = fill_unspecified_mesh_axes(
    dcn_parallelism, num_slices, 'DCN'
  )
  ici_parallelism = fill_unspecified_mesh_axes(
    ici_parallelism, num_devices_per_slice, 'ICI'
  )

  if multi_slice_env:
    mesh = mesh_utils.create_hybrid_device_mesh(
      ici_parallelism, dcn_parallelism
    )
  else:
    mesh = mesh_utils.create_device_mesh(ici_parallelism)

  logging.info(f'Decided on mesh: {mesh}')
  logging.info(f'Mesh shape: {mesh.shape}')

  return mesh
  


def fill_unspecified_mesh_axes(
  parallelism_vals, target_product, parallelism_type
):
  """Evaluates unspecified DCN/ICI parallelism values"""
  if -1 in parallelism_vals:
    assert parallelism_vals.count(-1) == 1, (
      f'Found unspecified values (-1) for more than one {parallelism_type}   '
      '   parallelism axis. At most one axis can be unspecified.'
    )

    determined_val = target_product / np.prod(parallelism_vals) * -1

    assert determined_val >= 1 and determined_val.is_integer, (
      'Unspecified value unable to be determined with the given     '
      f' {parallelism_type} parallelism values'
    )

    parallelism_vals[parallelism_vals.index(-1)] = int(determined_val)

  target_type = 'slices' if parallelism_type == 'DCN' else 'devices per slice'

  assert np.prod(parallelism_vals) == target_product, (
    f'Number of {target_type} {target_product} does not match    the product'
    f' of the {parallelism_type} parallelism {np.prod(parallelism_vals)}'
  )

  return parallelism_vals

def _to_array(x):
  if not isinstance(x, jax.Array):
    x = jnp.asarray(x)
  return x

def setup_initial_state(
  constructor: Callable[[TransformerConfig, jax.Array], TransformerLM],
  tx,
  config: TransformerConfig,
  rng: jax.Array,
  mesh: jax.sharding.Mesh,
) -> tuple[TrainState, TrainState]:
  """We initialize the model and optimizer state, and optionally load from a
  checkpoint as necessary.

  Args:
    constructor: the model constructor
    tx: the optax.GradientTransformation
    config: config object
    rng: jax.prng key
    mesh: jax.devices() mesh

  Returns:
    state: the initialized train state
    state_mesh_annotations: the mesh annotations for the train state
  """

  # Initialization

  # SHARDING: Establish mesh context - all operations inside will be aware of the device mesh
  with mesh:
    # SHARDING: Create model with sharding annotations attached via nnx.with_partitioning
    # At this point, parameters have partition metadata but are NOT actually sharded
    model = constructor(config, rng)
    
    # SHARDING: Split model into structure and parameters
    # The sharding annotations stay with the parameters in the 'params' object
    graphdef, params = nnx.split(model, nnx.Param)
    
    # SHARDING: Create training state - parameters are still replicated/on single device
    # TrainState.create() doesn't perform any sharding, just packages the data
    state = TrainState.create(
      apply_fn=graphdef.apply,
      params=params,  # Contains sharding annotations but data is not sharded yet
      tx=tx,
      graphdef=graphdef,
    )
    
    # SHARDING: Convert all data to JAX arrays (required for sharding operations)
    # Non-JAX arrays cannot be sharded, so this ensures compatibility
    state = jax.tree.map(_to_array, state)
    
    # SHARDING: Extract the sharding blueprint from the annotated parameters
    # This reads the PartitionSpec annotations but doesn't move any data
    state_spec = nnx.get_partition_spec(state)
    
    # SHARDING: Actually apply the sharding! This is where data gets moved between devices
    # Compares current sharding with desired state_spec and reshards accordingly
    state = jax.lax.with_sharding_constraint(state, state_spec)

  # SHARDING: Create the concrete sharding object for JIT compilation
  # This combines the partition specs with the mesh to create NamedSharding objects
  # that JAX can use for efficient distributed computation
  state_sharding = nnx.get_named_sharding(state, mesh)
  return state, state_sharding