from flax.training import train_state
from flax import nnx


class TrainState(train_state.TrainState):
    graphdef: nnx.GraphDef