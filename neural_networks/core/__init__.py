# coding: utf-8

"""Abstract class defining the API and common methods of neural networks."""

from ._build_layers import (
    build_layers_stack, get_layer_class, validate_layer_config
)
from ._components import (
    build_dynamic_weights_matrix, build_rmse_readouts, refine_signal
)
from ._dnn import DeepNeuralNetwork, load_dumped_model
