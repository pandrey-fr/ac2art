# coding: utf-8

"""Set of tensorflow-related utility functions."""

from ._tf_utils import (
    add_dynamic_features,
    batch_tensor_mean,
    binary_step,
    conv2d,
    get_activation_function_name,
    get_delta_features,
    get_simple_difference,
    get_rnn_cell_type_name,
    index_tensor,
    log_base,
    minimize_safely,
    reduce_finite_mean,
    run_along_first_dim,
    setup_activation_function,
    setup_rnn_cell_type,
    sinc,
    tensor_length
)
