# coding: utf-8

"""Set of tensorflow-related utility functions."""

import inspect

import tensorflow as tf
import numpy as np

from neural_networks.utils import (
    check_type_validity, get_object, get_object_name
)


def binary_step(tensor):
    """Return a binary output depending on an input's positivity."""
    return tf.cast(tensor > 0, tf.float32)


def conv2d(input_data, weights):
    """Convolute 2-D inputs to a 4-D weights matrix filter."""
    return tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], 'SAME')


ACTIVATION_FUNCTIONS = {
    'identity': tf.identity, 'binary': binary_step,
    'leaky_relu': tf.nn.leaky_relu, 'relu': tf.nn.relu,
    'sigmoid': tf.nn.sigmoid, 'softmax': tf.nn.softmax,
    'softplus': tf.nn.softplus, 'tanh': tf.nn.tanh
}


RNN_CELL_TYPES = {
    'lstm': tf.nn.rnn_cell.LSTMCell, 'gru': tf.nn.rnn_cell.GRUCell
}


def get_activation_function_name(function):
    """Return the short name or full import name of an activation function."""
    return get_object_name(function, ACTIVATION_FUNCTIONS)


def get_rnn_cell_type_name(cell_type):
    """Return the short name or full import name of a RNN cell type."""
    return get_object_name(cell_type, RNN_CELL_TYPES)


def index_tensor(tensor, start=0):
    """Add an index column to a given 1-D tensor.

    Return a 2-D tensor whose first column is an index ranging
    from `start` with a unit step, and whose second column is
    the initially provided `tensor`.
    """
    tf.assert_rank(tensor, 1)
    n_obs = tensor_length(tensor)
    count = tf.range(start, n_obs + start, dtype=tensor.dtype)
    return tf.concat(
        [tf.expand_dims(count, 1), tf.expand_dims(tensor, 1)], axis=1
    )


def log_base(tensor, base):
    """Compute the logarithm of a given tensorflow Tensor in a given base."""
    if isinstance(base, tf.Tensor) and base.dtype in [tf.int32, tf.int64]:
        base = tf.cast(base, tf.float32)
    elif isinstance(base, int):
        base = float(base)
    if tensor.dtype in [tf.int32, tf.int64]:
        tensor = tf.cast(tensor, tf.float32)
    return tf.log(tensor) / tf.log(base)


def setup_activation_function(activation):
    """Validate and return a tensorflow activation function.

    activation : either an actual function, returned as is,
                 or a function name, from which the actual
                 function is looked for and returned.
    """
    if isinstance(activation, str):
        return get_object(
            activation, ACTIVATION_FUNCTIONS, 'activation function'
        )
    elif inspect.isfunction(activation):
        return activation
    else:
        raise TypeError("'activation' should be a str or a function.")


def setup_rnn_cell_type(cell_type):
    """Validate and return a tensorflow RNN cell type.

    cell_type : either an actual cell type, returned as is,
                or a cell type name, from which the actual
                type is looked for and returned.
    """
    check_type_validity(cell_type, (str, type), 'cell_type')
    if isinstance(cell_type, str):
        return get_object(
            cell_type, RNN_CELL_TYPES, 'RNN cell type'
        )
    elif issubclass(cell_type, tf.nn.rnn_cell.RNNCell):
        return cell_type
    raise TypeError(
        "'cell_type' is not a tensorflow.nn.rnn_cell.RNNCell subclass."
        )


def sinc(tensor):
    """Compute the normalized sinc of a tensorflow Tensor."""
    normalized = np.pi * tensor
    is_zero = tf.cast(tf.equal(tensor, 0), tf.float32)
    return is_zero + tf.sin(normalized) / (normalized + 1e-30)


def tensor_length(tensor):
    """Return a Tensor recording the length of another Tensor."""
    sliced = tf.slice(
        tensor, [0] * len(tensor.shape), [-1] + [1] * (len(tensor.shape) - 1)
    )
    return tf.reduce_sum(tf.ones_like(sliced, dtype=tf.int32))
