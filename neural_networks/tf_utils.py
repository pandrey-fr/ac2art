# coding: utf-8

"""Set of tensorflow-related utility functions."""

import inspect

import tensorflow as tf
import numpy as np


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


def get_activation_function(function_name):
    """Return the tensorflow activation function of given name."""
    if function_name not in ACTIVATION_FUNCTIONS.keys():
        if not function_name.count('.'):
            raise KeyError(
                "Invalid activation function name: '%s'.\n"
                "A valid name should either belong to {'%s'} or "
                "consist of a full module name and function name."
                % (function_name, "', '".join(list(ACTIVATION_FUNCTIONS)))
            )
        module_name, function_name = function_name.rsplit('.', 1)
        module = __import__(module_name, fromlist=[module_name])
        return getattr(module, function_name)
    return ACTIVATION_FUNCTIONS[function_name]


def get_activation_function_name(function):
    """Return the name of a given activation function."""
    functions = list(ACTIVATION_FUNCTIONS.values())
    if function not in functions:
        return function.__module__ + '.' + function.__name__
    return list(ACTIVATION_FUNCTIONS.keys())[functions.index(function)]


def setup_activation_function(activation):
    """Validate and return a tensorflow activation function.

    activation : either an actual function, returned as is,
                 or a function name, from which the actual
                 function is looked for and returned.
    """
    if isinstance(activation, str):
        return get_activation_function(activation)
    elif inspect.isfunction(activation):
        return activation
    else:
        raise TypeError("'activation' should be a str or a function.")


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
