# coding: utf-8

"""Set of tensorflow-related utility functions."""

import inspect

import tensorflow as tf
import numpy as np

from utils import check_type_validity, get_object, get_object_name


def binary_step(tensor):
    """Return a binary output depending on an input's positivity."""
    return tf.cast(tensor > 0, tf.float32)


def conv2d(input_data, weights):
    """Convolute 2-D inputs to a 4-D weights matrix filter."""
    return tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], 'SAME')


ACTIVATION_FUNCTIONS = {
    'identity': tf.identity, 'binary': binary_step, 'relu': tf.nn.relu,
    'sigmoid': tf.nn.sigmoid, 'softmax': tf.nn.softmax,
    'softplus': tf.nn.softplus, 'tanh': tf.nn.tanh
}


RNN_CELL_TYPES = {
    'lstm': tf.nn.rnn_cell.LSTMCell, 'gru': tf.nn.rnn_cell.GRUCell
}


def get_activation_function_name(function):
    """Return the short name or full import name of an activation function."""
    return get_object_name(function, ACTIVATION_FUNCTIONS)


def get_delta_features(tensor, window=5):
    """Compute and return delta features, using a given time window.

    static_features : 2-D tensor of values whose delta to compute
    window          : half-size of the time window used (int, default 5)
    """
    norm = 2 * sum(i ** 2 for i in range(1, window + 1))
    delta = tf.add_n([
        get_simple_difference(tensor, lag=i) * i for i in range(1, window + 1)
    ])
    return delta / norm


def get_rnn_cell_type_name(cell_type):
    """Return the short name or full import name of a RNN cell type."""
    return get_object_name(cell_type, RNN_CELL_TYPES)


def get_simple_difference(tensor, lag):
    """Compute and return the simple difference of a series for a given lag.

    tensor : 2-D tensor whose first dimension is time
    lag    : lag to use, so that the difference at time t is
             between values at times t + lag and t - lag
    """
    padding = tf.ones((lag, tensor.shape[1].value))
    past = tf.concat([padding * tensor[0], tensor[:-lag]], axis=0)
    future = tf.concat([tensor[lag:], padding * tensor[-1]], axis=0)
    return future - past


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


def minimize_safely(optimizer, loss, var_list=None, reduce_fn=None):
    """Minimize a given loss function, making sure no NaN is propagated.

    optimizer : optimizer to use (tf.train.Optimizer instance)
    loss      : loss function to minimize (tf.Tensor)
    var_list  : optional list of tf.Variable elements to optimize
    reduce_fn : optional tensorflow operation to use so as to derive
                the default value by which to replace NaN values
                (by default, NaNs are replaced with 0.)
    """
    # Compute the gradients.
    gradients = optimizer.compute_gradients(loss=loss, var_list=var_list)

    # Set up a NaN values replacing function.
    def clean(gradient):
        """Replace all NaN values in a given gradient Tensor."""
        default = tf.cond(
            tf.reduce_sum(tf.cast(tf.is_finite(gradient), tf.float32)) > 0,
            lambda: (
                tf.zeros_like(gradient) if reduce_fn is None
                else tf.ones_like(gradient) * reduce_fn(gradient)
            ),
            lambda: tf.zeros_like(gradient),
        )
        return tf.where(tf.is_finite(gradient), gradient, default)

    # Replace all NaN values and apply the gradients.
    gradients = [
        (clean(gradient), variable) for gradient, variable in gradients
    ]
    return optimizer.apply_gradients(gradients)


def reduce_finite_mean(tensor, axis=None):
    """Compute the mean of finite elements across a tensor's dimensions.

    tensor : numeric-type tensor to reduce
    axis   : optional dimension index along which to reduce (int, default None)
    """
    # Check argument's type validity.
    check_type_validity(tensor, tf.Tensor, 'tensor')
    check_type_validity(axis, (int, type(None)), 'axis')
    # Compute the number of non-Nan elements across the reduction axis.
    is_finite = tf.is_finite(tensor)
    if axis is None:
        length = tf.reduce_sum(tf.ones_like(tensor, dtype=tf.int32))
    elif axis == 0:
        length = tensor_length(tensor)
    else:
        perm = [{0: axis, axis: 0}.get(i, i) for i in range(len(tensor.shape))]
        length = tensor_length(tf.transpose(tensor, perm=perm))
    n_obs = length - tf.reduce_sum(tf.cast(is_finite, tf.int32), axis=axis)
    # Compute the sum of non-Nan elements across the reduction axis.
    filled = tf.where(is_finite, tf.zeros_like(tensor), tensor)
    sums = tf.reduce_sum(filled, axis=axis)
    # Retun the mean(s) across the reduction axis.
    return sums / tf.cast(n_obs, tf.float32)


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
