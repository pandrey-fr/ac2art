# coding: utf-8

"""Set of auxiliary functions to build pieces of network architectures."""


import tensorflow as tf
import numpy as np

from neural_networks.core import build_layers_stack
from neural_networks.tf_utils import tensor_length


def build_dynamic_weights_matrix(size, window, complete=False):
    """Return a tensor matrix to produce dynamic features out of static ones.

    size     : size of the static features matrix (int or tf.int32 tensor)
    window   : (half) size of the time window to use (int)
    complete : whether to return the full weights matrix producing
               both static, delta and deltadelta features instead
               of the sole delta-computing weights (bool, default False)

    The returned tensor may be of variable size, e.g. by feeding
    the tensor returned by `neural_networks.tensor_length` applied
    to a variable-size placeholder of static features as `size`.
    """
    # Set up default weight arrays.
    w_norm = 3 / (window * (window + 1) * (2 * window + 1))
    w_future = tf.constant(
        np.array([i * w_norm for i in range(1, window + 1)]), dtype=tf.float32
    )
    w_past = tf.concat([-1 * w_future[::-1], [0]], axis=0)
    # Define functions building the weights.

    def past_weights(time):
        """Build and return the weights for indices 0 to time."""
        pads = tf.zeros(tf.maximum(0, time - window - 1))
        boundary = tf.maximum(0, window - time + 1)
        first = tf.reduce_sum(w_past[:boundary], keepdims=True)
        return tf.concat([pads, first, w_past[boundary:]], axis=0)[:size - 1]

    def future_weights(time):
        """Build and return the weights for indices time + 1 to size."""
        pads = tf.zeros(tf.maximum(0, size - window - time - 2))
        boundary = tf.maximum(0, size - window - time + 3)
        last = tf.reduce_sum(w_future[boundary:], keepdims=True)
        return tf.concat([w_future[:boundary], last, pads], axis=0)

    def get_weights(time):
        """Build and return a full row of weights."""
        return tf.expand_dims(
            tf.concat([past_weights(time), future_weights(time)], axis=0), 0
        )

    def build_step(weights, time):
        """Add a row to the weights matrix and increment the iterator."""
        weights = tf.concat([weights, get_weights(time)], axis=0)
        time += 1
        return weights, time

    # Build the dynamic weights matrix tensor.
    weights, _ = tf.while_loop(
        cond=lambda _, time: tf.less(time, size), body=build_step,
        loop_vars=[get_weights(0), tf.constant(1, dtype=tf.int32)],
        shape_invariants=[tf.TensorShape([None, None]), tf.TensorShape([])]
    )
    # Optionally build the full weights matrix.
    if complete:
        weights = tf.concat([
            tf.matrix_diag(tf.ones(size)), weights, tf.matmul(weights, weights)
        ], axis=0)
    return weights


def build_rmse_readouts(prediction, targets):
    """Return a dict of tensorflow Tensors associated with prediction error.

    prediction : Tensor of predicted values
    targets    : Tensor of true values

    Return a dict recording the initial prediction Tensor, the Tensor of
    prediction errors and that of the root mean square prediction error.
    """
    errors = prediction - targets
    rmse = tf.sqrt(tf.reduce_mean(tf.square(errors), axis=0))
    return {'prediction': prediction, 'errors': errors, 'rmse': rmse}


def refine_signal(
        signal, norm_params=None, filter_config=None, add_dynamic=False
    ):
    """Refine a multi-dimensional signal.

    signal        : bi-dimensional tensor containing the signal
    norm_params   : optional array of normalization parameters by
                    which to scale the signal's channels
    filter_config : optional tuple specifying a signal filter for smoothing
    add_dynamic   : whether to add delta and deltadelta features
                    to the refined signal (bool, default False)
    """
    # Optionally de-normalize the initial signal.
    if norm_params is not None:
        signal *= norm_params
    # Optionally filter the signal.
    if filter_config is None:
        top_filter = None
    else:
        top_filter = list(
            build_layers_stack(signal, [filter_config]).values()
        )[0]
        signal = top_filter.output
    # Optionally add dynamic features to the signal.
    if add_dynamic:
        dynamic_weights = build_dynamic_weights_matrix(
            tensor_length(signal), window=5, complete=False
        )
        delta_features = tf.matmul(dynamic_weights, signal)
        signal = tf.concat([
            signal, delta_features, tf.matmul(dynamic_weights, delta_features)
        ], axis=1)
    # Return the refined signal and the defined top filter, if any.
    return signal, top_filter
