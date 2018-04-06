# coding: utf-8

"""Set of auxiliary functions to build pieces of network architectures."""


import tensorflow as tf

from neural_networks.core import build_layers_stack


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
        signal, norm_params=None, filter_config=None, dynamic_weights=None
    ):
    """Refine a multi-dimensional signal.

    signal          : bi-dimensional tensor containing the signal
    norm_params     : optional array of normalization parameters by which
                      to scale the signal's channels
    filter_config   : optional tuple specifying a filter used to smooth
                      the predicted signal
    dynamic_weights : optional tensor holding a matrix of weights by which
                      dynamic features are added to the static ones
                      (see `data.commons.enhance.build_rmse_readouts`)
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
    if dynamic_weights is not None:
        signal = tf.matmul(dynamic_weights, signal)
        signal = tf.reshape(signal, (3, -1, signal.shape[1].value))
        signal = tf.concat(tf.unstack(signal), axis=1)
    # Return the refined signal and the defined top filter, if any.
    return signal, top_filter
