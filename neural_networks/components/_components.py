# coding: utf-8

"""Set of auxiliary functions to build pieces of network architectures."""

import tensorflow as tf


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
