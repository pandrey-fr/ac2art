# coding: utf-8
#
# Copyright 2018 Paul Andrey
#
# This file is part of ac2art.
#
# ac2art is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ac2art is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ac2art.  If not, see <http://www.gnu.org/licenses/>.

"""Set of functions to read the outputs of a neural network's layers stack."""


import tensorflow as tf

from ac2art.internal.network_bricks import build_layers_stack
from ac2art.internal.tf_utils import (
    add_dynamic_features, batch_tensor_mean, run_along_first_dim
)


def build_binary_classif_readouts(pred_proba, labels, batch_sizes=None):
    """Return a dict of tensors recording binary classification metrics.

    pred_proba  : tensor of predicted probabilities
    labels      : tensor of true labels
    batch_sizes : optional tensor of true sequences length,
                  for batched (or fixed-size) inputs

    Return a dict recording predicted probabilites, predicted
    labels, cross-entropy metrics and the overall accuracy.
    """
    # Compute unit-wise prediction, correctness and entropy.
    prediction = tf.round(pred_proba)
    correctness = tf.cast(tf.equal(prediction, labels), tf.float32)
    entropy = - 1 * (
        labels * tf.log(pred_proba + 1e-32)
        + (1 - labels) * tf.log(1 - pred_proba + 1e-32)
    )
    # Compute accuracy and cross_entropy scores.
    if batch_sizes is None:
        accuracy = tf.reduce_mean(correctness, axis=-2)
        cross_entropy = tf.reduce_mean(entropy, axis=-2)
    else:
        accuracy = batch_tensor_mean(correctness, batch_sizes)
        cross_entropy = batch_tensor_mean(entropy, batch_sizes)
    # Return a dict of computed metrics.
    return {
        'accuracy': accuracy, 'cross_entropy': cross_entropy,
        'predicted_proba': pred_proba, 'prediction': prediction
    }


def build_rmse_readouts(prediction, targets, batch_sizes=None):
    """Return a dict of tensorflow Tensors associated with prediction error.

    prediction  : tensor of predicted values
    targets     : tensor of true values
    batch_sizes : optional tensor of true sequences length,
                  for batched (or fixed-size) inputs

    Return a dict recording the initial prediction Tensor, the Tensor of
    prediction errors and that of the root mean square prediction error.
    """
    errors = prediction - targets
    mean_square_errors = (
        tf.reduce_mean(tf.square(errors), axis=-2) if batch_sizes is None
        else batch_tensor_mean(tf.square(errors), batch_sizes)
    )
    rmse = tf.sqrt(mean_square_errors)
    return {'prediction': prediction, 'errors': errors, 'rmse': rmse}


def refine_signal(
        signal, norm_params=None, filter_config=None, add_dynamic=False
    ):
    """Refine a multi-dimensional signal.

    signal        : bi-dimensional tensor containing the signal,
                    or rank 3 tensor batching such signals
    norm_params   : optional array of normalization parameters by
                    which to scale the signal's channels
    filter_config : optional tuple specifying a signal filter for smoothing
    add_dynamic   : whether to add delta and deltadelta features
                    to the refined signal (bool, default False)
    """
    tf.assert_rank_in(signal, (2, 3))
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
        signal = (
            add_dynamic_features(signal, window=5) if len(signal.shape) == 2
            else run_along_first_dim(add_dynamic_features, signal, window=5)
        )
    # Return the refined signal and the defined top filter, if any.
    return signal, top_filter
