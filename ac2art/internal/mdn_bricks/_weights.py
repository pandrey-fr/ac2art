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

"""Function to generate weights matrix for delta features computation.

Note that using such a matrix to compute delta features is highly
inefficient computationally speaking; this implementation should
merely be used as a dependency to using the MLPG algorithm.
"""


import tensorflow as tf
import numpy as np


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
        pads = tf.zeros(tf.maximum(0, time - window))
        boundary = tf.maximum(1, window - time + 1)
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
