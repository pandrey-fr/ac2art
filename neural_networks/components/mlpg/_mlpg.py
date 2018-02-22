# coding: utf-8

"""Auxiliary functions implementing the MLPG algorithm."""

import numpy as np
import tensorflow as tf


def build_dynamic_weights_matrix(size, window):
    """Return a numpy matrix to produce dynamic features out of static ones.

    size   : size of the static features matrix (int)
    window : (half) size of the time window to use (int)
    """
    # Declare stuff.
    w_norm = 3 / (window * (window + 1) * (2 * window + 1))
    w_future = np.array([i * w_norm for i in range(1, window + 1)])
    w_past = -1 * w_future[::-1]
    # Declare the weights matrix and fill it row by row.
    weights = np.zeros((size, size))
    for time in range(size):
        # Fill weights for past observations.
        if time < window:
            weights[time, 0] = w_past[:window - time + 1].sum()
            weights[time, 1:time] = w_past[window - time + 1:]
        else:
            weights[time, time - window:time] = w_past
        # Fill weights for future observations.
        if time >= size - window:
            weights[time, -1] = w_future[size - window - time - 2:].sum()
            weights[time, time + 1:-1] = w_future[:size - window - time - 2]
        else:
            weights[time, time + 1:time + window + 1] = w_future
    # Return the generated matrix of weights.
    return weights


def run_mlpg_algorithm(means, stds, weights):
    """Generate a trajectory out of a time sequence of gaussian parameters.

    The algorithm used is taken from Tokuda, K. et alii (2000). Speech
    Parameter Generation Algorithms for HMM-based speech synthesis. It
    aims at generating the most likely trajectory sequence based on
    gaussian parameters fitted to an input sequence of some kind.

    means   : sequence of multivariate means (2-D tensor)
    stds    : sequence of standard deviations (1-D tensor)
    weights : matrix of weights to derive successive orders
              of dynamic features out of static ones (2-D tensor)

    Each row of means must include means associated with (in that order)
    the static features, the first-order dynamic features and the second-
    order ones.
    """
    # Test arguments' rank validity.
    tf.assert_rank(stds, 1)
    tf.assert_rank(means, 2)
    tf.assert_rank(weights, 2)
    # Reshape the means and standard deviations tensors.
    n_targets = means.shape[1].value // 3
    means = tf.concat([
        means[:, i * n_targets:(i + 1) * n_targets] for i in range(3)
    ], axis=0)
    stds = tf.matrix_diag(tf.concat([stds for _ in range(3)], axis=0))
    # Solve the system using cholesky decomposition of the left term matrix.
    weighted_stds = tf.matmul(tf.matrix_transpose(weights), stds)
    return tf.cholesky_solve(
        tf.cholesky(tf.matmul(weighted_stds, weights)),
        tf.matmul(weighted_stds, means)
    )


def supertoto(priors, means, stds, delta_weights, n_steps):
    """Docstring."""
    # Set up various tensors repeatedly used.
    stddevs = tf.expand_dims(stds, 1)
    n_obs = tensor_length(priors)
    weights = tf.concat([
        tf.matrix_diag(tf.ones([n_obs])), delta_weights,
        tf.matmul(delta_weights, delta_weights)
    ], axis=0)
    # Choose a random components sequence to start with.
    sequence = index_tensor(
        tf.random_uniform([n_obs], 0, priors.shape[1], dtype=tf.int32)
    )
    means_sequence = tf.gather_nd(means, sequence)
    stds_sequence = tf.gather_nd(stds, sequence)
    # Iteratively run the MLPG algorithm, optimizing the components sequence.
    for i in range(n_steps):
        # Run the MLPG algorithm using the current components sequence.
        static = run_mlpg_algorithm(means_sequence, stds_sequence, weights)
        # Add static features to the generated static ones and reshape them.
        prediction = tf.matul(weights, static)
        prediction = tf.concat([
            features[:n_obs], features[n_obs:2 * n_obs], features[2*n_obs:]
        ], axis=1)
        prediction = tf.expand_dims(
            tf.concat([static, delta, deltadelta], axis=1), 1
        )
        # Select a new components sequence based on the density of the
        # gaussian components at the current predicted positions.
        sequence_weights = priors * tf.reduce_prod(
            gaussian_density(prediction, means, stddevs), axis=2
        )
        means_sequence = sequence_weights * means
        stds_sequence = sequence_weights * stds
