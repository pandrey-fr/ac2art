# coding: utf-8

"""Auxiliary functions implementing the MLPG algorithm."""

import numpy as np
import tensorflow as tf

from neural_networks.components.gaussian import gaussian_density
from neural_networks.tf_utils import index_tensor, tensor_length
from neural_networks.utils import check_positive_int


def build_dynamic_weights_matrix(size, window, complete=False):
    """Return a numpy matrix to produce dynamic features out of static ones.

    size     : size of the static features matrix (int)
    window   : (half) size of the time window to use (int)
    complete : whether to return the full weights matrix producing
               both static, delta and deltadelta features instead
               of the sole delta-computing weights (bool, default False)
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
    # Optionally build the full weights matrix.
    if complete:
        weights = np.concatenate(
            [np.identity(size), weights, np.dot(weights, weights)]
        )
    # Return the generated matrix of weights.
    return weights


def generate_trajectory_from_gaussian(means, stds, weights):
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
    # Long but explicit function name; pylint: disable=invalid-name
    # Test arguments' rank validity.
    tf.assert_rank(means, 2)
    tf.assert_rank(stds, 1)
    tf.assert_rank(weights, 2)
    # Reshape the means and standard deviations tensors.
    n_targets = means.shape[1].value // 3
    means = tf.concat([
        means[:, i * n_targets:(i + 1) * n_targets] for i in range(3)
    ], axis=0)
    stds = tf.matrix_diag(tf.concat([stds for _ in range(3)], axis=0))
    # Solve the system using cholesky decomposition of the left term matrix.
    weighted_stds = tf.matmul(tf.matrix_transpose(weights), stds)
    static_features = tf.cholesky_solve(
        tf.cholesky(tf.matmul(weighted_stds, weights)),
        tf.matmul(weighted_stds, means)
    )
    # Add dynamic features to the predicted static ones and return them.
    features = tf.matmul(weights, static_features)
    return tf.concat(
        tf.unstack(tf.reshape(features, (3, -1, n_targets))), axis=1
    )


def generate_trajectory_from_gaussian_mixture(
        priors, means, stds, weights, n_steps=10
    ):
    """Generate a trajectory out of a time sequence of gaussian mixture models.

    The algorithm used is taken from Tokuda, K. et alii (2000). Speech
    Parameter Generation Algorithms for HMM-based speech synthesis. It
    aims at generating the most likely trajectory sequence based on
    gaussian mixture density parameters fitted to an input sequence.

    priors  : sequence of mixture components' priors (2-D tensor)
    means   : sequence of componenents' multivariate means (3-D tensor)
    stds    : sequence of components' standard deviations (2-D tensor)
    weights : matrix of weights to derive successive orders
              of dynamic features out of static ones (2-D tensor)
    n_steps : maximum number of iteration when updating the selected
              trajectory through an E-M algorithm (int, default 10)

    Each row of means must include means associated with (in that order)
    the static features, the first-order dynamic features and the second-
    order ones.
    """
    # Long but explicit function name; pylint: disable=invalid-name
    # Test arguments' rank validity.
    tf.assert_rank(priors, 2)
    tf.assert_rank(means, 3)
    tf.assert_rank(stds, 2)
    tf.assert_rank(weights, 2)
    # Check additional arguments.
    check_positive_int(n_steps, 'n_steps')
    stddevs = tf.expand_dims(stds, 1)
    # Set up the expectation step function.
    def generate_trajectory(means_sequence, stds_sequence):
        """Generate a trajectory and density-based metrics."""
        features = generate_trajectory_from_gaussian(
            means_sequence, stds_sequence, weights
        )
        densities = priors * tf.reduce_prod(
            gaussian_density(tf.expand_dims(features, 1), means, stddevs),
            axis=2
        )
        log_likelihood = tf.reduce_sum(
            tf.log(tf.reduce_sum(densities, axis=1))
        )
        return features, densities, log_likelihood
    # Set up the maximization step function.
    def generate_parameters(densities):
        """Generate a parameters sequence using occupancy probabilities."""
        # Compute occupancy probabilities (i.e. posterior of components).
        norm = tf.expand_dims(tf.reduce_sum(densities, axis=1), 1)
        occupancy = densities / norm
        # Derive a weighted sequence of means and standard deviations.
        means_sequence = tf.matmul(tf.expand_dims(occupancy, 1), means)[:, 0]
        stds_sequence = tf.reduce_sum(occupancy * stds, axis=1)
        # Return generated parameters sequences.
        return means_sequence, stds_sequence
    # Set up a function running an E-M algorithm step.
    def run_step(index, previous_traject, previous_dens, previous_ll):
        """Run an iteration of the E-M algorithm for trajectory selection."""
        # Run the maximization and expectation steps.
        means_seq, stds_seq = generate_parameters(previous_dens)
        trajectory, densities, log_likelihood = (
            generate_trajectory(means_seq, stds_seq)
        )
        # Either return the updated results or interrupt the process.
        return tf.cond(
            log_likelihood > previous_ll,
            lambda: (index + 1, trajectory, densities, log_likelihood),
            lambda: (n_steps, previous_traject, previous_dens, previous_ll)
        )
    # Choose an initial trajectory, using a random sequence of components.
    sequence = index_tensor(tf.random_uniform(
        [tensor_length(priors)], 0, priors.shape[1], dtype=tf.int32
    ))
    init_trajectory, init_densities, init_ll = generate_trajectory(
        tf.gather_nd(means, sequence), tf.gather_nd(stds, sequence)
    )
    # Iteratively update the selected trajectory with an E-M algorithm.
    _, trajectory, _, _ = tf.while_loop(
        lambda i, *_: i < n_steps, run_step,
        [tf.constant(0), init_trajectory, init_densities, init_ll]
    )
    return trajectory
