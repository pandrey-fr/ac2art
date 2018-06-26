# coding: utf-8

"""Auxiliary functions implementing the MLPG algorithm."""


import tensorflow as tf

from ac2art.internal.mdn_bricks import gaussian_density
from ac2art.utils import check_positive_int


def mlpg_from_gaussian_mixture(
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
    # Test arguments validity.
    tf.assert_rank(priors, 2)
    tf.assert_rank(means, 3)
    tf.assert_rank(stds, 3)
    check_positive_int(n_steps, 'n_steps')

    # Set up the expectation step function.
    def generate_trajectory(means_sequence, stds_sequence):
        """Generate a trajectory and density-based metrics."""
        features = mlpg_from_gaussian(means_sequence, stds_sequence, weights)
        densities = priors * tf.reduce_prod(
            gaussian_density(tf.expand_dims(features, 1), means, stds), axis=2
        )
        log_likelihood = tf.reduce_sum(
            tf.log(tf.reduce_sum(densities, axis=1) + 1e-30)
        )
        return features, densities, log_likelihood

    # Set up the maximization step function.
    def generate_parameters(densities):
        """Generate a parameters sequence using occupancy probabilities."""
        # Compute occupancy probabilities (i.e. posterior of components).
        norm = tf.expand_dims(tf.reduce_sum(densities, axis=1), 1)
        occupancy = tf.expand_dims(densities / (norm + 1e-30), 2)
        # Derive a weighted sequence of means and standard deviations.
        return (
            tf.reduce_sum(occupancy * means, axis=1),
            tf.reduce_sum(occupancy * stds, axis=1)
        )

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

    # Choose an initial trajectory, using the component's priors as weights.
    init_trajectory, init_densities, init_ll = generate_trajectory(
        tf.reduce_sum(tf.expand_dims(priors, 2) * means, axis=1),
        tf.reduce_sum(tf.expand_dims(priors, 2) * stds, axis=1)
    )
    # Iteratively update the selected trajectory with an E-M algorithm.
    _, trajectory, _, _ = tf.while_loop(
        lambda i, *_: i < n_steps, run_step,
        [tf.constant(0), init_trajectory, init_densities, init_ll]
    )
    return trajectory


def mlpg_from_gaussian(means, stds, weights):
    """Generate a trajectory out of a time sequence of gaussian parameters.

    The algorithm used is taken from Tokuda, K. et alii (2000). Speech
    Parameter Generation Algorithms for HMM-based speech synthesis. It
    aims at generating the most likely trajectory sequence based on
    gaussian parameters fitted to an input sequence of some kind.

    means   : time sequence of multivariate means (2-D tensor)
    stds    : time sequence of dimension-wise standard deviations (2-D tensor)
    weights : matrix of weights to derive successive orders
              of dynamic features out of static ones (2-D tensor)

    The means and standard deviations should be organized as matrices
    where each row represents a given time, while the columns comprise
    the K parameters associated with static features, followed by those
    associated with the delta features and finally those associated with
    the delta delta features.
    """
    # Test arguments' rank validity.
    tf.assert_rank(means, 2)
    tf.assert_rank(stds, 2)
    tf.assert_rank(weights, 2)
    tf.assert_equal(means.shape[1], stds.shape[1])
    # Pile up the static and dynamic parameters.
    n_targets = means.shape[1].value // 3
    means = _reshape_moments_tensor(means, n_targets)
    stds = _reshape_moments_tensor(stds, n_targets)
    # Generate the most likely trajectory for each target dimension.
    features = tf.concat([
        mlpg_univariate(means[:, k], stds[:, k], weights)
        for k in range(n_targets)
    ], axis=1)
    # Properly reshape the results and return them.
    stacks = tf.unstack(tf.reshape(features, (3, -1, n_targets)))
    return tf.concat(stacks, axis=1)


def _reshape_moments_tensor(tensor, n_targets):
    """Reshape a tensor to stack vertically static and dynamic features."""
    return tf.concat([
        tensor[:, i:i + n_targets] for i in range(0, 3 * n_targets, n_targets)
    ], axis=0)


def mlpg_univariate(means, stds, weights):
    """Generate a trajectory out of a time sequence of gaussian parameters.

    The algorithm used is taken from Tokuda, K. et alii (2000). Speech
    Parameter Generation Algorithms for HMM-based speech synthesis. It
    aims at generating the most likely trajectory sequence based on
    gaussian parameters fitted to an input sequence of some kind.

    means   : time sequence of means (1-D tensor)
    stds    : time sequence of standard deviations (1-D tensor)
    weights : matrix of weights to derive successive orders
              of dynamic features out of static ones (2-D tensor)

    The means and standard deviations should consist of the time
    sequence of parameters for static features first, followed by the
    time sequence of delta features parameters and finally by that of
    delta delta features parameters.
    """
    # Test arguments' rank validity.
    tf.assert_rank(means, 1)
    tf.assert_rank(stds, 1)
    tf.assert_rank(weights, 2)
    # Compute the terms of the parameters generation system.
    inv_stds = tf.matrix_diag(1 / (tf.square(stds) + 1e-30))
    timed_variance = tf.matmul(tf.matrix_transpose(weights), inv_stds)
    left_term = tf.matmul(timed_variance, weights)
    right_term = tf.matmul(timed_variance, tf.expand_dims(means, 1))
    # Solve the system using cholesky decomposition.
    static_features = tf.cholesky_solve(tf.cholesky(left_term), right_term)
    # Add dynamic features to the predicted static ones and return them.
    return tf.matmul(weights, static_features)
