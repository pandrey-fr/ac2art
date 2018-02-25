# coding: utf-8

"""Auxiliary function to evaluate gaussian density functions."""

import numpy as np
import tensorflow as tf


def gaussian_density(data, mean, std):
    """Evaluate gaussian density of given parameters at given points.

    data : point(s) on which to evaluate the density function(s)
    mean : mean (scalar or vector) of the evaluated function(s)
    std  : standard deviation (scalar) of evaluated function(s)
    dim  : dimension of the distribution (int, default 1)

    This function can be used to evaluate a single density function
    on one or more data points, multiple density functions on the
    same point or each on a single point. The density function(s)
    may be univariate or multivariate. In the latter case, a single
    standard deviation value common to each dimension of the variables
    is used.
    """
    return (
        tf.exp(-1 * tf.square(data - mean) / (2 * tf.square(std)))
        / (tf.sqrt(2 * np.pi) * std)
    )


def gaussian_mixture_density(data, priors, means, stds):
    """Evaluate gaussian mixtures of given parameters' density at given points.

    The evaluated gaussian distributions may be univariate or multivariate.
    For multivariate cases, no covariance between terms is considered.

    data   : points at which to evaluate each density function
             (1-D or 2-D tensorflow.Tensor)
    priors : prior probability of each mixture component,
             for each mixture (2-D tensorflow.Tensor)
    means  : mean of each mixture component, for each mixture
             (2-D or 3-D tensorflow.Tensor)
    stds   : standard deviation of each mixture component,
             for each mixture (2-D tensorflow.Tensor)

    Return a 1-D Tensor (one row per mixture).
    """
    # Check ranks validity
    tf.assert_rank_in(data, [1, 2])
    tf.assert_rank(priors, 2)
    tf.assert_rank_in(means, [2, 3])
    tf.assert_rank_in(stds, [2, 3])
    # Handle the univariate density case.
    if len(data.shape) == 1 or data.shape[1].value == 1:
        return tf.reduce_sum(
            priors * gaussian_density(data, means, stds), axis=1
        )
    # Handle the multivariate density case.
    data = tf.expand_dims(data, 1)
    if len(stds.shape) == 2:
        stds = tf.expand_dims(stds, 2)
    densities = tf.reduce_prod(gaussian_density(data, means, stds), axis=2)
    return tf.reduce_sum(priors * densities, axis=1)
