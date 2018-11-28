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
    # Straight-forward naming. pylint: disable=invalid-name
    pi = np.float64(np.pi) if data.dtype is tf.float64 else np.pi
    return (
        tf.exp(-1 * tf.square(data - mean) / (2 * tf.square(std)))
        / (tf.sqrt(2 * pi) * std)
    )


def gaussian_mixture_density(data, priors, means, stds):
    """Evaluate gaussian mixtures of given parameters' density at given points.

    The evaluated gaussian distributions may be univariate or multivariate.
    For multivariate cases, no covariance between terms is considered.
    Data may also be made of batched sequences of multivariate points
    (3-D tensor of dimensions [n_batches, batches_length, n_dim]).

    data   : points at which to evaluate each density function
             (tensorflow.Tensor of rank r in [1, 3])
    priors : prior probability of each mixture component,
             for each mixture (tensorflow.Tensor of rank max(r, 2))
    means  : mean of each mixture component, for each mixture
             (tensorflow.Tensor of rank r + 1)
    stds   : standard deviation of each mixture component,
             for each mixture (tensorflow.Tensor of rank r + 1)

    Return a 1-D Tensor gathering point-wise density for data made
    of a single sequence (rank in [1, 2]), or a 2-D Tensor gathering
    sequence-wise point-wise density for batched data (rank 3).
    """
    tf.assert_rank_in(data, [1, 2, 3])
    rank = tf.rank(data) + 1
    tf.control_dependencies([
        tf.assert_rank(priors, tf.maximum(2, rank - 1)),
        tf.assert_rank(means, rank), tf.assert_rank(stds, rank)
    ])
    # Handle the univariate density case.
    if len(data.shape) == 1 or data.shape[1].value == 1:
        return tf.reduce_sum(
            priors * gaussian_density(data, means, stds), axis=1
        )
    # Handle the multivariate density case.
    data = tf.expand_dims(data, -2)
    if len(stds.shape) == 2:
        stds = tf.expand_dims(stds, 2)
    densities = tf.reduce_prod(gaussian_density(data, means, stds), axis=-1)
    return tf.reduce_sum(priors * densities, axis=-1)
