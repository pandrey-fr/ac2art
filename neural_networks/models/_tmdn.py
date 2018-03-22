# coding: utf-8

"""Class implementing Trajectory Mixture Density Networks."""

import tensorflow as tf
import scipy.sparse
import numpy as np

from data.commons.enhance import build_dynamic_weights_matrix
from neural_networks.components.mlpg import (
    generate_trajectory_from_gaussian_mixture
)
from neural_networks.models import MixtureDensityNetwork
from neural_networks.utils import onetimemethod


class TrajectoryMDN(MixtureDensityNetwork):
    """Class for trajectory mixture density networks in tensorflow.

    A Trajectory Mixture Density Network (TMDN) is a special kind
    of MDN where the Maximum Likelihood Parameter Generation (MLPG)
    algorithm is used to produce a trajectory out of the produced
    gaussian mixture parameters.

    As the MLPG algorithm makes use of both static and dynamic (delta
    and deltadelta) articulatory features, a TMDN may gain from being
    first trained at maximizing the likelihood of the gaussian mixtures
    it produces for both static and dynamic features, before being
    trained at minimizing the root mean square error of prediction
    associated with the sole static features.

    The so-called MLPG algorithm is based on Tokuda, K. et alii (2000).
    Speech Parameter Generation Algorithms for HMM-based speech synthesis.

    The TMDN was first proposed in Richmond, K. (2006). A Trajectory Mixture
    Density Network for the Acoustic-Articulatory Inversion Mapping.
    """

    @onetimemethod
    def _validate_args(self):
        """Process the initialization arguments of the instance."""
        # Control arguments common the any mixture density network.
        super()._validate_args()
        # Control n_targets argument.
        if self.n_targets % 3:
            raise ValueError(
                "'n_targets' must be a multiple of 3, comprising "
                + "first and second order dynamic features count."
            )

    @onetimemethod
    def _build_placeholders(self):
        """Build the instance's placeholders."""
        super()._build_placeholders()
        self._holders['_delta_weights'] = (
            tf.sparse_placeholder(tf.float64, [None, None])
        )

    @onetimemethod
    def _build_initial_prediction(self):
        """Build a trajectory prediction using the MLPG algorithm."""
        trajectory = generate_trajectory_from_gaussian_mixture(
            self._readouts['priors'], self._readouts['means'],
            self._readouts['std_deviations'], self._holders['_delta_weights']
        )
        self._readouts['raw_prediction'] = tf.cast(trajectory, tf.float32)

    def _get_feed_dict(
            self, input_data, targets=None, keep_prob=1, fit='likelihood'
        ):
        """Return a tensorflow feeding dictionary out of provided arguments.

        input_data : data to feed to the network
        targets    : optional true targets associated with the inputs
        keep_prob  : probability to use for the dropout layers (default 1)
        fit        : output quantity used (str in {'likelihood', 'trajectory'})
        """
        feed_dict = super()._get_feed_dict(input_data, targets, keep_prob)
        # If needed, generate a delta weights matrix and set it to be fed.
        if fit == 'trajectory':
            # Build a weights matrix producing delta features.
            delta = build_dynamic_weights_matrix(len(input_data), window=5)
            # Zero-pad the matrix to fit the trajectory generation algorithm.
            n_targets = self.n_targets // 3
            identity = scipy.sparse.identity(n_targets)
            zeros = scipy.sparse.coo_matrix((n_targets, n_targets))
            delta = scipy.sparse.vstack([
                scipy.sparse.hstack([
                    coeff * identity if coeff != 0 else zeros for coeff in row
                ])
                for row in delta
            ])
            # Build the full matrix of weights for static and dynamic features.
            weights = scipy.sparse.vstack([
                scipy.sparse.identity(n_targets * len(input_data)),
                delta, delta.dot(delta)
            ])
            # Build a sparse tensor out of the weights matrix and feed it.
            weights_tensor = tf.SparseTensorValue(
                indices=np.array([weights.row, weights.col]).T,
                values=weights.data, dense_shape=weights.shape
            )
            feed_dict[self._holders['_delta_weights']] = weights_tensor
        return feed_dict
