# coding: utf-8

"""Class implementing Trajectory Mixture Density Networks."""

import tensorflow as tf

from data.commons.enhance import build_dynamic_weights_matrix
from neural_networks.components.mlpg import (
    expand_tmdn_standard_deviations,
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
        # Override the parent class's number of parameters.
        self.n_parameters = self.n_components * (4 + self.n_targets)

    def _build_placeholders(self):
        """Build the instance's placeholders."""
        super()._build_placeholders()
        self._holders['_delta_weights'] = (
            tf.placeholder(tf.float64, [None, None])
        )

    @onetimemethod
    def _build_parameters_readouts(self):
        """Build wrappers reading the produced density mixture parameters."""
        super()._build_parameters_readouts()
        stds = tf.reshape(
            self._readouts['std_deviations'], (-1, self.n_components, 3)
        )
        self._readouts['std_deviations_raw'] = stds
        self._readouts['std_deviations'] = expand_tmdn_standard_deviations(
            stds, self.n_components, self.n_targets
        )

    @onetimemethod
    def _build_initial_prediction(self):
        """Build a trajectory prediction using the MLPG algorithm."""
        trajectory = generate_trajectory_from_gaussian_mixture(
            self._readouts['priors'],
            self._readouts['means'],
            self._readouts['std_deviations_raw'],
            self._holders['_delta_weights']
        )
        self._readouts['raw_prediction'] = trajectory

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
        if fit == 'trajectory':
            weights = build_dynamic_weights_matrix(
                len(input_data), window=5, complete=True
            )
            feed_dict[self._holders['_delta_weights']] = weights
        return feed_dict
