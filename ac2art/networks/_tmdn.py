# coding: utf-8

"""Class implementing Trajectory Mixture Density Networks."""


import tensorflow as tf

from ac2art.internal.mdn_bricks import (
    build_dynamic_weights_matrix, mlpg_from_gaussian_mixture
)
from ac2art.internal.tf_utils import run_along_first_dim
from ac2art.networks import MixtureDensityNetwork
from ac2art.utils import onetimemethod


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
        if len(self.input_shape) == 3 and self.input_shape[1] is not None:
            delta_weights = build_dynamic_weights_matrix(
                size=self.input_shape[1], window=5, complete=True
            )
            self.holders['_delta_weights'] = tf.constant(
                delta_weights, dtype=tf.float64
            )
        else:
            self.holders['_delta_weights'] = (
                tf.placeholder(tf.float64, [None, None])
            )

    @onetimemethod
    def _build_initial_prediction(self):
        """Build a trajectory prediction using the MLPG algorithm."""
        parameters = tuple(
            self.readouts[key] for key in ('priors', 'means', 'std_deviations')
        )
        if len(self.input_shape) == 2:
            trajectory = mlpg_from_gaussian_mixture(
                *parameters, weights=self.holders['_delta_weights']
            )
        else:
            trajectory = run_along_first_dim(
                mlpg_from_gaussian_mixture, tensors=parameters,
                weights=self.holders['_delta_weights']
            )
        self.readouts['raw_prediction'] = tf.cast(trajectory, tf.float32)

    def get_feed_dict(
            self, input_data, targets=None, keep_prob=1, loss='rmse'
        ):
        """Return a tensorflow feeding dictionary out of provided arguments.

        input_data : data to feed to the network
        targets    : optional true targets associated with the inputs
        keep_prob  : probability to use for the dropout layers (default 1)
        loss       : loss computed (str in {'likelihood', 'rmse'})
        """
        feed_dict = super().get_feed_dict(input_data, targets, keep_prob)
        # If needed, generate a delta weights matrix and set it to be fed.
        if loss == 'rmse':
            if len(self.input_shape) == 2 or self.input_shape[1] is None:
                length = feed_dict[self.holders['input']].shape[-2]
                weights = build_dynamic_weights_matrix(
                    length, window=5, complete=True
                )
                feed_dict[self.holders['_delta_weights']] = weights
        return feed_dict
