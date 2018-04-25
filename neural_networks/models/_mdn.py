# coding: utf-8

"""Class implementing Mixture Density Networks."""

import tensorflow as tf
import numpy as np

from neural_networks.components.gaussian import (
    gaussian_density, gaussian_mixture_density
)
from neural_networks.components.dense_layer import DenseLayer
from neural_networks.core import DeepNeuralNetwork
from neural_networks.models import MultilayerPerceptron
from neural_networks.tf_utils import minimize_safely
from utils import check_type_validity, check_positive_int, onetimemethod


class MixtureDensityNetwork(MultilayerPerceptron):
    """Base class for mixture density networks in tensorflow.

    A Mixture Density Network (MDN) is a special kind of multilayer
    perceptron (MLP) which outputs parameters of a gaussian mixture
    density fitted on the inputs, supposed to model the output's
    probability density function.

    The most common way to fit a MDN is to seek to maximize the mean
    log-likelihood of the mixture density it models from input samples
    on associated target points. Another way to train it is to derive
    a prediction from the modeled probability density function, and to
    minimize the root mean square error of this prediction.

    The MDN was first introduced in Bishop, C. (1994). Mixture Density
    Networks.
    """

    def __init__(
            self, input_shape, n_targets, n_components, layers_config,
            top_filter=None, norm_params=None, optimizer=None
        ):
        """Instantiate the mixture density network.

        input_shape   : shape of the 2-D input data fed to the network,
                        with the number of samples as first component
        n_targets     : number of real-valued targets to predict
        n_components  : number of mixture components to model
        layers_config : list of tuples specifying a layer configuration,
                        made of a layer class (or short name), a number
                        of units (or a cutoff frequency for filters) and
                        an optional dict of keyword arguments
        top_filter    : optional tuple specifying a SignalFilter to use
                        on top of the network's raw prediction
        norm_params   : optional normalization parameters of the targets
                        (np.ndarray)
        optimizer     : tensorflow.train.Optimizer instance (by default,
                        Adam optimizer with 1e-3 learning rate)
        """
        # Arguments serve modularity; pylint: disable=too-many-arguments
        # Use the basic API init instead of that of the direct parent.
        # pylint: disable=super-init-not-called, non-parent-init-called
        self.n_parameters = None
        DeepNeuralNetwork.__init__(
            self, input_shape, n_targets, layers_config, top_filter,
            False, norm_params, optimizer=optimizer, n_components=n_components
        )

    @onetimemethod
    def _validate_args(self):
        """Process the initialization arguments of the instance."""
        # Control arguments common the any multilayer perceptron.
        super()._validate_args()
        # Control n_components argument and compute n_parameters.
        check_positive_int(self.n_components, 'n_components')
        self.n_parameters = self.n_components * (1 + 2 * self.n_targets)

    @onetimemethod
    def _build_readout_layer(self):
        """Build the readout layer of the mixture density network."""
        self.layers['readout_layer'] = DenseLayer(
            self._top_layer.output, self.n_parameters, 'identity'
        )

    @onetimemethod
    def _build_readouts(self):
        """Build wrappers around the produced GMM parameters and likelihood."""
        # Extract GMM parameters and build associated likelihood readouts.
        self._build_parameters_readouts()
        self._build_likelihood_readouts()
        # Build initial prediction, refine it and build error readouts.
        super()._build_readouts()

    @onetimemethod
    def _build_parameters_readouts(self):
        """Build wrappers reading the produced density mixture parameters."""
        raw_parameters = tf.cast(
            self.layers['readout_layer'].output, tf.float64
        )
        self.readouts['priors'] = (
            tf.nn.softmax(raw_parameters[..., :self.n_components])
        )
        if len(self.input_shape) == 2:
            moments_shape = (-1, self.n_components, self.n_targets)
        else:
            moments_shape = (
                -1, self.input_shape[1], self.n_components, self.n_targets
            )
        n_means = self.n_components * self.n_targets
        self.readouts['means'] = tf.reshape(
            raw_parameters[..., self.n_components:self.n_components + n_means],
            moments_shape
        )
        self.readouts['std_deviations'] = tf.reshape(
            tf.exp(raw_parameters[..., self.n_components + n_means:]),
            moments_shape
        )

    @onetimemethod
    def _build_likelihood_readouts(self):
        """Build wrappers computing the likelihood of the produced GMM."""
        # Define the network's likelihood.
        targets = (
            self.holders['targets'] if self.norm_params is None
            else self.holders['targets'] / self.norm_params
        )
        self.readouts['likelihood'] = gaussian_mixture_density(
            tf.cast(targets, tf.float64), self.readouts['priors'],
            self.readouts['means'], self.readouts['std_deviations']
        )
        # Define the error function and training step optimization program.
        self.readouts['mean_log_likelihood'] = tf.reduce_mean(
            tf.log(self.readouts['likelihood'] + 1e-62), axis=-1
        )

    @onetimemethod
    def _build_initial_prediction(self):
        """Build a simple trajectory prediction out of GMM parameters.

        A time sequence of mixture components is selected based on
        the mixtures' priors at each time. The returned trajectory
        is made of the expected value (i.e. mean) of each selected
        component.
        """
        # Gather parameters for better code readability.
        priors = tf.expand_dims(self.readouts['priors'], -1)
        means = self.readouts['means']
        stds = self.readouts['std_deviations']
        # Compute the mean of the components' means, weighted by priors.
        # Use this as an initial prediction.
        initial = tf.expand_dims(tf.reduce_sum(priors * means, axis=-2), -2)
        # Compute the occupancy probabilities of the mixture's components.
        densities = tf.reduce_prod(
            gaussian_density(initial, means, stds), axis=-1
        )
        norm = tf.expand_dims(tf.reduce_sum(densities, axis=-1), -1)
        occupancy = tf.expand_dims(densities / (norm + 1e-30), -1)
        # Compute the mean of the component's means, weighted by occupancy.
        # Use this as the targets' prediction.
        prediction = tf.reduce_sum(occupancy * means, axis=-2)
        self.readouts['raw_prediction'] = tf.cast(prediction, tf.float32)

    @onetimemethod
    def _build_training_function(self):
        """Build the model's training step method."""
        # Build a function to maximize the network's mean log-likelihood.
        maximize_likelihood = minimize_safely(
            self.optimizer, loss=-1 * self.readouts['mean_log_likelihood'],
            var_list=self._neural_weights
        )
        # Build a function to minimize the prediction error.
        super()._build_training_function()
        minimize_rmse = self.training_function

        # Declare a two-fold train step function.
        def train_step(loss='likelihood'):
            """Modular training step, depending on the metric to use."""
            nonlocal maximize_likelihood
            nonlocal minimize_rmse
            if loss == 'likelihood':
                return maximize_likelihood
            elif loss == 'rmse':
                return minimize_rmse
            else:
                raise ValueError("Unknown loss quantity '%s'." % loss)

        # Assign the network's train step function.
        self.training_function = train_step

    def get_feed_dict(
            self, input_data, targets=None, keep_prob=1, loss='rmse'
        ):
        """Return a tensorflow feeding dictionary out of provided arguments.

        input_data : data to feed to the network
        targets    : optional true targets associated with the inputs
        keep_prob  : probability to use for the dropout layers (default 1)
        loss       : loss computed (str in {'likelihood', 'rmse'})
        """
        # Add an argument unneeded by parents; pylint: disable=arguments-differ
        # 'loss' argument is for subclasses; pylint: disable=unused-argument
        return super().get_feed_dict(input_data, targets, keep_prob)

    def run_training_function(
            self, input_data, targets, keep_prob=1, loss='likelihood'
        ):
        """Run a training step of the model.

        input_data : a 2-D numpy.ndarray (or pandas.DataFrame) where
                     each row is a sample to feed to the model
        targets    : target values associated with the input data
                     (numpy.ndarray or pandas structure)
        keep_prob  : probability for each unit to have its outputs used in
                     the training procedure (float in [0., 1.], default 1.)
        loss       : loss quantity to minimize ; either produced GMM's
                     'likelihood' or derived trajectory's 'rmse'
        """
        # Add an argument unneeded by parents; pylint: disable=arguments-differ
        feed_dict = self.get_feed_dict(input_data, targets, keep_prob, loss)
        self.session.run(self.training_function(loss), feed_dict)

    def score(self, input_data, targets, loss='rmse'):
        """Return a given metric evaluating the network.

        input_data : input data sample to evalute the model on which
        targets    : true targets associated with the input dataset
        loss       : quantity to score ; either 'likelihood' of the
                     produced GMM or 'rmse' of the derived prediction
        """
        # Add an argument unneeded by parents; pylint: disable=arguments-differ
        # Check 'loss' argument validity and select the associated metric.
        check_type_validity(loss, str, 'loss')
        if loss == 'rmse':
            metric = self.readouts['rmse']
        elif loss == 'likelihood':
            metric = self.readouts['mean_log_likelihood']
        else:
            raise ValueError("Unknown loss quantity: '%s'.")
        # Evaluate the selected metric.
        feed_dict = self.get_feed_dict(input_data, targets, loss=loss)
        return metric.eval(feed_dict, self.session)

    def score_corpus(self, input_corpus, targets_corpus, loss='rmse'):
        """Iteratively compute the network's likelihood or prediction error.

        input_corpus   : sequence of input data arrays
        targets_corpus : sequence of true targets arrays
        loss           : quantity to score ; either 'likelihood' of the
                         produced GMM or 'rmse' of the derived prediction

        Corpora of input and target data must include numpy arrays
        of values to feed to the network and to evaluate against,
        without any nested arrays structure.
        """
        # Add an argument unneeded by parent; pylint: disable=arguments-differ
        # Check 'loss' argument validity. Handle the rmse metric case.
        check_type_validity(loss, str, 'loss')
        if loss == 'rmse':
            return super().score_corpus(input_corpus, targets_corpus)
        elif loss != 'likelihood':
            raise ValueError("Unknown loss quantity: '%s'.")
        # Handle the likelihood metric case.
        # Compute sample-wise likelihoods.
        scores = np.concatenate([
            self.score(input_data, targets, loss='likelihood')
            for input_data, targets in zip(input_corpus, targets_corpus)
        ])
        # Gather samples' lengths.
        sizes = self._get_corpus_sizes(input_corpus)
        # Reduce scores and return them.
        return np.sum(scores * sizes, axis=0) / sizes.sum()
