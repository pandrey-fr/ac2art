# coding: utf-8

"""Class implementing Mixture Density Networks."""

import tensorflow as tf

from neural_networks.components.gaussian import (
    gaussian_density, gaussian_mixture_density
)
from neural_networks.components.layers import DenseLayer
from neural_networks.core import DeepNeuralNetwork
from neural_networks.models import MultilayerPerceptron
from neural_networks.tf_utils import minimize_safely
from neural_networks.utils import (
    check_type_validity, check_positive_int, onetimemethod
)


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
    Netowrks.
    """

    def __init__(
            self, input_shape, n_targets, n_components, layers_config,
            top_filter=None, norm_params=None, optimizer=None
        ):
        """Instanciate the mixture density network.

        input_shape   : shape of the input data fed to the network,
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
            norm_params, optimizer=optimizer, n_components=n_components
        )

    def _validate_args(self):
        """Process the initialization arguments of the instance."""
        # Control arguments common the any multilayer perceptron.
        super()._validate_args()
        # Control n_components argument and compute n_parameters.
        check_positive_int(self.n_components, 'n_components')
        self.n_parameters = (
            self.n_components * (2 + self.n_targets)
        )

    @onetimemethod
    def _build_readout_layer(self):
        """Build the readout layer of the mixture density network."""
        self._layers['readout_layer'] = DenseLayer(
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
        raw_parameters = self._layers['readout_layer'].output
        self._readouts['priors'] = (
            tf.nn.softmax(raw_parameters[:, :self.n_components])
        )
        n_means = self.n_components * self.n_targets
        self._readouts['means'] = tf.reshape(
            raw_parameters[:, self.n_components:self.n_components + n_means],
            (-1, self.n_components, self.n_targets)
        )
        self._readouts['std_deviations'] = tf.expand_dims(
            tf.exp(raw_parameters[:, self.n_components + n_means:]), 2
        )

    @onetimemethod
    def _build_likelihood_readouts(self):
        """Build wrappers computing the likelihood of the produced GMM."""
        # Define the network's likelihood.
        targets = (
            self._holders['targets'] if self.norm_params is None
            else self._holders['targets'] / self.norm_params
        )
        self._readouts['likelihood'] = gaussian_mixture_density(
            targets, self._readouts['priors'],
            self._readouts['means'], self._readouts['std_deviations']
        )
        # Define the error function and training step optimization program.
        self._readouts['mean_log_likelihood'] = (
            tf.reduce_mean(tf.log(self._readouts['likelihood'] + 1e-30))
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
        priors = tf.expand_dims(self._readouts['priors'], 2)
        means = self._readouts['means']
        stds = self._readouts['std_deviations']
        # Compute the mean of the components' means, weighted by priors.
        # Use this as an initial prediction.
        initial = tf.expand_dims(tf.reduce_sum(priors * means, axis=1), 1)
        # Compute the occupancy probabilities of the mixture's components.
        densities = tf.reduce_prod(
            gaussian_density(initial, means, stds), axis=2
        )
        norm = tf.expand_dims(tf.reduce_sum(densities, axis=1), 1)
        occupancy = tf.expand_dims(densities / (norm + 1e-30), 2)
        # Compute the mean of the component's means, weighted by occupancy.
        # Use this as the targets' prediction.
        prediction = tf.reduce_sum(occupancy * means, axis=1)
        self._readouts['raw_prediction'] = prediction

    @onetimemethod
    def _build_training_function(self):
        """Build the model's training step method."""
        # Build a function to maximize the network's mean log-likelihood.
        maximize_likelihood = minimize_safely(
            self.optimizer, loss=-1 * self._readouts['mean_log_likelihood'],
            var_list=self._neural_weights, reduce_fn=tf.reduce_max
        )
        # Build a function to minimize the prediction error.
        super()._build_training_function()
        minimize_rmse = self._training_function
        # Declare a two-fold train step function.
        def train_step(fit='likelihood'):
            """Modular training step, depending on the metric to use."""
            nonlocal maximize_likelihood
            nonlocal minimize_rmse
            if fit == 'likelihood':
                return maximize_likelihood
            elif fit == 'trajectory':
                return minimize_rmse
            else:
                raise ValueError("Unknown cost function '%s'." % fit)
        # Assign the network's train step function.
        self._training_function = train_step

    def _get_feed_dict(
            self, input_data, targets=None, keep_prob=1, fit='likelihood'
        ):
        """Return a tensorflow feeding dictionary out of provided arguments.

        input_data : data to feed to the network
        targets    : optional true targets associated with the inputs
        keep_prob  : probability to use for the dropout layers (default 1)
        fit        : output quantity used (str in {'likelihood', 'trajectory'})
        """
        # Add an argument unneeded by parents; pylint: disable=arguments-differ
        # 'fit' argument is for subclasses; pylint: disable=unused-argument
        return super()._get_feed_dict(input_data, targets, keep_prob)

    def run_training_function(
            self, input_data, targets, keep_prob=1, fit='likelihood'
        ):
        """Run a training step of the model.

        input_data : a 2-D numpy.ndarray (or pandas.DataFrame) where
                     each row is a sample to feed to the model
        targets    : target values associated with the input data
                     (numpy.ndarray or pandas structure)
        keep_prob  : probability for each unit to have its outputs used in
                     the training procedure (float in [0., 1.], default 1.)
        fit        : kind of output to use to as to fit the model ; either
                     'trajectory' RMSE or produced GMM 'likelihood'
        """
        # Add an argument unneeded by parents; pylint: disable=arguments-differ
        feed_dict = self._get_feed_dict(input_data, targets, keep_prob, fit)
        self.session.run(self._training_function(fit), feed_dict)

    def predict(self, input_data):
        """Predict the targets associated with a given set of inputs."""
        feed_dict = self._get_feed_dict(input_data, fit='trajectory')
        return self._readouts['prediction'].eval(feed_dict, self.session)

    def score(self, input_data, targets, score='likelihood'):
        """Return a given metric evaluating the network.

        input_data : input data sample to evalute the model on which
        targets    : true targets associated with the input dataset
        score      : choice of quantity to score ; may be 'likelihood'
                     of the produced GMM or root mean square error of
                     the predicted 'trajectory'
        """
        # Add an argument unneeded by parents; pylint: disable=arguments-differ
        # Check 'score' argument validity and select the associated metric.
        check_type_validity(score, str, 'score')
        if score == 'likelihood':
            metric = self._readouts['mean_log_likelihood']
        elif score == 'trajectory':
            metric = self._readouts['rmse']
        else:
            raise ValueError("Unknown score method: '%s'.")
        # Evaluate the selected metric.
        feed_dict = self._get_feed_dict(input_data, targets, fit=score)
        return metric.eval(feed_dict, self.session)
