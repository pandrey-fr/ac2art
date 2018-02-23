# coding: utf-8

"""Set of classes implementing Multilayer Perceptron models."""

import inspect

import tensorflow as tf

from neural_networks.components import build_rmse_readouts
from neural_networks.components.filtering import lowpass_filter
from neural_networks.core import DeepNeuralNetwork
from neural_networks.layers import DenseLayer
from neural_networks.utils import (
    check_positive_int, check_type_validity, raise_type_error, onetimemethod
)


class MultilayerPerceptron(DeepNeuralNetwork):
    """Class implementing the multilayer perceptron for regression."""

    def __init__(
            self, input_shape, n_targets, layers_shape,
            activation='relu', optimizer=None
        ):
        """Instanciate a multilayer perceptron for regression tasks.

        input_shape  : shape of the input data fed to the network,
                       with the number of samples as first component
        n_targets    : number of real-valued targets to predict
        layers_shape : a tuple of int defining the hidden layers' sizes
        activation   : either an activation function or its name
                       (default 'relu', i.e. tensorflow.nn.relu)
        optimizer    : tensorflow.train.Optimizer instance (by default,
                       SGD optimizer with 1e-3 learning rate)
        """
        super().__init__(
            input_shape, n_targets, activation,
            layers_shape=layers_shape, optimizer=optimizer
        )

    def _adjust_init_arguments_for_saving(self):
        """Adjust `_init_arguments` attribute before dumping the model.

        Return a tuple of two dict. The first is a copy of the keyword
        arguments used at initialization, containing only values which
        numpy.save can serialize. The second associates to non-serializable
        arguments' names a dict enabling their (recursive) reconstruction
        using the `neural_networks.utils.instanciate` function.
        """
        # Pop out the optimizer instance from the initialization arguments.
        init_arguments = self._init_arguments.copy()
        optimizer = init_arguments.pop('optimizer')
        # Gather the optimizer's class and initialization arguments.
        optimizer_config = {
            'class_name': (
                optimizer.__module__ + '.' + optimizer.__class__.__name__
            ),
            'init_kwargs': {
                key: optimizer.__dict__['_' + key]
                for key in inspect.signature(optimizer.__class__).parameters
            }
        }
        # Return the serializable arguments and the optimizer configuration.
        return init_arguments, {'optimizer': optimizer_config}

    def _validate_args(self):
        """Process the initialization arguments of the instance."""
        # Control input shape, number of targets and activation function.
        super()._validate_args()
        # Control class-specific arguments.
        check_type_validity(self.layers_shape, tuple, 'layers_shape')
        try:
            [check_positive_int(shape, '') for shape in self.layers_shape]
        except (TypeError, ValueError):
            raise TypeError(
                "'layers_shape' must contain positive integers only."
            )
        if self.optimizer is None:
            self._init_arguments['optimizer'] = (
                tf.train.GradientDescentOptimizer(1e-3)
            )
        elif not isinstance(self.optimizer, tf.train.Optimizer):
            raise_type_error(
                'optimizer', (type(None), dict, 'tensorflow.train.Optimizer'),
                type(self.optimizer).__name__
            )

    @onetimemethod
    def _build_layers(self):
        """Build the layers stack of the multilayer perceptron."""
        # Define a dropout pooling function.
        keep_prob = self._holders['keep_prob']
        def dropout(input_data):
            """Dropout filter pooling the output of a neural layer."""
            nonlocal keep_prob
            return tf.nn.dropout(input_data, keep_prob=keep_prob)
        # Build the network's hidden layers.
        self._layers['dense_layer_0'] = DenseLayer(
            self._holders['input'], self.layers_shape[0],
            self.activation, pooling=dropout
        )
        for i, shape in enumerate(self.layers_shape[1:], 1):
            self._layers['dense_layer_%s' % i] = DenseLayer(
                self._top_layer.output, shape, self.activation, pooling=dropout
            )
        # Build the network's readout layer.
        self._build_readout_layer()

    @onetimemethod
    def _build_readout_layer(self):
        """Build the readout layer of the multilayer perceptron."""
        self._layers['readout_layer'] = DenseLayer(
            self._top_layer.output, self.n_targets, 'identity', bias=False
        )

    @onetimemethod
    def _build_readouts(self):
        """Build wrappers on top of the network's readout layer."""
        self._readouts['raw_prediction'] = self._layers['readout_layer'].output
        prediction = lowpass_filter(
            signal=self._layers['readout_layer'].output, cutoff=20,
            sampling_rate=200, window=5
        )
        self._readouts.update(
            build_rmse_readouts(prediction, self._holders['targets'])
        )

    @onetimemethod
    def _build_training_function(self):
        """Build the train step function of the network."""
        self._training_function = (
            self.optimizer.minimize(self._readouts['rmse'])
        )

    def _get_feed_dict(self, input_data, targets=None, keep_prob=1):
        """Build a tensorflow feeding dictionary out of provided arguments.

        input_data : data to feed to the network
        targets    : optional true targets associated with the inputs
        keep_prob  : probability to use for the dropout layers (default 1)
        """
        feed_dict = {
            self._holders['input']: input_data,
            self._holders['keep_prob']: keep_prob
        }
        if targets is not None:
            feed_dict[self._holders['targets']] = targets
        return feed_dict

    def run_training_function(self, input_data, targets, keep_prob=1):
        """Run a training step of the model.

        input_data : a 2-D numpy.ndarray (or pandas.DataFrame) where
                     each row is a sample to feed to the model
        targets    : target values associated with the input data
                     (numpy.ndarray or pandas structure)
        keep_prob  : probability for each unit to have its outputs used in
                     the training procedure (float in [0., 1.], default 1.)
        """
        feed_dict = self._get_feed_dict(input_data, targets, keep_prob)
        self._training_function.run(feed_dict, self.session)

    def predict(self, input_data):
        """Predict the targets associated with a given set of inputs."""
        feed_dict = self._get_feed_dict(input_data)
        return self._readouts['prediction'].eval(feed_dict, self.session)

    def score(self, input_data, targets):
        """Return the root mean square prediction error of the network.

        input_data : input data sample to evalute the model on which
        targets    : true targets associated with the input dataset
        """
        feed_dict = self._get_feed_dict(input_data, targets)
        return self._readouts['rmse'].eval(feed_dict, self.session)
