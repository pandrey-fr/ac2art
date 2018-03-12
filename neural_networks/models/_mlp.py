# coding: utf-8

"""Set of classes implementing Multilayer Perceptron models."""

import inspect

import tensorflow as tf

from neural_networks.components import build_rmse_readouts
from neural_networks.components.filters import LowpassFilter
from neural_networks.components.layers import DenseLayer
from neural_networks.core import DeepNeuralNetwork
from neural_networks.utils import raise_type_error, onetimemethod


class MultilayerPerceptron(DeepNeuralNetwork):
    """Class implementing the multilayer perceptron for regression."""

    def __init__(
            self, input_shape, n_targets, layers_config, norm_params=None,
            filter_kwargs=None, optimizer=None
        ):
        """Instanciate a multilayer perceptron for regression tasks.

        input_shape   : shape of the input data fed to the network,
                        with the number of samples as first component
        n_targets     : number of real-valued targets to predict
        layers_config : list of tuples specifying a layer configuration,
                        made of a layer class (or short name), a number
                        of units (or a cutoff frequency for filters) and
                        an optional dict of keyword arguments
        norm_params   : optional normalization parameters of the targets
                        (np.ndarray)
        filter_kwargs : dict of keyword arguments setting up a final
                        low-pass filter (by default, learnable filter
                        initialized at 20 Hz, with a 200 hz sampling rate)
        optimizer     : tensorflow.train.Optimizer instance (by default,
                        SGD optimizer with 1e-3 learning rate)
        """
        # Arguments serve modularity; pylint: disable=too-many-arguments
        super().__init__(
            input_shape, n_targets, layers_config, norm_params,
            filter_kwargs=filter_kwargs, optimizer=optimizer
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
        # Control arguments common to any DeepNeuralNetwork subclass.
        super()._validate_args()
        # Control filter kwargs argument.
        filter_params = {
            'cutoff':20, 'learnable':True, 'sampling_rate':200, 'window':5
        }
        if isinstance(self.filter_kwargs, dict):
            filter_params.update(self.filter_kwargs)
        elif self.filter_kwargs is not None:
            raise_type_error(
                'filter_kwargs', (dict, type(None)), type(self.filter_kwargs)
            )
        self._init_arguments['filter_kwargs'] = filter_params
        # Control optimizer argument.
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
    def _build_readout_layer(self):
        """Build the readout layer of the multilayer perceptron."""
        self._layers['readout_layer'] = DenseLayer(
            self._top_layer.output, self.n_targets, 'identity'
        )
        self._layers['readout_filter'] = LowpassFilter(
            signal=self._layers['readout_layer'].output, **self.filter_kwargs
        )

    @onetimemethod
    def _build_readouts(self):
        """Build wrappers on top of the network's readout layer."""
        self._readouts['raw_prediction'] = self._layers['readout_layer'].output
        prediction = self._layers['readout_filter'].output
        if self.norm_params is not None:
            prediction *= self.norm_params
        self._readouts.update(
            build_rmse_readouts(prediction, self._holders['targets'])
        )

    @onetimemethod
    def _build_training_function(self):
        """Build the train step function of the network."""
        # Build a function optimizing the layer's weights.
        fit_weights = self.optimizer.minimize(
            self._readouts['rmse'], var_list=self._layer_weights
        )
        # If the readout filter is learnable, optimize its cutoff frequency.
        filt = self._layers['readout_filter']
        if filt.learnable:
            fit_filter = filt.get_cutoff_training_function(
                self._readouts['rmse'], learning_rate=1
            )
            self._training_function = [fit_weights, fit_filter]
        else:
            self._training_function = fit_weights

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
        self.session.run(self._training_function, feed_dict)

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
