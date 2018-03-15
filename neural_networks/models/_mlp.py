# coding: utf-8

"""Set of classes implementing Multilayer Perceptron models."""

import inspect

import tensorflow as tf

from neural_networks.components.layers import DenseLayer
from neural_networks.core import DeepNeuralNetwork
from neural_networks.tf_utils import minimize_safely
from neural_networks.utils import raise_type_error, onetimemethod


class MultilayerPerceptron(DeepNeuralNetwork):
    """Class implementing the multilayer perceptron for regression."""

    def __init__(
            self, input_shape, n_targets, layers_config, top_filter=None,
            norm_params=None, optimizer=None
        ):
        """Instanciate a multilayer perceptron for regression tasks.

        input_shape   : shape of the input data fed to the network,
                        with the number of samples as first component
        n_targets     : number of real-valued targets to predict
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
        super().__init__(
            input_shape, n_targets, layers_config, top_filter, norm_params,
            optimizer=optimizer,
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
                key: (
                    optimizer.__dict__['_' + key] if key != 'learning_rate'
                    else optimizer.__dict__.get(
                        '_learning_rate', optimizer.__dict__.get('_lr', .001)
                    )
                )
                for key in inspect.signature(optimizer.__class__).parameters
            }
        }
        # Return the serializable arguments and the optimizer configuration.
        return init_arguments, {'optimizer': optimizer_config}

    def _validate_args(self):
        """Process the initialization arguments of the instance."""
        # Control arguments common to any DeepNeuralNetwork subclass.
        super()._validate_args()
        # Control optimizer argument.
        if self.optimizer is None:
            self._init_arguments['optimizer'] = (
                tf.train.AdamOptimizer(1e-3)
            )
        elif not isinstance(self.optimizer, tf.train.Optimizer):
            raise_type_error(
                'optimizer', (type(None), dict, 'tensorflow.train.Optimizer'),
                type(self.optimizer).__name__
            )

    @onetimemethod
    def _build_readout_layer(self):
        """Build the readout layer of the multilayer perceptron."""
        # Build the readout layer.
        self._layers['readout_layer'] = DenseLayer(
            self._top_layer.output, self.n_targets, 'identity'
        )

    @onetimemethod
    def _build_initial_prediction(self):
        """Build the network's initial prediction.

        For the basic MLP, this is simply the readout layer's output.
        This method should be overriden by subclasses to define more
        complex predictions.
        """
        self._readouts['raw_prediction'] = self._layers['readout_layer'].output

    @onetimemethod
    def _build_training_function(self):
        """Build the train step function of the network."""
        # Build a function optimizing the neural layers' weights.
        fit_weights = minimize_safely(
            self.optimizer, loss=self._readouts['rmse'],
            var_list=self._neural_weights, reduce_fn=tf.reduce_max
        )
        # If appropriate, build a function optimizing the filters' cutoff.
        if self._filter_cutoffs:
            fit_filter = tf.train.GradientDescentOptimizer(.9).minimize(
                self._readouts['rmse'], var_list=self._filter_cutoffs
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
