# coding: utf-8

"""Set of classes implementing Multilayer Perceptron models."""

import inspect

import tensorflow as tf
import numpy as np

from neural_networks.components.layers import DenseLayer
from neural_networks.core import DeepNeuralNetwork, build_rmse_readouts
from neural_networks.tf_utils import minimize_safely, reduce_finite_mean
from utils import raise_type_error, onetimemethod


class MultilayerPerceptron(DeepNeuralNetwork):
    """Class implementing the multilayer perceptron for regression."""

    def __init__(
            self, input_shape, n_targets, layers_config, top_filter=None,
            use_dynamic=True, norm_params=None, optimizer=None
        ):
        """Instantiate a multilayer perceptron for regression tasks.

        input_shape   : shape of the input data fed to the network,
                        of either [n_samples, input_size] shape
                        or [n_batches, max_length, input_size],
                        where the first may be variable (None)
                        (tuple, list, tensorflow.TensorShape)
        n_targets     : number of real-valued targets to predict
        layers_config : list of tuples specifying a layer configuration,
                        made of a layer class (or short name), a number
                        of units (or a cutoff frequency for filters) and
                        an optional dict of keyword arguments
        top_filter    : optional tuple specifying a SignalFilter to use
                        on top of the network's raw prediction
        use_dynamic   : whether to produce dynamic features and use them
                        when training the model (bool, default True)
        norm_params   : optional normalization parameters of the targets
                        (np.ndarray)
        optimizer     : tensorflow.train.Optimizer instance (by default,
                        Adam optimizer with 1e-3 learning rate)
        """
        # Arguments serve modularity; pylint: disable=too-many-arguments
        super().__init__(
            input_shape, n_targets, layers_config, top_filter,
            use_dynamic, norm_params, optimizer=optimizer
        )

    def _adjust_init_arguments_for_saving(self):
        """Adjust `_init_arguments` attribute before dumping the model.

        Return a tuple of two dict. The first is a copy of the keyword
        arguments used at initialization, containing only values which
        numpy.save can serialize. The second associates to non-serializable
        arguments' names a dict enabling their (recursive) reconstruction
        using the `neural_networks.utils.instantiate` function.
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

    @onetimemethod
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
        self.layers['readout_layer'] = DenseLayer(
            self._top_layer.output, self.n_targets, 'identity'
        )

    @onetimemethod
    def _build_initial_prediction(self):
        """Build the network's initial prediction.

        For the basic MLP, this is simply the readout layer's output.
        This method should be overridden by subclasses to define more
        complex predictions.
        """
        self.readouts['raw_prediction'] = self.layers['readout_layer'].output

    @onetimemethod
    def _build_error_readouts(self):
        """Build error readouts of the network's prediction."""
        readouts = build_rmse_readouts(
            self.readouts['prediction'], self.holders['targets'],
            self.holders.get('batch_sizes')
        )
        self.readouts.update(readouts)

    @onetimemethod
    def _build_training_function(self):
        """Build the train step function of the network."""
        # Build a function optimizing the neural layers' weights.
        fit_weights = minimize_safely(
            self.optimizer, loss=self.readouts['rmse'],
            var_list=self._neural_weights, reduce_fn=reduce_finite_mean
        )
        # If appropriate, build a function optimizing the filters' cutoff.
        if self._filter_cutoffs:
            fit_filters = minimize_safely(
                tf.train.GradientDescentOptimizer(.9),
                loss=self.readouts['rmse'], var_list=self._filter_cutoffs
            )
            self.training_function = [fit_weights, fit_filters]
        else:
            self.training_function = fit_weights

    def score(self, input_data, targets):
        """Return the root mean square prediction error of the network.

        input_data : input data sample to evalute the model on which
        targets    : true targets associated with the input dataset
        """
        feed_dict = self.get_feed_dict(input_data, targets)
        return self.readouts['rmse'].eval(feed_dict, self.session)

    def score_corpus(self, input_corpus, targets_corpus):
        """Iteratively compute the network's root mean square prediction error.

        input_corpus   : sequence of input data arrays
        targets_corpus : sequence of true targets arrays

        Return the channel-wise root mean square prediction error
        of the network on the full set of samples.
        """
        # Compute sample-wise scores.
        scores = np.array([
            np.square(self.score(input_data, targets)) * len(input_data)
            for input_data, targets in zip(input_corpus, targets_corpus)
        ])
        # Reduce scores and return them.
        return np.sqrt(np.mean(scores, axis=0))
