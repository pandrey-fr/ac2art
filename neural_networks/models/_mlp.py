# coding: utf-8

"""Set of classes implementing Multilayer Perceptron models."""

import inspect

import tensorflow as tf
import numpy as np

from neural_networks.components.dense_layer import DenseLayer
from neural_networks.core import (
    DeepNeuralNetwork, build_binary_classif_readouts, build_rmse_readouts
)
from neural_networks.tf_utils import minimize_safely, reduce_finite_mean
from utils import raise_type_error, onetimemethod


class MultilayerPerceptron(DeepNeuralNetwork):
    """Class implementing the multilayer perceptron for regression.

    Note that strictly speaking, this class implements more than
    the sole Multilayer Perceptron: RNN units may be used (as well
    as signal filters), and mixed tasks of regression and binary
    classification may be specified.

    This is basically a framework for defining end-to-end models
    comprising feed-forward layers and recurrent units (note that
    other kinds of layers may be added as well through the parent
    abstract `DeepNeuralNetwork` class) for tasks of (channel-wise)
    regression and/or binary classification.

    The implementation tends to be general, but echoes the fact
    that it was developed for acoustic-to-articulatory inversion
    purposes, thus adopting certain choices that may not be fit
    for other contexts. Additionally, given details were guided
    by the implementation of other models (inheriting from this
    class), thus causing design choices that may seem odd when
    considered out of context.
    """

    def __init__(
            self, input_shape, n_targets, layers_config, top_filter=None,
            use_dynamic=True, binary_tracks=None, norm_params=None,
            optimizer=None
        ):
        """Instantiate a multilayer perceptron for regression tasks.

        input_shape   : shape of the input data fed to the network,
                        of either [n_samples, input_size] shape
                        or [n_batches, max_length, input_size],
                        where the last axis must be fixed (non-None)
                        (tuple, list, tensorflow.TensorShape)
        n_targets     : number of targets to predict,
                        notwithstanding dynamic features
        layers_config : list of tuples specifying a layer configuration,
                        made of a layer class (or short name), a number
                        of units (or a cutoff frequency for filters) and
                        an optional dict of keyword arguments
        top_filter    : optional tuple specifying a SignalFilter to use
                        on top of the network's raw prediction
        use_dynamic   : whether to produce dynamic features and use them
                        when training the model (bool, default True)
        binary_tracks : optional list of targets which are binary-valued
                        (and should therefore not have delta counterparts)
        norm_params   : optional normalization parameters of the targets
                        (np.ndarray)
        optimizer     : tensorflow.train.Optimizer instance (by default,
                        Adam optimizer with 1e-3 learning rate)
        """
        # Arguments serve modularity; pylint: disable=too-many-arguments
        super().__init__(
            input_shape, n_targets, layers_config, top_filter,
            use_dynamic, binary_tracks, norm_params, optimizer=optimizer
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
        hidden_output = self._top_layer.output
        n_binary = len(self.binary_tracks) if self.binary_tracks else 0
        # Build the readout layer for continuous target points.
        self.layers['readout_layer'] = DenseLayer(
            hidden_output, self.n_targets - n_binary, 'identity'
        )
        # If any, build the readout layer for binary target points.
        if n_binary:
            self.layers['readout_layer_binary'] = DenseLayer(
                hidden_output, n_binary, 'sigmoid'
            )

    @onetimemethod
    def _build_initial_prediction(self):
        """Build the network's initial prediction.

        For the basic MLP, this is simply the readout layer's output.
        This method should be overridden by subclasses to define more
        complex predictions.
        """
        self.readouts['raw_prediction'] = self.layers['readout_layer'].output
        if self.binary_tracks is not None:
            binary_proba = self.layers['readout_layer_binary'].output
            self.readouts['_binary_probability'] = binary_proba
            self.readouts['_binary_prediction'] = tf.round(binary_proba)

    @onetimemethod
    def _build_error_readouts(self):
        """Build error readouts of the network's prediction."""
        if self.binary_tracks is None:
            rmse_readouts = build_rmse_readouts(
                self.readouts['prediction'], self.holders['targets'],
                self.holders.get('batch_sizes')
            )
            self.readouts.update(rmse_readouts)
        else:
            # Build the readouts associated with continuous targets.
            continuous_targets = tf.concat([
                self.holders['targets'][..., i:i+1]
                for i in range(self.holders['targets'].shape[-1].value)
                if i not in self.binary_tracks
            ], axis=-1)
            rmse_readouts = build_rmse_readouts(
                self.readouts['_continuous_prediction'], continuous_targets,
                self.holders.get('batch_sizes')
            )
            rmse_readouts.pop('prediction')
            self.readouts.update(rmse_readouts)
            # Build the readouts associated with binary-valued targets.
            binary_targets = tf.concat([
                self.holders['targets'][..., i:i+1] for i in self.binary_tracks
            ], axis=-1)
            accuracy_readouts = build_binary_classif_readouts(
                self.readouts['_binary_probability'], binary_targets,
                self.holders.get('batch_sizes')
            )
            # Record and interlace error metrics of both kinds of targets.
            self.readouts['_rmse'] = self.readouts.pop('rmse')
            for key in ('accuracy', 'cross_entropy'):
                self.readouts['_' + key] = accuracy_readouts[key]
            self.readouts['rmse'] = self._interlace(
                self.readouts['_rmse'], self.readouts['_cross_entropy']
            )

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

        For binary-valued targets, cross-entropy is returned.
        """
        feed_dict = self.get_feed_dict(input_data, targets)
        return self.readouts['rmse'].eval(feed_dict, self.session)

    def score_corpus(self, input_corpus, targets_corpus):
        """Iteratively compute the network's root mean square prediction error.

        input_corpus   : sequence of input data arrays
        targets_corpus : sequence of true targets arrays

        Corpora of input and target data must include numpy arrays
        of values to feed to the network and to evaluate against,
        without any nested arrays structure.

        Return the channel-wise root mean square prediction error
        of the network on the full set of samples.
        """
        # Compute sample-wise scores.
        aggregate = np.array if len(self.input_shape) == 2 else np.concatenate
        scores = aggregate([
            self.score(input_data, targets)
            for input_data, targets in zip(input_corpus, targets_corpus)
        ])
        # Gather samples' lengths.
        sizes = self._get_corpus_sizes(input_corpus)
        # Reduce scores and return them.
        return self._reduce_sample_prediction_scores(scores, sizes)

    def _reduce_sample_prediction_scores(self, scores, sample_sizes):
        """Aggregate sample-wise prediction errors."""
        # Compute the square of sample-wise root mean square error terms.
        binary = tuple() if self.binary_tracks is None else self.binary_tracks
        non_binary = [i for i in range(self.n_targets) if i not in binary]
        scores[:, non_binary] = np.square(scores[:, non_binary])
        # Compute the weighted average of channel-wise scores.
        scores = np.sum(scores * sample_sizes, axis=0) / sample_sizes.sum()
        # Root scale mean squared error terms.
        scores[non_binary] = np.sqrt(scores[non_binary])
        # Return the corpus-wise scores.
        return scores


    def _get_corpus_sizes(self, corpus):
        """Return an array gathering the lengths of a corpus' sequences.

        Auxiliary method to `score_corpus`, to be used only as such.
        """
        if len(self.input_shape) == 2:
            sizes = np.array([len(data) for data in corpus])
        elif self.input_shape[1] is None:
            sizes = np.array([len(data) for data in corpus])
        else:
            max_length = self.input_shape[1]
            sizes = np.array([min(len(data), max_length) for data in corpus])
        return np.expand_dims(sizes, 1)
