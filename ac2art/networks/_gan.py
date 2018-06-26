# coding: utf-8

"""Class implementing Generative Adversarial Nets for inversion."""


import tensorflow as tf
import numpy as np

from ac2art.internal.network_bricks import (
    build_layers_stack, build_binary_classif_readouts,
    get_layer_class, validate_layer_config
)
from ac2art.internal.neural_layers import DenseLayer, SignalFilter
from ac2art.internal.tf_utils import minimize_safely
from ac2art.networks import NeuralNetwork, MultilayerPerceptron
from ac2art.utils import check_type_validity, onetimemethod


class GenerativeAdversarialNets(MultilayerPerceptron):
    """Class to define generative adversarial networks in tensorflow.

    GANs are made of a generator network, which based on some input
    aims at generating some target data, and a discriminator network,
    whose goal is to distinguish the actual target data from that
    produced by the generator network. These network's loss functions
    and training procedures are then brought together so that their
    joint training may help improve performance of both networks, and
    especially that of the generator.
    """

    def __init__(
            self, input_shape, n_targets, layers_config, discr_config,
            top_filter=None, use_dynamic=True, binary_tracks=None,
            norm_params=None, optimizer=None
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
        discr_config  : list of tuples specifying the discriminator's layers
                        (similar format as `layers_config`)
        top_filter    : optional tuple specifying a SignalFilter to use
                        on top of the generator network's raw prediction
        use_dynamic   : whether to produce dynamic features and use them
                        when training the model (bool, default True)
        binary_tracks : optional list of targets which are binary-valued
                        (and should therefore not have delta counterparts)
        norm_params   : optional normalization parameters of the targets
                        (np.ndarray)
        optimizer     : tensorflow.train.Optimizer instance (by default,
                        Adam optimizer with 1e-3 learning rate)
        """
        # Arguments serve modularity ; pylint: disable=too-many-arguments
        # Use the basic API init instead of that of the direct parent.
        # pylint: disable=super-init-not-called, non-parent-init-called
        NeuralNetwork.__init__(
            self, input_shape, n_targets, layers_config,
            top_filter, use_dynamic, binary_tracks, norm_params,
            discr_config=discr_config, optimizer=optimizer
        )

    @onetimemethod
    def _validate_args(self):
        """Process the initialization arguments of the instance."""
        # Validate arguments defining the generator network.
        super()._validate_args()
        # Validate the discriminator network's hidden layers config.
        check_type_validity(self.discr_config, list, 'discr_config')
        for i, config in enumerate(self.discr_config):
            validated = validate_layer_config(config)
            if isinstance(get_layer_class(validated[0]), SignalFilter):
                raise ValueError(
                    'Discrimnator layers may not contain signal filters.'
                )
            self.discr_config[i] = validated

    @onetimemethod
    def _build_hidden_layers(self):
        """Build the network's hidden layers and readouts.

        The name of this private method is thus inaccurate,
        but mandatory to preserve API standards.
        """
        self._build_generator()
        self._build_discriminator()

    @onetimemethod
    def _build_generator(self):
        """Build the layers and readouts of the generator network."""
        super()._build_hidden_layers()
        super()._build_readout_layer()
        super()._build_readouts()

    @onetimemethod
    def _build_discriminator(self):
        """Build the layers and readouts of the discriminator network."""
        # Aggregate the generator's predictions and the true targets.
        true_data = self.holders['targets']
        false_data = self.readouts['prediction']
        if len(self.input_shape) == 2:
            true_data = tf.expand_dims(true_data, 0)
            false_data = tf.expand_dims(false_data, 0)
        input_tensor = tf.concat([true_data, false_data], axis=0)
        # Build the discriminator network's layers.
        discr_layers = build_layers_stack(input_tensor, self.discr_config)
        for name, layer in discr_layers.items():
            self.layers['discrim_' + name] = layer


    @onetimemethod
    def _build_readout_layer(self):
        """Build the discriminator network's readout layer."""
        self.layers['discrim_readout_layer'] = DenseLayer(
            self._top_layer.output, n_units=1, activation='sigmoid'
        )

    @onetimemethod
    def _build_readouts(self):
        """Build wrappers of the network's predictions and errors."""
        # Gather the discriminator's output and the true labels tensor.
        readout = self.layers['discrim_readout_layer'].output
        targets = self.holders['targets']
        batched = (len(self.input_shape) == 3)
        labels_shape = targets[:, 0, 0] if batched else [1.]
        labels_tensor = tf.concat([
            tf.ones_like(labels_shape), tf.zeros_like(labels_shape)
        ], axis=0)
        # Build the discriminator's probability predictions.
        if batched:
            sizes_tensor = self.holders.get('batch_sizes')
            sizes_tensor = tf.concat([sizes_tensor, sizes_tensor], axis=0)
            mask = tf.sequence_mask(
                sizes_tensor, maxlen=targets.shape[-2].value, dtype=tf.float32
            )
            predicted_proba = (
                tf.reduce_sum(readout * tf.expand_dims(mask, 2), axis=1)
                / tf.reduce_sum(mask, axis=1, keepdims=True)
            )
        else:
            predicted_proba = tf.reduce_mean(readout, axis=1)
        # Build and record the discriminator's readouts.
        discriminator_readouts = build_binary_classif_readouts(
            predicted_proba, tf.expand_dims(labels_tensor, 1)
        )
        for key, readout in discriminator_readouts.items():
            self.readouts['discrim_' + key] = readout

    @onetimemethod
    def _build_training_function(self):
        """Build the adversarial training functions of the networks."""
        generat_loss = self.readouts['rmse']
        discrim_loss = self.readouts['discrim_cross_entropy']
        # Build adversarial training functions of the two networks.
        fit_discriminator = minimize_safely(
            self.optimizer, loss=discrim_loss,
            var_list=[
                self.get_weights(name) for name in self.layers
                if name.startswith('discrim_')
            ]
        )
        fit_generator = minimize_safely(
            self.optimizer, loss=generat_loss + (1 - discrim_loss),
            var_list=[
                self.get_weights(name) for name, layer in self.layers.items()
                if not name.startswith('discrim_')
                and not isinstance(layer, SignalFilter)
            ]
        )
        # Add a function to optimize the generator's filter layer(s), if any.
        if self._filter_cutoffs:
            fit_filter = minimize_safely(
                tf.train.GradientDescentOptimizer(.9),
                loss=generat_loss, var_list=self._filter_cutoffs
            )
            fit_generator = (fit_generator, fit_filter)

        # Build a wrapper for the network's training functions.
        def training_function(model):
            """Return the specified training function(s)."""
            if model == 'discriminator':
                return fit_discriminator
            elif model == 'generator':
                return fit_generator
            elif model == 'both':
                return (fit_discriminator, fit_generator)
            else:
                raise KeyError("Invalid model key: '%s'." % model)

        # Assign the wrapper as the instance's training function.
        self.training_function = training_function

    def run_training_function(
            self, input_data, target_data, keep_prob=1, network='both'
        ):
        """Run a training step, fitting both networks adversarially.

        input_data  : input data to feed to the generator network
        keep_prob   : probability for each unit to have its outputs used in
                      the training procedure (float in [0., 1.], default 1.)
        network     : str identifying the model(s) to fit, among
                      {'discriminator', 'generator', 'both'}
        """
        # Add an argument unneeded by parents; pylint: disable=arguments-differ
        feed_dict = self.get_feed_dict(input_data, target_data, keep_prob)
        self.session.run(self.training_function(network), feed_dict)

    def score(self, input_data, target_data, network='generator'):
        """Score the network(s) based on the provided data.

        input_data : input data sample to evalute the model on which
        targets    : true targets associated with the input dataset
        network    : network(s) to score ; either 'generator',
                     'discriminator' or 'both'

        The generator's metrics include the channel-wise prediction
        root mean square error (or binary cross-entropy, for binary
        channels).
        The discriminator's metric is the prediction accuracy,
        evaluated against both the generator's predictions and
        the true targets (i.e. balanced dataset).
        """
        # Add an argument unneeded by parents; pylint: disable=arguments-differ
        # Select the metric(s) to return.
        if network == 'generator':
            metric = self.readouts['rmse']
        elif network == 'discriminator':
            metric = self.readouts['discrim_accuracy']
        elif network == 'both':
            metric = (self.readouts['rmse'], self.readouts['discrim_accuracy'])
        else:
            raise KeyError("Invalid 'network' argument: '%s'." % network)
        # Evaluate the selected metric(s).
        feed_dict = self.get_feed_dict(input_data, target_data)
        return self.session.run(metric, feed_dict)

    def score_corpus(self, input_corpus, targets_corpus, network='generator'):
        """Iteratively compute the network's root mean square prediction error.

        input_corpus   : sequence of input data arrays
        targets_corpus : sequence of true targets arrays
        network        : network(s) to score ; either 'generator',
                         'discriminator' or 'both'

        Corpora of input and target data must include numpy arrays
        of values to feed to the network and to evaluate against,
        without any nested arrays structure.

        See `score` method for details on the metrics returned
        depending on the selected `network` argument.
        """
        # Add an argument unneeded by parents; pylint: disable=arguments-differ
        # Case when scoring the generator only.
        if network == 'generator':
            return super().score_corpus(input_corpus, targets_corpus)
        # Compute sample-wise scores.
        scores = [
            self.score(input_data, targets, network)
            for input_data, targets in zip(input_corpus, targets_corpus)
        ]
        # Case when scoring the discriminator only.
        if network == 'discriminator':
            return np.mean(scores)
        # Case when scoring both networks.
        discriminator_score = np.mean([score for _, score in scores])
        aggregate = np.array if len(self.input_shape) == 2 else np.concatenate
        generator_score = self._reduce_sample_prediction_scores(
            aggregate([score for score, _ in scores]),
            self._get_corpus_sizes(input_corpus)
        )
        return generator_score, discriminator_score
