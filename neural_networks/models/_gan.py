# coding: utf-8

"""Set of classes implementing Generative Adversarial Nets for inversion."""

import os
import time

import tensorflow as tf
import numpy as np

from neural_networks.components.layers import DenseLayer
from neural_networks.core import DeepNeuralNetwork, build_layers_stack
from neural_networks.tf_utils import minimize_safely
from neural_networks.utils import check_type_validity, onetimemethod


class Discriminator:
    """Structure designing a binary classifier.

    This class partly mimics the API of the DeepNeuralNetwork class, but
    it does not inherit from it nor implement all of its functionalities.
    """
    # Structure rather than class; pylint: disable=too-few-public-methods

    def __init__(self, input_tensor, labels_tensor, layers_config):
        """Instantiate the discriminator network.

        input_tensor  : input samples to discriminate (tensorflow.Tensor)
        labels_tensor : labels indicating which input samples are real
                        and which are synthetic (tensorflow.Tensor)
        layers_config : list of tuples specifying the discriminator
                        network's architecture; each tuple should
                        be composed of a number of units (int) and
                        a dict of keyword arguments used when
                        building the fully-connected layers
        """
        # Assign the arguments as attributes and control their validity.
        self.input_tensor = input_tensor
        self.labels_tensor = labels_tensor
        self.layers_config = layers_config
        self._validate_args()
        # Build the network.
        self.layers = None
        self.readouts = {}
        self._build_layers()
        self._build_readouts()

    @onetimemethod
    def _validate_args(self):
        """Control the validity of the initialization arguments."""
        check_type_validity(self.input_tensor, tf.Tensor, 'input_tensor')
        check_type_validity(self.labels_tensor, tf.Tensor, 'labels_tensor')
        check_type_validity(self.layers_config, list, 'layers_config')
        for i, config in enumerate(self.layers_config):
            if not isinstance(config, tuple) and len(config) == 2:
                raise TypeError(
                    '`layers_config` should contain two-elements tuples.'
                )
            self.layers_config[i] = ('dense_layer', *config)

    @property
    def _weights(self):
        """Return the weight and biases tensors of all neural layers."""
        return [
            (layer.weight, layer.bias) if layer.bias else layer.weight
            for layer in self.layers.values()
        ]

    @onetimemethod
    def _build_layers(self):
        """Build the discriminator network's layers stack."""
        self.layers = build_layers_stack(self.input_tensor, self.layers_config)
        self.layers['readout_layer'] = DenseLayer(
            self.layers[list(self.layers.keys())[-1]].output,
            n_units=2, activation='sigmoid', bias=False
        )

    @onetimemethod
    def _build_readouts(self):
        """Build wrappers on top of the network's readout layer."""
        readout = self.layers['readout_layer'].output
        self.readouts['cross_entropy'] = (
            tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.one_hot(self.labels_tensor, 2), logits=readout
            )
        )
        prediction = tf.argmax(
            tf.nn.softmax(readout), axis=1, output_type=tf.int32
        )
        right = tf.equal(prediction, self.labels_tensor)
        self.readouts['accuracy'] = tf.reduce_mean(tf.cast(right, tf.float32))


class GenerativeAdversarialNets:
    """Class to define generative adversarial networks in tensorflow.

    GANs are made of a generator network, which based on some input
    aims at generating some target data, and a discriminator network,
    whose goal is to distinguish the actual target data from that
    produced by the generator network. These network's loss functions
    and training procedures are then brought together so that their
    joint training may help improve performance of both networks, and
    especially that of the generator.

    This class does not inherit from DeepNeuralNetwork, but mimics
    bits of its API while not reproducing all of its functionalities.
    """

    def __init__(self, generator_model, discriminator_layers):
        """Instantiate the adversarial training instance.

        generator_model      : pre-instantiated DeepNeuralNetwork inheriting
                               generative network
        discriminator_layers : list of tuples specifying the discriminator
                               network's fully-connected layers' architecture
        """
        check_type_validity(
            generator_model, DeepNeuralNetwork, 'generator_model'
        )
        self.generator = generator_model
        self.session = self.generator.session
        self.discriminator = None
        self._build_discriminator(discriminator_layers)
        self.training_function = None
        self._build_training_function()
        self._initialize_model()

    @onetimemethod
    def _build_discriminator(self, layers_config):
        """Build the discriminator network."""
        # Gather the true data and that generated by the other network.
        true_data = self.generator.holders['targets']
        false_data = self.generator.readouts['prediction']
        # Concatenate both data tensors and build the associated labels
        input_tensor = tf.concat([true_data, false_data], axis=0)
        labels_tensor = tf.concat([
            tensor_like(true_data[:, 0], dtype=tf.int32)
            for tensor_like in (tf.ones_like, tf.zeros_like)
        ], axis=0)
        # Instantiate a binary classifier network fed by the previous.
        self.discriminator = Discriminator(
            input_tensor, labels_tensor, layers_config
        )

    @onetimemethod
    def _initialize_model(self):
        """Initialize the models' weights and nodes."""
        # Save the current state of the generator network.
        tempfile = 'tmp_%s.npy' % int(time.time())
        self.generator.save_model(tempfile)
        # Initialize the discriminator, thus resetting the generator.
        self.session.run(tf.global_variables_initializer())
        # Restore the generator's weights. Remove its temporary dump.
        self.generator.restore_model(tempfile)
        os.remove(tempfile)

    @onetimemethod
    def _build_training_function(self):
        """Build the adversarial training functions of the networks."""
        rmse = tf.reduce_mean(self.generator.readouts['rmse'])
        cross_entropy = self.discriminator.readouts['cross_entropy']
        # Build adversarial training functions of the two networks.
        # Access hidden properties; pylint: disable=protected-access
        fit_discriminator = minimize_safely(
            self.generator.optimizer, loss=cross_entropy,
            var_list=self.discriminator._weights
        )
        fit_generator = minimize_safely(
            self.generator.optimizer, loss=rmse + (1 - cross_entropy),
            var_list=self.generator._neural_weights
        )
        # Add a function to optimize the generator's top filter, if any.
        learnable_filter = (
            self.generator.top_filter is not None
            and self.generator.layers['top_filter'].learnable
        )
        if learnable_filter:
            fit_filter = minimize_safely(
                tf.train.AdamOptimizer(.9), loss=rmse,
                var_list=self.generator._filter_cutoffs
            )
            fit_generator = (fit_generator, fit_filter)
        # pylint: enable=protected-access
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

    def run_training_step(self, input_data, target_data, n_iter=10):
        """Run a training step, fitting both networks adversarially.

        input_data  : input data to feed to the generator network
        target_data : true data the generator network should produce
        n_iter      : number of subsamples to draw from the data
                      and iterate over so as to fit both networks
                      (positive int, default 10)
        """
        # Cut the data into subsamples and define a training procedure.
        size = len(input_data) // n_iter
        samples = [
            list(range(i, i + size)) for i in range(0, len(input_data), size)
        ]
        def train_network(network):
            """Iteratively fit a network using the different data samples."""
            for sample in np.random.permutation(samples):
                feed = self.generator.get_feed_dict(
                    input_data[sample], target_data[sample]
                )
                self.session.run(self.training_function(network), feed)
        # Train the discriminator network, and then the generator one.
        train_network('discriminator')
        train_network('generator')


    def score(self, input_data, target_data):
        """Score both networks based on the provided data.

        Return the generator's per-channel root mean square
        prediction error and the discriminator's accuracy.
        """
        return self.session.run([
            self.generator.readouts['rmse'],
            self.discriminator.readouts['accuracy']
        ], self.generator.get_feed_dict(input_data, target_data))
