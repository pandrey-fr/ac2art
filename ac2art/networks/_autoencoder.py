# coding: utf-8
#
# Copyright 2018 Paul Andrey
#
# This file is part of ac2art.
#
# ac2art is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ac2art is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ac2art.  If not, see <http://www.gnu.org/licenses/>.

"""Class implementing auto-encoder networks in tensorflow."""


import tensorflow as tf
import numpy as np

from ac2art.internal.network_bricks import (
    build_layers_stack, build_rmse_readouts,
    refine_signal, validate_layer_config
)
from ac2art.internal.neural_layers import DenseLayer
from ac2art.networks import NeuralNetwork, MultilayerPerceptron
from ac2art.utils import check_type_validity, onetimemethod


class AutoEncoder(MultilayerPerceptron):
    """Class implementing auto-encoder networks in tensorflow.

    An auto-encoder is a network composed of two stacked sub-networks,
    and encoder and a decoder. The former aims at producing a given
    representation (or prediction) out of some input data, while the
    latter reconstructs the initial data from this representation.
    Both network are trained jointly, through a loss function combining
    their prediction and reconstruction errors.
    """

    def __init__(
            self, input_shape, n_targets, encoder_config, decoder_config,
            encoder_filter=None, decoder_filter=None, use_dynamic=True,
            binary_tracks=None, norm_params=None, optimizer=None
        ):
        """Instantiate the auto-encoder network.

        input_shape    : shape of the input data fed to the network,
                         of either [n_samples, input_size] shape
                         or [n_batches, max_length, input_size],
                         where the last axis must be fixed (non-None)
                         (tuple, list, tensorflow.TensorShape)
        n_targets      : number of targets to predict,
                         notwithstanding dynamic features
        encoder_config : list of tuples specifying the encoder's architecture
        decoder_config : list of tuples specifying the decoder's architecture
        encoder_filter : optional tuple specifying a top filter for the encoder
        decoder_filter : optional tuple specifying a top filter for the decoder
        use_dynamic    : whether to produce dynamic features and use them
                         when training the model (bool, default True)
        binary_tracks  : optional list of targets which are binary-valued
                         (and should therefore not have delta counterparts)
        norm_params    : optional normalization parameters of the targets
                         (np.ndarray)
        optimizer      : tensorflow.train.Optimizer instance (by default,
                         Adam optimizer with 1e-3 learning rate)

        The `encoder_config` and `decoder_config` arguments should
        be of a similar form as the `layers_config` argument of any
        `NeuralNetwork` instance, i.e. lists of tuples specifying
        a layer configuration each, made of a layer class (or short name),
        a number of units (or a cutoff frequency for filters) and an
        optional dict of keyword arguments.

        Readout layers are automatically added between the encoder and
        decoder parts, and on top of the latter. They are fully-connected
        layers with identity activation.
        """
        # Arguments serve modularity ; pylint: disable=too-many-arguments
        # Use the basic API init instead of that of the direct parent.
        # pylint: disable=super-init-not-called, non-parent-init-called
        # Initialize the auto-encoder network.
        NeuralNetwork.__init__(
            self, input_shape, n_targets, layers_config=[],
            use_dynamic=use_dynamic, binary_tracks=binary_tracks,
            encoder_config=encoder_config, encoder_filter=encoder_filter,
            decoder_config=decoder_config, decoder_filter=decoder_filter,
            optimizer=optimizer
        )
        # Remove unused inherited argument.
        self._init_arguments.pop('top_filter')

    def _adjust_init_arguments_for_saving(self):
        """Adjust `_init_arguments` attribute before dumping the model.

        Return a tuple of two dict. The first is a copy of the keyword
        arguments used at initialization, containing only values which
        numpy.save can serialize. The second associates to non-serializable
        arguments' names a dict enabling their (recursive) reconstruction
        using the `neural_networks.utils.instantiate` function.
        """
        init_args, rebuild_init = super()._adjust_init_arguments_for_saving()
        init_args.pop('layers_config')
        return init_args, rebuild_init

    @onetimemethod
    def _validate_args(self):
        """Process the initialization arguments of the instance."""
        # Validate arguments that do not define the model's layers.
        super()._validate_args()
        # Check input_shape and use_dynamic parameters' compatibility.
        if self.use_dynamic and self.input_shape[-1] % 3:
            raise ValueError(
                "Wrong `input_shape` with `use_dynamic=True`: "
                "dim 1 should a divisor of 3."
            )
        # Validate and alter if needed the network's layers' configuration.
        for half in ('encoder', 'decoder'):
            # Check the network's half's hidden layers' configuration.
            layers_config = self._init_arguments[half + '_config']
            check_type_validity(layers_config, list, half + '_config')
            for i, config in enumerate(layers_config):
                layers_config[i] = validate_layer_config(config)
            # Check the network's half's top filter's configuration.
            top_filter = self._init_arguments[half + '_filter']
            if top_filter is not None:
                self._init_arguments[half + '_filter'] = (
                    validate_layer_config(top_filter)
                )

    @onetimemethod
    def _build_hidden_layers(self):
        """Build the network's hidden layers and readouts.

        The name of this private method is thus inaccurate,
        but mandatory to preserve API standards.
        """
        self._build_encoder()
        self._build_decoder()

    @onetimemethod
    def _build_encoder(self):
        """Build the layers and readouts of the network's encoder half."""
        # Build the encoder's hidden layers' stack.
        hidden_layers = build_layers_stack(
            self.holders['input'], self.encoder_config,
            self.holders['keep_prob'], self.holders.get('batch_sizes'),
            check_config=False
        )
        for key, layer in hidden_layers.items():
            self.layers['encoder_' + key] = layer
        # Build the encoder's readout layer(s) and derived readouts.
        super()._build_readout_layer()
        self._init_arguments['top_filter'] = self.encoder_filter
        super()._build_readouts()
        # Rename the encoder's readout layers' storage keys.
        self.layers['encoder_readout_layer'] = self.layers.pop('readout_layer')
        if self.binary_tracks:
            self.layers['encoder_readout_layer_binary'] = (
                self.layers.pop('readout_layer_binary')
            )
        if self.encoder_filter:
            for key in list(self.layers.keys())[::-1]:
                if key.startswith('top_filter'):
                    self.layers['encoder_top_filter'] = self.layers.pop(key)
                    break
        # Rename all generated readouts associated with the encoder.
        for key in list(self.readouts.keys()):
            self.readouts['encoder_' + key] = self.readouts.pop(key)

    @onetimemethod
    def _build_decoder(self):
        """Build the layers and readouts of the network's decoder half."""
        # Build the decoder's hidden layers' stack.
        hidden_layers = build_layers_stack(
            self.readouts['encoder_prediction'], self.decoder_config,
            self.holders['keep_prob'], self.holders.get('batch_sizes'),
            check_config=False
        )
        for key, layer in hidden_layers.items():
            self.layers['decoder_' + key] = layer
        # Build the decoder's readout layer.
        n_targets = self.input_shape[-1]
        if self.use_dynamic:
            n_targets //= 3
        self.layers['decoder_readout_layer'] = readout_layer = DenseLayer(
            self._top_layer.output, n_targets, 'identity'
        )
        # Build and refine the decoder's prediction.
        self.readouts['decoder_raw_prediction'] = readout_layer.output
        prediction, top_filter = refine_signal(
            readout_layer.output, None, self.decoder_filter, self.use_dynamic
        )
        if self.decoder_filter:
            self.layers['decoder_top_filter'] = top_filter
        # Build the decoder's error readouts.
        rmse_readouts = build_rmse_readouts(
            prediction, self.holders['input'], self.holders.get('batch_sizes')
        )
        for key, readout in rmse_readouts.items():
            self.readouts['decoder_' + key] = readout

    @onetimemethod
    def _build_readout_layer(self):
        """Empty method, solely included to respect the API standards."""
        pass

    @onetimemethod
    def _build_readouts(self):
        """Build wrappers of the network's predictions and errors."""
        self.readouts['rmse'] = tf.concat([
            self.readouts['encoder_rmse'], self.readouts['decoder_rmse']
        ], axis=-1)
        self.readouts['prediction'] = tf.concat([
            self.readouts['encoder_prediction'],
            self.readouts['decoder_prediction']
        ], axis=-1)

    def predict(self, input_data):
        """Predict the targets associated with a given set of inputs."""
        prediction = super().predict(input_data)
        if len(prediction.shape) > 1:
            return self._split_metrics(prediction)
        # Handle batched predictions case.
        encoder_pred = [None] * len(prediction)
        decoder_pred = [None] * len(prediction)
        for i, pred in enumerate(prediction):
            enc_pred, dec_pred = self._split_metrics(pred)
            encoder_pred[i] = enc_pred
            decoder_pred[i] = dec_pred
        return encoder_pred, decoder_pred

    def score(self, input_data, targets):
        """Compute the root mean square prediction errors of the network.

        input_data : input data to be rebuilt by the network's decoder part
        targets    : target data to be rebuilt by the network's encoder part

        Return a couple of numpy arrays containing respectively
        the prediction and reconstruction by-channel root mean
        square errors.
        """
        scores = super().score(input_data, targets)
        return self._split_metrics(scores)

    def score_corpus(self, input_corpus, targets_corpus):
        """Iteratively compute the network's root mean square prediction error.

        input_corpus   : sequence of input data arrays
        targets_corpus : sequence of true targets arrays

        Corpora of input and target data must include numpy arrays
        of values to feed to the network and to evaluate against,
        without any nested arrays structure

        Return a couple of numpy arrays containing respectively
        the prediction and reconstruction by-channel root mean
        square errors on the full set of sample pairs.
        """
        # Compute sample-wise scores.
        get_score = super().score
        scores = np.concatenate([
            np.square(get_score(input_data, targets))
            for input_data, targets in zip(input_corpus, targets_corpus)
        ])
        # Reduce scores and return them.
        sizes = self._get_corpus_sizes(input_corpus)
        scores = np.sqrt(np.sum(scores * sizes, axis=0) / sizes.sum())
        return self._split_metrics(scores)

    def _split_metrics(self, metrics):
        """Split an array aggregating encoder and decoder outputs."""
        n_input = self.input_shape[-1]
        return metrics[..., :-n_input], metrics[..., -n_input:]
