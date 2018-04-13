# coding: utf-8

"""Class implementing auto-encoder networks in tensorflow."""


import tensorflow as tf
import numpy as np

from neural_networks.core import (
    build_rmse_readouts, refine_signal, validate_layer_config
)
from neural_networks.models import MultilayerPerceptron
from neural_networks.utils import check_type_validity, onetimemethod


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
            norm_params=None, optimizer=None
        ):
        """Instantiate the auto-encoder network.

        input_shape    : shape of the input data fed to the network,
                         with the number of samples as first component
                         and including dynamic features
        n_targets      : number of real-valued targets to predict,
                         notwithstanding dynamic features
        encoder_config : list of tuples specifying the encoder's architecture
        decoder_config : list of tuples specifying the decoder's architecture
        encoder_filter : optional tuple specifying a top filter for the encoder
        decoder_filter : optional tuple specifying a top filter for the decoder
        use_dynamic    : whether to produce dynamic features and use them
                         when training the model (bool, default True)
        norm_params    : optional normalization parameters of the targets
                         (np.ndarray)
        optimizer      : tensorflow.train.Optimizer instance (by default,
                         Adam optimizer with 1e-3 learning rate)

        The `encoder_config` and `decoder_config` arguments should
        be of a similar form as the `layers_config` argument of any
        `DeepNeuralNetwork` instance, i.e. lists of tuples specifying
        a layer configuration each, made of a layer class (or short name),
        a number of units (or a cutoff frequency for filters) and an
        optional dict of keyword arguments.

        Readout layer are automatically added between the encoder and
        decoder parts, and on top of the latter. They are fully-connected
        layers with identity activation.
        """
        # Arguments serve modularity ; pylint: disable=too-many-arguments
        # Compute the number of elements to reconstruct.
        check_type_validity(input_shape, (list, tuple), 'input_shape')
        check_type_validity(use_dynamic, bool, 'use_dynamic')
        n_inputs = input_shape[1]
        if use_dynamic:
            if n_inputs % 3:
                raise ValueError(
                    "Wrong `input_shape` with `use_dynamic=True`: "
                    "dim 1 should a divisor of 3."
                )
            n_inputs //= 3
        # Build the network's full layers configuration.
        layers_config = self._build_layers_config(
            n_inputs, n_targets, encoder_config, decoder_config,
            encoder_filter, decoder_filter
        )
        # Initialize the auto-encoder network.
        super().__init__(
            input_shape, n_targets, layers_config, use_dynamic=use_dynamic,
            optimizer=optimizer
        )
        # Record additionnal initialization arguments.
        self._init_arguments['encoder_config'] = encoder_config
        self._init_arguments['decoder_config'] = decoder_config
        self._init_arguments['encoder_filter'] = encoder_filter
        self._init_arguments['decoder_filter'] = decoder_filter
        # Remove unused inherited argument.
        self._init_arguments.pop('top_filter')

    @staticmethod
    def _build_layers_config(
            n_inputs, n_targets, encoder_config, decoder_config,
            encoder_filter, decoder_filter
        ):
        """Build and return the list specifying all network layers."""
        # Conduct minimal necessary tests. More are run later on.
        check_type_validity(encoder_config, list, 'encoder_config')
        check_type_validity(decoder_config, list, 'decoder_config')

        # Define a function specifying readout layers.
        def get_readout(part, n_units, top_filter):
            """Return the configuration of a network part's readout layer."""
            kwargs = {'activation': 'identity', 'name': part + '_readout'}
            readout_layer = ('dense_layer', n_units, kwargs)
            if top_filter is None:
                return [readout_layer]
            top_filter = validate_layer_config(top_filter)
            top_filter[2].setdefault('name', part + '_top_filter')
            return [readout_layer, top_filter]

        # Aggregate the encoder's and decoder's layers.
        encoder_readout = get_readout('encoder', n_targets, encoder_filter)
        decoder_readout = get_readout('decoder', n_inputs, decoder_filter)
        return (
            encoder_config + encoder_readout + decoder_config + decoder_readout
        )

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
    def _build_readout_layer(self):
        """Empty method, solely included to respect the API standards."""
        pass

    @onetimemethod
    def _build_readouts(self):
        """Build wrappers of the network's predictions and errors."""
        def build_readouts(part, true_data, norm_params=None):
            """Build the error readouts of a part of the network."""
            raw_output = self.layers.get(
                part + '_top_filter', self.layers[part + '_readout']
            ).output
            output, _ = refine_signal(
                raw_output, norm_params, None, self.use_dynamic
            )
            readouts = build_rmse_readouts(output, true_data)
            for name, readout in readouts.items():
                self.readouts[part + '_' + name] = readout
        # Use the previous function to build partial RMSE readouts.
        build_readouts('encoder', self.holders['targets'], self.norm_params)
        build_readouts('decoder', self.holders['input'])
        # Build wrappers aggregating predictions and scores of the model.
        self.readouts['rmse'] = tf.concat([
            self.readouts['encoder_rmse'], self.readouts['decoder_rmse']
        ], axis=0)
        self.readouts['prediction'] = tf.concat([
            self.readouts['encoder_prediction'],
            self.readouts['decoder_prediction']
        ], axis=1)

    def predict(self, input_data):
        """Predict the targets associated with a given set of inputs."""
        prediction = super().predict(input_data)
        encoded, decoded = self._split_metrics(prediction.T)
        return encoded.T, decoded.T

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

        Return the channel-wise root mean square prediction error
        of the network on the full set of samples.
        """
        # Compute sample-wise errors.
        predict = super().predict
        errors = np.concatenate([
            predict(input_data) - np.concatenate([targets, input_data], axis=1)
            for input_data, targets in zip(input_corpus, targets_corpus)
        ])
        # Reduce scores and return them.
        scores = np.sqrt(np.square(errors).mean(axis=0))
        return self._split_metrics(scores)

    def _split_metrics(self, metrics):
        """Split an array aggregating encoder and decoder outputs."""
        n_targets = self.n_targets * (3 if self.use_dynamic else 1)
        return metrics[:n_targets], metrics[n_targets:]
