# coding: utf-8

"""Abstract neural network class and model dump loading function."""

from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import tensorflow as tf
import numpy as np

from neural_networks.components.filters import LowpassFilter, SignalFilter
from neural_networks.components.layers import DenseLayer, NeuralLayer
from neural_networks.components.rnn import (
    AbstractRNN, RecurrentNeuralNetwork, BidirectionalRNN
)
from neural_networks.utils import (
    check_positive_int, check_type_validity,
    get_object, instanciate, onetimemethod
)


def get_layer_class(layer_class):
    """Validate and return a layer class.

    layer_class : either a subclass from AbstractRNN, NeuralLayer
                  or SignalFilter, or the (short) name of one such
                  class
    """
    if isinstance(layer_class, str):
        reference_dict = {
            'dense_layer': DenseLayer, 'rnn_stack': RecurrentNeuralNetwork,
            'bi_rnn_stack': BidirectionalRNN, 'lowpass_filter': LowpassFilter
        }
        return get_object(layer_class, reference_dict, 'layer class')
    elif issubclass(layer_class, (AbstractRNN, NeuralLayer, SignalFilter)):
        return layer_class
    else:
        raise TypeError("'layer_class' should be a str or an adequate class.")


class DeepNeuralNetwork(metaclass=ABCMeta):
    """Abstract class for deep neural network models in tensorflow.

    *** document API and abstract methods ***
    """

    def __init__(
            self, input_shape, n_targets, layers_config, norm_params=None,
            **kwargs
        ):
        """Initialize the neural network.

        input_shape   : shape of the input data fed to the network, with
                        the number of samples as first component
                        (tuple, list, tensorflow.TensorShape)
        n_targets     : number of real-valued targets to predict
        layers_config : list of tuples specifying a layer configuration,
                        made of a layer class (or short name), a number
                        of units (or a cutoff frequency for filters) and
                        an optional dict of keyword arguments
        norm_params   : optional normalization parameters of the targets
                        (np.ndarray)
        """
        # Record and process initialization arguments.
        self._init_arguments = {
            'input_shape': input_shape, 'n_targets': n_targets,
            'layers_config': layers_config, 'norm_params': norm_params
        }
        self._init_arguments.update(kwargs)
        self._validate_args()
        # Declare common attributes to contain the network's structure.
        self._holders = {}
        self._layers = OrderedDict()
        self._readouts = {}
        self._training_function = None
        # Build the network's tensorflow architecture.
        self._build_placeholders()
        self._build_hidden_layers()
        self._build_readout_layer()
        self._build_readouts()
        self._build_training_function()
        # Assign a tensorflow session to the instance.
        if 'session' in kwargs.keys():
            session = kwargs['session']
            check_type_validity(session, tf.Session, 'session')
            self.session = session
        else:
            self.session = tf.Session()
            self.reset_model()

    def __getattr__(self, name):
        """Return initialization arguments when looked up for as attributes.

        Note: if an attribute and an initialization argument share the same
        name, the former will be the one returned by both `getattr` and the
        dot syntax.
        """
        if not hasattr(self, '_init_arguments'):
            raise AttributeError("'%s' object has no attribute '%s'.")
        if name in self._init_arguments.keys():
            return self._init_arguments[name]
        else:
            raise AttributeError(
                "'%s' object has no attribute '%s' nor initialization "
                "argument of such name" % (self.__class__.__name__, name)
            )

    @property
    def architecture(self):
        """Dict describing the network's architecture."""
        return OrderedDict([
            (name, layer.configuration)
            for name, layer in self._layers.items()
        ])

    def get_values(self):
        """Return the current values of the network's layers' parameters."""
        return {
            name: layer.get_values(self.session)
            for name, layer in self._layers.items()
        }

    @property
    def _layer_weights(self):
        """Return the weight and biases tensors of all network layers."""
        def get_weights(layer):
            """Return the weight variables of a given layer."""
            if isinstance(layer, NeuralLayer):
                return (
                    (layer.weight, layer.bias) if layer.bias else layer.weight
                )
            elif isinstance(layer, AbstractRNN):
                return layer.weights
            raise TypeError("Unknown layer class: '%s'." % type(layer))
        return [
            get_weights(layer) for layer in self._layers.values()
            if not isinstance(layer, SignalFilter)
        ]

    def _adjust_init_arguments_for_saving(self):
        """Adjust `_init_arguments` attribute before dumping the model.

        Return a tuple of two dict. The first is a copy of the keyword
        arguments used at initialization, containing only values which
        numpy.save can serialize. The second associates to non-serializable
        arguments' names a dict enabling their (recursive) reconstruction
        using the `neural_networks.utils.instanciate` function.
        """
        return self.init_arguments, None

    def save_model(self, filename):
        """Save the network's configuration and current weights on disk."""
        init_arguments, rebuild_init = self._adjust_init_arguments_for_saving()
        model = {
            '__init__': init_arguments,
            '__class__': self.__module__ + '.' + self.__class__.__name__,
            '__rebuild_init__': rebuild_init,
            'architecture': self.architecture,
            'values': self.get_values(),
        }
        np.save(filename, model)

    def restore_model(self, filename):
        """Restore the networks' weights from disk."""
        load_dumped_model(filename, model=self)

    def reset_model(self, restart_session=False):
        """Reset the network's parameters. Optionally restart its session."""
        if restart_session:
            self.session.close()
            self.session = tf.Session(self.session.sess_str)
        self.session.run(tf.global_variables_initializer())

    @property
    def _top_layer(self):
        """Return the layer on top of the network's architecture."""
        return self._layers[list(self._layers.keys())[-1]]

    @onetimemethod
    def _validate_args(self):
        """Process the initialization arguments of the instance."""
        # Validate the model's input layer shape.
        check_type_validity(
            self.input_shape, (tuple, list, tf.TensorShape), 'input_shape'
        )
        if len(self.input_shape) == 1:
            raise TypeError("'input_shape' must be at least bi-dimensional.")
        # Validate the model's layers configuration.
        check_type_validity(self.layers_config, list, 'layers_config')
        for i, config in enumerate(self.layers_config):
            # Check that the configuration is a three-elements tuple.
            if not isinstance(config, tuple):
                raise TypeError("'layers_config' elements should be tuples.")
            if len(config) == 2:
                self.layers_config[i] = config = (*config, {})
            elif len(config) != 3:
                raise TypeError("Wrong 'layers_config' element tuple length.")
            # Check sub-elements type validity.
            check_type_validity(config[0], (str, type), 'layer class')
            check_type_validity(
                config[1], (int, list, tuple), 'layer config primary parameter'
            )
            check_type_validity(config[2], dict, 'layer config kwargs')
        # Validate the model's number of targets.
        check_positive_int(self.n_targets, 'n_targets')
        # Validate the model's normalization parameters.
        norm_params = self.norm_params
        check_type_validity(
            norm_params, (np.ndarray, type(None)), 'norm_params'
        )
        if norm_params is not None and norm_params.shape != (self.n_targets,):
            raise TypeError(
                "Wrong 'norm_params' shape: %s instead of (%s,)"
                % (norm_params.shape, self.n_targets)
            )

    @onetimemethod
    def _build_placeholders(self):
        """Build the network's placeholders."""
        self._holders['input'] = tf.placeholder(tf.float32, self.input_shape)
        self._holders['targets'] = tf.placeholder(
            tf.float32, [self.input_shape[0], self.n_targets]
        )
        self._holders['keep_prob'] = tf.placeholder(tf.float32, ())

    @onetimemethod
    def _build_hidden_layers(self):
        """Build the network's hidden layers."""
        # Gather the input placeholder. Declare a layers counter.
        input_tensor = self._holders['input']
        layers_count = {}
        # Iteratively build the layers.
        for name, n_units, kwargs in self.layers_config:
            # Get the layer's class and instanciate it.
            layer_class = get_layer_class(name)
            if issubclass(layer_class, DenseLayer):
                kwargs = kwargs.copy()
                kwargs['keep_prob'] = kwargs.get(
                    'keep_prob', self._holders['keep_prob']
                )
            layer = layer_class(input_tensor, n_units, **kwargs)
            # Update the layers counter.
            index = layers_count.setdefault(name, 0)
            layers_count[name] += 1
            # Add the layer on top of the stack and use its output as next input.
            self._layers[name + '_%s' % index] = layer
            input_tensor = layer.output

    @abstractmethod
    @onetimemethod
    def _build_readout_layer(self):
        """Build the network's readout layer.

        This method should update the `_layers_stack` list attribute.
        """
        return NotImplemented

    @abstractmethod
    @onetimemethod
    def _build_readouts(self):
        """Build wrappers on top of the network's readout layer.

        This method should update the `_readouts` dict attribute.
        """
        return NotImplemented

    @abstractmethod
    @onetimemethod
    def _build_training_function(self):
        """Build the train step function of the network.

        This method should assign a tensorflow operation
        to the `_training_function` attribute.
        """
        return NotImplemented


def load_dumped_model(filename, model=None):
    """Restore a neural network model from a .npy dump.

    filename : path to a .npy file containing a model's configuration
    model    : optional instanciated model whose weights to restore
               (default None, implying that a model is instanciated
               based on the dumped configuration and returned)
    """
    # Access private attributes by design; pylint: disable=protected-access
    # Load the dumped model configuration.
    config = np.load(filename).tolist()
    check_type_validity(config, dict, 'loaded configuration')
    # If no model was provided, instanciate one.
    new_model = model is None
    if new_model:
        model = instanciate(
            config['__class__'], config['__init__'], config['__rebuild_init__']
        )
        if 'session' not in config['__init__'].keys():
            model.reset_model()
    # Check that the model's architecture is coherent with the dump.
    if model.architecture != config['architecture']:
        raise TypeError("Invalid network architecture.")
    # Restore the model's weights.
    for name, layer in model._layers.items():
        layer.set_values(config['values'][name], model.session)
    # If the model was instanciated within this function, return it.
    return model if new_model else None
