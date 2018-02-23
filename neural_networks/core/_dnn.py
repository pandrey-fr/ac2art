# coding: utf-8

"""Abstract neural network class and model dump loading function."""

from abc import ABCMeta, abstractmethod
import inspect
from collections import OrderedDict

import tensorflow as tf
import numpy as np

from neural_networks.utils import (
    check_positive_int, check_type_validity, instanciate,
    raise_type_error, onetimemethod
)


class DeepNeuralNetwork(metaclass=ABCMeta):
    """Abstract class for deep neural network models in tensorflow.

    *** document API and abstract methods ***
    """

    def __init__(self, input_shape, n_targets, activation, **kwargs):
        """Initialize the neural network.

        input_shape : shape of the input data fed to the network, with
                      the number of samples as first component
                      (tuple, list, tensorflow.TensorShape)
        n_targets   : number of real-valued targets to predict
        activation  : default activation function to use (or its name)
        """
        # Record and process initialization arguments.
        self._init_arguments = {
            'input_shape': input_shape, 'n_targets': n_targets,
            'activation': activation
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
        self._build_layers()
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

    @abstractmethod
    @onetimemethod
    def _validate_args(self):
        """Process the initialization arguments of the instance."""
        # Validate the model's input layer shape.
        check_type_validity(
            self.input_shape, (tuple, list, tf.TensorShape), 'input_shape'
        )
        if len(self.input_shape) == 1:
            raise TypeError("'input_shape' must be at least bi-dimensional.")
        # Validate the model's activation function.
        activation = self.activation
        if not (isinstance(activation, str) or inspect.isfunction(activation)):
            raise_type_error('activation', (str, 'function'), type(activation))
        # Validate the model's number of targets.
        check_positive_int(self.n_targets, 'n_targets')

    @onetimemethod
    def _build_placeholders(self):
        """Build the network's placeholders."""
        self._holders['input'] = tf.placeholder(tf.float32, self.input_shape)
        self._holders['targets'] = tf.placeholder(
            tf.float32, [self.input_shape[0], self.n_targets]
        )
        self._holders['keep_prob'] = tf.placeholder(tf.float32, ())

    @abstractmethod
    @onetimemethod
    def _build_layers(self):
        """Build the network's layers.

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
        weight, bias = config['values'][name]
        model.session.run(layer.weight.assign(weight))
        if bias is not None:
            model.session.run(layer.bias.assign(bias))
    # If the model was instanciated within this function, return it.
    return model if new_model else None
