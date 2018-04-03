# coding: utf-8

"""Abstract neural network class and model dump loading function."""

from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import os

import tensorflow as tf
import numpy as np

from neural_networks.core import build_layers_stack, validate_layer_config
from neural_networks.components.filters import SignalFilter
from neural_networks.components.layers import NeuralLayer
from neural_networks.components.rnn import AbstractRNN
from neural_networks.utils import (
    check_positive_int, check_type_validity, instantiate, onetimemethod
)


class DeepNeuralNetwork(metaclass=ABCMeta):
    """Abstract class for deep neural network models in tensorflow.

    This class defines both an API and a building procedure for neural
    networks aimed at solving supervised learning problems.

    {API_DOCSTRING}
    """

    def __init__(
            self, input_shape, n_targets, layers_config,
            top_filter=None, norm_params=None, **kwargs
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
        top_filter    : optional tuple specifying a SignalFilter to use
                        on top of the network's raw prediction
        norm_params   : optional normalization parameters of the targets
                        (np.ndarray)
        """
        # Record and process initialization arguments.
        self._init_arguments = {
            'input_shape': input_shape, 'n_targets': n_targets,
            'layers_config': layers_config, 'top_filter': top_filter,
            'norm_params': norm_params
        }
        self._init_arguments.update(kwargs)
        self._validate_args()
        # Declare common attributes to contain the network's structure.
        self.holders = {}
        self.layers = OrderedDict()
        self.readouts = {}
        self.training_function = None
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
            (name, layer.configuration) for name, layer in self.layers.items()
        ])

    def get_values(self):
        """Return the current values of the network's layers' parameters."""
        return {
            name: layer.get_values(self.session)
            for name, layer in self.layers.items()
        }

    @property
    def _neural_weights(self):
        """Return the weight and biases tensors of all neural layers."""
        return [
            layer.weights if isinstance(layer, AbstractRNN) else (
                (layer.weight, layer.bias) if layer.bias else layer.weight
            )
            for layer in self.layers.values()
            if isinstance(layer, (AbstractRNN, NeuralLayer))
        ]

    @property
    def _filter_cutoffs(self):
        """Return the cutoff tensors of all learnable filter layers."""
        return [
            layer.cutoff for layer in self.layers.values()
            if isinstance(layer, SignalFilter) and layer.learnable
        ]

    def _adjust_init_arguments_for_saving(self):
        """Adjust `_init_arguments` attribute before dumping the model.

        Return a tuple of two dict. The first is a copy of the keyword
        arguments used at initialization, containing only values which
        numpy.save can serialize. The second associates to non-serializable
        arguments' names a dict enabling their (recursive) reconstruction
        using the `neural_networks.utils.instantiate` function.
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
        return self.layers[list(self.layers.keys())[-1]]

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
            self.layers_config[i] = validate_layer_config(config)
        # Validate the model's optional top layer configuration.
        if self.top_filter is not None:
            self._init_arguments['top_filter'] = (
                validate_layer_config(self.top_filter)
            )
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
        self.holders['input'] = tf.placeholder(tf.float32, self.input_shape)
        self.holders['targets'] = tf.placeholder(
            tf.float32, [self.input_shape[0], self.n_targets]
        )
        self.holders['keep_prob'] = tf.placeholder(tf.float32, ())

    @onetimemethod
    def _build_hidden_layers(self):
        """Build the network's hidden layers."""
        hidden_layers = build_layers_stack(
            self.holders['input'], self.layers_config,
            self.holders['keep_prob'], check_config=False
        )
        self.layers.update(hidden_layers)

    @abstractmethod
    @onetimemethod
    def _build_readout_layer(self):
        """Build the network's readout layer.

        This method should add a 'readout_layer' element
        on top of the `layers` OrderedDict attribute.
        """
        return NotImplemented

    @onetimemethod
    def _build_readouts(self):
        """Build wrappers on top of the network's readout layer."""
        self._build_initial_prediction()
        self._build_refined_prediction()
        self._build_error_readouts()

    @abstractmethod
    @onetimemethod
    def _build_initial_prediction(self):
        """Build the network's initial prediction.

        This method should add a 'raw_prediction'
        Tensor to the `readouts` dict attribute.
        """
        return NotImplemented

    @onetimemethod
    def _build_refined_prediction(self):
        """Refine the network's initial prediction."""
        prediction = self.readouts['raw_prediction']
        # Optionally de-normalize the initial prediction.
        if self.norm_params is not None:
            prediction *= self.norm_params
        # Optionally filter the prediction.
        if self.top_filter is not None:
            self.layers['top_filter'] = list(
                build_layers_stack(prediction, [self.top_filter]).values()
            )[0]
            prediction = self.layers['top_filter'].output
        # Assign the refined prediction to the _readouts attribute.
        self.readouts['prediction'] = prediction

    @abstractmethod
    @onetimemethod
    def _build_error_readouts(self):
        """Build error readouts of the network's prediction.

        This method should assign any tensorflow.Tensor to
        the `readouts` dict atttribute necessary to define
        the network's training function.
        """
        return NotImplemented

    @abstractmethod
    @onetimemethod
    def _build_training_function(self):
        """Build the train step function of the network.

        This method should assign a tensorflow operation
        to the `training_function` attribute.
        """
        return NotImplemented

    def _get_feed_dict(self, input_data, targets=None, keep_prob=1):
        """Build a tensorflow feeding dictionary out of provided arguments.

        input_data : data to feed to the network
        targets    : optional true targets associated with the inputs
        keep_prob  : probability to use for the dropout layers (default 1)
        """
        feed_dict = {
            self.holders['input']: input_data,
            self.holders['keep_prob']: keep_prob
        }
        if targets is not None:
            feed_dict[self.holders['targets']] = targets
        return feed_dict

    def run_training_function(self, input_data, targets, keep_prob=1):
        """Run a training step of the model.

        input_data : samples to feed to the network (2-D numpy.ndarray)
        targets    : target values associated with the input data
        keep_prob  : probability for each unit to have its outputs used in
                     the training procedure (float in [0., 1.], default 1.)
        """
        feed_dict = self._get_feed_dict(input_data, targets, keep_prob)
        self.session.run(self.training_function, feed_dict)

    def predict(self, input_data):
        """Predict the targets associated with a given set of inputs."""
        feed_dict = self._get_feed_dict(input_data)
        return self.readouts['prediction'].eval(feed_dict, self.session)

    @abstractmethod
    def score(self, input_data, targets):
        """Return the root mean square prediction error of the network.

        input_data : input data sample to evalute the model on which
        targets    : true targets associated with the input dataset
        """
        return NotImplemented


# Load up the full DeepNeuralNetwork docstring.
DOC_PATH = os.path.join(os.path.dirname(__file__), 'dnn_api_doc.md')
with open(DOC_PATH, encoding='utf-8') as doc:
    API_DOCSTRING = doc.read().replace('\n', '\n    ')
    DeepNeuralNetwork.__doc__ = (
        DeepNeuralNetwork.__doc__.format(API_DOCSTRING=API_DOCSTRING)
    )


def load_dumped_model(filename, model=None):
    """Restore a neural network model from a .npy dump.

    filename : path to a .npy file containing a model's configuration
    model    : optional instantiated model whose weights to restore
               (default None, implying that a model is instantiated
               based on the dumped configuration and returned)
    """
    # Load the dumped model configuration and check its validity.
    config = np.load(filename).tolist()
    check_type_validity(config, dict, 'loaded configuration')
    missing_keys = [
        key for key in
        ('__init__', '__class__', '__rebuild_init__', 'architecture', 'values')
        if key not in config.keys()
    ]
    if missing_keys:
        raise KeyError("Invalid model dump. Missing key(s): %s." % missing_keys)
    # If no model was provided, instantiate one.
    new_model = model is None
    if new_model:
        model = instantiate(
            config['__class__'], config['__init__'], config['__rebuild_init__']
        )
        if 'session' not in config['__init__'].keys():
            model.reset_model()
    # Check that the provided or rebuild model is indeed a neural network.
    check_type_validity(
        model, DeepNeuralNetwork, 'rebuilt model' if new_model else 'model'
    )
    # Check that the model's architecture is coherent with the dump.
    if model.architecture != config['architecture']:
        raise TypeError("Invalid network architecture.")
    # Restore the model's weights.
    for name, layer in model.layers.items():
        layer.set_values(config['values'][name], model.session)
    # If the model was instantiated within this function, return it.
    return model if new_model else None
