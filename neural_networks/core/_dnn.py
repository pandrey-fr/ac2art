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


class DeepNeuralNetwork(metaclass=ABCMeta):
    """Abstract class for deep neural network models in tensorflow.

    This class defines both an API and a building procedure for neural
    networks aimed at solving supervised learning problems.

    [Public API]

    * Initialization and layers configuration (Public API, 1/3)

    - An `__init__` method allows the user to fully configure the
      network's architecture and specificities, noticeably through
      the `layers_config` argument presented below. The `__init__`
      is further discussed in the "Network building" section.

    - The `layers_config` argument used at the initialization step
      fully specifies the network's hidden layers stack, and should
      do so in any subclass. Subclasses should in turn specify the
      readout layer, the algorithm generating a prediction out of
      it and the training function(s) used to fit the model.

    - The structure of `layers_config` is rather straight-forward: it
      consists of a list of tuples, each of which specifies a layer of
      the network, ordered from input to readout and stacked on top of
      each other. These layers may either be an actual neural layer, a
      stack of RNN layers or a signal filtering process. Each layer is
      specified as a tuple containing the layer's class (or a keyword
      designating it), its number of units (or cutoff frequency, for
      signal filters) and an optional dict of keyword arguments used
      to instanciate the layer.


    * Training, predicting and scoring methods (Public API, 2/3)

    - The `run_training_function` should be used to train the model.
      It requires both some input data and the associated targets to
      run. Additionally, the `keep_prob` argument may be set to any
      float between 0 and 1 to use dropout when training the layers.
      Note that by default, all dense layers are set to be affected
      by this dropout, with a shared probability parameter ; this may
      be changed by explicitely setting 'keep_prob' to None in these
      layers' keyword arguments dict in `layers_config` at `__init__`.

    - The `predict` method requires only input data and returns the
      network's prediction as a numpy.array.

    - The `score` method returns a subclass-specific evaluation metric
      of the model's outputs based on some input data and the target
      values associated with it.


    * Saving, restoring and resetting the model (Public API, 3/3)

    - The `save_model` method allows to save the network's weights as
      well as its full specification to a simple .npy file. The stored
      values may also be accessed through the `architecture` attribute
      and the `get_values` method.

    - The `restore_model` method allows to restore and instanciated
      model's weights from a .npy dump. More generally, the function
      `load_dumped_model` may be used to fully instanciate a dumped
      model.

    - The `reset_model` method may be used at any moment to reset the
      model's weights to their initial (randomized) value.


    [Network building]

    Apart from desining some common arguments, the `__init__` method
    includes both an arguments-processing procedure which enables its
    call by any subclass (bypassing intermediary subclasses if needed)
    and a network building procedure. The latter is made of multiple
    private methods called successively and protected against being
    called more than once. This section aims at presenting the design
    of these hidden methods, some of which are meant to be overriden
    by subclasses.

    * Setting up the network's basics and hidden stack (Network building, 1/2)

    - The `_validate_args` method is first run to ensure that all
      arguments provided to instanciate a network are of expected
      type and/or values. Subclasses should override this method
      to validate any non-basic `__init__` parameter they introduce.

    - The `_build_placeholders` method is then run to assign tensorflow
      placeholders to the dict attribute `_holders`. Those are used to
      pass on input and target data, but also to specify parameters
      such as dropout. Subclasses may have to override this, either
      to introduce additional placeholders or alter the shape of the
      basic ones.

    - The `_build_hidden_layers` method comes next, and is run to
      instanciate forward layers, recurrent units' stacks and signal
      filters following the architecture specified through the
      `layers_config` attribute. This method should not need overriding
      by any subclass, as it is a general way to build up hidden layers
      sequentially, handling some technicalities such as setting dropout
      (unless explicitely told not to) or assigning unique names to
      rnn stacks in order to avoid tensorflow scope issues. The hidden
      layers are stored in the `_layers` OrderedDict attribute.


    * From readouts to training - abstract methods (Network building, 2/2)

    - The `_build_readout_layer` is an abstract method that needs
      implementing by subclasses. It should design the network's final
      hidden layer (typically using an identity activation function),
      whose purpose is to produce an output of proper dimension to be
      then used to derive a prediction, or any kind of metric useful
      to train the network.

    - The `_build_readouts` method is run after the previous, and is
      basically a caller of other hidden methods used sequentially
      to fill the `_readouts` dict attribute with tensors useful to
      train and/or evaluate the network's performances. This method
      may be overriden by subclasses which may need to add up steps
      (i.e. additional methods) to this end. In its basic deifinition,
      this method calls, in that order, the following methods:
        1. `build_initial_prediction`, an abstract method which should
        assign a tensor to the `_readouts` attribute under the
        'raw_prediction' key.

        2. `build_refined_prediction`, an implemented method which aims
        at improving the raw prediction, through optional steps of
        de-normalization and signal filtering (smoothing).

        3. `build_error_readouts`, an abstract method which should assign
        to the `_readouts` attribute any tensor necessary to building
        the training function.

    - Finally, the `_build_training_function` is run. This abstract
      method should build one or more tensorflow operations that
      need running so as to update the network's weights (or signal
      cutoff frequencies) and assign them (e.g. as a list) to the
      `_training_function` attribute.


    [Network training and scoring]

    - `run_training_function`
    - `predict`
    - `score`
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
    def _neural_weights(self):
        """Return the weight and biases tensors of all neural layers."""
        return [
            layer.weights if isinstance(layer, AbstractRNN) else (
                (layer.weight, layer.bias) if layer.bias else layer.weight
            )
            for layer in self._layers.values()
            if isinstance(layer, (AbstractRNN, NeuralLayer))
        ]

    @property
    def _filter_cutoffs(self):
        """Return the cutoff tensors of all learnable filter layers."""
        return [
            layer.cutoff for layer in self._layers.values()
            if isinstance(layer, SignalFilter) and layer.learnable
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
            # Get the layer's class and update the layers counter.
            layer_class = get_layer_class(name)
            layer_name = kwargs.pop(
                'name', name + '_%s' % layers_count.setdefault(name, 0)
            )
            # Handle dropout and avoid RNN scope issues, if relevant.
            if issubclass(layer_class, (DenseLayer, AbstractRNN)):
                kwargs = kwargs.copy()
                if issubclass(layer_class, AbstractRNN):
                    kwargs['name'] = layer_name + '_%s' % id(self)
                kwargs.setdefault('keep_prob', self._holders['keep_prob'])
            # Instanciate the layer.
            layer = layer_class(input_tensor, n_units, **kwargs)
            # Add the layer to the stack and use its output as next input.
            self._layers[layer_name] = layer
            layers_count[name] += 1
            input_tensor = layer.output

    @abstractmethod
    @onetimemethod
    def _build_readout_layer(self):
        """Build the network's readout layer.

        This method should add a 'readout_layer' element
        on top of the `_layers` OrderedDict attribute.
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
        Tensor to the `_readouts` dict attribute.
        """
        return NotImplemented

    @onetimemethod
    def _build_refined_prediction(self):
        """Refine the network's initial prediction."""
        prediction = self._readouts['raw_prediction']
        # Optionally de-normalize the initial prediction.
        if self.norm_params is not None:
            prediction *= self.norm_params
        # Optionally filter the prediction.
        if self.top_filter is not None:
            filter_class, cutoff, kwargs = self.top_filter
            filter_class = get_layer_class(filter_class)
            self._layers['top_filter'] = (
                filter_class(prediction, cutoff, **kwargs)
            )
            prediction = self._layers['top_filter'].output
        # Assign the refined prediction to the _readouts attribute.
        self._readouts['prediction'] = prediction

    @abstractmethod
    @onetimemethod
    def _build_error_readouts(self):
        """Build error readouts of the network's prediction.

        This method should assign any tensorflow.Tensor to
        the `_readouts` dict atttribute necessary to define
        the network's training function.
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

        input_data : samples to feed to the network (2-D numpy.ndarray)
        targets    : target values associated with the input data
        keep_prob  : probability for each unit to have its outputs used in
                     the training procedure (float in [0., 1.], default 1.)
        """
        feed_dict = self._get_feed_dict(input_data, targets, keep_prob)
        self.session.run(self._training_function, feed_dict)

    def predict(self, input_data):
        """Predict the targets associated with a given set of inputs."""
        feed_dict = self._get_feed_dict(input_data)
        return self._readouts['prediction'].eval(feed_dict, self.session)

    @abstractmethod
    def score(self, input_data, targets):
        """Return the root mean square prediction error of the network.

        input_data : input data sample to evalute the model on which
        targets    : true targets associated with the input dataset
        """
        return NotImplemented


def get_layer_class(layer_class):
    """Validate and return a layer class.

    layer_class : either a subclass from AbstractRNN, NeuralLayer or
                  SignalFilter, or the (short) name of one such class
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


def validate_layer_config(config):
    """Validate that a given object is fit as a layer's configuration.

    Return the validated object, which may have been extended with an
    empty dict.
    """
    # Check that the configuration is a three-elements tuple.
    if not isinstance(config, tuple):
        raise TypeError("Layer configuration elements should be tuples.")
    if len(config) == 2:
        config = (*config, {})
    elif len(config) != 3:
        raise TypeError(
            "Wrong layer configuration tuple length: should be 2 or 3."
        )
    # Check sub-elements types.
    check_type_validity(config[0], (str, type), 'layer class')
    check_type_validity(
        config[1], (int, list, tuple), 'layer primary parameter'
    )
    check_type_validity(config[2], dict, 'layer config kwargs')
    # Return the config tuple.
    return config


def load_dumped_model(filename, model=None):
    """Restore a neural network model from a .npy dump.

    filename : path to a .npy file containing a model's configuration
    model    : optional instanciated model whose weights to restore
               (default None, implying that a model is instanciated
               based on the dumped configuration and returned)
    """
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
    # Restore the model's weights. pylint: disable=protected-access
    for name, layer in model._layers.items():
        layer.set_values(config['values'][name], model.session)
    # If the model was instanciated within this function, return it.
    return model if new_model else None
