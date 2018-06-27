# coding: utf-8

"""Abstract neural network class and dumped models loading function."""

from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import tensorflow as tf
import numpy as np

from ac2art.internal.data_utils import batch_to_sequences, sequences_to_batch
from ac2art.internal.network_bricks import (
    build_layers_stack, refine_signal, validate_layer_config
)
from ac2art.internal.neural_layers import AbstractRNN, DenseLayer, SignalFilter
from ac2art.utils import (
    check_positive_int, check_type_validity, instantiate, onetimemethod
)


class NeuralNetwork(metaclass=ABCMeta):
    """Abstract class to design neural networks using tensorflow.

    This class defines both an API and a building procedure for neural
    networks aimed at learning acoustic-to-articulatory inversion.
    """

    def __init__(
            self, input_shape, n_targets, layers_config, top_filter=None,
            use_dynamic=True, binary_tracks=None, norm_params=None, **kwargs
        ):
        """Initialize the neural network.

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
        """
        # Arguments serve modularity; pylint: disable=too-many-arguments
        # Record and process initialization arguments.
        self._init_arguments = {
            'input_shape': input_shape, 'n_targets': n_targets,
            'layers_config': layers_config, 'top_filter': top_filter,
            'use_dynamic': use_dynamic, 'binary_tracks': binary_tracks,
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
            raise AttributeError(
                "'%s' object has no attribute '%s'."
                % (self.__class__.__name__, name)
            )
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

    def get_weights(self, layer_name):
        """Return the tensor(s) of weights of a layer of given name."""
        layer = self.layers[layer_name]
        if isinstance(layer, SignalFilter):
            return layer.cutoff
        elif isinstance(layer, AbstractRNN) or layer.bias is None:
            return layer.weights
        return layer.weights

    @property
    def _neural_weights(self):
        """Return the weight and biases tensors of all neural layers."""
        return [
            self.get_weights(name) for name, layer in self.layers.items()
            if isinstance(layer, (AbstractRNN, DenseLayer))
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

    def _interlace(self, main_tensor, binary_tensor):
        """Interlace tensors associated with continuous and binary targets."""
        if self.binary_tracks is None:
            raise RuntimeError(
                "'_interlace' method was called, but "
                + "'self.binary_tracks' is None."
            )
        start = 0
        stacks = []
        for i, end in enumerate(self.binary_tracks):
            stacks.append(main_tensor[..., start:end])
            stacks.append(binary_tensor[..., i:i+1])
            start = end
        stacks.append(main_tensor[..., start:])
        return tf.concat(stacks, axis=-1)

    @onetimemethod
    def _validate_args(self):
        """Process the initialization arguments of the instance."""
        # Validate the model's input layer shape.
        check_type_validity(
            self.input_shape, (tuple, list, tf.TensorShape), 'input_shape'
        )
        if len(self.input_shape) not in [2, 3]:
            raise TypeError("'input_shape' must be of length 2 or 3.")
        if self.input_shape[-1] is None:
            raise ValueError("Last 'input_shape' dimension must be fixed.")
        # Validate the model's layers configuration.
        check_type_validity(self.layers_config, list, 'layers_config')
        for i, config in enumerate(self.layers_config):
            self.layers_config[i] = validate_layer_config(config)
        # Validate the model's optional top layer configuration.
        if self.top_filter is not None:
            self._init_arguments['top_filter'] = (
                validate_layer_config(self.top_filter)
            )
        # Validate the model's number of targets and their specification.
        check_positive_int(self.n_targets, 'n_targets')
        check_type_validity(self.use_dynamic, bool, 'use_dynamic')
        check_type_validity(
            self.binary_tracks, (list, type(None)), 'binary_tracks'
        )
        if self.binary_tracks is not None:
            if self.binary_tracks:
                invalid = not all(
                    isinstance(track, int) and 0 <= track < self.n_targets
                    for track in self.binary_tracks
                )
                if invalid:
                    raise TypeError(
                        "'binary_tracks' should be a list of int in [0, %s]"
                        % (self.n_targets - 1)
                    )
            else:
                self._init_arguments['binary_tracks'] = None
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
        n_targets = self.n_targets
        if self.use_dynamic:
            n_binary = len(self.binary_tracks) if self.binary_tracks else 0
            n_targets += 2 * (self.n_targets - n_binary)
        self.holders['targets'] = tf.placeholder(
            tf.float32, [*self.input_shape[:-1], n_targets]
        )
        self.holders['keep_prob'] = tf.placeholder(tf.float32, ())
        if len(self.input_shape) == 3:
            self.holders['batch_sizes'] = (
                tf.placeholder(tf.int32, [self.input_shape[0]])
            )

    @onetimemethod
    def _build_hidden_layers(self):
        """Build the network's hidden layers."""
        hidden_layers = build_layers_stack(
            self.holders['input'], self.layers_config,
            self.holders['keep_prob'], self.holders.get('batch_sizes'),
            check_config=False
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

        This method should add a 'raw_prediction' Tensor to the `readouts`
        dict attribute, plus a '_binary_prediction' one if `binary_tracks`
        attribute is not None.
        """
        return NotImplemented

    @onetimemethod
    def _build_refined_prediction(self):
        """Refine the network's initial prediction."""
        # Refine the raw prediction of continuous targets.
        prediction, top_filter = refine_signal(
            self.readouts['raw_prediction'], self.norm_params,
            self.top_filter, self.use_dynamic
        )
        # Optionally stack binary predictions with the continuous ones.
        if self.binary_tracks is not None:
            self.readouts['_continuous_prediction'] = prediction
            binary_prediction = self.readouts['_binary_prediction']
            self.readouts['raw_prediction'] = self._interlace(
                self.readouts['raw_prediction'], binary_prediction
            )
            prediction = self._interlace(prediction, binary_prediction)
        # Record the full consolidated prediction.
        self.readouts['prediction'] = prediction
        # If a top filter was built, store it in the layers stack.
        if top_filter is not None:
            n_filters = sum(
                name.startswith('top_filter') for name in self.layers.keys()
            )
            name = ('top_filter_%s' % n_filters) if n_filters else 'top_filter'
            self.layers[name] = top_filter

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

    def get_feed_dict(self, input_data, targets=None, keep_prob=1):
        """Build a tensorflow feeding dictionary out of provided arguments.

        input_data : data to feed to the network
        targets    : optional true targets associated with the inputs
        keep_prob  : dropout keep-probability to use (default 1)
        """
        # Build the basic feed dict.
        feed_dict = {
            self.holders['input']: input_data,
            self.holders['keep_prob']: keep_prob
        }
        # Alter data and update the feed dict when using batches of sequences.
        if len(self.input_shape) == 3:
            length = self.input_shape[1]
            input_data, batch_sizes = sequences_to_batch(input_data, length)
            if targets is not None:
                targets, _ = sequences_to_batch(targets, length)
            feed_dict.update({
                self.holders['input']: input_data,
                self.holders['batch_sizes']: batch_sizes
            })
        # Add the target data to the feed dict, if any.
        if targets is not None:
            feed_dict[self.holders['targets']] = targets
        # Return the defined feed dict.
        return feed_dict

    def run_training_function(self, input_data, targets, keep_prob=1):
        """Run a training step of the model.

        input_data : samples to feed to the network (2-D numpy.ndarray)
        targets    : target values associated with the input data
        keep_prob  : probability for each unit to have its outputs used in
                     the training procedure (float in [0., 1.], default 1.)
        """
        feed_dict = self.get_feed_dict(input_data, targets, keep_prob)
        self.session.run(self.training_function, feed_dict)

    def predict(self, input_data):
        """Predict the targets associated with a given set of inputs."""
        feed_dict = self.get_feed_dict(input_data)
        prediction = self.readouts['prediction'].eval(feed_dict, self.session)
        # De-batch the prediction, if relevant.
        if len(self.input_shape) == 3:
            prediction = batch_to_sequences(
                prediction, feed_dict[self.holders['batch_sizes']]
            )
        # Return the predicted sequence(s).
        return prediction

    @abstractmethod
    def score(self, input_data, targets):
        """Return the root mean square prediction error of the network.

        input_data : input data sample to evalute the model on which
        targets    : true targets associated with the input dataset
        """
        return NotImplemented


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
        raise KeyError(
            "Invalid model dump. Missing key(s): %s." % missing_keys
        )
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
        model, NeuralNetwork, 'rebuilt model' if new_model else 'model'
    )
    # Check that the model's architecture is coherent with the dump.
    if model.architecture != config['architecture']:
        raise TypeError("Invalid network architecture.")
    # Restore the model's weights.
    for name, layer in model.layers.items():
        layer.set_values(config['values'][name], model.session)
    # If the model was instantiated within this function, return it.
    return model if new_model else None
