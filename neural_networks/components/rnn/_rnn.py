# coding: utf-8

"""Class wrapping recurrent neural network layer stacks in tensorflow."""

from abc import ABCMeta, abstractmethod

import tensorflow as tf
import numpy as np

from neural_networks.tf_utils import (
    get_activation_function_name, get_rnn_cell_type_name,
    setup_activation_function, setup_rnn_cell_type
)
from neural_networks.utils import check_positive_int, check_type_validity


class AbstractRNN(metaclass=ABCMeta):
    """Abstract class defining an API for recurrent neural network stacks."""
    # Attributes serve clarity; pylint: disable=too-many-instance-attributes

    @abstractmethod
    def __init__(
            self, input_data, layers_shape, cell_type='lstm',
            activation='tanh', name='rnn', keep_prob=None
        ):
        """Instanciate the recurrent neural network.

        input_data   : input data of the network (tensorflow.Tensor,
                       either of shape [n_batches, max_time, input_size]
                       or [len_sequence, input_size]
        layers_shape : number of units per layer (int or tuple of int)
        cell_type    : type of recurrent cells to use (short name (str)
                       or tensorflow.nn.rnn_cell.RNNCell subclass,
                       default 'lstm', i.e. LSTMCell)
        activation   : activation function of the cell units (function
                       or function name, default 'tanh')
        name         : name of the stack (using the same name twice will
                       cause tensorflow to raise an exception)
        keep_prob    : optional Tensor recording a keep probability to use
                       as a dropout parameter

        This needs overriding by subclasses to actually build the network
        out of the pre-validated arguments. The `weights` argument should
        also be filled in by subclasses.
        """
        # Arguments serve modularity; pylint: disable=too-many-arguments
        # Check name validity.
        check_type_validity(name, str, 'name')
        self.name = name
        # Check input data valitidy.
        check_type_validity(input_data, tf.Tensor, 'input_data')
        if len(input_data.shape) not in [2, 3]:
            raise TypeError("Invalid 'input_data' rank: should be 2 or 3.")
        if len(input_data.shape) == 2:
            self.input_data = tf.expand_dims(input_data, 0)
            self._single_batch = True
        else:
            self.input_data = input_data
            self._single_batch = False
        # Check layers shape validity.
        check_type_validity(layers_shape, (tuple, int), 'layers_shape')
        if isinstance(layers_shape, int):
            layers_shape = (layers_shape,)
        for value in layers_shape:
            check_positive_int(value, "layer's number of units")
        self.layers_shape = layers_shape
        # Check keep_prob validity.
        check_type_validity(keep_prob, (tf.Tensor, type(None)), 'keep_prob')
        self.keep_prob = keep_prob
        # Set up the RNN cell type and activation function.
        self.cell_type = setup_rnn_cell_type(cell_type)
        self.activation = setup_activation_function(activation)
        # Set up an argument that needs assigning by subclasses.
        self.weights = None

    @property
    def configuration(self):
        """Return a dict specifying the network's configuration."""
        return {
            'activation': get_activation_function_name(self.activation),
            'cell_type': get_rnn_cell_type_name(self.cell_type),
            'class': self.__class__.__module__ + '.' + self.__class__.__name__,
            'layers_shape': self.layers_shape,
            'dropout': self.keep_prob is not None
        }

    @abstractmethod
    def get_values(self, session):
        """Return the network's cells' weights' current values."""
        raise NotImplementedError("No 'get_values' method defined.")

    @abstractmethod
    def set_values(self, weights, session):
        """Set the recurrent neural network's weights to given values."""
        raise NotImplementedError("No 'set_values' method defined.")


def assign_rnn_weights(container, weights, session):
    """Assign their values to a recurrent neural network's cells' weights.

    container : a list of tuples containing kernel and bias weights
                tensors whose values to update
    weights   : a list of tuples containing kernel and bias weights
                numpy.ndarray containing values to assign
    session   : a tensorflow.Session in the context of which
                the assignment is to be performed
    """
    # Check input weights' conformity.
    conform = (
        isinstance(weights, list)
        and len(weights) == len(container)
        and all(
            isinstance(weight, tuple) and len(weight) == 2
            and isinstance(weight[0], np.ndarray)
            and isinstance(weight[1], np.ndarray)
            for weight in weights
        )
    )
    if not conform:
        raise TypeError("Invalid 'weights' argument.")
    # Assign weights.
    index = 0
    for weight, bias in weights:
        session.run(tf.assign(container[index][0], weight))
        session.run(tf.assign(container[index][1], bias))
        index += 1


def build_cells_wrapper(cell_type, layers_shape, activation, keep_prob):
    """Build and return a tensorflow multi RNN cell wrapper.

    cell_type    : type of units (tensorflow.nn.rnn_cell.RNNCell subclass)
    layers_shape : sequence of int representing layers' number of units
    activation   : activation function of the units
    keep_prob    : keep_prob Tensor used to specify output dropout, or None
    """
    cells = [
        cell_type(n_units, activation=activation) for n_units in layers_shape
    ]
    if keep_prob is None:
        cells = [
            tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
            for cell in cells
        ]
    return tf.nn.rnn_cell.MultiRNNCell(cells)


class RecurrentNeuralNetwork(AbstractRNN):
    """Class wrapping Recurrent Neural Network stacks in tensorflow."""

    def __init__(
            self, input_data, layers_shape, cell_type='lstm',
            activation='tanh', name='rnn', keep_prob=None
        ):
        """Instanciate the recurrent neural network.

        input_data   : input data of the network (tensorflow.Tensor,
                       either of shape [n_batches, max_time, input_size]
                       or [len_sequence, input_size]
        layers_shape : number of units per layer (int or tuple of int)
        cell_type    : type of recurrent cells to use (short name (str)
                       or tensorflow.nn.rnn_cell.RNNCell subclass,
                       default 'lstm', i.e. LSTMCell)
        name         : name of the stack (using the same name twice will
                       cause tensorflow to raise an exception)
        activation   : activation function of the cell units (function
                       or function name, default 'tanh')
        """
        # Arguments serve modularity; pylint: disable=too-many-arguments
        super().__init__(
            input_data, layers_shape, cell_type, activation, name, keep_prob
        )
        # Build a wrapper for the network's cells.
        self.cells = build_cells_wrapper(
            self.cell_type, self.layers_shape, self.activation, self.keep_prob
        )
        # Build the recurrent unit.
        output, state = tf.nn.dynamic_rnn(
            self.cells, self.input_data, scope=self.name, dtype=tf.float32
        )
        self.output = output[0] if self._single_batch else output
        self.state = state[0] if self._single_batch else state
        # Assign the cells' weights to the weights attribute.
        weights = self.cells.weights
        self.weights = [
            (weights[i], weights[i + 1]) for i in range(0, len(weights), 2)
        ]

    def get_values(self, session):
        """Return the network's cells' kernel and bias weights' current values.

        session : a tensorflow.Session in the context of which
                  the evaluation is to be performed

        Return a list containing the tuple of kernel and bias weights
        associated to each cell of the network.
        """
        return session.run(self.weights)

    def set_values(self, weights, session):
        """Set the recurrent neural network's weights to given values.

        weights : a list of tuples containing kernel and bias weights
                  of the network's cells, each as a numpy.ndarray
        session : a tensorflow.Session in the context of which
                  the assignment is to be performed
        """
        assign_rnn_weights(self.weights, weights, session)


class BidirectionalRNN(AbstractRNN):
    """Class wrapping bidirectional RNN stacks in tensorflow."""
    # Attributes serve clarity; pylint: disable=too-many-instance-attributes

    def __init__(
            self, input_data, layers_shape, cell_type='lstm',
            activation='tanh', name='bi_rnn', aggregate='concatenate',
            keep_prob=None
        ):
        """Instanciate the bidirectional recurrent neural network.

        input_data   : input data of the network (tensorflow.Tensor,
                       either of shape [n_batches, max_time, input_size]
                       or [len_sequence, input_size]
        layers_shape : number of units per layer (int or tuple of int),
                       common to both forward and backward units
        cell_type    : type of recurrent cells to use (short name (str)
                       or tensorflow.nn.rnn_cell.RNNCell subclass,
                       default 'lstm', i.e. LSTMCell)
        activation   : activation function of the cell units (function
                       or function name, default 'tanh')
        name         : name of the stack (using the same name twice will
                       cause tensorflow to raise an exception)
        aggregate    : aggregation method for the network's outputs
                       (one of {'concatenate', 'mean', 'min', 'max'},
                       default 'concatenate')
        """
        # Arguments serve modularity; pylint: disable=too-many-arguments
        super().__init__(
            input_data, layers_shape, cell_type, activation, name, keep_prob
        )
        # Check aggregate argument validity.
        check_type_validity(aggregate, str, 'aggregate')
        if aggregate not in ['concatenate', 'mean', 'min', 'max']:
            raise TypeError("Unknown aggregation method: '%s'." % aggregate)
        self.aggregate = aggregate
        # Build forward and backward cells' wrappers.
        self.forward_cells = build_cells_wrapper(
            self.cell_type, self.layers_shape, self.activation, self.keep_prob
        )
        self.backward_cells = build_cells_wrapper(
            self.cell_type, self.layers_shape, self.activation, self.keep_prob
        )
        # Build the bidirectional recurrent unit.
        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            self.forward_cells, self.backward_cells, self.input_data,
            scope=self.name, dtype=tf.float32
        )
        # Unpack the network's outputs and aggregate them.
        fw_output = outputs[0][0] if self._single_batch else outputs[0]
        bw_output = outputs[1][0] if self._single_batch else outputs[1]
        if self.aggregate == 'concatenate':
            self.output = tf.concat([fw_output, bw_output], axis=-1)
        elif self.aggregate == 'mean':
            self.output = tf.add(fw_output, bw_output) / 2
        elif self.aggregate == 'min':
            self.output = tf.minimum(fw_output, bw_output)
        elif self.aggregate == 'max':
            self.output = tf.maximum(fw_output, bw_output)
        # Unpack the network's state outputs.
        self.states = (
            (states[0][0], states[1][0]) if self._single_batch else states
        )
        # Wrap the forward and backward cells' weights into attributes.
        def get_cells_weights(cells):
            """Return the weights of given cells as a list of tuples."""
            weights = cells.weights
            return [
                (weights[i], weights[i + 1]) for i in range(0, len(weights), 2)
            ]
        self._forward_weights = get_cells_weights(self.forward_cells)
        self._backward_weights = get_cells_weights(self.backward_cells)
        self.weights = [self._forward_weights, self._backward_weights]

    @property
    def configuration(self):
        """Return a dict specifying the network's configuration."""
        configuration = super().configuration
        configuration.update({'aggregate': self.aggregate})
        return configuration

    def get_values(self, session):
        """Return the network's cells' kernel and bias weights' current values.

        session : a tensorflow.Session in the context of which
                  the evaluation is to be performed

        Return a list containing the tuple of kernel and bias weights
        associated to each cell of the network.
        """
        return session.run(self.weights)

    def set_values(self, weights, session):
        """Set the recurrent neural network's weights to given values.

        weights : a tuple containing two lists of tuples containing kernel and
                  bias weights of the network's cells, each as a numpy.ndarray
        session : a tensorflow.Session in the context of which
                  the assignment is to be performed
        """
        # Unpack forward and backward weights.
        if not isinstance(weights, tuple) and len(weights) == 2:
            raise TypeError(
                "Invalid 'weights' argument: should be a two-elements tuple."
            )
        forward_weights, backward_weights = weights
        # Make a safe save of the current forward weights.
        current_forward = self.get_values(session)[0]
        # Assign the forward cells' weights.
        assign_rnn_weights(self._forward_weights, forward_weights, session)
        # Assign the backward cells' weights.
        try:
            assign_rnn_weights(
                self._backward_weights, backward_weights, session
            )
        # In case of error, restore the initial forward weights.
        except Exception as exception:  # pylint: disable=broad-except
            assign_rnn_weights(self._forward_weights, current_forward, session)
            raise exception
