# coding: utf-8

"""Module docstring."""

from abc import ABCMeta, abstractmethod

import tensorflow as tf
import numpy as np

from neural_networks.utils import check_positive_int, check_type_validity


class AbstractRNN(metaclass=ABCMeta):
    """Abstract class defining an API for recurrent neural network stacks."""

    @abstractmethod
    def __init__(self, input_data, layers_shape, cell_type):
        """Instanciate the recurrent neural network.

        input_data   : input data of the network (tensorflow.Tensor,
                       either of shape [n_batches, max_time, input_size]
                       or [len_sequence, input_size]
        layers_shape : number of units per layer (int or tuple of int)
        cell_type    : type of recurrent cells to use
                       (tf.nn.rnn_cell.RNNCell subclass)

        This needs overriding by subclasses to actually build the network
        out of the pre-validated arguments.
        """
        # Check input data valitidy.
        check_type_validity(input_data, tf.Tensor, 'input_data')
        if len(input_data.shape) not in [2, 3]:
            raise TypeError("Invalid 'input_data' rank: should be 2 or 3.")
        if len(input_data.shape) == 2:
            self.input_data = tf.expand_dims(input_data, 0)
            self.single_batch = True
        else:
            self.input_data = input_data
            self.single_batch = False
        # Check layers shape validity.
        check_type_validity(layers_shape, (tuple, int), 'layers_shape')
        if isinstance(layers_shape, int):
            layers_shape = (layers_shape,)
        for value in layers_shape:
            check_positive_int(value, "layer's number of units")
        self.layers_shape = layers_shape
        # Check cell type validity.
        if not issubclass(cell_type, tf.nn.rnn_cell.RNNCell):
            raise TypeError(
                "'cell_type' is not a tensorflow.nn.rnn_cell.RNNCell subclass."
            )
        self.cell_type = cell_type.__module__ + '.' + cell_type.__name__

    @property
    def configuration(self):
        """Return a dict specifying the network's configuration."""
        return {
            'cell_type': self.cell_type,
            'class': self.__class__.__module__ + '.' + self.__class__.__name__,
            'layers_shape': self.layers_shape
        }

    @abstractmethod
    def get_values(self, session):
        """Return the network's cells' weights' current values."""
        raise NotImplementedError("No 'get_values' method defined.")

    @abstractmethod
    def set_values(self, weights, session):
        """Set the recurrent neural network's weights to given values."""
        raise NotImplementedError("No 'set_values' method defined.")


class RecurrentNeuralNetwork(AbstractRNN):
    """Class wrapping Recurrent Neural Network units in tensorflow."""

    def __init__(self, input_data, layers_shape, cell_type):
        """Instanciate the recurrent neural network.

        input_data   : input data of the network (tensorflow.Tensor,
                       either of shape [n_batches, max_time, input_size]
                       or [len_sequence, input_size]
        layers_shape : number of units per layer (int or tuple of int)
        cell_type    : type of recurrent cells to use
                       (tf.nn.rnn_cell.RNNCell subclass)
        """
        super().__init__(input_data, layers_shape, cell_type)
        # Build a wrapper for the network's cells.
        self.cells = tf.nn.rnn_cell.MultiRNNCell(
            [cell_type(n_units) for n_units in self.layers_shape]
        )
        # Build the recurrent unit.
        output, state = tf.nn.dynamic_rnn(
            self.cells, self.input_data, dtype=tf.float32
        )
        self.output = output[0] if self.single_batch else output
        self.state = state[0] if self.single_batch else state
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
        # Check input weights' conformity.
        conform = (
            isinstance(weights, list)
            and len(weights) != len(self.weights)
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
            session.run(tf.assign(self.weights[index][0], weight))
            session.run(tf.assign(self.weights[index][1], bias))
            index += 1


class BidirectionalRNN(AbstractRNN):
    """Class wrapping bidirectional RNN units in tensorflow."""

    def __init__(self, input_data, layers_shape, cell_type):
        """Instanciate the bidirectional recurrent neural network.

        input_data   : input data of the network (tensorflow.Tensor,
                       either of shape [n_batches, max_time, input_size]
                       or [len_sequence, input_size]
        layers_shape : number of units per layer (int or tuple of int),
                       common to both forward and backward units
        cell_type    : type of recurrent cells to use
                       (tf.nn.rnn_cell.RNNCell subclass)
        """
        super().__init__(input_data, layers_shape, cell_type)
        # Build forward and backward cells' wrappers.
        def build_cells():
            """Build a multi RNN cell wrapper adjusted to this instance."""
            return tf.nn.rnn_cell.MultiRNNCell([
                cell_type(n_units) for n_units in self.layers_shape
            ])
        self.forward_cells = build_cells()
        self.backward_cells = build_cells()
        # Build the bidirectional recurrent unit.
        output, states = tf.nn.bidirectional_dynamic_rnn(
            self.forward_cells, self.backward_cells, self.input_data,
            dtype=tf.float32
        )
        self.output = output  # TODO: implement output reshaping options
        fw_state, bw_state = states
        self.forward_state = fw_state[0] if self.single_batch else fw_state
        self.backward_state = bw_state[0] if self.single_batch else bw_state
        # Wrap the forward and backward cells' weights into attributes.
        def get_cells_weights(cells):
            """Return the weights of given cells as a list of tuples."""
            weights = cells.weights
            return [
                (weights[i], weights[i + 1]) for i in range(0, len(weights), 2)
            ]
        self.forward_weights = get_cells_weights(self.forward_cells)
        self.backward_weights = get_cells_weights(self.backward_cells)

    def get_values(self, session):
        """Return the network's cells' kernel and bias weights' current values.

        session : a tensorflow.Session in the context of which
                  the evaluation is to be performed

        Return a list containing the tuple of kernel and bias weights
        associated to each cell of the network.
        """
        return session.run([self.forward_weights, self.backward_weights])

    def set_values(self, weights, session):
        """Set the recurrent neural network's weights to given values.

        weights : a list of tuples containing kernel and bias weights
                  of the network's cells, each as a numpy.ndarray
        session : a tensorflow.Session in the context of which
                  the assignment is to be performed
        """
        # FIXME: adjust this implementation.
        raise NotImplementedError('Unfinished method.')
        # Check input weights' conformity.
        conform = (
            isinstance(weights, list)
            and len(weights) != len(self.weights)
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
            session.run(tf.assign(self.weights[index][0], weight))
            session.run(tf.assign(self.weights[index][1], bias))
            index += 1
