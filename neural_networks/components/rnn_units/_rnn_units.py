# coding: utf-8

"""Module docstring."""

import tensorflow as tf
import numpy as np

from neural_networks.utils import check_positive_int



class AbstractRNN(metaclass=ABCMeta):
    """Docstring."""

    def __init__(self, input_data, layers_shape, cell_type):
        


class RecurrentNeuralNetwork:
    """Class wrapping Recurrent Neural Network units in tensorflow."""

    def __init__(self, input_data, layers_shape, cell_type):
        """Instanciate the recurrent neural network."""
        # Check cell type validity.
        if not issubclass(cell_type, tf.nn.rnn_cell.RNNCell):
            raise TypeError(
                "'cell_type' is not a tensorflow.nn.rnn_cell.RNNCell subclass."
            )
        self.cell_type = cell_type.__module__ + '.' + cell_type.__name__
        # Check layers shape validity.
        check_type_validity(layers_shape, (tuple, int), 'layers_shape')
        if isinstance(layers_shape, int):
            layers_shape = (layers_shape,)
        for value in layers_shape:
            check_positive_int(value, "layer's number of units")
        self.layers_shape = layers_shape
        # Build a wrapper for the network's cells.
        self.cells = tf.nn.rnn_cell.MultiRNNCell(
            [cell_type(n_units) for n_units in self.layers_shape]
        )
        # Build the network, thus filling the output and state attributes.
        if len(input_data.shape) == 2:
            input_data = tf.expand_dims(input_data, 0)
        output, state = tf.nn.dynamic_rnn(
            self.cells, input_data, dtype=tf.float32
        )
        self.output = output
        self.state = state
        # Assign the cells' weights to the weights attribute.
        weights = self.cells.weights
        self.weights = [
            (weights[i], weights[i + 1]) for i in range(0, len(weights), 2)
        ]

    @property
    def configuration(self):
        """Return a dict specifying the network's configuration."""
        return {
            'cell_type': self.cell_type,
            'class': self.__class__.__module__ + '.' + self.__class__.__name__
            'layers_shape': self.layers_shape
        }

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



class LSTMLayer(RecurrentLayer):
    """Docstring."""

    def __init__(self, input_data, n_units):
        super().__init__(input_data, n_units, tf.nn.rnn_cells.LSTMCell)


class GRULayer(RecurrentLayer):
    """Docstring."""

    def __init__(self, input_data, n_units):
        super().__init__(input_data, n_units, tf.nn.rnn_cells.GRUCell)
