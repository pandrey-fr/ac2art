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

"""Class implementing fully-connected feed-forward layers in tensorflow."""

import tensorflow as tf
import numpy as np

from ac2art.internal.tf_utils import (
    get_activation_function_name, setup_activation_function,
    run_along_first_dim
)
from ac2art.utils import check_positive_int, check_type_validity


class DenseLayer:
    """Class implementing layers of fully-connected units."""

    def __init__(
            self, input_data, n_units, activation='relu', bias=True,
            name='DenseLayer', keep_prob=None
        ):
        """Initialize the fully-connected neural layer.

        input_data : tensorflow variable or placeholder to use as input
        n_units    : number of units of the layer
        activation : activation function or activation function name
                     (default 'relu', i.e. `tensorflow.nn.relu`)
        bias       : whether to add a bias constant to the transformed data
                     passed to the activation function (bool, default True)
        name       : optional name to give to the layer's inner operations
        keep_prob  : optional Tensor recording a keep probability to use
                     as a dropout parameter
        """
        # Arguments serve modularity; pylint: disable=too-many-arguments
        # Check the arguments' validity.
        check_type_validity(input_data, tf.Tensor, 'input_data')
        if len(input_data.shape) not in (2, 3):
            raise TypeError("`input_data` must be of rank 2 or 3.")
        check_positive_int(n_units, 'n_units')
        check_type_validity(bias, bool, 'bias')
        check_type_validity(name, str, 'name')
        check_type_validity(keep_prob, (tf.Tensor, type(None)), 'keep_prob')
        # Set up the layer's activation function.
        self.activation = setup_activation_function(activation)
        # Set up the layer's weights, adjusting their initial value.
        # note: design loosely based on Glorot, X. & Bengio, Y. (2010)
        if self.activation is tf.nn.relu:
            stddev = np.sqrt(2 / n_units)
        elif self.activation in [tf.nn.tanh, tf.nn.softmax]:
            stddev = np.sqrt(3 / n_units)
        else:
            stddev = .1
        initial = tf.truncated_normal(
            [input_data.shape[-1].value, n_units], mean=0, stddev=stddev
        )
        self.weights = tf.Variable(initial, name=name + '_weight')
        # Optionally set up a learnable bias term.
        if bias:
            self.bias = tf.Variable(
                tf.constant(0.1, shape=[n_units]), name=name + '_bias'
            )
        else:
            self.bias = None
        # Build the layer's processing of the inputs.
        if len(input_data.shape) == 2:
            self.output = self._feed_tensor(input_data)
        else:
            self.output = run_along_first_dim(self._feed_tensor, input_data)
        # Optionally set up dropout on top of the layer.
        self.keep_prob = keep_prob
        if self.keep_prob is not None:
            self.output = tf.nn.dropout(self.output, keep_prob=keep_prob)

    def _feed_tensor(self, tensor):
        """Compute the layer's output for a 2-D input tensor."""
        values = tf.matmul(tensor, self.weights)
        if self.bias is not None:
            values += self.bias
        return self.activation(values)

    @property
    def configuration(self):
        """Dict describing the layer's configuration."""
        def get_name(function):
            """Get a function's name, including its parent module name."""
            return (
                getattr(function, '__module__', '') + '.'
                + getattr(function, '__name__', '')
            ).strip('.')
        return {
            'activation': get_activation_function_name(self.activation),
            'bias': self.bias is not None,
            'class': get_name(self.__class__),
            'dimensions': tuple(self.weights.shape.as_list()),
            'dropout': self.keep_prob is not None
        }

    def initialize(self, session):
        """Initialize the layer in the context of a given session."""
        if self.bias is not None:
            session.run(self.bias.initializer)
        session.run(self.weights.initializer)

    def get_values(self, session):
        """Return the layer's weight and bias current values.

        session : a tensorflow.Session in the context of which
                  the evaluation is to be performed

        Return a tuple containing the weight matrix and the bias
        scalar (or None if the layer has no bias component).
        """
        if self.bias is None:
            return (session.run(self.weights), None)
        return session.run((self.weights, self.bias))

    def set_values(self, weights, session):
        """Set the layer's weight and bias to given values.

        weights : a tuple containing the values to assign as
                  numpy.ndarray (or None if there is no bias)
        session : a tensorflow.Session in the context of which
                  the assignment is to be performed
        """
        conform = (
            isinstance(weights, tuple) and len(weights) == 2
            and isinstance(weights[0], np.ndarray)
            and (
                weights[1] is None if self.bias is None
                else isinstance(weights[1], np.ndarray)
            )
        )
        if not conform:
            raise TypeError("Invalid 'weights' argument.")
        session.run(tf.assign(self.weights, weights[0]))
        if self.bias is not None:
            session.run(tf.assign(self.bias, weights[1]))
