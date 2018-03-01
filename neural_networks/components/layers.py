# coding: utf-8

"""Set of wrapper classes for neural network layers in tensorflow."""

import inspect

import tensorflow as tf
import numpy as np

from neural_networks.tf_utils import (
    conv2d, get_activation_function, get_activation_function_name
)


class NeuralLayer:
    """Generic wrapper class for neural network layers in tensorflow."""

    def __init__(
            self, input_data, activation, operation, weight_dim,
            bias=True, pooling=None
        ):
        """Initialize the neural layer.

        input_data : tensorflow variable or placeholder to use as input
        activation : activation function or activation function name
        operation  : function of input_data and units' weight whose result
                     to feed to the activation function
        weight_dim : list recording the dimensions of the weight matrix
        bias       : whether to add a bias constant to the transformed data
                     passed to the activation function (bool, default True)
        pooling    : optional pooling (or reshaping) function to apply
                     on top of this layer (function, default None)
        """
        # Meant to be wrapped by children. pylint: disable=too-many-arguments
        # Set up the activation function.
        if isinstance(activation, str):
            self.activation = get_activation_function(activation)
        elif inspect.isfunction(activation):
            self.activation = activation
        else:
            raise TypeError("'activation' should be a str or a function.")
        # Set up the input-weighting function.
        if not inspect.isfunction(operation):
            raise TypeError("'operation' should be a function.")
        self.operation = operation
        # Set up the optional pooling function.
        if pooling is not None and not inspect.isfunction(pooling):
            raise TypeError("'pooling' should be a function or None.")
        self.pooling = pooling
        # Set up the weight and bias terms of the layer's units.
        self.weight = tf.Variable(tf.truncated_normal(weight_dim, stddev=.1))
        self.bias = (
            tf.Variable(tf.constant(0.1, shape=weight_dim[-1:]))
            if bias else None
        )
        # Set up the operation yielding the layer's outputs.
        output = self.activation(
            self.operation(input_data, self.weight) if not bias
            else self.operation(input_data, self.weight) + self.bias
        )
        self.output = (
            output if self.pooling is None else self.pooling(output)
        )

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
            'dimensions': tuple(self.weight.shape.as_list()),
            'operation': get_name(self.operation),
            'pooling': (
                get_name(self.pooling) if self.pooling is not None else None
            )
        }

    def get_values(self, session):
        """Return the layer's weight and bias current values.

        session : a tensorflow.Session in the context of which
                  the evaluation is to be performed

        Return a tuple containing the weight matrix and the bias
        scalar (or None if the layer has no bias component).
        """
        if self.bias is None:
            return (session.run(self.weight), None)
        return session.run((self.weight, self.bias))

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
        session.run(tf.assign(self.weight, weights[0]))
        if self.bias is not None:
            session.run(tf.assign(self.bias, weights[1]))


class DenseLayer(NeuralLayer):
    """Dense layer of fully-connected units."""

    def __init__(
            self, input_data, n_units, activation, bias=True, pooling=None
        ):
        """Initialize the fully-connected neural layer.

        input_data : tensorflow variable or placeholder to use as input
        n_units    : number of units on the layer
        activation : activation function or activation function name
        bias       : whether to add a bias constant to the transformed data
                     passed to the activation function (bool, default True)
        pooling    : optional pooling (or reshaping) function to apply
                     on top of this layer (function, default None)
        """
        super().__init__(
            input_data, activation, operation=tf.matmul,
            weight_dim=[input_data.shape[1].value, n_units],
            bias=bias, pooling=pooling
        )


class ConvolutionalLayer(NeuralLayer):
    """Layer of 2-D convolutional units."""

    def __init__(
            self, input_data, units_shape, activation, bias=True, pooling=None
        ):
        """Initialize the 2-D convolutional neural layer.

        input_data  : tensorflow variable or placeholder to use as input
        units_shape : shape of the units' 4-D filtering matrix, defined
                      by the filter's height, its width, the number of
                      input channels and the number of features to output
                      `[filt_height, filt_width, in_chans, out_chans]`
        activation  : activation function or activation function name
        bias        : whether to add a bias constant to the transformed data
                      passed to the activation function (bool, default True)
        pooling     : optional pooling (or reshaping) function to apply
                      on top of this layer (function, default None)
        """
        super().__init__(
            input_data, activation, operation=conv2d,
            weight_dim=units_shape, bias=bias, pooling=pooling
        )
