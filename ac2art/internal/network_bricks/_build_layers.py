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

"""Set of functions to build neural networks' layers stacks."""

import time
from collections import OrderedDict


from ac2art.internal.neural_layers import (
    DenseLayer, LowpassFilter, SignalFilter,
    AbstractRNN, RecurrentNeuralNetwork, BidirectionalRNN
)
from ac2art.utils import check_type_validity, get_object


LAYER_CLASSES = {
    'dense_layer': DenseLayer, 'rnn_stack': RecurrentNeuralNetwork,
    'bi_rnn_stack': BidirectionalRNN, 'lowpass_filter': LowpassFilter
}


def build_layers_stack(
        input_tensor, layers_config, keep_prob=None,
        batch_sizes=None, check_config=True
    ):
    """Build a stack of neural layers, rnn substacks and signal filters.

    input_tensor  : tensorflow.Tensor fed to the first layer of the stack
    layers_config : list of tuples specifying the stack's layers ; each
                    tuple should contain a layer type (or its shortname),
                    a primary argument (number of units, or cutoff frequency
                    for signal filters) and an optional dict of keyword
                    arguments used to instantiate the layer
    keep_prob     : optional tensor specifying dropout keep probability
    batch_sizes   : optional tensor specifying true sequences lengths
                    when using batches of data sequences
    check_config  : whether to check `layers_config` to be valid
                    (bool, default True)
    """
    # Optionally check the layers_config argument's validity.
    if check_config:
        check_type_validity(layers_config, list, 'layers_config')
        for i, config in enumerate(layers_config):
            layers_config[i] = validate_layer_config(config)
    # Build the layers' stack container and a type-wise layers counter.
    layers_stack = OrderedDict([])
    layers_counter = {}
    # Iteratively build the layers.
    for name, n_units, kwargs in layers_config:
        # Get the layer's class and give the layer a name.
        layer_class = get_layer_class(name)
        layer_name = kwargs.pop(
            'name', name + '_%s' % layers_counter.setdefault(name, 0)
        )
        # Handle dropout and naming, if relevant.
        if issubclass(layer_class, (DenseLayer, AbstractRNN)):
            kwargs = kwargs.copy()
            kwargs.setdefault('keep_prob', keep_prob)
            kwargs['name'] = layer_name
            # Avoid RNN scope issues. Feed batch sizes, if any.
            if issubclass(layer_class, AbstractRNN):
                kwargs['name'] += '_%s' % int(time.time())
                kwargs['batch_sizes'] = batch_sizes
        # Instantiate the layer.
        layer = layer_class(input_tensor, n_units, **kwargs)
        # Add the layer to the stack and use its output as next input.
        layers_stack[layer_name] = layer
        layers_counter[name] += 1
        input_tensor = layer.output
    # Return the layers stack.
    return layers_stack


def get_layer_class(layer_class):
    """Validate and return a layer class.

    layer_class : either a subclass from AbstractRNN, DenseLayer or
                  SignalFilter, or the (short) name of one such class
    """
    if isinstance(layer_class, str):
        return get_object(layer_class, LAYER_CLASSES, 'layer class')
    if issubclass(layer_class, (AbstractRNN, DenseLayer, SignalFilter)):
        return layer_class
    raise TypeError("'layer_class' should be a str or an adequate class.")


def validate_layer_config(config):
    """Validate that a given object is fit as a layer's configuration.

    Return the validated input, which may be extended with an empty dict.
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
