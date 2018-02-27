# coding: utf-8

"""Base class and functions to filter signals in tensorflow."""

from abc import ABCMeta, abstractmethod

import tensorflow as tf
import numpy as np

from neural_networks.utils import check_type_validity, onetimemethod


def filter_1d_signal(signal, filt):
    """Apply a filter to a one-dimensional signal.

    signal : signal to filter, of shape (signal length, n_channels)
             (or (signal length,) in case of a single channel)
    filt   : filter to apply, of shape (n_channels, filter width)
    """
    check_type_validity(signal, tf.Tensor, 'signal')
    check_type_validity(filt, tf.Tensor, 'filt')
    # Check the signal's shape and adjust it if needed.
    one_dimensional = (len(signal.shape) == 1)
    if one_dimensional:
        signal = tf.expand_dims(signal, 1)
    elif len(signal.shape) > 2:
        raise ValueError("'signal' rank is not in [1, 2].")
    # Convolve the signal and the filter.
    convolved = tf.nn.conv1d(
        tf.expand_dims(tf.transpose(signal), -1),
        tf.expand_dims(tf.transpose(filt), 1),
        stride=1, padding='SAME'
    )
    # Gather the results and return them.
    index = tf.expand_dims(tf.range(signal.shape[1], dtype=tf.int32), 1)
    filtered = tf.gather_nd(
        tf.transpose(convolved, [0, 2, 1]), tf.concat([index, index], axis=1)
    )
    return filtered[0] if one_dimensional else tf.transpose(filtered)


class SignalFilter(metaclass=ABCMeta):
    """Abstract class for 1D signal filters in tensorflow.

    Subclasses should override the `_build_filter` method, and add any
    necessary keyword arguments to the `__init__`, which may then be
    called as-is.
    """
    # More of a structure than a class. pylint: disable=too-few-public-methods

    def __init__(self, signal, cutoff, learnable=True, **kwargs):
        """Initialize the filter.

        signal    : signal to filter (1-D or 2-D tensorflow.Tensor)
        cutoff    : cutoff frequency (or frequencies) of the filter, in Hz
                    (positive float or list, array or Tensor of such values)
        learnable : whether the cutoff frequency may be adjusted, e.g. in
                    backpropagation procedures (bool, default True)

        Subclasses may pass on any filter-designing keyword arguments.
        """
        # Check signal validity and shape.
        check_type_validity(signal, tf.Tensor, 'signal')
        if len(signal.shape) not in [1, 2]:
            raise ValueError("'signal' rank is not in [1, 2].")
        self.n_channels = signal.shape[1] if len(signal.shape) == 2 else 1
        # Check and assign the cutoff frequencies of the filter.
        self.cutoff = None
        self._build_cutoff(cutoff)
        # Optionally set the cutoff frequencies to be learnable.
        check_type_validity(learnable, bool, 'learnable')
        self.learnable = learnable
        if self.learnable:
            self.cutoff = tf.Variable(self.cutoff)
        # Set up the actual filter.
        self.filter = None
        self._build_filter(**kwargs)
        # Compute the filter's output.
        self.output = filter_1d_signal(signal, self.filter)
        # Record the instance's configuration.
        self.configuration = {
            'class': self.__class__.__module__ + '.' + self.__class__.__name__,
            'learnable': self.learnable
        }
        self.configuration.update(kwargs)

    @onetimemethod
    def _build_cutoff(self, cutoff):
        """Assign a cutoff frequency container as the 'cutoff' attribute."""
        # Check cutoff argument type.
        check_type_validity(
            cutoff, (tf.Tensor, np.ndarray, list, int), 'cutoff'
        )
        if not isinstance(cutoff, tf.Tensor):
            cutoff = tf.constant(cutoff, dtype=tf.float32)
        if cutoff.dtype != tf.float32:
            cutoff = tf.cast(cutoff, tf.float32)
        # Check cutoff tensor rank.
        if not len(cutoff.shape):  # pylint: disable=len-as-condition
            cutoff = tf.expand_dims(cutoff, 0)
        elif len(cutoff.shape) != 1:
            raise TypeError("'cutoff' rank is not in [0, 1].")
        # Check number of cutoff frequencies.
        if cutoff.shape[0] == 1:
            cutoff = tf.concat(
                [cutoff for _ in range(self.n_channels)], axis=0
            )
        elif cutoff.shape[0] != self.n_channels:
            raise TypeError("Invalid 'cutoff' shape: %s" % cutoff.shape)
        # Assign the (adjusted) cutoff object as an attribute.
        self.cutoff = cutoff

    @abstractmethod
    @onetimemethod
    def _build_filter(self, **kwargs):
        """Docstring."""
        raise NotImplementedError("No '_build_filter' method defined.")

    def get_values(self, session):
        """Return the current cutoff value.

        session: a tensorflow.Session in the context of which
                 the evaluation is to be performed
        """
        return session.run(self.cutoff)
