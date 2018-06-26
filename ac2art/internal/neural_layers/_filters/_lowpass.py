# coding: utf-8

"""Class implementing learnable lowpass filters in tensorflow."""

import tensorflow as tf

from ac2art.internal.neural_layers._filters import SignalFilter
from ac2art.internal.tf_utils import sinc
from ac2art.utils import check_positive_int, onetimemethod


class LowpassFilter(SignalFilter):
    """Class implementing a 1d low-pass filter in tensorflow."""
    # More of a structure than a class; pylint: disable=too-few-public-methods

    def __init__(
            self, signal, cutoff, learnable=True, sampling_rate=200, window=5
        ):
        """Initialize the filter.

        signal        : signal to filter (tensorflow.Tensor of rank 1 to 3)
        cutoff        : cutoff frequency (or frequencies) of the filter, in Hz
                        (positive int or list, array or Tensor of such values)
        learnable     : whether the cutoff frequency may be adjusted, e.g. in
                        backpropagation procedures (bool, default True)
        sampling_rate : sampling rate of the signal, in Hz (int, default 200)
        window        : half-size of the filtering window (int, default 5)

        Note: three-dimensional signals are treated as a batch of 2-D
              signals stacked along the first dimension, and are filtered
              as such, i.e. independently.
        """
        check_positive_int(sampling_rate, 'sampling_rate')
        check_positive_int(window, 'window')
        super().__init__(
            signal, cutoff, learnable,
            sampling_rate=sampling_rate, window=window
        )

    @onetimemethod
    def _build_filter(self, sampling_rate, window):
        """Build the low-pass filter, based on a Hamming window."""
        # Explicit class-specific arguments, pylint: disable=arguments-differ
        # Turn parameters into tensors.
        sampling_rate = tf.constant(float(sampling_rate))
        window = tf.constant(window)
        # Compute the ideal filter.
        nyq = tf.expand_dims(2 * self.cutoff / sampling_rate, -1)
        filt_supp = tf.cast(tf.range(- window, window + 1), tf.float32)
        ideal = nyq * sinc(nyq * filt_supp)
        # Generate a Hamming window and return the low-pass filter.
        # False positive; pylint: disable=no-member
        hamming = tf.contrib.signal.hamming_window(2 * window + 1)
        self.filter = hamming * ideal
