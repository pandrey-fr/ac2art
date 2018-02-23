# coding: utf-8

"""Set of functions to filter signals in tensorflow."""

import tensorflow as tf
import numpy as np

from neural_networks.tf_utils import log_base, sinc, tensor_length
from neural_networks.utils import check_positive_int, check_type_validity


def get_lowpass_filter(cutoff, sampling_rate, window):
    """Return a Hamming-based low-frequency filtering window."""
    # Handle the cases when a unique (flat) cutoff frequency is provided.
    if isinstance(cutoff, (int, list)):
        cutoff = np.array(list(cutoff))
    elif isinstance(cutoff, tf.Tensor) and not len(cutoff.shape):
        cutoff = tf.expand_dims(cutoff, 1)
    # Compute and return the filter.
    nyq = tf.expand_dims(tf.cast(2 * cutoff / sampling_rate, tf.float32), -1)
    filt_supp = tf.range((1 - window) / 2, (window + 1) / 2)
    ideal = nyq * sinc(nyq * filt_supp)
    hamming = tf.contrib.signal.hamming_window(window)
    return hamming * ideal


def lowpass_filter(signal, cutoff, sampling_rate, window=5):
    """Low-pass filter a signal in tensorflow.

    signal        : signal to filter (1-D or 2-D tensorflow.Tensor)
    cutoff        : cutoff frequency (or frequencies) of the filter, in Hz
                    (positive int or list, array or Tensor of such values)
    sampling_rate : sampling rate of the signal, in Hz (positive int)
    window        : half-size of the filtering window (int, default 5)

    If a single cutoff frequency is provided but the signal
    is multivariate, it will be used for each component.
    """
    # Check signal and cutoff arguments. Adjust them if needed.
    signal, cutoff, one_dimensional = _check_signal_and_cutoff(signal, cutoff)
    # Check additional arguments' validity.
    check_positive_int(sampling_rate, 'sampling_rate')
    check_positive_int(window, 'window')
    window = 1 + 2 * window
    # Set up the low-pass filter and convolve it to the signal.
    filt = get_lowpass_filter(cutoff, sampling_rate, window)
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


def _check_signal_and_cutoff(signal, cutoff):
    """Control the signal and cutoff arguments of a filtering operation.

    Return both arguments, adjusted if needed, as well as a bool
    indicating whether the initial signal was one-dimensional.
    """
    # Check signal validity and shape.
    check_type_validity(signal, tf.Tensor, 'signal')
    one_dimensional = (len(signal.shape) == 1)
    if one_dimensional:
        signal = tf.expand_dims(signal, 1)
    elif len(signal.shape) > 2:
        raise ValueError("'signal' rank is not in [1, 2].")
    # Check cutoff argument type.
    check_type_validity(cutoff, (tf.Tensor, np.ndarray, list, int), 'cutoff')
    if not isinstance(cutoff, tf.Tensor):
        cutoff = tf.constant(cutoff)
    # Check cutoff tensor rank.
    if not len(cutoff.shape):
        cutoff = tf.expand_dims(cutoff, 0)
    elif len(cutoff.shape) != 1:
        raise TypeError("'cutoff' rank is not in [0, 1].")
    # Check number of cutoff frequencies.
    if cutoff.shape[0] == 1:
        cutoff = tf.concat(
            [cutoff for _ in range(signal.shape[1])], axis=0
        )
    elif cutoff.shape[0] != signal.shape[1]:
        raise TypeError("Invalid 'cutoff' shape: %s" % cutoff.shape)
    # Return the (adjusted) arguments and signal flatness boolean.
    return signal, cutoff, one_dimensional
