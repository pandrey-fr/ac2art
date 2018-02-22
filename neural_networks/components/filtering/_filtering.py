# coding: utf-8

"""Set of functions to filter signals in tensorflow."""

import tensorflow as tf

from neural_networks.tf_utils import log_base, sinc, tensor_length
from neural_networks.utils import check_positive_int, check_type_validity


def get_lowpass_filter(cutoff, sampling_rate, window):
    """Return a Hamming-based low-frequency filtering window."""
    nyq = 2 * cutoff / sampling_rate
    filt_supp = tf.range((1 - window) / 2, (window + 1) / 2)
    ideal = nyq * sinc(nyq * filt_supp)
    hamming = tf.contrib.signal.hamming_window(window)
    return hamming * ideal


def lowpass_filter(signal, cutoff, sampling_rate, window=5):
    """Low-pass filter a signal in tensorflow.

    signal        : signal to filter (1-D or 2-D tensorflow.Tensor)
    cutoff        : cutoff frequency of the filter, in Hz (positive int)
    sampling_rate : sampling rate of the signal, in Hz (positive int)
    window        : half-size of the filtering window (int, default 5)
    """
    # Check arguments validity.
    check_type_validity(signal, tf.Tensor, 'signal')
    rank = len(signal.shape)
    if rank == 1:
        one_dimensional = True
        signal = tf.expand_dims(signal, 1)
    elif rank == 2:
        one_dimensional = False
    else:
        raise ValueError(
            "'signal' rank is not in [1, 2]: rank %s." % signal.get_shape()
        )
    check_positive_int(cutoff, 'cutoff')
    check_positive_int(sampling_rate, 'sampling_rate')
    check_positive_int(window, 'window')
    window = 1 + 2 * window
    # Set up the low-pass filter.
    filt = tf.reshape(
        get_lowpass_filter(cutoff, sampling_rate, window), (-1, 1, 1)
    )
    # Reshape the signal and convolve it with the filter.
    signal = tf.expand_dims(tf.transpose(signal), -1)
    filtered = tf.nn.conv1d(signal, filt, stride=1, padding='SAME')
    # Reshape the filtered signal and return it.
    return tf.transpose(tf.squeeze(filtered))
