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

"""Set of functions to enhance acoustic and articulatory data."""


import numpy as np
import scipy.signal
import scipy.interpolate

from ac2art.utils import check_positive_int, check_type_validity


def add_dynamic_features(static_features, window=5):
    """Build delta and deltadelta features on top of static ones.

    static_features : 2-D numpy.ndarray of static features to enhance
    window          : half-size of the time window used (int, default 5)
    """
    delta = get_delta_features(static_features, window)
    deltadelta = get_delta_features(delta, window)
    return np.concatenate([static_features, delta, deltadelta], axis=1)


def get_delta_features(array, window=5):
    """Compute and return delta features, using a given time window.

    array  : 2-D numpy.ndarray of values whose delta to compute
    window : half-size of the time window used (int, default 5)
    """
    norm = 2 * np.sum(i ** 2 for i in range(1, window + 1))
    return np.sum(
        get_simple_difference(array, lag) * lag for lag in range(1, window + 1)
    ) / norm


def get_simple_difference(array, lag):
    """Compute and return the simple difference of a series for a given lag.

    array : 2-D numpy.ndarray whose first dimension is time
    lag   : lag to use, so that the difference at time t is
            between values at times t + lag and t - lag
    """
    padding = np.ones((lag, array.shape[1]))
    past = np.concatenate([padding * array[0], array[:-lag]])
    future = np.concatenate([array[lag:], padding * array[-1]])
    return future - past


def build_context_windows(audio_frames, window=5, zero_padding=True):
    """Build context windows out of a series of audio frames.

    audio_frames : 2-D numpy.ndarray of frames of audio features
    window       : half-size of the context window built
    zero_padding : whether to zero-pad the data instead of using
                   the edge frames only as context (bool, default True)

    The context windows are built using past and future frames
    symmetrically around a central frame.
    """
    # Optionally zero-pad the signal.
    if zero_padding:
        padding = np.zeros((window, audio_frames.shape[1]))
        frames = np.concatenate([padding, audio_frames, padding])
        length = len(audio_frames)
    else:
        frames = audio_frames
        length = len(audio_frames) - 2 * window
    # Build the windows of frames and return them.
    full_window = 1 + 2 * window
    return np.concatenate(
        [frames[i:i + length] for i in range(full_window)], axis=1
    )


def build_dynamic_weights_matrix(size, window, complete=False):
    """Return a numpy matrix to produce dynamic features out of static ones.

    size     : size of the static features matrix (int)
    window   : (half) size of the time window to use (int)
    complete : whether to return the full weights matrix producing
               both static, delta and deltadelta features instead
               of the sole delta-computing weights (bool, default False)
    """
    # Declare default non-zero weights arrays.
    w_norm = 3 / (window * (window + 1) * (2 * window + 1))
    w_future = np.array([i * w_norm for i in range(1, window + 1)])
    w_past = -1 * w_future[::-1]
    # Declare the weights matrix and fill it row by row.
    weights = np.zeros((size, size))
    for time in range(size):
        # Fill weights for past observations.
        if time < window:
            weights[time, 0] = w_past[:window - time + 1].sum()
            weights[time, 1:time] = w_past[window - time + 1:]
        else:
            weights[time, time - window:time] = w_past
        # Fill weights for future observations.
        if time >= size - window:
            weights[time, -1] = w_future[size - window - time - 2:].sum()
            weights[time, time + 1:-1] = w_future[:size - window - time - 2]
        else:
            weights[time, time + 1:time + window + 1] = w_future
    # Optionally build the full weights matrix.
    if complete:
        weights = np.concatenate(
            [np.identity(size), weights, np.dot(weights, weights)]
        )
    # Return the generated matrix of weights.
    return weights


def interpolate_missing_values(array):
    """Fill NaN values in a 1-D numpy array by cubic spline interpolation."""
    # Check array's type validity.
    check_type_validity(array, np.ndarray, 'array')
    if array.ndim > 1:
        raise TypeError("'array' must be one-dimensional.")
    # Identify NaN values. If there aren't any, simply return the array.
    is_nan = np.isnan(array)
    if is_nan.sum() == 0:
        return array
    array = array.copy()
    not_nan = ~ is_nan
    # Build a cubic spline out of non-NaN values.
    spline = scipy.interpolate.splrep(
        np.argwhere(not_nan).ravel(), array[not_nan], k=3
    )
    # Interpolate missing values and replace them.
    for i in np.argwhere(is_nan).ravel():
        array[i] = scipy.interpolate.splev(i, spline)
    return array


def lowpass_filter(signal, cutoff, sample_rate, order=5):
    """Low-pass filter a signal at a given cutoff frequency.

    signal      : single or multi-channel signal (1-D or 2-D numpy.array)
    cutoff      : cutoff frequency, in Hz (positive int)
    sample_rate : sampling rate of the signal, in Hz (positive int)
    order       : half-order of the butterworth filter used (int, default 5)

    Filtering is conducted using a butterworth digital filter,
    applied forward then backward so as to be zero-phased.
    """
    # Build the initial butterworth low-pass filter.
    filt_b, filt_a = scipy.signal.butter(
        order, 2 * cutoff / sample_rate, btype='low', analog=False
    )
    # If the signal has a single channel, filter it out and return it.
    if len(signal.shape) == 1:
        return scipy.signal.filtfilt(filt_b, filt_a, signal)
    # Otehrwise, conduct channel-wise filtering
    return np.concatenate([
        np.expand_dims(scipy.signal.filtfilt(filt_b, filt_a, channel), 1)
        for channel in signal.T
    ], axis=1)


def sequences_to_batch(sequences, length=None):
    """Batch a set of data sequences into a three-dimensional array.

    sequences : list of array of two-dimensional numpy arrays sharing
                the same shape on their last dimension
    length    : optional size of the batched array's second dimension
                (otherwise, maximum sample length is used)

    Return a numpy.array of shape [n_sequences, length, sequences_width].
    """
    # Check sequences argument validity.
    check_type_validity(sequences, (list, np.ndarray), 'sequences')
    if isinstance(sequences, np.ndarray):
        if len(sequences.shape) == 2:
            sequences = [sequences]
        elif len(sequences.shape) != 1:
            raise TypeError(
                "'sequences' must be a list or a flat numpy array."
            )
    width = sequences[0].shape[1]
    valid = all(
        isinstance(sequence, np.ndarray) and len(sequence.shape) == 2
        and sequence.shape[1] == width
        for sequence in sequences
    )
    if not valid:
        raise TypeError(
            "All sequences must be 2-D numpy arrays of same number of columns."
        )
    # Gather the length of each and every sequence, truncated if needed.
    if length is None:
        batch_sizes = np.array([len(sequence) for sequence in sequences])
        length = np.max(batch_sizes)
    else:
        check_positive_int(length, 'length')
        batch_sizes = np.array([
            min(len(sequence), length) for sequence in sequences
        ])
    # Zero-pad the sequences and concatenate them.
    batched = np.array([
        np.concatenate([
            seq[:length], np.zeros((length - seq_length, seq.shape[1]))
        ])
        for seq, seq_length in zip(sequences, batch_sizes)
    ])
    # Return the batched sequences and the true sequence lengths.
    return batched, batch_sizes


def batch_to_sequences(batch, batch_sizes):
    """Split an array of batched sequences into a list of sequences.

    batch       : three-dimensional numpy array of shape
                  [n_sequences, max_length, sequences_width]
    batch_sizes : list of true lengths of the batched sequences
                  (i.e. notwithstanding zero padding)
    """
    sequences = np.array([
        sequence[:batch_sizes[i]] for i, sequence in enumerate(batch)
    ])
    if batch.shape[0] == 1:
        sequences = sequences[0]
    return sequences
