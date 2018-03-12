# coding: utf-8

"""Set of functions to enhance acoustic and articulatory data."""

import numpy as np


def add_dynamic_features(static_features, window=5):
    """Build delta and deltadelta features on top of static ones.

    static_features : 2-D numpy.ndarray of static features to enhance
    window          : half-size of the window used in the computation
    """
    weights = build_dynamic_weights_matrix(len(static_features), window)
    delta = np.dot(weights, static_features)
    deltadelta = np.dot(weights, delta)
    return np.concatenate([static_features, delta, deltadelta], axis=1)


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
    # Declare stuff.
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
