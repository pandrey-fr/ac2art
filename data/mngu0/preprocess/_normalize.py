# coding: utf-8

"""Set of functions to normalize data already extracted from the mngu0 data."""

import os


from data.utils import load_data_paths
from neural_networks.components.mlpg import build_dynamic_weights_matrix


_, NEW_FOLDER = load_data_paths('mngu0')


def compute_files_moments(file_type, store=True):
    """Compute file-wise and global means and standard deviations of a dataset.

    The dataset must have been produced through pre-processing operations
    on the initial .ema and .wav files of the mngu0 dataset.

    file_type : type of data (str in {'ema', 'energy', 'lsf', 'lpc', 'mfcc'})
    store     : whether to store the computed values to disk, each to
                its own .npy file (bool, default True)

    Return a dict containing the computed values.
    """
    # Compute file-wise means and standard deviations.
    dataset = np.array([
        np.load(os.path.join(NEW_FOLDER, filename))
        for filename in os.listdir(NEW_FOLDER)
        if filename.endswith('_%s.npy' % file_type)
    ])
    moments = {
        'file_means': np.array([np.mean(data) for data in dataset]),
        'file_stds': np.array([np.std(data) for data in dataset])
    }
    # Compute corpus-wide means and standard deviations.
    dataset = np.concatenate(dataset)
    moments.update({
        'global_means': np.mean(dataset, axis=0),
        'global_stds': np.std(dataset, axis=0)
    })
    # Optionally store the computed values to disk.
    if store:
        dirname = os.path.join(NEW_FOLDER, file_type, 'norm_params/')
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        for key, array in moments.items():
            np.save(os.path.join(dirname, key + '.npy'), array)
    # Return the dict of computed values.
    return moments


def normalize_data():
    pass


def _load_utterance(filename, dtype):
    """Docstring."""
    path = os.path.join(NEW_FOLDER, dtype, filename + '_%s.npy' % dtype)
    return np.load(path)


def load_utterance(filename, audio='lsf', dynamic=False, window=None):
    """Docstring."""
    audio = _load_utterance(filename, audio)
    ema = _load_utterance(filename, 'ema')
    # TODO: normalization
    if window is not None:
        audio = build_context_windows(audio, window)
    if dynamic:
        ema = add_dynamic_features(ema)
    return audio, ema


def add_dynamic_features(static_features, window=5):
    """Docstring."""
    weights = build_dynamic_weights_matrix(len(static_features), window)
    delta = np.dot(weights, static_features)
    deltadelta = np.dot(weights, delta)
    return np.concatenate([static_features, delta, deltadelta], axis=1)


def build_context_windows(audio_frames, window=5):
    """Docstring."""
    padding = np.zeros((window, audio_frames.shape[1]))
    padded = np.concatenate([padding, audio_frames, padding])
    full_window = 1 + 2 * window
    return np.concatenate(
        [padded[i:i + len(audio_frames)] for i in range(full_window)], axis=1
    )
