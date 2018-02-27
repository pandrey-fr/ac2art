# coding: utf-8

"""Set of functions to normalize data already extracted from the mngu0 data."""

import os

import numpy as np

from data.utils import load_data_paths


_, FOLDER = load_data_paths('mngu0')


def compute_files_moments(file_type, store=True):
    """Compute file-wise and global mean, deviation and spread of a dataset.

    The dataset must have been produced through extracting operations
    on the initial .ema and .wav files of the mngu0 dataset.

    file_type : type of data (str in {'ema', 'energy', 'lsf', 'lpc', 'mfcc'})
    store     : whether to store the computed values to disk
                (bool, default True)

    Return a dict containing the computed values.
    Optionally write it to a dedicated .npy file.
    """
    folder = os.path.join(FOLDER, file_type)
    # Compute file-wise means and standard deviations.
    dataset = np.array([
        np.load(os.path.join(folder, filename))
        for filename in sorted(os.listdir(folder))
        if filename.endswith('_%s.npy' % file_type)
    ])
    moments = {
        'file_means': np.array([data.mean(axis=0) for data in dataset]),
        'file_stds': np.array([data.std(axis=0) for data in dataset]),
        'file_spread': np.array([
            data.max(axis=0) - data.min(axis=0) for data in dataset
        ])
    }
    # Compute corpus-wide means, standard deviations and spread.
    dataset = np.concatenate(dataset)
    moments.update({
        'global_means': dataset.mean(axis=0),
        'global_stds': dataset.std(axis=0),
        'global_spread': dataset.max(axis=0) - dataset.min(axis=0)
    })
    # Optionally store the computed values to disk.
    if store:
        dirname = os.path.join(FOLDER, 'norm_params')
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        np.save(os.path.join(dirname, 'norm_%s.npy' % file_type), moments)
    # Return the dict of computed values.
    return moments


def normalize_files(file_type, norm_type='spread'):
    """Normalize pre-extracted mngu0 data of a given type.

    Normalization includes de-meaning (based on dataset-wide mean)
    and division by a dataset-wide computed value, which may either
    be four standard-deviations or the difference between the
    extremum points (distribution spread).

    Normalized utterances are stored as .npy files in a properly-named folder.

    file_type : one in {'ema', 'energy', 'lpc', 'lsf', 'mfcc'}
    norm_type : normalization divisor to use ('spread' or 'stds')
    """
    # Gather files list.
    input_folder = os.path.join(FOLDER, file_type)
    files = sorted([
        filename for filename in os.listdir(input_folder)
        if filename.endswith('_%s.npy' % file_type)
    ])
    # Gather files' moments. Compute them if needed.
    path = os.path.join(FOLDER, 'norm_params', 'norm_%s.npy' % file_type)
    if os.path.isfile(path):
        moments = np.load(path).tolist()
    else:
        moments = compute_files_moments(file_type, store=True)
    means = moments['global_means']
    norm = moments['global_%s' % norm_type] * (4 if norm_type == 'stds' else 1)
    # Build the output directory, if needed.
    output_folder = os.path.join(FOLDER, file_type + '_norm_' + norm_type)
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    # Iteratively normalize the utterances.
    for filename in files:
        data = np.load(os.path.join(input_folder, filename))
        data = (data - means) / norm
        np.save(os.path.join(output_folder, filename), data)
