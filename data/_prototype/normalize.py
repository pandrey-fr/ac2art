# coding: utf-8

"""Prototype and wrapper functions to normalize datasets."""

import os

import numpy as np

from data.utils import CONSTANTS, import_from_string


def build_normalization_functions(dataset):
    """Define and return functions to normalize datasets."""
    # Gather dataset-specific dependencies.
    main_folder = CONSTANTS['%s_processed_folder' % dataset]
    get_utterances_list, speakers = import_from_string(
        'data.%s.raw._loaders' % dataset, ['get_utterances_list', 'SPEAKERS']
    )

    # Wrap the normalization parameters computing function.
    def compute_moments(file_type, by_speaker=False, store=True):
        """Compute files moments."""
        nonlocal speakers
        # Optionally compute speaker-wise normalization parameters.
        if by_speaker:
            return {
                speaker: _compute_moments(
                    file_type, speaker, store, main_folder, get_utterances_list
                )
                for speaker in speakers
            }
        # Otherwise, compute corpus-wide parameters.
        return _compute_moments(
            file_type, None, store, main_folder, get_utterances_list
        )

    # Wrap the files normalization functon.
    def normalize_files(file_type, norm_type, by_speaker=False):
        """Normalize a set of files."""
        nonlocal speakers
        # Optionally normalize utterances using speaker-wise values.
        if by_speaker:
            for speaker in speakers:
                _normalize_files(
                    file_type, norm_type, speaker, main_folder,
                    get_utterances_list, compute_moments
                )
            return None
        # Otherwise, normalize all utterances using corpus-wide values.
        return _normalize_files(
            file_type, norm_type, None, main_folder,
            get_utterances_list, compute_moments
        )
    # Adjust the functions' docstrings and return them.
    compute_moments.__doc__ = _compute_moments.__doc__.format(dataset)
    normalize_files.__doc__ = _normalize_files.__doc__.format(dataset)
    return compute_moments, normalize_files


def _get_normfile_path(main_folder, file_type, speaker):
    """Get the path to a norm parameters file."""
    name = file_type if speaker is None else '%s_%s' % (file_type, speaker)
    return os.path.join(main_folder, 'norm_params', 'norm_%s.npy' % name)


def _compute_moments(
        file_type, speaker, store, main_folder, get_utterances_list
    ):
    """Compute file-wise and global mean, deviation and spread of a dataset.

    The dataset must have been produced through extracting operations
    on the initial .ema and .wav files of the {0} dataset.

    file_type  : one of {{'ema', 'energy', 'lsf', 'lpc', 'mfcc'}}
    by_speaker : whether to compute speaker-wise normalization parameters
                 instead of corpus-wide ones (bool, default False)
    store      : whether to store the computed values (bool, default True)

    Return a dict containing the computed values.
    Optionally write it to a dedicated .npy file.
    """
    folder = os.path.join(main_folder, file_type)
    # Compute file-wise means and standard deviations.
    dataset = np.array([
        np.load(os.path.join(folder, name + '_%s.npy' % file_type))
        for name in get_utterances_list(speaker)
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
        path = _get_normfile_path(main_folder, file_type, speaker)
        folder = os.path.dirname(path)
        if not os.path.isdir(folder):
            os.makedirs(folder)
        np.save(path, moments)
    # Return the dict of computed values.
    return moments


def _normalize_files(
        file_type, norm_type, speaker, main_folder,
        get_utterances_list, compute_moments
    ):
    """Normalize pre-extracted {0} data of a given type.

    Normalization includes de-meaning (based on dataset-wide mean)
    and division by a dataset-wide computed value, which may either
    be four standard-deviations or the difference between the
    extremum points (distribution spread).

    Normalized utterances are stored as .npy files in a properly-named folder.

    file_type  : one of {{'ema', 'energy', 'lpc', 'lsf', 'mfcc'}}
    norm_type  : normalization divisor to use ('spread' or 'stds')
    by_speaker : whether to use speaker-wise normalization parameters
                 instead of corpus-wide ones (bool, default False)
    """
    input_folder = os.path.join(main_folder, file_type)
    # Gather files' moments. Compute them if needed.
    path = _get_normfile_path(main_folder, file_type, speaker)
    if os.path.isfile(path):
        moments = np.load(path).tolist()
    elif speaker is None:
        moments = compute_moments(file_type, by_speaker=False)
    else:
        moments = compute_moments(file_type, by_speaker=True)[speaker]
    means = moments['global_means']
    norm = moments['global_%s' % norm_type] * (4 if norm_type == 'stds' else 1)
    # Build the output directory, if needed.
    norm_type += ('' if speaker is None else '_byspeaker')
    output_folder = os.path.join(main_folder, file_type + '_norm_' + norm_type)
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    # Iteratively normalize the utterances.
    files = [
        name + '_%s.npy' % file_type for name in get_utterances_list(speaker)
    ]
    for filename in files:
        data = np.load(os.path.join(input_folder, filename))
        data = (data - means) / norm
        np.save(os.path.join(output_folder, filename), data)
