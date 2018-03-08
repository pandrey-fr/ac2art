# coding: utf-8

"""Set of functions to load raw data from mngu0 in adequate formats."""

import os

import numpy as np

from data.commons import Wav
from data.mngu0.raw import EstTrack
from data.utils import (
    check_type_validity, interpolate_missing_values, load_data_paths
)


RAW_FOLDER, _ = load_data_paths('mngu0')


def get_utterances_list():
    """Return the full list of mngu0 utterances' names."""
    wav_folder = os.path.join(RAW_FOLDER, 'wav_16kHz')
    return sorted([
        name[:-4] for name in os.listdir(wav_folder) if name.endswith('.wav')
    ])


def load_wav(filename, frame_size=200, hop_time=5):
    """Load data from a mngu0 waveform (.wav) file.

    filename   : name of the utterance whose audio data to load (str)
    frame_size : number of samples to include per frame (int, default 200)
    hop_time   : duration of the shift step between frames, in milliseconds
                 (int, default 5)

    Return a `data.commons.Wav` instance, containing the audio waveform
    and an array of frames grouping samples based on the `frame_size`
    and `hop_time` arguments. The default values of the latter are those
    used in most papers relying on mngu0 data.
    """
    path = os.path.join(RAW_FOLDER, 'wav_16kHz/', filename + '.wav')
    return Wav(path, frame_size, hop_time)


def load_ema(filename, columns_to_keep=None):
    """Load data from a mngu0 EMA (.ema) file. Interpolate any missing values.

    filename        : name of the utterance whose raw EMA data to load (str)
    columns_to_keep : optional list of columns to keep

    Return a 2-D numpy.ndarray where each row represents a sample (recorded
    at 200Hz) and each column is represents a 1-D coordinate of an articulator.
    Also return the list of column names.
    """
    # Import data from file.
    path = os.path.join(RAW_FOLDER, 'ema_basic_data/', filename + '.ema')
    track = EstTrack(path)
    # Optionally select the data columns kept.
    check_type_validity(columns_to_keep, (list, type(None)), 'columns_to_keep')
    column_names = list(track.column_names.values())
    if columns_to_keep is not None:
        cols_index = [column_names.index(col) for col in columns_to_keep]
        ema_data = track.data[:, cols_index]
        column_names = columns_to_keep
    else:
        ema_data = track.data
    # Interpolate NaN values using cubic splines.
    ema_data = np.concatenate([
        interpolate_missing_values(data_column).reshape(-1, 1)
        for data_column in np.transpose(ema_data)
    ], axis=1)
    # Return the EMA data and a list of columns names.
    return ema_data, column_names


def load_phone_labels(filename):
    """Load data from a mngu0 phone labels (.lab) file.

    Return a list of tuples, where each tuple represents a phoneme
    as a pair of an ending time in seconds (float) and a symbol (str).
    """
    path = os.path.join(RAW_FOLDER, 'phone_labels/', filename + '.lab')
    with open(path) as file:
        while next(file) != '#\n':
            pass
        labels = [
            row.strip('\n').strip('\t').replace(' 26 ', '').split('\t')
            for row in file
        ]
    return [(float(label[0]), label[1]) for label in labels]
