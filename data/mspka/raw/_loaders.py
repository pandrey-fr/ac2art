# coding: utf-8

"""Set of functions to load raw data from mspka in adequate formats."""

import os

import pandas as pd

from data.commons.loaders import Wav
from data.utils import alphanum_sort, check_type_validity, CONSTANTS


RAW_FOLDER = CONSTANTS['mspka_raw_folder']


def get_utterances_list(speaker=None):
    """Return the full list of mspka utterances' names."""
    speakers = ['cnz', 'lls', 'olm']
    # If no speaker is targetted, return all speakers' utterances list.
    if speaker is None:
        return [
            utterance for speaker in speakers
            for utterance in get_utterances_list(speaker)
        ]
    # Otherwise, load the list of utterances available for the speaker.
    folder = os.path.join(RAW_FOLDER, speaker + '_1.0.0', 'lab_1.0.0')
    return alphanum_sort([
        name[:-4] for name in os.listdir(folder) if name.endswith('.lab')
    ])


def load_wav(filename, frame_size=200, hop_time=2.5):
    """Load data from a mspka waveform (.wav) file.

    filename   : name of the utterance whose audio data to load (str)
    frame_size : number of samples to include per frame (int, default 200)
    hop_time   : duration of the shift step between frames, in milliseconds
                 (int, default 2.5)

    Return a `data.commons.Wav` instance, containing the audio waveform
    and an array of frames grouping samples based on the `frame_size`
    and `hop_time` arguments.
    """
    speaker = filename.split('_')[0]
    path = os.path.join(
        RAW_FOLDER, speaker + '_1.0.0', 'wav_1.0.0', filename + '.wav'
    )
    return Wav(path, 22050, frame_size, hop_time)


def load_ema(filename, columns_to_keep=None):
    """Load data from a mspka EMA (.ema) file.

    filename        : name of the utterance whose raw EMA data to load (str)
    columns_to_keep : optional list of columns to keep

    Return a 2-D numpy.ndarray where each row represents a sample (recorded
    at 400Hz) and each column is represents a 1-D coordinate of an articulator,
    in centimeters. Also return the list of column names.
    """
    # Import data from file.
    speaker = filename.split('_')[0]
    path = os.path.join(
        RAW_FOLDER, speaker + '_1.0.0', 'ema_1.0.0', filename + '.ema'
    )
    ema_data = pd.read_csv(path, sep=' ', header=None).T.values / 10
    column_names = [
        'ul_x', 'ul_y', 'ul_z', 'll_x', 'll_y', 'll_z', 'ui_x',
        'ui_y', 'ui_z', 'li_x', 'li_y', 'li_z', 'tb_x', 'tb_y',
        'tb_z', 'td_x', 'td_y', 'td_z', 'tt_x', 'tt_y', 'tt_z'
    ]
    # Optionally select the data columns kept.
    check_type_validity(columns_to_keep, (list, type(None)), 'columns_to_keep')
    if columns_to_keep is not None:
        cols_index = [column_names.index(col) for col in columns_to_keep]
        ema_data = ema_data[:, cols_index]
        column_names = columns_to_keep
    # Return the EMA data and a list of columns names.
    return ema_data, column_names


def load_phone_labels(filename):
    """Load data from a mspka phone labels (.lab) file.

    Return a list of tuples, where each tuple represents a phoneme
    as a pair of an ending time in seconds (float) and a symbol (str).
    """
    speaker = filename.split('_')[0]
    path = os.path.join(
        RAW_FOLDER, speaker + '_1.0.0', 'lab_1.0.0', filename + '.lab'
    )
    with open(path) as file:
        labels = [
            row.strip('\n').replace(' sil ', ' # ').split(' ') for row in file
        ]
    return [(float(label[1]), label[2]) for label in labels]
