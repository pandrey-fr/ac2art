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

"""Set of functions to load raw data from mspka in adequate formats."""

import os

import pandas as pd

from ac2art.corpora.prototype.raw import (
    build_utterances_getter, build_ema_loaders
)
from ac2art.internal.data_loaders import Wav
from ac2art.utils import alphanum_sort, CONSTANTS


RAW_FOLDER = CONSTANTS['mspka_raw_folder']
SPEAKERS = ['cnz', 'lls', 'olm']


def get_speaker_utterances(speaker):
    """Return the list of mocha-timit utterances for a given speaker."""
    folder = os.path.join(RAW_FOLDER, speaker + '_1.0.0', 'lab_1.0.0')
    return alphanum_sort([
        name[:-4] for name in os.listdir(folder) if name.endswith('.lab')
    ])


def load_wav(filename, frame_time=25, hop_time=10):
    """Load data from a mspka waveform (.wav) file.

    filename   : name of the utterance whose audio data to load (str)
    frame_time : frames duration, in milliseconds (int, default 25)
    hop_time   : duration of the shift step between frames, in milliseconds
                 (int, default 10)

    Return a `data.commons.Wav` instance, containing the audio waveform
    and an array of frames grouping samples based on the `frame_time`
    and `hop_time` arguments.
    """
    speaker = filename.split('_')[0]
    path = os.path.join(
        RAW_FOLDER, speaker + '_1.0.0', 'wav_1.0.0', filename + '.wav'
    )
    return Wav(path, 22050, frame_time, hop_time)


def load_ema_base(filename, columns_to_keep=None):
    """Load data from a mspka EMA (.ema) file."""
    # Unused argument kept for API compliance; pylint: disable=unused-argument
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


# Define functions through wrappers; pylint: disable=invalid-name
get_utterances_list = (
    build_utterances_getter(get_speaker_utterances, SPEAKERS, corpus='mspka')
)
load_ema, load_voicing = build_ema_loaders(
    'mspka', 400, load_ema_base, load_phone_labels
)
