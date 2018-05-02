# coding: utf-8

"""Set of functions to load raw data from mngu0 in adequate formats."""

import os


from data.commons.loaders import EstTrack, Wav
from data._prototype.raw import build_ema_loaders
from data.utils import CONSTANTS

RAW_FOLDER = CONSTANTS['mngu0_raw_folder']
SPEAKERS = [None]


def get_utterances_list(speaker=None):
    """Return the full list of mngu0 utterances' names."""
    # Argument merely serves compatibility; pylint: disable=unused-argument
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
    return Wav(path, 16000, frame_size, hop_time)


def load_ema_base(filename, columns_to_keep=None):
    """Load data from a mngu0 EMA (.ema) file."""
    # Unused argument kept for API compliance; pylint: disable=unused-argument
    path = os.path.join(RAW_FOLDER, 'ema_basic_data/', filename + '.ema')
    track = EstTrack(path)
    ema_data = track.data
    column_names = list(track.column_names.values())
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


# Define a function through a wrapper; pylint: disable=invalid-name
load_ema, load_voicing = build_ema_loaders(
    'mngu0', 200, load_ema_base, load_phone_labels
)
