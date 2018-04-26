# coding: utf-8

"""Set of functions to load raw mocha-timit data in adequate formats."""

import os

import numpy as np
from sphfile import SPHFile

from data.commons.enhance import lowpass_filter
from data.commons.loaders import EstTrack, Wav
from data._prototype.raw import build_utterances_getter, build_voicing_loader
from data.utils import CONSTANTS
from utils import check_type_validity


RAW_FOLDER = CONSTANTS['mocha_raw_folder']
SPEAKERS = ['fsew0', 'msak0']


def get_speaker_utterances(speaker):
    """Return the list of mocha-timit utterances for a given speaker."""
    folder = os.path.join(RAW_FOLDER, speaker)
    return sorted([
        name[:-4] for name in os.listdir(folder) if name.endswith('.wav')
    ])


def load_sphfile(path, sampling_rate, frame_size, hop_time):
    """Return a Wav instance based on the data stored in a Sphere file."""
    # Build a temporary copy of the file, converted to actual waveform format.
    tmp_path = path[:-4] + '_tmp.wav'
    SPHFile(path).write_wav(tmp_path)
    # Load data from the waveform file, and then remove the latter.
    wav = Wav(tmp_path, sampling_rate, frame_size, hop_time)
    os.remove(tmp_path)
    return wav


def load_wav(filename, frame_size=200, hop_time=2):
    """Load data from a mocha-timit waveform (.wav) file.

    filename   : name of the utterance whose audio data to load (str)
    frame_size : number of samples to include per frame (int, default 200)
    hop_time   : duration of the shift step between frames, in milliseconds
                 (int, default 2)

    Return a `data.commons.Wav` instance, containing the audio waveform
    and an array of frames grouping samples based on the `frame_size`
    and `hop_time` arguments. The default values of the latter are set
    to match the initial EMA sample rate and the frame size used in
    papers dealing with the mngu0 corpus.
    """

    # Load phone labels and compute frames index so as to trim silences.
    speaker = filename.split('_')[0]
    path = os.path.join(RAW_FOLDER, speaker, filename + '.wav')
    return load_sphfile(path, 16000, frame_size, hop_time)


def load_larynx(filename):
    """Load laryngograph data from a mocha-timit .lar file.

    filename : name of the utterance whose laryngograph data to load (str)

    Return laryngograph data resampled from 16 000 kHz to 500 Hz
    so as to match the EMA data's sampling rate.
    """
    speaker = filename.split('_')[0]
    path = os.path.join(RAW_FOLDER, speaker, filename + '.lar')
    return load_sphfile(path, 16000, 200, 2).get_rms_energy()


def load_ema(filename, columns_to_keep=None):
    """Load articulatory data associated with a mocha-timit utterance.

    filename        : name of the utterance whose raw EMA data to load (str)
    columns_to_keep : optional list of columns to keep

    Return a 2-D numpy.ndarray where each row represents a sample (recorded
    at 500Hz) and each column is represents a 1-D coordinate of an articulator,
    in centimeter. Also return the list of column names.
    """
    # Import data from file.
    speaker = filename.split('_')[0]
    path = os.path.join(RAW_FOLDER, speaker, filename + '.ema')
    track = EstTrack(path)
    # Optionally select the data columns kept.
    check_type_validity(columns_to_keep, (list, type(None)), 'columns_to_keep')
    column_names = list(track.column_names.values())
    if columns_to_keep is None:
        ema_data = track.data / 1000
        column_names.append('larynx')
    else:
        cols_index = [
            column_names.index(col) for col in columns_to_keep
            if col != 'larynx'
        ]
        column_names = columns_to_keep
        ema_data = track.data[:, cols_index] / 1000
    # Optionally add the laryngograph data to the articulatory data.
    # NOTE: EMA tracks last longer than laryngograph (and audio) ones,
    # thus cutting its final edge causes no issue.
    if 'larynx' in column_names:
        larynx = load_larynx(filename) * 10
        if len(larynx) > len(ema_data):
            larynx = larynx[:len(ema_data)]
        else:
            ema_data = ema_data[:len(larynx)]
        ema_data = np.concatenate([ema_data, larynx], axis=1)
    # Smooth the signal, as recordings are pretty bad.
    ema_data = lowpass_filter(ema_data, cutoff=20, sample_rate=500)
    # Return the EMA data and a list of columns names.
    return ema_data, column_names


def load_phone_labels(filename):
    """Load data from a mspka phone labels (.lab) file.

    Return a list of tuples, where each tuple represents a phoneme
    as a pair of an ending time in seconds (float) and a symbol (str).
    """
    # Load provided labels.
    speaker = filename.split('_')[0]
    path = os.path.join(RAW_FOLDER, speaker, filename + '.lab')
    with open(path) as file:
        labels = [row.strip('\n').split(' ') for row in file]
    # Replace silence and breath labels' symbols.
    symbols = {'sil': '#', 'breath': '##'}
    labels = [
        (float(label[1]), symbols.get(label[2], label[2])) for label in labels
    ]
    # Trim the initial silent breathing labels when returning them.
    return labels[2:]


# Define functions through wrappers; pylint: disable=invalid-name
get_utterances_list = (
    build_utterances_getter(get_speaker_utterances, SPEAKERS, corpus='mocha')
)
load_voicing = build_voicing_loader('mocha', 500, load_phone_labels)
