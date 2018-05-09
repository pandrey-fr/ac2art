# coding: utf-8

"""Set of functions to load raw mocha-timit data in adequate formats."""

import os

import numpy as np
from sphfile import SPHFile

from data.commons.enhance import lowpass_filter
from data.commons.loaders import EstTrack, Wav
from data._prototype.raw import build_ema_loaders, build_utterances_getter
from data.utils import CONSTANTS


RAW_FOLDER = CONSTANTS['mocha_raw_folder']
SPEAKERS = ['fsew0', 'msak0']


def get_speaker_utterances(speaker):
    """Return the list of mocha-timit utterances for a given speaker."""
    folder = os.path.join(RAW_FOLDER, speaker)
    return sorted([
        name[:-4] for name in os.listdir(folder) if name.endswith('.wav')
    ])


def get_transcription(utterance):
    """Return the transcription of a given mocha-timit utterance."""
    utt_id = int(utterance.split('_', 1)[1])
    path = os.path.join(RAW_FOLDER, 'mocha-timit.txt')
    transcription = ''
    with open(path) as transcripts:
        for _ in range(2 * (utt_id - 1)):
            next(transcripts)
        for row in transcripts:
            if row.strip(' ') != '\n' and int(row[:3]) == utt_id:
                transcription += row[5:-1]
                if utt_id < 460:
                    row = next(transcripts)
                    if row != '\n' and int(row[:3]) == utt_id:
                        transcription += row[5:-1]
                break
    return transcription


def load_sphfile(path, sampling_rate, frame_time, hop_time):
    """Return a Wav instance based on the data stored in a Sphere file."""
    # Build a temporary copy of the file, converted to actual waveform format.
    tmp_path = './' + os.path.basename(path[:-4]) + '_tmp.wav'
    SPHFile(path).write_wav(tmp_path)
    # Load data from the waveform file, and then remove the latter.
    wav = Wav(tmp_path, sampling_rate, frame_time, hop_time)
    os.remove(tmp_path)
    return wav


def load_wav(filename, frame_time=25, hop_time=10):
    """Load data from a mocha-timit waveform (.wav) file.

    filename   : name of the utterance whose audio data to load (str)
    frame_time : frames duration, in milliseconds (int, default 25)
    hop_time   : duration of the shift step between frames, in milliseconds
                 (int, default 10)

    Return a `data.commons.Wav` instance, containing the audio waveform
    and an array of frames grouping samples based on the `frame_time`
    and `hop_time` arguments.
    """

    # Load phone labels and compute frames index so as to trim silences.
    speaker = filename.split('_')[0]
    path = os.path.join(RAW_FOLDER, speaker, filename + '.wav')
    return load_sphfile(path, 16000, frame_time, hop_time)


def load_larynx(filename):
    """Load laryngograph data from a mocha-timit .lar file.

    filename : name of the utterance whose laryngograph data to load (str)

    Return laryngograph data resampled from 16 000 kHz to 500 Hz
    so as to match the EMA data's sampling rate.
    """
    speaker = filename.split('_')[0]
    path = os.path.join(RAW_FOLDER, speaker, filename + '.lar')
    return load_sphfile(path, 16000, 200, 2).get_rms_energy()


def load_ema_base(filename, columns_to_keep=None):
    """Load articulatory data associated with a mocha-timit utterance."""
    # Import data from file.
    speaker = filename.split('_')[0]
    path = os.path.join(RAW_FOLDER, speaker, filename + '.ema')
    track = EstTrack(path)
    ema_data = track.data / 1000
    column_names = list(track.column_names.values())
    # Optionally add laryngograph data.
    if columns_to_keep is None or 'larynx' in columns_to_keep:
        larynx = load_larynx(filename) * 10
        if len(larynx) > len(ema_data):
            larynx = larynx[:len(ema_data)]
        else:
            ema_data = ema_data[:len(larynx)]
        ema_data = np.concatenate([ema_data, larynx], axis=1)
        column_names.append('larynx')
    # Smooth the data, as recordings are pretty bad.
    ema_data = lowpass_filter(ema_data, cutoff=20, sample_rate=500)
    # Return the EMA data and the list of columns names.
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
load_ema, load_voicing = build_ema_loaders(
    'mocha', 500, load_ema_base, load_phone_labels
)
