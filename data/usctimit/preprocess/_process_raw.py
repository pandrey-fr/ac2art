# coding: utf-8

"""Set of functions to transform raw USC Timit data."""

import librosa
from scipy.io.wavfile import write as write_wav
import numpy as np
import pandas as pd

from data.usctimit.raw import get_filepath, load_phone_labels, UscTimitEMA


def load_all_phone_labels(speaker):
    """Docstring."""
    def load_labels(name):
        """Docstring."""
        labels = pd.read_csv(
            get_filepath(name, 'trans'), header=None,
            names=['start', 'end', 'phone', 'word', 'sentence', 'utterance']
        )
        labels['file'] = name
        return labels
    name = 'usctimit_ema_%s_{0:003}_{1:003}' % speaker.lower()
    return pd.concat([
        load_labels(name.format(i, i + 4)) for i in range(1, 461, 5)
    ])




def split_wav(filename):
    """Docstring."""
    # Get the path, base name and starting utterance index of the treated file.
    path = get_filepath(filename, 'wav')
    base_name, index, _ = filename.rsplit('_', 2)
    base_name += '_{0:03}.wav'
    index = int(index)
    # Load the waveform audio data and the phone labels transcription.
    wav, sampling_rate = librosa.load(path)
    labels = load_phone_labels(filename)
    # Compute the boundaries of the utterances, keeping silent edges.
    silences = np.array([
        np.mean(label[:2]) for label in labels if label[-1] == 0
    ])
    boundaries = np.int32(sampling_rate * silences)
    # Iterate over the utterances. Write them and their transcription to disk.
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i:i+2]
        utterance = wav[start:end]
        utterance_path = os.path.join(
            DATA_FOLDER, 'wav_split', base_name.format(i + index)
        )
        wavfile.write(utterance_path, sampling_rate, utterance)
