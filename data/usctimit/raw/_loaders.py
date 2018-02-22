# coding: utf-8

"""Set of functions to load raw data from USC Timit EMA in adequate formats."""

import os

import numpy as np
from scipy.io import loadmat

from data.commons import AbstractDataLoader
from data.utils import load_data_paths


DATA_FOLDER, _ = load_data_paths('usctimit')


def get_filepath(filename, filetype):
    """Return the path to a USC Timit file of given name and type."""
    speaker = filename.split('_', 2)[-1][:2].upper()
    filename += ('.' + filetype)
    return os.path.join(DATA_FOLDER, speaker, filetype, filename)


def load_phone_labels(filename):
    """Load data from a USC Timit phone labels (.trans) file.

    Return a list of tuples, where each tuple represents a phoneme
    as a pair of an ending time in seconds (float) and a symbol (str).
    """
    path = get_filepath(filename, 'trans')
    with open(path) as file:
        labels = [_format_phone_labels_row(row) for row in file]
    return labels


def _format_phone_labels_row(row):
    """Format a row read from a USC Timit phone labels (.trans) file."""
    start, end, phone, word, sentence, utterance = row.strip('\n').split(',')
    utterance = int(utterance) if len(utterance) else 0
    return (float(start), float(end), phone, word, sentence, utterance)


def load_wav(utterance, frame_size=200, hop_time=5):
    """Load data from a waveform (.wav) file.

    filename   : name of the utterance whose audio data to load (str)
    frame_size : number of samples to include per frame (int, default 200)
    hop_time   : duration of the shift step between frames, in milliseconds
                 (int, default 5)

    Return a `data.commons.Wav` instance, containing the audio waveform
    and an array of frames grouping samples based on the `frame_size`
    and `hop_time` arguments. The default values of the latter are those
    used in most papers relying on mngu0 data.
    """
    speaker = filename.split('_', 2)[-1][:2].upper()
    path = os.path.join(DATA_FOLDER, 'EMA', 'Data', speaker, 'wav', filename + '.wav')
    return Wav(path, frame_size, hop_time)




class UscTimitEMA(AbstractDataLoader):
    """Class to load EMA articulatory data from the USC Timit database.

    The USC Timit EMA data is stored in .mat files of specific format,
    to which this class's `load` method is (and only is) fit.
    """

    def load(self):
        """Load the articulatory data from the file."""
        # Load the .mat file. Extract the articulatory data out of it.
        filename = os.path.splitext(os.path.basename(self.filename))[0]
        data_array = loadmat(self.filename)[filename][0, 1:]
        # Extract articulator names and record column names accordingly.
        articulator_names = [articulator[0][0] for articulator in data_array]
        for i, name in enumerate(articulator_names):
            self.column_names[3 * i] = name + '_x'
            self.column_names[(3 * i) + 1] = name + '_y'
            self.column_names[(3 * i) + 2] = name + '_z'
        # Extract articulatory data.
        self.data = np.concatenate(
            [articulator[2] for articulator in data_array], axis=1
        )
        # Infer time index from the track's duration.
        self.time_index = np.arange(.01, .01 + .01 * len(self.data), .01)
