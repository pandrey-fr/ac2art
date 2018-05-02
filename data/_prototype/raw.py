# coding: utf-8

"""Wrapper functions to build raw data loaders."""

import numpy as np
import pandas as pd

from data.utils import CONSTANTS
from utils import check_type_validity


def build_utterances_getter(get_speaker_utterances, speakers, corpus):
    """Build a function returning lists of utterances of a corpus."""
    def get_utterances_list(speaker=None):
        """Return the list of {0} utterances from a given speaker."""
        nonlocal speakers, get_speaker_utterances
        if speaker is None:
            return [
                utterance for speaker in speakers
                for utterance in get_speaker_utterances(speaker)
            ]
        elif speaker in speakers:
            return get_speaker_utterances(speaker)
        raise KeyError("Invalid speaker: '%s'." % speaker)
    # Adjust the function's docstring and return it.
    get_utterances_list.__doc__ = get_utterances_list.__doc__.format(corpus)
    return get_utterances_list


def build_voicing_loader(corpus, initial_sr, load_phone_labels):
    """Define and return a function generating binary voicing data."""
    # Load auxiliary symbols voicing reference chart.
    voiced = pd.read_csv(CONSTANTS['symbols_file'], index_col=corpus)['voiced']

    # Define the corpus-specific voicing function.
    def load_voicing(filename, sampling_rate=initial_sr):
        """Return theoretical voicing binary data of a {0} utterance.

        filename      : name of the utterance whose voicing to return
        sampling_rate : sampling rate (in Hz) of the data the voicing
                        is to be added to (positive int, default {1})

        The returned voicing is derived from the phoneme transcription
        of the utterance, and may thus contain mistakes as to the actual
        voicing of the audio track.
        """
        nonlocal load_phone_labels, voiced
        labels = load_phone_labels(filename)
        voicing = np.zeros(int(np.ceil(labels[-1][0] * sampling_rate)))
        start = 0
        for time, label in labels:
            end = round(time * sampling_rate)
            if voiced[label]:
                voicing[start:end] = 1
            start = end
        return np.expand_dims(voicing, 1)

    # Adjust the function's docstring and return it.
    load_voicing.__doc__ = load_voicing.__doc__.format(corpus, initial_sr)
    return load_voicing


def build_ema_loaders(corpus, initial_sr, load_ema_base, load_phone_labels):
    """Define and return functions to load raw EMA and binary voicing data.

    corpus            : name of the corpus (str)
    initial_sr        : initial sampling_rate of the corpus' EMA data, in Hz
    load_ema_base     : base (corpus-specific) EMA data loading function,
                        expecting an utterance's name and returning raw EMA
                        data (2-D numpy.ndarray) and the list of column names
    load_phone_labels : corpus-specific phone labels loading function
    """
    # Define the auxiliary binary voicing data loader.
    load_voicing = build_voicing_loader(corpus, initial_sr, load_phone_labels)

    # Define the corpus-specific raw EMA data loading function.
    def load_ema(filename, columns_to_keep=None):
        """Load data from a {0} EMA (.ema) file.

        filename        : utterance whose raw EMA data to load (str)
        columns_to_keep : optional list of columns to keep

        Return a 2-D numpy.ndarray where each row represents a sample
        (recorded at {1} Hz) and each column is represents a 1-D coordinate
        of an articulator, in centimeter. Also return the list of column names.
        """
        nonlocal load_ema_base
        check_type_validity(
            columns_to_keep, (list, type(None)), 'columns_to_keep'
        )
        ema_data, column_names = load_ema_base(filename, columns_to_keep)
        # Optionally select the articulatory data points to keep.
        if columns_to_keep is not None:
            cols_index = [column_names.index(col) for col in columns_to_keep]
            ema_data = ema_data[:, cols_index]
            column_names = columns_to_keep
        # Return the EMA data and the list of columns names.
        return ema_data, column_names

    # Adjust the previous function's docstring. Return both functions.
    load_ema.__doc__ = load_ema.__doc__.format(corpus, initial_sr)
    return load_ema, load_voicing
