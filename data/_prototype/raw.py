# coding: utf-8

"""Wrapper functions to build raw data loaders."""

import numpy as np
import pandas as pd

from data.utils import CONSTANTS


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
