# coding: utf-8

"""Set of dependency functions to build mngu0 abkhazia corpus files."""

import os
import shutil


from data.mngu0.raw import get_utterances_list
from data._prototype.utils import read_transcript
from data.utils import CONSTANTS


def copy_wavs(dest_folder):
    """Copy mngu0 wav files to a given folder."""
    utterances = get_utterances_list()
    wav_folder = os.path.join(CONSTANTS['mngu0_raw_folder'], 'wav_16kHz')
    for name in utterances:
        name += '.wav'
        shutil.copyfile(
            os.path.join(wav_folder, name), os.path.join(dest_folder, name)
        )
    # Return the list of copied utterances.
    return [name for name in utterances]


def get_transcription(utterance, phonetic=False):
    """Return the transcription of a given mngu0 utterance."""
    path = os.path.join(
        CONSTANTS['mngu0_processed_folder'], 'labels', utterance + '.lab'
    )
    return read_transcript(path, phonetic, silences=['#'], fields=2)
