# coding: utf-8

"""Set of dependency functions to build mngu0 abkhazia corpus files."""

import os
import re
import shutil


from data.mngu0.raw import get_utterances_list
from data.utils import CONSTANTS


RAW_FOLDER = CONSTANTS['mngu0_raw_folder']


def copy_wavs(dest_folder):
    """Copy mngu0 wav files to a given folder."""
    utterances = get_utterances_list()
    wav_folder = os.path.join(RAW_FOLDER, 'wav_16kHz')
    for name in utterances:
        name += '.wav'
        shutil.copyfile(
            os.path.join(wav_folder, name), os.path.join(dest_folder, name)
        )
    # Return the list of copied utterances.
    return [name for name in utterances]


def get_transcription(utterance, phonetic=False):
    """Return the transcription of a given mngu0 utterance.

    Note: as labelling of mngu0 data is not over yet, this
          function returns a pseudo-phonetic transcription
          instead of the actual one.
    """
    # Gather words transcript.
    path = os.path.join(RAW_FOLDER, 'phone_labels', utterance + '.utt')
    with open(path) as utt_file:
        for _ in range(4):
            next(utt_file)
        transcript = next(utt_file).split('\\"')[1].strip('.')
    # Optionally pseudo-phonetize the utterance.
    if phonetic:
        # Use regex; pylint: disable=anomalous-backslash-in-string
        transcript = re.sub('[^\w ]', ' ', transcript.lower())
        transcript = re.sub('  +', ' ', transcript)
        transcript = ' '.join(map('-'.join, transcript.split(' ')))
    return transcript
