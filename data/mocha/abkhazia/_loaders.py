# coding: utf-8

"""Set of dependency functions to build mocha-timit abkhazia corpus files."""

import os

from sphfile import SPHFile

from data.mocha.raw._loaders import SPEAKERS
from data._prototype.utils import read_transcript
from data.utils import CONSTANTS


def copy_wavs(dest_folder):
    """Copy wav files, converting them from sph file format on the go."""
    utterances = []
    for speaker in SPEAKERS:
        folder = os.path.join(CONSTANTS['mocha_raw_folder'], speaker)
        spk_utterances = sorted(
            [name for name in os.listdir(folder) if name.endswith('.wav')]
        )
        for filename in spk_utterances:
            SPHFile(os.path.join(folder, filename)).write_wav(
                os.path.join(dest_folder, filename)
            )
        utterances.extend(spk_utterances)
    # Return the list of copied utterances.
    return [name[:-4] for name in utterances]


def get_transcription(utterance, phonetic=False):
    """Return the transcription of a given mocha-timit utterance."""
    path = os.path.join(
        CONSTANTS['mocha_processed_folder'], 'labels', utterance + '.lab'
    )
    return read_transcript(path, phonetic, silences=['sil', 'breath'])
