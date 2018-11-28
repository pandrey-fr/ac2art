# coding: utf-8
#
# Copyright 2018 Paul Andrey
#
# This file is part of ac2art.
#
# ac2art is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ac2art is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ac2art.  If not, see <http://www.gnu.org/licenses/>.

"""Set of dependency functions to build mocha-timit abkhazia corpus files."""

import os

from sphfile import SPHFile

from ac2art.corpora.mocha.raw._loaders import SPEAKERS
from ac2art.corpora.prototype.utils import read_transcript
from ac2art.utils import CONSTANTS


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
