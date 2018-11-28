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

"""Set of dependency functions to build mngu0 abkhazia corpus files."""

import os
import shutil


from ac2art.corpora.mngu0.raw import get_utterances_list
from ac2art.corpora.prototype.utils import read_transcript
from ac2art.utils import CONSTANTS


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
