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

"""Set of dependency functions to build mspka abkhazia corpus files."""

import os


from ac2art.corpora.mspka.raw._loaders import SPEAKERS
from ac2art.corpora.prototype.utils import read_transcript
from ac2art.utils import CONSTANTS


RAW_FOLDER = CONSTANTS['mspka_raw_folder']


def copy_wavs(dest_folder):
    """Copy mspka wav files to a given folder, resampling them on the go."""
    utterances = []
    for speaker in SPEAKERS:
        wav_folder = os.path.join(RAW_FOLDER, speaker + '_1.0.0', 'wav_1.0.0')
        spk_utterances = sorted(
            [name for name in os.listdir(wav_folder) if name.endswith('.wav')]
        )
        for name in spk_utterances:
            status = os.system('sox %s -r 16000 %s' % (
                os.path.join(wav_folder, name), os.path.join(dest_folder, name)
            ))
            if status != 0:
                raise RuntimeError(
                    'Copy and resampling of file %s.wav exited with code %s.'
                    % (name, status)
                )
        utterances.extend(spk_utterances)
    # Return the list of copied utterances.
    return [name[:-4] for name in utterances]


def get_transcription(utterance, phonetic=False):
    """Return the transcription of a given mspka utterance."""
    speaker = utterance.split('_', 1)[0]
    path = os.path.join(
        RAW_FOLDER, speaker + '_1.0.0', 'lab_1.0.0', utterance + '.lab'
    )
    return read_transcript(path, phonetic, silences=['silence'])
