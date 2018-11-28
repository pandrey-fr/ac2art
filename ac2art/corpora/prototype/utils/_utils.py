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

"""Set of utility functions used internally in the prototype submodule."""

import os


from ac2art.utils import CONSTANTS


def _get_normfile_path(main_folder, file_type, speaker):
    """Get the path to a norm parameters file."""
    name = file_type if speaker is None else '%s_%s' % (file_type, speaker)
    return os.path.join(main_folder, 'norm_params', 'norm_%s.npy' % name)


def load_articulators_list(corpus, norm_type=None):
    """Load the list of articulators contained in a corpus's data."""
    folder = 'ema_norm_%s' % norm_type if norm_type else 'ema'
    folder = os.path.join(CONSTANTS['%s_processed_folder' % corpus], folder)
    with open(os.path.join(folder, 'articulators'), encoding='utf-8') as file:
        articulators = [row.strip('\n') for row in file]
    return articulators


def read_transcript(path, phonetic=False, silences=None, fields=3):
    """Generic function to read an utterance's transcript.

    path     : path to the .lab file to read (str)
    phonetic : whether to return phonetic transcription
               instead of word one (bool, default False)
    silences : optional list of silence markers to ignore
    fields   : number of space-separated fields on rows
               which contain a word beginning (int, default 3)

    This function is compatible with mspka .lab files and
    manually-enhanced mocha-timit and mngu0 .lab files.
    """
    if silences is None:
        silences = []
    # Case when reading phonetic transcript of the utterance.
    if phonetic:
        with open(path) as lab_file:
            words = []
            word = ''
            for row in lab_file:
                row = row.strip(' \n')
                # Ignore rows indexing silences.
                if row.rsplit(' ', 1)[1] in silences:
                    continue
                n_fields = row.count(' ')
                # Case when the row marks the beginning of a new word.
                if n_fields == fields:
                    if word:
                        words.append(word.strip('-'))
                    word = row.rsplit(' ', 2)[1]
                # Case when the row marks the continuation of a word.
                elif n_fields == (fields - 1):
                    word += '-' + row.rsplit(' ', 1)[1]
            if word:
                words.append(word)
    # Case when reading words transcript of the utterance.
    else:
        with open(path) as lab_file:
            words = [
                row.strip(' \n').rsplit(' ', 1)[1]
                for row in lab_file if row.count(' ') == fields
            ]
    # Return the transcription as a string.
    return ' '.join(words)
