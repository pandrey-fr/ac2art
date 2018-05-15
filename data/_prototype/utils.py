# coding: utf-8

"""Utility functions used internally in the _prototype submodule."""

import os


from data.utils import CONSTANTS


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


def read_transcript(path, phonetic=False, silences=None):
    """Generic function to read an utterance's transcript.

    path     : path to the .lab file to read (str)
    phonetic : whether to return phonetic transcription
               instead of word one (bool, default False)
    silences : optional list of silence markers to ignore

    This function is compatible with mspka .lab files and
    manually-enhanced mocha-timit .lab files.
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
                if n_fields == 3:
                    if word:
                        words.append(word)
                    word = row.rsplit(' ', 2)[1]
                # Case when the row marks the continuation of a word.
                elif n_fields == 2:
                    word += '-' + row.rsplit(' ', 1)[1]
            if word:
                words.append(word)
    # Case when reading words transcript of the utterance.
    else:
        with open(path) as lab_file:
            words = [
                row.strip(' \n').rsplit(' ', 1)[1]
                for row in lab_file if row.count(' ') == 3
            ]
    # Return the transcription as a string.
    return ' '.join(words)
