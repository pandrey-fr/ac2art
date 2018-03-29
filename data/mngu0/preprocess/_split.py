# coding: utf-8

"""Set of functions to split the mngu0 dataset ensuring triphones coverage"""

import os


from data._prototype.split import build_split_corpus, store_filesets
from data.mngu0.raw import get_utterances_list
from data.utils import CONSTANTS


FOLDER = CONSTANTS['mngu0_raw_folder']

# Define functions through wrappers; pylint: disable=invalid-name
split_corpus = build_split_corpus('mngu0', lowest_limit=9)


def adjust_provided_filesets():
    """Adjust the filesets lists provided with mngu0 to fit the raw data.

    As the provided filesets lists are based on a extracted version of
    the mngu0 corpus anterior to the splitting of some of the utterances
    within the raw data, running this function is necessary so as to use
    similar train, validation and test utterances of newly-processed data
    as in most papers using this database.
    """
    # Load the full list of utterances.
    utterances = get_utterances_list()
    # Iteratively produce the adjusted filesets.
    filesets = {}
    for set_name in ['train', 'validation', 'test']:
        # Load the raw fileset list.
        path = os.path.join(FOLDER, 'ema_filesets', set_name + 'files.txt')
        with open(path) as file:
            raw_fileset = [row.strip('\n') for row in file]
        # Derive the correct fileset of newly processed utterances.
        filesets[set_name + '_initial'] = [
            utt for utt in utterances if utt.strip('abcdef') in raw_fileset
        ]
    # Write the filesets to txt files.
    store_filesets(filesets, corpus='mngu0')
