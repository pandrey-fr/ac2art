# coding: utf-8

"""Utility functions used internally in the _prototype submodule."""

import os


from data.utils import CONSTANTS


def _get_normfile_path(main_folder, file_type, speaker):
    """Get the path to a norm parameters file."""
    name = file_type if speaker is None else '%s_%s' % (file_type, speaker)
    return os.path.join(main_folder, 'norm_params', 'norm_%s.npy' % name)


def load_articulators_list(corpus):
    """Load the list of articulators contained in a corpus's data."""
    folder = os.path.join(CONSTANTS['%s_processed_folder' % corpus], 'ema')
    with open(os.path.join(folder, 'articulators'), encoding='utf-8') as file:
        articulators = [row.strip('\n') for row in file]
    return articulators
