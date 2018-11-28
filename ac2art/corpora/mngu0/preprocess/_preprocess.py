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

"""Set of functions to pre-process raw mngu0 data."""

import os


from ac2art.corpora.prototype.preprocess import (
    build_features_extraction_functions, build_normalization_functions,
    build_corpus_splitting_function, store_filesets
)
from ac2art.corpora.mngu0.raw import get_utterances_list
from ac2art.utils import CONSTANTS


EXTRACTION_DOC_DETAILS = """
    The default values of the last three arguments echo those used in most
    papers making use of the mngu0 data. The articulators kept correspond
    to the front-to-back and top-to-bottom (resp. py and pz) coordinates of
    six articulators : tongue tip (T1), tongue dorsum (T2), tongue back (T3),
    jaw, upperlip and lowerlip.
"""


DEFAULT_ARTICULATORS = [
    'T3_py', 'T3_pz', 'T2_py', 'T2_pz', 'T1_py', 'T1_pz',
    'jaw_py', 'jaw_pz', 'upperlip_py', 'upperlip_pz',
    'lowerlip_py', 'lowerlip_pz'
]


# Define functions through wrappers; pylint: disable=invalid-name
extract_utterances_data = (
    build_features_extraction_functions(
        corpus='mngu0', initial_sampling_rate=200,
        default_articulators=DEFAULT_ARTICULATORS,
        docstring_details=EXTRACTION_DOC_DETAILS
    )
)

compute_moments, normalize_files = build_normalization_functions('mngu0')

split_corpus = build_corpus_splitting_function(
    corpus='mngu0', lowest_limit=9, same_speaker_data=False
)
# pylint: enable=invalid-name


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
        folder = os.path.join(CONSTANTS['mngu0_raw_folder'], 'ema_filesets')
        with open(os.path.join(folder, set_name + 'files.txt')) as file:
            raw_fileset = [row.strip('\n') for row in file]
        # Derive the correct fileset of newly processed utterances.
        filesets[set_name + '_initial'] = [
            utt for utt in utterances if utt.strip('abcdef') in raw_fileset
        ]
    # Write the filesets to txt files.
    store_filesets(filesets, corpus='mngu0')
