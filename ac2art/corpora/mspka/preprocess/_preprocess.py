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

"""Set of functions to pre-process raw mspka data."""


from ac2art.corpora.prototype.preprocess import (
    build_features_extraction_functions,
    build_normalization_functions, build_corpus_splitting_function
)


EXTRACTION_DOC_DETAILS = """
    The default values of the last three arguments echo those used in the
    paper introducing the mspka data. The articulators kept correspond
    to the front-to-back and top-to-bottom (resp. x and z) coordinates of
    six articulators : tongue tip (tt), tongue dorsum (td), tongue back (tb),
    lower incisor (li), upperlip (ul) and lowerlip (ll).
"""


DEFAULT_ARTICULATORS = [
    'tt_x', 'tt_z', 'td_x', 'td_z', 'tb_x', 'tb_z',
    'li_x', 'li_z', 'ul_x', 'ul_z', 'll_x', 'll_z'
]


# Define functions through wrappers; pylint: disable=invalid-name
extract_utterances_data = (
    build_features_extraction_functions(
        corpus='mspka', initial_sampling_rate=400,
        default_articulators=DEFAULT_ARTICULATORS,
        docstring_details=EXTRACTION_DOC_DETAILS
    )
)

compute_moments, normalize_files = build_normalization_functions('mspka')

split_corpus = build_corpus_splitting_function(
    corpus='mspka', lowest_limit=9, same_speaker_data=False
)
# pylint: enable=invalid-name
