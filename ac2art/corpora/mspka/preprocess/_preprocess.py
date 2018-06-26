# coding: utf-8

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
