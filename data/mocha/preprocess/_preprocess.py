# coding: utf-8

"""Set of functions to pre-process raw mocha-timit data."""


from data._prototype.extract import build_features_extraction_functions
from data._prototype.normalize import build_normalization_functions
from data._prototype.split import build_split_corpus


EXTRACTION_DOC_DETAILS = """
    The default values of the last three arguments echo those used in most
    papers making use of the mocha-timit corpus. The articulators kept
    correspond to the front-to-back and top-to-bottom (resp. x and y)
    coordinates of seven articulators : tongue tip (tt), tongue dorsum (td),
    tongue back (tb), lower incisor (li), upperlip (ul), lowerlip (ul)
    and velum (v). Laryngograph data (in one dimension) is also included.
"""


DEFAULT_ARTICULATORS = [
    'tt_x', 'tt_y', 'td_x', 'td_y', 'tb_x', 'tb_y', 'li_x', 'li_y',
    'ul_x', 'ul_y', 'll_x', 'll_y', 'v_x ', 'v_y ', 'larynx'
]


# Define functions through wrappers; pylint: disable=invalid-name
extract_utterance_data, extract_all_utterances = (
    build_features_extraction_functions(
        dataset='mocha', initial_sampling_rate=500,
        default_articulators=DEFAULT_ARTICULATORS,
        docstring_details=EXTRACTION_DOC_DETAILS
    )
)

compute_moments, normalize_files = build_normalization_functions('mocha')

split_corpus = build_split_corpus('mocha', lowest_limit=9)
