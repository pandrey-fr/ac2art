# coding: utf-8

"""Set of functions to extract raw mspka data."""


from data._prototype.extract import build_features_extraction_functions


MSPKA_DOCSTRING_DETAILS = """
    The default values of the last three arguments echo those used in the
    paper introducing the mspka data. The articulators kept correspond
    to the front-to-back and top-to-bottom (resp. x and z) coordinates of
    six articulators : tongue tip (tt), tongue dorsum (td), tongue back (tb),
    lower incisor (li), upperlip (ul) and lowerlip (ll).
"""


MSPKA_DEFAULT_ARTICULATORS = [
    'tt_x', 'tt_z', 'td_x', 'td_z', 'tb_x', 'tb_z',
    'li_x', 'li_z', 'ul_x', 'ul_z', 'll_x', 'll_z'
]


# Define functions through wrappers; pylint: disable=invalid-name
extract_utterance_data, extract_all_utterances = (
    build_features_extraction_functions(
        dataset='mspka', initial_sampling_rate=400, default_frames_size=200,
        default_articulators=MSPKA_DEFAULT_ARTICULATORS,
        docstring_details=MSPKA_DOCSTRING_DETAILS
    )
)
