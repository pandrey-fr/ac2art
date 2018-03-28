# coding: utf-8

"""Set of functions to extract raw mngu0 data."""


from data._prototype.extract import build_features_extraction_functions


MNGU0_DOCSTRING_DETAILS = """
    The default values of the last three arguments echo those used in most
    papers making use of the mngu0 data. The articulators kept correspond
    to the front-to-back and top-to-bottom (resp. py and pz) coordinates of
    six articulators : tongue tip (T1), tongue dorsum (T2), tongue back (T3),
    jaw, upperlip and lowerlip.
"""


MNGU0_DEFAULT_ARTICULATORS = [
    'T3_py', 'T3_pz', 'T2_py', 'T2_pz', 'T1_py', 'T1_pz',
    'jaw_py', 'jaw_pz', 'upperlip_py', 'upperlip_pz',
    'lowerlip_py', 'lowerlip_pz'
]


# Define functions through wrappers; pylint: disable=invalid-name
extract_utterance_data, extract_all_utterances = (
    build_features_extraction_functions(
        dataset='mngu0', initial_sampling_rate=200, default_frames_size=200,
        default_articulators=MNGU0_DEFAULT_ARTICULATORS,
        docstring_details=MNGU0_DOCSTRING_DETAILS
    )
)
