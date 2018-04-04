# coding: utf-8

"""Set of functions to load pre-processed mocha-timit data."""

from data._prototype.load import build_loading_functions


# Define functions through a generic wrapper; pylint: disable=invalid-name
(
    change_loading_setup, get_loading_setup, get_norm_parameters,
    get_utterances, load_acoustic, load_ema, load_utterance, load_dataset
) = build_loading_functions('mocha')
