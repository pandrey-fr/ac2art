# coding: utf-8

"""Set of functions to normalize data already extracted from the mngu0 data."""

from data._prototype.normalize import build_normalization_functions


# Define functions through wrappers; pylint: disable=invalid-name
compute_moments, normalize_files = build_normalization_functions('mngu0')
