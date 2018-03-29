# coding: utf-8

"""Set of functions to normalize data already extracted from the mspka data."""

from data._prototype.normalize import build_normalization_functions


# Define functions through wrappers; pylint: disable=invalid-name
compute_files_moments, normalize_files = build_normalization_functions('mspka')
