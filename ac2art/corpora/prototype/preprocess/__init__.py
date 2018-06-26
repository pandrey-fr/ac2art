# coding: utf-8

"""Wrappers to build corpus-specific data preprocessing functions."""

from ._extract import build_features_extraction_functions
from ._normalize import build_normalization_functions
from ._split import build_corpus_splitting_function, store_filesets
