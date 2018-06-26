# coding: utf-8

"""Wrappers to help define corpus-specific raw data loading functions.

These are the only wrappers from ac2art.corpora.prototype whose
use is optional when implementing support of a new corpus.

The common API, which implies defining a strict set of raw data
loading functions, must however be respected.
"""

from ._raw import (
    build_ema_loaders, build_utterances_getter, build_voicing_loader
)
