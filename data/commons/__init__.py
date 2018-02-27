# coding: utf-8

"""Set of cross-dataset classes used to load acoustic and articulatory data."""

from ._dataloader import AbstractDataLoader
from ._enhance import (
    add_dynamic_features, build_context_windows, build_dynamic_weights_matrix
)
from ._wav import Wav, linear_predictive_coding, lpc_to_lsf, lsf_to_lpc
