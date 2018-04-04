# coding: utf-8

"""Set of functions to enhance acoustic and articulatory data."""

from ._enhance import (
    add_dynamic_features, build_context_windows,
    build_dynamic_weights_matrix, lowpass_filter
)
