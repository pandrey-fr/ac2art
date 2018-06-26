# coding: utf-8

"""Set of functions to enhance acoustic and articulatory data."""

from ._data_utils import (
    add_dynamic_features,
    build_context_windows,
    build_dynamic_weights_matrix,
    interpolate_missing_values,
    lowpass_filter,
    sequences_to_batch,
    batch_to_sequences
)
