# coding: utf-8

"""Set of utilitarian functions for the data submodule.

Note: some functions implemented here are copied or adapted
      from the YAPTools package, written by the same author
      (https://github.com/pandrey-fr/yaptools/)
"""

import json
import os

import numpy as np
import scipy.interpolate

from utils import check_type_validity


def __load_constants():
    """Load the constants stored in the package's 'config.json' file."""
    path = os.path.join(os.path.dirname(__file__), '..', '..', 'config.json')
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "The 'config.json' file is missing in folder '%s'."
            % os.path.dirname(path)
        )
    with open(path) as file:
        config = json.load(file)
    return config


CONSTANTS = __load_constants()


def interpolate_missing_values(array):
    """Fill NaN values in a 1-D numpy array by cubic spline interpolation."""
    # Check array's type validity.
    check_type_validity(array, np.ndarray, 'array')
    if array.ndim > 1:
        raise TypeError("'array' must be one-dimensional.")
    # Identify NaN values. If there aren't any, simply return the array.
    is_nan = np.isnan(array)
    if is_nan.sum() == 0:
        return array
    array = array.copy()
    not_nan = ~ is_nan
    # Build a cubic spline out of non-NaN values.
    spline = scipy.interpolate.splrep(
        np.argwhere(not_nan).ravel(), array[not_nan], k=3
    )
    # Interpolate missing values and replace them.
    for i in np.argwhere(is_nan).ravel():
        array[i] = scipy.interpolate.splev(i, spline)
    return array
