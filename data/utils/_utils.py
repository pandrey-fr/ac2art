# coding: utf-8

"""Set of utilitarian classes.

Note: some functions implemented here are copied or adapted
      from the YAPTools package, written by the same author
      (https://github.com/pandrey-fr/yaptools/)
"""

import json
import os

import numpy as np
import scipy.interpolate


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


def check_batch_type(valid_type, **kwargs):
    """Check that a batch of keyword arguments are of a given type."""
    invalid = [
        name for name, arg in kwargs.items() if not isinstance(arg, valid_type)
    ]
    if invalid:
        raise TypeError(
            "Invalid argument%s: '%s'. Should be of type %s."
            % ('s' * (len(invalid) > 1), ', '.join(invalid), valid_type)
        )


def check_positive_int(instance, var_name):
    """Check that a given variable is a positive integer."""
    check_type_validity(instance, int, var_name)
    if instance <= 0:
        raise ValueError("'%s' must be positive." % var_name)


def check_type_validity(instance, valid_types, var_name):
    """Raise a TypeError if a given variable instance is not of expected type.

    instance    : instance whose type to check
    valid_types : expected type (or tuple of types)
    var_name    : variable name to use in the exception's message
    """
    if isinstance(valid_types, type):
        valid_types = (valid_types,)
    elif not isinstance(valid_types, tuple):
        raise AssertionError("Invalid 'valid_types' argument.")
    if float in valid_types and int not in valid_types:
        valid_types = (*valid_types, int)
    if not isinstance(instance, valid_types):
        raise_type_error(var_name, valid_types, type(instance).__name__)


def import_from_string(module, elements):
    """Import and return elements from a module based on their names.

    module   : name of the module from which to import elements (str)
    elements : name or list of names of the elements to import
    """
    check_type_validity(module, str, 'module_name')
    check_type_validity(elements, (list, str), elements)
    lib = __import__(module, fromlist=module)
    if isinstance(elements, str):
        return getattr(lib, elements)
    return tuple(getattr(lib, element) for element in elements)


def raise_type_error(var_name, valid_types, var_type):
    """Raise a custom TypeError.

    var_name    : name of the variable causing the exception (str)
    valid_types : tuple of types or type names to list as valid options
    var_type    : type of the variable causing the exception (type or str)
    """
    valid_names = [
        str(getattr(valid, '__name__', valid)) for valid in valid_types
    ]
    names_string = (
        valid_names[0] if len(valid_names) == 1
        else ', '.join(valid_names[:-1]) + ' or ' + valid_names[-1]
    )
    raise TypeError(
        "Expected '%s' to be of type %s, not %s."
        % (var_name, names_string, getattr(var_type, '__name__', var_type))
    )


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
