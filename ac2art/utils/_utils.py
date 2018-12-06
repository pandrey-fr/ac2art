# coding: utf-8
#
# Copyright 2018 Paul Andrey
#
# This file is part of ac2art.
#
# ac2art is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ac2art is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ac2art.  If not, see <http://www.gnu.org/licenses/>.
#
# Some code from this specific file is copied or adapted from yaptools,
# a Python package from the same author, distributed under the MIT license.

"""Set of broadly used and generic utilitarian classes.

Note: some functions implemented here are copied or adapted
      from the YAPTools package, written by the same author
      (https://github.com/pandrey-fr/yaptools/)
"""

import re
import inspect
import functools


def alphanum_sort(string_list):
    """Sort a list of strings using the alphanum algorithm.
    Dave Koelle's Alphanum algorithm sorts names containing integer
    in a more human-intuitive way than the usual Ascii-based way.
    E.g. sorting ['2', '1', '10'] results in ['1', '2', '10'],
         whereas built-in sorted() results in ['1', '10', '2'].
    """
    check_type_validity(string_list, list, 'string_list')
    if not all(isinstance(string, str) for string in string_list):
        raise TypeError('The provided list contains non-string elements.')
    return sorted(string_list, key=_alphanum_key)


def _alphanum_key(string):
    """Parse a string into string and integer components."""
    parity = int(string[0] in '0123456789')
    return [
        int(x) if i % 2 != parity else x
        for i, x in enumerate(re.split('([0-9]+)', string)[parity:])
    ]

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


def get_object(name, reference_dict, object_name):
    """Return a function or type of given name.

    name           : either a valid key to the `reference_dict` or
                     the full import name of the object to return
    reference_dict : dict associating return values to str keys
    object_name    : string designating the kind of object being
                     reached, used to disambiguate error messages
    """
    check_type_validity(name, str, 'name')
    check_type_validity(reference_dict, dict, 'reference_dict')
    check_type_validity(object_name, str, 'object_name')
    if name not in reference_dict.keys():
        if not name.count('.'):
            raise KeyError(
                "Invalid %s name: '%s'.\n" % (object_name, name)
                + "A valid name should either belong to {'%s'} or "
                "consist of a full module name and function name."
                % "', '".join(list(reference_dict))
            )
        module_name, name = name.rsplit('.', 1)
        return import_from_string(module_name, name)
    return reference_dict[name]


def get_object_name(func_or_type, reference_dict):
    """Return the name of a given function or type.

    If the function or type belongs to a reference dict's values,
    return the key to which it is associated ; otherwise, return
    its full import path.

    func_or_type   : the function or type whose name to return
    reference_dict : a dict associating some expected functions
                     or types to str keys
    """
    valid = inspect.isfunction(func_or_type) or isinstance(func_or_type, type)
    if not valid:
        raise_type_error(
            'func_or_type', ('function', type), type(func_or_type)
        )
    check_type_validity(reference_dict, dict, 'reference_dict')
    keys = list(reference_dict.values())
    if func_or_type not in keys:
        return func_or_type.__module__ + '.' + func_or_type.__name__
    return list(reference_dict.keys())[keys.index(func_or_type)]


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


def instantiate(class_name, init_kwargs, rebuild_init=None):
    """Instantiate an object given a class name and initial arguments.

    The initialization arguments of the object to instantiate may
    be instantiated recursively using the exact same function.

    class_name   : full module and class name of the object (str)
    init_kwargs  : dict of keyword arguments to use for instanciation
    rebuild_init : dict associating dict of `instantiate` arguments to
                   arguments name, used to recursively instantiate any
                   object nececessary to instantiate the main one
    """
    # Check arguments validity.
    check_type_validity(class_name, str, 'class_name')
    check_type_validity(init_kwargs, dict, 'init_kwargs')
    check_type_validity(rebuild_init, (type(None), dict), 'rebuild_init')
    # Optionally instantiate init arguments, recursively.
    if rebuild_init is not None:
        for key, arguments in rebuild_init.items():
            init_kwargs[key] = instantiate(**arguments)
    # Gather the class constructor of the object to instantiate.
    module, name = class_name.rsplit('.', 1)
    constructor = import_from_string(module, name)
    # Instantiate the object and return it.
    return constructor(**init_kwargs)


def onetimemethod(method):
    """Decorator for methods which need to be executable only once."""
    if not inspect.isfunction(method):
        raise TypeError('Not a function.')
    has_run = {}
    @functools.wraps(method)
    def wrapped(self, *args, **kwargs):
        """Wrapped method being run once and only once."""
        nonlocal has_run
        if has_run.setdefault(id(self), False):
            raise RuntimeError(
                "One-time method '%s' cannot be re-run for this instance."
                % method.__name__
            )
        has_run[id(self)] = True
        return method(self, *args, **kwargs)
    return wrapped


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
