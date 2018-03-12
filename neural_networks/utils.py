# coding: utf-8

"""Set of utility functions.

Note: some functions implemented here are copied or adapted
      from the YAPTools package, written by the same author
      (https://github.com/pandrey-fr/yaptools/)
"""

import inspect
import functools


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
    if not (inspect.isfunction(func_or_type) or isinstance(func_or_type, type)):
        raise_type_error('func_or_type', ('function', type), type(func_or_type))
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


def instanciate(class_name, init_kwargs, rebuild_init=None):
    """Instanciate an object given a class name and initial arguments.

    The initialization arguments of the object to instanciate may
    be instanciated recursively using the exact same function.

    class_name   : full module and class name of the object (str)
    init_kwargs  : dict of keyword arguments to use for instanciation
    rebuild_init : dict associating dict of `instanciate` arguments to
                   arguments name, used to recursively instanciate any
                   object nececessary to instanciate the main one
    """
    # Check arguments validity.
    check_type_validity(class_name, str, 'class_name')
    check_type_validity(init_kwargs, dict, 'init_kwargs')
    check_type_validity(rebuild_init, (type(None), dict), 'rebuild_init')
    # Optionally instanciate init arguments, recursively.
    if rebuild_init is not None:
        for key, arguments in rebuild_init.items():
            init_kwargs[key] = instanciate(**arguments)
    # Gather the class constructor of the object to instanciate.
    module, name = class_name.rsplit('.', 1)
    constructor = import_from_string(module, name)
    # Instanciate the object and return it.
    return constructor(**init_kwargs)


def onetimemethod(method):
    """Decorator for method which need to be executable only once."""
    if not inspect.isfunction(method):
        raise TypeError('Not a function.')
    has_run = {}
    @functools.wraps(method)
    def wrapped(self, *args, **kwargs):
        """Wrapped method being run once and only once."""
        nonlocal has_run
        if has_run.setdefault(id(self), False):
            raise RuntimeError(
                "One-time method '%s' has already been called for this instance."
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
