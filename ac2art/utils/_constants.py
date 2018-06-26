# coding: utf-8

"""Load the constants stored in the package's config.json file."""

import json
import os


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
