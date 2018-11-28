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
    if 'symbols_file' not in config:
        __add_symbols_file(config, path)
    return config


def __add_symbols_file(config, path):
    """Attempt to fill missing symbols_file enty in config.json."""
    symbols_file = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..', 'phone_symbols.csv'
    ))
    if not os.path.isfile(symbols_file):
        raise FileNotFoundError(
            'Missing symbols_file entry in config.json.\n'
            'Attempt to use default location (below) failed:\n'
            + symbols_file
        )
    config['symbols_file'] = symbols_file
    with open(path, 'w') as file:
        json.dump(config, file, indent=2, sort_keys=True)


CONSTANTS = __load_constants()
