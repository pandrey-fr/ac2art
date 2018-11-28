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
