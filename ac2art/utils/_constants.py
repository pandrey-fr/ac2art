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
    path = os.path.abspath(os.path.join(__file__, '../../../config.json'))
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "Missing configuration file '%s'" % path
        )
    with open(path) as file:
        config = json.load(file)
    return config


CONSTANTS = __load_constants()


def update_constants(**kwargs):
    """Add or update an entry from the config.json file.

    Note : this might require you to be running Python with root rights.
    """
    for key, arg in kwargs.items():
        CONSTANTS[key] = arg
    path = os.path.abspath(os.path.join(__file__, '../../../config.json'))
    with open(path, 'w') as file:
        json.dump(CONSTANTS, file, indent=2, sort_keys=True)


def __add_default_symbols_file():
    """Attempt to fill missing symbols_file enty in config.json."""
    path = os.path.abspath(os.path.join(__file__, '../../../phone_symbols.csv'))
    if not os.path.isfile(path):
        raise FileNotFoundError(
            'Missing symbols_file entry in config.json.\n'
            'Attempt to use default location (below) failed:\n' + path
        )
    update_constants(symbols_file=path)


if 'symbols_file' not in CONSTANTS:
    __add_default_symbols_file()
