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

"""Pre-installation check-up script for Python 2 dependencies."""

import json
import os
import subprocess
import sys


def _check_python3():
    """Check that Python 3 is being run."""
    if sys.version_info[0] != 3 or sys.version_info[1] < 4:
        raise RuntimeError('ac2art is a Python >= 3.4 package.')


def _check_python2():
    """Check that Python 2 is installed and can be run."""
    if os.system('python2 -V'):
        raise RuntimeError(
            'python2 could not be called. Make sure it is installed '
            'and included in your PATH environment variable.'
        )


def _check_python2_deps():
    """Check mandatory Python 2.7 third-party dependencies."""
    packages = subprocess.check_output(['python2', '-m', 'pip', 'freeze'])
    packages = [
        name.split('==', 1)[0] for name in packages.decode().split('\n')
        if name and name[0].lower() == 'a'
    ]
    missing = [k for k in ('abkhazia', 'ABXpy') if k not in packages]
    if missing:
        raise RuntimeError(
            "Missing mandatory python2 package(s): %s" % missing
        )


def _load_config():
    """Load the config.json file."""
    path = os.path.join(os.path.dirname(__file__), 'config.json')
    if not os.path.isfile(path):
        raise FileNotFoundError("The 'config.json' file is missing.")
    with open(path) as file:
        return json.load(file)


def _check_folders(config):
    """Check that mandatory entries of config.json are filled."""
    for key in ("abxpy_folder", "kaldi_folder"):
        path = config.get(key, None)
        if path is None:
            raise KeyError(
                "Missing entry in the 'config.json' file: %s" % key
            )
        if not os.path.isdir(path):
            raise FileNotFoundError(
                "%s directory not found at specified %s." % (key, path)
            )


def main():
    """Check some key points before installing ac2art."""
    _check_python3()
    _check_python2()
    _check_python2_deps()
    config = _load_config()
    _check_folders(config)


if __name__ == '__main__':
    main()
