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

"""Set of generic functions to call ABXpy module scripts.

ABXpy is a Python 2.7 package to conduct ABX discrimination tasks,
developped by the Bootphon team and distributed under MIT license
at https://github.com/bootphon/ABXpy.

Running functions from this submodule requires to have downloaded
a copy of the ABXpy Git repository, installed its dependencies for
python 2.7 and referenced its path under the 'abxpy_folder' key in
this package's 'config.json' file.

At the time of the latest revision of this submodule, the version
of ABXpy was 0.1.0, and the latest commit to the ABXpy repository
was that of SHA 0fe520e09d1bf3f580706f67412890815f2f3c93.
"""

from ._abxpy import (
    abxpy_pipeline, abxpy_task, abxpy_distance, abxpy_score, abxpy_analyze
)
