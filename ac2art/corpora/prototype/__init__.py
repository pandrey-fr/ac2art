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

"""Set of functions creating wrappers prototyping corpora processing.

This submodule provides with wrappers which may be used to easily set
up corpus-specific data processing pipelines. These wrappers are found
in subparts of this submodule, which mirror the structure of corpora'
code folders:

raw        : raw data loaders (prototypes useful to some corpora only)
preprocess : extract and normalize the data, split the corpus
load       : load the extracted (normalized) data in a modular way
abx        : set up and run ABXpy tasks on the corpus's data

An additional `_utils` subpart acts as a dependency to the previous.
"""

from . import utils
from . import raw
from . import preprocess
from . import load
from . import abx
