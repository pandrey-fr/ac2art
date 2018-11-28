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

"""Set of functions to run ABXpy tasks on mocha-timit data."""

from ac2art.corpora.prototype.abx import (
    build_h5features_extractor, build_abxpy_callers
)


# Define functions through wrappers; pylint: disable=invalid-name
extract_h5_features = build_h5features_extractor('mocha')

abx_from_features, make_abx_task, make_itemfile, load_abx_scores = (
    build_abxpy_callers('mocha')
)
