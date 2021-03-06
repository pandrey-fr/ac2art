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

"""Set of functions to load pre-processed mspka data."""

from ac2art.corpora.prototype.load import build_loading_functions


# Define functions through a generic wrapper; pylint: disable=invalid-name
(
    change_loading_setup, get_loading_setup, get_norm_parameters,
    get_utterances, load_acoustic, load_ema, load_utterance, load_dataset
) = build_loading_functions('mspka', default_byspeaker=True)
