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

"""Set of broadly used and generic utilitarian classes.

Note: most functions implemented here are copied or adapted
      from the YAPTools package, written by the same author
      (https://github.com/pandrey-fr/yaptools/)
"""

from ._utils import (
    alphanum_sort, check_batch_type, check_positive_int,
    check_type_validity, get_object, get_object_name, import_from_string,
    instantiate, onetimemethod, raise_type_error
)
from ._constants import CONSTANTS, update_constants
