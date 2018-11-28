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

"""Set of tensorflow-related utility functions."""

from ._tf_utils import (
    add_dynamic_features,
    batch_tensor_mean,
    binary_step,
    conv2d,
    get_activation_function_name,
    get_delta_features,
    get_simple_difference,
    get_rnn_cell_type_name,
    index_tensor,
    log_base,
    minimize_safely,
    reduce_finite_mean,
    run_along_first_dim,
    setup_activation_function,
    setup_rnn_cell_type,
    sinc,
    tensor_length
)
