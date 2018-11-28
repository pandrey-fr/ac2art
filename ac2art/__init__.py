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

"""ac2art - acoustic-to-articulatory inversion using neural networks.

This package implements tools to set up neural networks for learning
acousitc-to-articulatory inversion tasks, process data corpora used
to train and evaluate such networks, and run the inversion task on
additional data.

Users should focus on the `corpora` and `networks` submodules, which
respectively implement processing pipelines for corpora of parallel
acoustic and articulatory data recordings, and neural networks of
various and modular architecture that may learn the inversion task.

The `run_inversion` function may also be used to invert pre-computed
acoustic features using a pre-trained inverter. The generation of the
acoustic features may also be assisted by functions implemented under
the `ac2art.external.abkhazia` and `ac2art.internal.data_loaders`
submodules.
"""

from . import utils
from . import external
from . import internal
from . import corpora
from . import networks
from ._invert import run_inversion
