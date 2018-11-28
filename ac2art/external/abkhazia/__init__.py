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

"""Set of functions to interact with Abkhazia (and Kaldi).

Abkhazia is a Python 2.7 package wrapping speech processing
pipelines for ASR taks using Kaldi and ABX tasks using ABXpy,
developped by the Bootphon team and distributed under GPL 3
license at https://github.com/bootphon/abkhazia.

Running functions from this submodule requires to have installed
abkhazia for python 2.7 and its dependencies. Additionally, the
path to Kaldi must be referenced under the 'kaldi_folder' key in
this package's 'config.json' file.

At the time of the latest revision of this submodule, the version
of abkhazia was 0.3, and the latest commit to the git repository
was that of SHA 0d622c7c93d4fc55f79211f1b0ab9bfe06c4ebf1.
"""

from ._files import ark_to_npy, copy_feats, read_ark_file, update_scp
from ._features import compute_mfcc
from ._prepare import prepare_abkhazia_corpus
