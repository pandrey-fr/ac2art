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

"""Abstract class defining an API for data-loading classes.

Note: as of the last restructuration of this package, this
      class is only inherited from by a single class; it
      may however be useful to handle additional corpora,
      such as the Usc Timit one.
"""

from abc import ABCMeta, abstractmethod
import os

import pandas as pd

from ac2art.utils import check_type_validity


class AbstractDataLoader(metaclass=ABCMeta):
    """Abstract class to load data arrays from corpus-specific files.

    This abstract class defines a basic initialization procedure
    as well as a `to_dataframe` method returning a conveniently
    labeled `pandas.DataFrame` containing the data.

    Each inheriting class should override the `load` method so as
    to define the data loading procedure, which should fill in the
    `data`, `column_names` and `time_index` attributes.
    """

    def __init__(self, filename):
        """Initialize the instance.

        filename : path to the target file (str)
        """
        check_type_validity(filename, str, 'filename')
        # Set up the instance's attributes.
        self.filename = os.path.abspath(filename)
        self.data = None
        self.column_names = {}
        self.time_index = None
        # Load the attributes' values from the file.
        self.load()

    def to_dataframe(self):
        """Return the data as a pandas.DataFrame."""
        return pd.DataFrame(
            self.data, index=self.time_index,
            columns=list(self.column_names.values())
        )

    @abstractmethod
    def load(self):
        """Load the data from file."""
        raise NotImplementedError('This abstract method needs overriding.')
