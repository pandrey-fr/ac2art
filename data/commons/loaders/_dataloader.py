# coding: utf-8

"""Abstract class defining an API for data-loading classes."""

from abc import ABCMeta, abstractmethod
import os

import pandas as pd

from data.utils import check_type_validity


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
