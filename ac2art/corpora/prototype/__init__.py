# coding: utf-8

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
