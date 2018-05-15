# coding: utf-8

"""Set of generic functions to call abkhazia module scripts.

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

from . import utils
from ._features import compute_mfcc
from ._prepare import prepare_abkhazia_corpus
