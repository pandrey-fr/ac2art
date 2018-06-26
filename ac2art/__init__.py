# coding: utf-8

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
