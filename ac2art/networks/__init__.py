"""Set of classes implementing various types of neural networks.
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


The implemented classes are the following:

* `NeuralNetwork` is an abstract class defining a common API and
implementing key bricks to build neural networks. Details on the
API are provided below in this document.


* `MultilayerPerceptron` is the base class for end-to-end models ;
despite its name, this class allows to use recurrent unit stacks
(RNN) as part of the designed models' hidden layers.


* `MixtureDensityNetwork` is a class of networks that output the
parameters to a gaussian mixture density function instead of a
single prediction; it however disposes of a method to derive a
prediction out of the predicted parameters allowing its use as
an end-to-end model.


* `TrajectoryMDN` is a mixture density network which uses a more
complex prediction function than the one implemented in the base
class, which supposedly should take time dependencies into account
when run over a time sequence of data points.

_Note: the current implementation is unexpectedly unstable, and
has never yielded satisfactory performance in regard with what is
reported in the litterature, although it seems to be a correct
as to the algorithms used._


* `AutoEncoder` models could be called 'end-to-end-to-initial-end',
as they consist of two stacked neural networks that respectively
learn to (1) derive a given representation of the input data and
(2) rebuild the inputs based on the produced representation.


* `GenerativeAdversarialNets` implements a specific way to train
an end-to-end (`MultilayerPerceptron`-like) model, by penalizing
its loss function with the ability of another network to separate
its outputs from the true outputs associated with the input data.
Both models are trained jointly, and therefore against each other.

_Note: the current implementation is not satisfactory and should
be revised, noticeably as to the combining of the two networks'
loss functions._


Additionally, the `load_dumped_model` function allows to restore
any model previously dumped to a .npy file using the `save_model`
which is inherited from the abstract `NeuralNetwork` class.
"""

from._abstract import NeuralNetwork, load_dumped_model
from ._mlp import MultilayerPerceptron
from ._mdn import MixtureDensityNetwork
from ._tmdn import TrajectoryMDN
from ._autoencoder import AutoEncoder
from ._gan import GenerativeAdversarialNets
