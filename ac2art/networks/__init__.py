"""Set of neural networks' generic and specific classes.""" # FIXME: enhance

from._abstract import NeuralNetwork, load_dumped_model
from ._mlp import MultilayerPerceptron
from ._mdn import MixtureDensityNetwork
from ._tmdn import TrajectoryMDN
from ._autoencoder import AutoEncoder
from ._gan import GenerativeAdversarialNets
