# coding: utf-8

"""Set of functions used to build (trajectory) mixture density networks."""

from ._gaussian import gaussian_density, gaussian_mixture_density
from ._mlpg import mlpg_from_gaussian_mixture
from ._weights import build_dynamic_weights_matrix
