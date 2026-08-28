"""GONNECT: biologically-informed autoencoders constrained by the Gene Ontology.

The names re-exported here are the ones the experiment scripts under ``src/`` and
the figure pipeline build on. Everything else stays reachable at its full path,
e.g. ``from gonnect.data_processing.go_preprocessing import construct_go_bp``.
"""

from importlib.metadata import PackageNotFoundError, version

from gonnect.model.Autoencoder import Autoencoder
from gonnect.model.build_model import build_model
from gonnect.train.loss import (
    MSE,
    MSE_L1,
    MSE_Masked,
    MSE_Soft_Link_Proxyless,
    MSE_Soft_Link_Sum,
)
from gonnect.train.train import (
    make_data_splits,
    split_data,
    test,
    train,
    train_with_validation,
)

try:
    __version__ = version("gonnect")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0.0.0.dev0"

__all__ = [
    "Autoencoder",
    "build_model",
    "MSE",
    "MSE_L1",
    "MSE_Masked",
    "MSE_Soft_Link_Proxyless",
    "MSE_Soft_Link_Sum",
    "make_data_splits",
    "split_data",
    "test",
    "train",
    "train_with_validation",
    "__version__",
]
