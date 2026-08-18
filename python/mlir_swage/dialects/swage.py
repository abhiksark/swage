# python/mlir_swage/dialects/swage.py
"""Python access to the swage dialect: generated ops plus registration."""

from .._mlir_libs._swageDialectsNanobind.swage import *  # noqa: F401,F403
from ._swage_ops_gen import *  # noqa: F401,F403
