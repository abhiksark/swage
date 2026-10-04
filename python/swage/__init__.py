# python/swage/__init__.py
"""Swage: turn variable-sized dense segments into GPU tile tasks."""

from . import _runtime  # noqa: F401 (samples the process start time)
from ._frontend import CompilationError, jit
from ._segments import segment_reduce, segment_softmax

__version__ = "0.5.2"

__all__ = ["CompilationError", "jit", "segment_reduce", "segment_softmax"]
