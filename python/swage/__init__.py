# python/swage/__init__.py
"""Swage: compile fixed vector add or multiply for CPU and CUDA execution."""

from ._errors import BackendUnavailableError, CompilationError, SwageError
from ._frontend import jit

__version__ = "0.5.2"

__all__ = ["BackendUnavailableError", "CompilationError", "SwageError", "jit"]
