# python/swage/__init__.py
"""Swage: turn variable-sized dense segments into GPU tile tasks."""

from ._frontend import CompilationError, jit

__version__ = "0.0.1.dev0"

__all__ = ["CompilationError", "jit"]
