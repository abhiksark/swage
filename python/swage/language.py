# python/swage/language.py
"""Public markers for the restricted Swage kernel language."""

from enum import Enum


class _ScalarType(Enum):
    """Scalar types accepted by the first frontend slice."""

    FLOAT32 = "float32"
    INT32 = "int32"


class _Marker(Enum):
    """Non-runtime parameter markers."""

    CONSTEXPR = "constexpr"


class _PointerType:
    """Lightweight pointer type descriptor used by kernel signatures."""

    __slots__ = ("element_type",)

    def __init__(self, element_type):
        self.element_type = element_type

    def __eq__(self, other):
        return (
            isinstance(other, _PointerType)
            and self.element_type is other.element_type
        )

    def __hash__(self):
        return hash(self.element_type)

    def __repr__(self):
        return f"pointer({self.element_type.value})"


constexpr = _Marker.CONSTEXPR
float32 = _ScalarType.FLOAT32
int32 = _ScalarType.INT32


def pointer(element_type):
    """Describe a pointer to a scalar element type."""
    return _PointerType(element_type)


def _symbolic_only(name):
    raise RuntimeError(
        f"swage.language.{name} is only available inside @swage.jit kernels"
    )


def program_id(axis):
    """Return a logical program coordinate inside a compiled kernel."""
    _symbolic_only("program_id")


def arange(start, end):
    """Return a compile-time-sized index vector inside a compiled kernel."""
    _symbolic_only("arange")


def load(pointer_value, *, mask=None, other=None):
    """Load a masked vector inside a compiled kernel."""
    _symbolic_only("load")


def store(pointer_value, value, *, mask=None):
    """Store a masked vector inside a compiled kernel."""
    _symbolic_only("store")


__all__ = [
    "arange",
    "constexpr",
    "float32",
    "int32",
    "load",
    "pointer",
    "program_id",
    "store",
]
