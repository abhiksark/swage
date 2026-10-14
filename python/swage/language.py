# python/swage/language.py
"""Public markers for the restricted Swage kernel language."""

from enum import Enum


class _ScalarType(Enum):
    """Scalar types accepted by the first frontend slice."""

    FLOAT32 = "float32"
    FLOAT16 = "float16"
    FLOAT8_E4M3FN = "float8_e4m3fn"
    FLOAT8_E5M2 = "float8_e5m2"
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
float16 = _ScalarType.FLOAT16
float8_e4m3fn = _ScalarType.FLOAT8_E4M3FN
float8_e5m2 = _ScalarType.FLOAT8_E5M2
int32 = _ScalarType.INT32

_FLOAT_TYPES = (float32, float16, float8_e4m3fn, float8_e5m2)


def _torch_float_type(dtype, torch):
    """Map optional PyTorch metadata without importing the dependency."""
    if dtype is torch.float32:
        return float32
    if dtype is not None:
        for scalar_type in _FLOAT_TYPES:
            if dtype is getattr(torch, scalar_type.value, None):
                return scalar_type
    return None


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


def load(pointer_value, *, mask, other):
    """Load a masked vector inside a compiled kernel.

    The kernel language requires both keywords, so neither has a default.
    """
    _symbolic_only("load")


def store(pointer_value, value, *, mask):
    """Store a masked vector inside a compiled kernel.

    The kernel language requires the mask, so it has no default.
    """
    _symbolic_only("store")


__all__ = [
    "arange",
    "constexpr",
    "float16",
    "float8_e4m3fn",
    "float8_e5m2",
    "float32",
    "int32",
    "load",
    "pointer",
    "program_id",
    "store",
]
