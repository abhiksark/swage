# python/swage/_native.py
"""Lazy native imports and immutable packaged compiler identity."""

import importlib.resources
import json
import re
from typing import NamedTuple

from ._errors import BackendUnavailableError

_REMEDIATION = (
    "install a supported swage-compiler wheel for Linux x86-64, "
    "glibc >=2.28 and CPython 3.10-3.13, or build the pinned LLVM/MLIR "
    "toolchain and add build/python_packages to PYTHONPATH"
)


class _IRBindings(NamedTuple):
    ir: object
    arith: object
    func: object
    swage: object
    vector: object


def _unavailable(backend):
    return BackendUnavailableError(
        "Swage native compiler bindings are unavailable",
        code="native-unavailable",
        backend=backend,
        remediation=_REMEDIATION,
    )


def load_ir(backend="native"):
    """Load the complete frontend binding bundle or a chained import error."""
    try:
        from mlir_swage import ir
        from mlir_swage.dialects import arith, func, swage, vector
    except (ImportError, OSError) as error:
        raise _unavailable(backend) from error
    return _IRBindings(ir, arith, func, swage, vector)


def load_extension(backend="native"):
    """Load the native Swage extension without masking compiler exceptions."""
    try:
        from mlir_swage._mlir_libs._swageDialectsNanobind import swage
    except (ImportError, OSError) as error:
        raise _unavailable(backend) from error
    return swage


def validate_build_info(info):
    """Validate metadata without reflecting its potentially sensitive values."""
    fields = {
        "schema_version",
        "package_version",
        "source_revision",
        "source_clean",
        "llvm_version",
        "build_type",
    }
    if not isinstance(info, dict) or set(info) != fields:
        raise ValueError("invalid native build metadata: schema fields")
    if type(info["schema_version"]) is not int or info["schema_version"] != 1:
        raise ValueError("invalid native build metadata: schema_version")
    for name, pattern in (
        ("package_version", r"[0-9]+\.[0-9]+\.[0-9]+"),
        ("source_revision", r"[0-9a-f]{40}"),
        ("llvm_version", r"llvmorg-[0-9]+\.[0-9]+\.[0-9]+"),
        ("build_type", r"Release|RelWithDebInfo|Debug|MinSizeRel"),
    ):
        if not isinstance(info[name], str) or not re.fullmatch(
            pattern, info[name]
        ):
            raise ValueError(f"invalid native build metadata: {name}")
    if type(info["source_clean"]) is not bool:
        raise ValueError("invalid native build metadata: source_clean")
    return info


def build_info():
    """Return packaged identity, None if absent, or a safe validation error."""
    try:
        package = importlib.resources.files("mlir_swage")
    except ModuleNotFoundError as error:
        if error.name == "mlir_swage":
            return None
        raise ValueError(
            "native build metadata package is unreadable"
        ) from error
    except (ImportError, OSError) as error:
        raise ValueError(
            "native build metadata package is unreadable"
        ) from error
    try:
        text = package.joinpath("_build_info.json").read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError) as error:
        raise ValueError("native build metadata is unreadable") from error
    try:
        info = json.loads(text)
    except (ValueError, RecursionError) as error:
        raise ValueError("invalid native build metadata: JSON") from error
    return validate_build_info(info)
