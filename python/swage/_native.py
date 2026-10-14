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
    """Validate metadata without reflecting its potentially sensitive values.

    Schema 2 records the build of the native package: the `swage` version,
    the source revision (40 hex digits, or null for a build from sources
    without git), whether the sources were clean, the digest of the Python
    frontend the build packaged, the LLVM release, and the build type. The
    compiled-in identity of the extension spells an unknown revision
    `unknown` and a dirty one with a `-dirty` suffix; this record keeps the
    plain revision and `source_clean` apart instead.
    """
    fields = {
        "schema_version",
        "package_version",
        "source_revision",
        "source_clean",
        "frontend_digest",
        "llvm_version",
        "build_type",
    }
    if not isinstance(info, dict) or set(info) != fields:
        raise ValueError("invalid native build metadata: schema fields")
    if type(info["schema_version"]) is not int or info["schema_version"] != 2:
        raise ValueError("invalid native build metadata: schema_version")
    for name, pattern in (
        ("package_version", r"[0-9]+\.[0-9]+\.[0-9]+"),
        ("frontend_digest", r"[0-9a-f]{64}"),
        ("llvm_version", r"llvmorg-[0-9]+\.[0-9]+\.[0-9]+"),
        ("build_type", r"Release|RelWithDebInfo|Debug|MinSizeRel"),
    ):
        if not isinstance(info[name], str) or not re.fullmatch(
            pattern, info[name]
        ):
            raise ValueError(f"invalid native build metadata: {name}")
    revision = info["source_revision"]
    if revision is not None and (
        not isinstance(revision, str)
        or not re.fullmatch(r"[0-9a-f]{40}", revision)
    ):
        raise ValueError("invalid native build metadata: source_revision")
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
