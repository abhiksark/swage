# python/swage/env.py
"""Environment diagnostics for Swage.

Run as a module to print a report::

    python -m swage.env

The report never fails: components that are unavailable are reported as
such instead of raising.
"""

import importlib.util
import pathlib
import platform
import sys

import swage

# Enough of the commit hash to identify a checkout in a bug report.
_REVISION_LENGTH = 12


def _torch_info() -> dict:
    """Collect PyTorch and CUDA facts, degrading gracefully without torch."""
    if importlib.util.find_spec("torch") is None:
        return {
            "torch": None,
            "torch_cuda_build": None,
            "cuda_driver": None,
            "cuda": False,
            "gpu": None,
        }
    import torch

    info = {
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_driver": None,
        "cuda": torch.cuda.is_available(),
        "gpu": None,
    }
    if info["cuda"]:
        from ._runtime import driver_version

        info["cuda_driver"] = driver_version()
        major, minor = torch.cuda.get_device_capability()
        info["gpu"] = {
            "name": torch.cuda.get_device_name(),
            "compute_capability": f"{major}.{minor}",
        }
    return info


def _llvm_pin() -> str | None:
    """Return the pinned LLVM tag when running from a repository checkout."""
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    pin = repo_root / "cmake" / "llvm-version.txt"
    if pin.is_file():
        return pin.read_text().strip()
    return None


def _native_info() -> dict:
    """Probe the build-tree bindings, degrading gracefully without them.

    The probe runs only when a report is requested, never when `swage` is
    imported, so the pure-Python package keeps working without `mlir_swage`.
    """
    try:
        from mlir_swage._mlir_libs._swageDialectsNanobind import (
            swage as native_swage,
        )
    except Exception:
        # Any failure to load the bindings, not only a missing package, is
        # an unavailable backend; the report must describe it, not raise.
        return {"available": False, "llvm_linked": None}
    return {
        "available": True,
        "llvm_linked": getattr(native_swage, "__llvm_version__", None),
    }


def _mlir_backend(native: dict) -> str:
    """Describe the MLIR backend from the native binding probe."""
    if not native["available"]:
        return "unavailable (build-tree mlir_swage bindings not importable)"
    return f"available (linked LLVM {native['llvm_linked'] or 'unknown'})"


def _revision() -> str | None:
    """Return the short checkout HEAD, marked `-dirty` for a modified tree.

    Returns:
        The abbreviated commit hash, with `-dirty` appended when tracked
        files differ from it, or `None` when the package is not running
        from an identifiable git checkout.
    """
    try:
        from . import _runtime

        identity = _runtime._cached_identity()
        revision, clean = identity["revision"], identity["clean"]
    except Exception:
        # The report never fails; an unidentifiable build has no revision.
        return None
    if not revision:
        return None
    short = revision[:_REVISION_LENGTH]
    return short if clean else f"{short}-dirty"


def report() -> dict:
    """Build the full environment report as a dictionary."""
    info = _torch_info()
    native = _native_info()
    return {
        "swage": swage.__version__,
        "revision": _revision(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": info["torch"],
        "torch_cuda_build": info["torch_cuda_build"],
        "cuda_driver": info["cuda_driver"],
        "cuda": info["cuda"],
        "gpu": info["gpu"],
        "llvm_pin": _llvm_pin(),
        "llvm_linked": native["llvm_linked"],
        "backends": {"mlir": _mlir_backend(native)},
    }


def main() -> None:
    """Print the environment report as flat key/value lines."""
    for key, value in report().items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
