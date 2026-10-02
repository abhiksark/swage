# python/swage/env.py
"""Environment diagnostics for Swage.

Run as a module to print a report::

    python -m swage.env

The report never fails: components that are unavailable are reported as
such instead of raising.
"""

import importlib
import importlib.util
import pathlib
import platform
import sys

import swage

# Enough of the commit hash to identify a checkout in a bug report.
_REVISION_LENGTH = 12
_NATIVE_EXTENSION = "mlir_swage._mlir_libs._swageDialectsNanobind"
# The NVPTX processors the compiler admits: the processor list of the target
# description in lib/Target/NVIDIATarget.cpp. Any other target is rejected
# during compilation.
_ADMITTED_TARGETS = frozenset(
    f"sm_{number}"
    for number in (80, 86, 87, 88, 89, 90, 100, 101, 103, 110, 120, 121)
)
# The admitted targets that the GPU test tier executes on. README.md and
# ROADMAP.md record the evidence, which comes from one NVIDIA RTX A6000.
_QUALIFIED_TARGETS = frozenset({"sm_86"})


def _target(major: int, minor: int) -> str:
    """Name the target of a compute capability and its qualification.

    Returns:
        The `sm_` target followed by `(qualified)` when the GPU tests
        execute on it, `(admitted, not qualified)` when it compiles with no
        execution evidence, or `(not admitted)` when compilation rejects it.
    """
    target = f"sm_{major}{minor}"
    if target in _QUALIFIED_TARGETS:
        return f"{target} (qualified)"
    if target in _ADMITTED_TARGETS:
        return f"{target} (admitted, not qualified)"
    return f"{target} (not admitted)"


def _torch_info() -> dict:
    """Collect PyTorch and CUDA facts, degrading gracefully without torch."""
    if importlib.util.find_spec("torch") is None:
        return {
            "torch": None,
            "torch_cuda_build": None,
            "cuda": False,
            "gpu": None,
            "target": None,
        }
    import torch

    info = {
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda": torch.cuda.is_available(),
        "gpu": None,
        "target": None,
    }
    if info["cuda"]:
        major, minor = torch.cuda.get_device_capability()
        info["gpu"] = {
            "name": torch.cuda.get_device_name(),
            "compute_capability": f"{major}.{minor}",
        }
        info["target"] = _target(major, minor)
    return info


def _cuda_driver() -> str | None:
    """Return the CUDA driver version, or `None` when it cannot be read.

    The version comes from `libcuda.so.1`, so it is reported with a
    CPU-only PyTorch and with no PyTorch at all.
    """
    try:
        from . import _runtime

        return _runtime.driver_version()
    except Exception:
        # The report never fails; an unreadable driver has no version.
        return None


def _llvm_pin() -> str | None:
    """Return the pinned LLVM tag when running from a repository checkout."""
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    pin = repo_root / "cmake" / "llvm-version.txt"
    if pin.is_file():
        return pin.read_text().strip()
    return None


def _native_info() -> dict:
    """Probe the native bindings, degrading gracefully without them.

    The probe runs only when a report is requested, never when `swage` is
    imported, so the pure-Python package keeps working without `mlir_swage`.

    Returns:
        `available`, whether the bindings load and match this `swage`;
        `problem`, why bindings that load are refused, or None; `version`
        and `revision`, the `swage` version and source revision the
        bindings were built from; `llvm_linked`; and `file`, the path of
        the loaded extension. Unknown values are None.
    """
    missing = {
        "available": False,
        "problem": None,
        "version": None,
        "revision": None,
        "llvm_linked": None,
        "file": None,
    }
    from . import _runtime

    try:
        native_swage = _runtime._native_bindings()
        extension = importlib.import_module(_NATIVE_EXTENSION)
    except _runtime._BindingsMismatch as error:
        return {**missing, "problem": str(error)}
    except Exception:
        # Any other failure to load the bindings, not only a missing
        # package, is an unavailable backend; the report must describe it,
        # not raise.
        return missing
    return {
        "available": True,
        "problem": None,
        "version": getattr(native_swage, "__version__", None),
        "revision": getattr(native_swage, "__source_revision__", None),
        "llvm_linked": getattr(native_swage, "__llvm_version__", None),
        # `mlir_swage` is a namespace package with no file of its own, so
        # the loaded extension is what identifies the build.
        "file": getattr(extension, "__file__", None),
    }


def _mlir_backend(native: dict) -> str:
    """Describe the MLIR backend from the native binding probe."""
    if native["problem"] is not None:
        return f"rejected ({native['problem']})"
    if not native["available"]:
        return "unavailable (mlir_swage bindings not importable)"
    return f"available (linked LLVM {native['llvm_linked'] or 'unknown'})"


def _revision() -> str | None:
    """Return the short checkout HEAD, marked `-dirty` for a modified tree.

    Returns:
        The abbreviated commit hash, with `-dirty` appended when tracked
        files differ from it, or `None` when the package is not running
        from an identifiable Swage git checkout. A copy of the package
        inside another repository has no revision.
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


def _cache_info() -> dict:
    """Describe the persistent cache as the reporting process would use it.

    Returns:
        `cache_dir`, the cache root; `cache`, which reads `active`, `off`,
        or `rejected` with the mode or the reason in parentheses; and
        `compile_on_miss`, which says whether a kernel that is not cached is
        compiled. A cache variable that a launch would reject is reported
        as `unknown` with the error.
    """
    try:
        from . import _runtime

        status = _runtime._cache_status()
    except Exception as error:
        # The report never fails; it names the setting a launch rejects.
        return {
            "cache_dir": None,
            "cache": f"unknown ({error})",
            "compile_on_miss": None,
        }
    if status.rejected:
        state = f"rejected ({status.problem})"
    elif status.problem is not None:
        state = f"off ({status.problem})"
    elif status.writes:
        state = (
            f"active (reads and writes; {status.entries} of at most "
            f"{status.max_entries} entries)"
        )
    else:
        state = f"active (reads only; {status.entries} entries)"
    return {
        "cache_dir": str(status.directory),
        "cache": state,
        "compile_on_miss": (
            "allowed" if status.compiles else "refused (SWAGE_NO_COMPILE=1)"
        ),
    }


def report() -> dict:
    """Build the full environment report as a dictionary."""
    info = _torch_info()
    native = _native_info()
    return {
        "swage": swage.__version__,
        "revision": _revision(),
        "swage_file": swage.__file__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": info["torch"],
        "torch_cuda_build": info["torch_cuda_build"],
        "cuda_driver": _cuda_driver(),
        "cuda": info["cuda"],
        "gpu": info["gpu"],
        "target": info["target"],
        "llvm_pin": _llvm_pin(),
        "llvm_linked": native["llvm_linked"],
        "native_version": native["version"],
        "native_revision": native["revision"],
        "mlir_swage_file": native["file"],
        "backends": {"mlir": _mlir_backend(native)},
        **_cache_info(),
    }


def main() -> None:
    """Print the environment report as flat key/value lines."""
    for key, value in report().items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
