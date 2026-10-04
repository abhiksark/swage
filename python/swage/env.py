# python/swage/env.py
"""Non-throwing environment diagnostics and opt-in backend health checks.

Run as a module to print a report::

    python -m swage.env [--json] [--check native|cpu|cuda]

The report never fails: a component that is unavailable is reported with
the reason instead of raising. Only a requested `--check` of an unavailable
component makes the command exit with status 1.
"""

import argparse
import importlib
import json
import pathlib
import platform
import sys

import swage

from . import _native

_SCHEMA_VERSION = 2
# Enough of the commit hash to identify a checkout in a bug report.
_REVISION_LENGTH = 12
_NATIVE_EXTENSION = "mlir_swage._mlir_libs._swageDialectsNanobind"
# The NVPTX processors the compiler admits: the processor list of the target
# description in lib/Target/NVIDIATarget.cpp. Any other target is rejected
# during compilation. Release qualification is narrower.
_ADMITTED_CUDA_TARGETS = frozenset(
    f"sm_{number}"
    for number in (80, 86, 87, 88, 89, 90, 100, 101, 103, 110, 120, 121)
)
_NATIVE_FIELDS = (
    "package_version",
    "source_revision",
    "source_clean",
    "frontend_digest",
    "llvm_version",
    "build_type",
)


def _bindings_info(extension):
    """Describe the loaded bindings and whether they pair with `swage`.

    Returns:
        `version` and `revision`, the `swage` version and source revision
        the bindings were compiled for; `llvm_linked`; `file`, the path of
        the loaded extension; and `problem`, why bindings that load are
        refused, or None.
    """
    from . import _runtime

    info = {
        "version": getattr(extension, "__version__", None),
        "revision": getattr(extension, "__source_revision__", None),
        "llvm_linked": getattr(extension, "__llvm_version__", None),
        "file": None,
        "problem": None,
    }
    try:
        _runtime._verify_bindings(extension)
    except _runtime._BindingsMismatch as error:
        info["problem"] = str(error)
    except Exception as error:
        # The report never fails, also when the check's warning about
        # stale sources is turned into an error.
        info["problem"] = f"bindings probe failed ({type(error).__name__})"
    try:
        # `mlir_swage` is a namespace package with no file of its own, so
        # the loaded extension is what identifies the build.
        info["file"] = getattr(
            importlib.import_module(_NATIVE_EXTENSION), "__file__", None
        )
    except Exception:
        info["file"] = None
    return info


def _native_info():
    """Probe the packaged build record and the native bindings.

    The probe runs only when a report is requested, never when `swage` is
    imported, so the pure-Python package keeps working without `mlir_swage`.
    """
    info = dict.fromkeys(_NATIVE_FIELDS)
    info.update(available=False, error=None, bindings=None)
    try:
        build = _native.build_info()
        if build is not None:
            info.update({key: build[key] for key in _NATIVE_FIELDS})
    except ValueError as error:
        info["error"] = str(error)
    except Exception as error:
        info["error"] = f"native metadata probe failed ({type(error).__name__})"
    try:
        _native.load_ir()
        extension = _native.load_extension()
    except Exception as error:
        code = getattr(error, "code", "native-probe-failed")
        info["error"] = info["error"] or code
        return info
    info["bindings"] = _bindings_info(extension)
    problem = info["bindings"]["problem"]
    info["available"] = problem is None
    if problem is not None:
        info["error"] = info["error"] or "native-mismatch"
    return info


def _torch_info():
    """Collect PyTorch and CUDA facts, degrading gracefully without torch."""
    info = {
        "torch": None,
        "torch_cuda_build": None,
        "cuda": False,
        "gpu": None,
        "target": None,
        "error": None,
    }
    try:
        import torch

        info["torch"] = str(torch.__version__)
        info["torch_cuda_build"] = torch.version.cuda
        info["cuda"] = bool(torch.cuda.is_available())
        if info["cuda"]:
            major, minor = torch.cuda.get_device_capability()
            info["target"] = f"sm_{major}{minor}"
            info["gpu"] = {
                "name": torch.cuda.get_device_name(),
                "compute_capability": f"{major}.{minor}",
            }
    except Exception as error:
        info["error"] = f"PyTorch probe failed ({type(error).__name__})"
    return info


def _driver_info():
    """Return the CUDA driver version and why it is missing, if it is.

    The version comes from `libcuda.so.1`, so it is reported with a
    CPU-only PyTorch and with no PyTorch at all.
    """
    try:
        from ._cuda_backend import driver_version

        version = driver_version()
        return version, None if version else "CUDA driver is unavailable"
    except Exception as error:
        return None, f"CUDA driver probe failed ({type(error).__name__})"


def _llvm_pin(native):
    """Return the LLVM release of the build record or of the checkout."""
    if native["llvm_version"] is not None:
        return native["llvm_version"]
    try:
        pin = pathlib.Path(__file__).resolve().parents[2] / "cmake"
        return (pin / "llvm-version.txt").read_text().strip()
    except (OSError, UnicodeError, IndexError):
        return None


def _revision():
    """Return the short checkout HEAD, marked `-dirty` for a modified tree.

    Returns:
        The abbreviated commit hash, with `-dirty` appended when tracked
        files differ from it, or `None` when the package is not running
        from an identifiable Swage git checkout or build. A copy of the
        package inside another repository has no revision.
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


def _cache_info():
    """Describe the persistent cache as the reporting process would use it.

    Returns:
        `directory`, the cache root; `state`, which reads `active`, `off`,
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
            "directory": None,
            "state": f"unknown ({error})",
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
        "directory": str(status.directory),
        "state": state,
        "compile_on_miss": (
            "allowed" if status.compiles else "refused (SWAGE_NO_COMPILE=1)"
        ),
    }


def _artifact_info():
    """Describe the artifact the reporting process would run from.

    Returns:
        The directory that `SWAGE_ARTIFACT_DIR` selects with its manifest
        format, target, kernels, and origin. It reads `none` when the
        variable is unset and `rejected` with the reason when a segmented
        call would refuse the directory.
    """
    try:
        from . import _artifact

        artifact = _artifact.selected()
    except Exception as error:
        # The report never fails; it states the refusal a call would raise.
        return f"rejected ({error})"
    if artifact is None:
        return f"none ({_artifact._ENVIRONMENT} is unset)"
    manifest = artifact.manifest
    return (
        f"{artifact.directory} (format {manifest['format_version']}, "
        f"target {artifact.target}, {len(manifest['kernels'])} kernels "
        f"of {', '.join(artifact.programs)}, written by swage "
        f"{manifest.get('swage_version')} at revision "
        f"{manifest.get('source_revision')})"
    )


def report() -> dict:
    """Return the schema 2 facts without exposing inputs or hiding failures.

    Schema 2 adds `source` (the file `swage` was imported from and its
    revision), `native.bindings` (the identity of the loaded bindings and
    whether they pair with this `swage`), `cache`, and `artifact` to the
    facts of schema 1, and `native.frontend_digest` to the build record.
    """
    native = _native_info()
    torch = _torch_info()
    driver, driver_error = _driver_info()
    cpu_reasons = []
    if not native["available"]:
        cpu_reasons.append(
            "native-unavailable: install a supported native wheel"
        )
    if torch["torch"] is None:
        cpu_reasons.append(
            "pytorch-unavailable: install swage-compiler[pytorch]"
        )
    cuda_reasons = list(cpu_reasons)
    if not torch["cuda"]:
        cuda_reasons.append(
            "cuda-unavailable: select a CUDA-enabled PyTorch build"
        )
    if torch["error"]:
        cuda_reasons.append(torch["error"])
    if driver_error:
        cuda_reasons.append(driver_error)
    target = torch["target"]
    if torch["cuda"] and target not in _ADMITTED_CUDA_TARGETS:
        cuda_reasons.append(
            "CUDA target is not admitted by the pinned compiler"
        )
    return {
        "schema_version": _SCHEMA_VERSION,
        "swage": swage.__version__,
        "source": {"file": swage.__file__, "revision": _revision()},
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "torch": torch["torch"],
        "torch_cuda_build": torch["torch_cuda_build"],
        "cuda_driver": driver,
        "cuda": torch["cuda"],
        "gpu": torch["gpu"],
        "llvm_pin": _llvm_pin(native),
        "native": native,
        "backends": {
            "cpu": {
                "available": not cpu_reasons,
                "reason": "; ".join(cpu_reasons) or None,
            },
            "cuda": {
                "available": not cuda_reasons,
                "qualified": (
                    target == "sm_86"
                    and torch["gpu"] is not None
                    and torch["gpu"]["name"] == "NVIDIA RTX A6000"
                ),
                "target": target,
                "reason": "; ".join(cuda_reasons) or None,
            },
        },
        "cache": _cache_info(),
        "artifact": _artifact_info(),
    }


def main(argv=None) -> int:
    """Print the complete report; only a requested unavailable check fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print sorted JSON")
    parser.add_argument("--check", choices=("native", "cpu", "cuda"))
    args = parser.parse_args(argv)
    result = report()
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        for key, value in result.items():
            print(f"{key}: {value}")
    if args.check is None:
        return 0
    component = (
        result["native"]
        if args.check == "native"
        else result["backends"][args.check]
    )
    return 0 if component["available"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
