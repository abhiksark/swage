# python/swage/env.py
"""Non-throwing, value-free diagnostics and opt-in backend health checks."""

import argparse
import json
import pathlib
import platform
import sys

import swage

from . import _native

# Release qualification is narrower than the compiler's admitted processors.
_ADMITTED_CUDA_TARGETS = frozenset(
    f"sm_{sm}" for sm in (80, 86, 87, 88, 89, 90, 100, 101, 103, 110, 120, 121)
)
_NATIVE_FIELDS = (
    "package_version",
    "source_revision",
    "source_clean",
    "llvm_version",
    "build_type",
)


def _native_info():
    info = dict.fromkeys(_NATIVE_FIELDS)
    info.update(available=False, error=None)
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
        _native.load_extension()
        info["available"] = True
    except Exception as error:
        code = getattr(error, "code", "native-probe-failed")
        info["error"] = info["error"] or code
    return info


def _torch_info():
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
    try:
        from ._cuda_backend import driver_version

        version = driver_version()
        return version, None if version else "CUDA driver is unavailable"
    except Exception as error:
        return None, f"CUDA driver probe failed ({type(error).__name__})"


def _llvm_pin(native):
    if native["llvm_version"] is not None:
        return native["llvm_version"]
    try:
        pin = pathlib.Path(__file__).resolve().parents[2] / "cmake"
        return (pin / "llvm-version.txt").read_text().strip()
    except (OSError, UnicodeError, IndexError):
        return None


def report() -> dict:
    """Return schema-v1 facts without exposing inputs or hiding failures."""
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
        "schema_version": 1,
        "swage": swage.__version__,
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
