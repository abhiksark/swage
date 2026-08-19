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


def report() -> dict:
    """Build the full environment report as a dictionary."""
    info = _torch_info()
    return {
        "swage": swage.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": info["torch"],
        "torch_cuda_build": info["torch_cuda_build"],
        "cuda_driver": info["cuda_driver"],
        "cuda": info["cuda"],
        "gpu": info["gpu"],
        "llvm_pin": _llvm_pin(),
        "backends": {
            # No backend can execute kernels yet; say so instead of guessing.
            "mlir": "unavailable (native components not built into the "
            "Python package yet)",
        },
    }


def main() -> None:
    """Print the environment report as flat key/value lines."""
    for key, value in report().items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
