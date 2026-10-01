# python/tests/mlir/test_examples.py
"""Run the committed emit-only example with the native build alone."""

import os
import pathlib
import subprocess
import sys

import pytest

# The source tree holds a `mlir_swage` namespace directory, so the package
# name imports without a build. The `ir` module exists only in a build tree.
native_ir = pytest.importorskip(
    "mlir_swage.ir", reason="the native mlir_swage package is not importable"
)
import swage  # noqa: E402

_EXAMPLE = (
    pathlib.Path(__file__).resolve().parents[3]
    / "examples"
    / "emit_fixed_vector_add.py"
)
# A None entry in sys.modules makes `import torch` raise ImportError, which
# is what an interpreter without PyTorch does.
_RUN_WITHOUT_TORCH = (
    "import runpy, sys\n"
    "sys.modules['torch'] = None\n"
    "runpy.run_path(sys.argv[1], run_name='__main__')\n"
)


def _import_root(module):
    """Return the path entry that one imported module was found under.

    The path is not resolved: a build tree links `mlir_swage/ir.py` to the
    LLVM install, and the resolved location is not the build-tree package.
    """
    return str(pathlib.Path(module.__file__).absolute().parents[1])


def test_emit_only_example_runs_without_a_gpu_or_pytorch():
    """Emit the documented module with no visible device and no PyTorch."""
    environment = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        PYTHONPATH=os.pathsep.join(
            [_import_root(swage), _import_root(native_ir)]
        ),
    )

    completed = subprocess.run(
        [sys.executable, "-c", _RUN_WITHOUT_TORCH, str(_EXAMPLE)],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "func.func @add_kernel" in completed.stdout
    assert "swage.program_id" in completed.stdout
