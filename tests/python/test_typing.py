# tests/python/test_typing.py
"""Exercise the shipped public stubs without native bindings or PyTorch."""

import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURES = _REPO_ROOT / "tests" / "typing"


def _type_check(fixture, tmp_path):
    config = tmp_path / "mypy.ini"
    config.write_text("[mypy]\n", encoding="utf-8")
    environment = {
        **os.environ,
        "MYPYPATH": os.pathsep.join(
            (str(_REPO_ROOT / "python"), str(_FIXTURES))
        ),
    }
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--strict",
            "--python-version",
            "3.10",
            "--show-error-codes",
            "--no-pretty",
            "--no-error-summary",
            "--no-incremental",
            "--cache-dir",
            str(tmp_path / "mypy-cache"),
            "--config-file",
            str(config),
            str(_FIXTURES / fixture),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_fixed_kernel_supports_both_backends_and_mlir_emission(tmp_path):
    """The constexpr-annotated DSL and CPU/CUDA public calls type-check."""
    result = _type_check("fixed_vector_add.py", tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


def test_unsupported_backend_is_an_argument_type_error(tmp_path):
    """An unsupported backend fails by error category, not exact wording."""
    result = _type_check("invalid_backend.py", tmp_path)
    diagnostics = result.stdout + result.stderr
    assert result.returncode == 1, diagnostics
    errors = [line for line in diagnostics.splitlines() if ": error:" in line]
    assert len(errors) == 1, diagnostics
    assert "invalid_backend.py:" in errors[0], diagnostics
    assert "[arg-type]" in errors[0], diagnostics
