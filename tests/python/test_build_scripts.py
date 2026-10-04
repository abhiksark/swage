# tests/python/test_build_scripts.py
"""Tests for the interpreter the native build scripts hand to CMake.

The documented commands call the interpreter `python`, so the scripts must
configure the MLIR and Swage Python bindings for that interpreter and fall
back to `python3` only where `python` is absent or too old. Each test runs
the real script with a PATH that holds nothing but stand-ins: `cmake` records
its configure arguments and does nothing for `--build`, and `ninja` does
nothing, so no build starts.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TAG = (_REPO_ROOT / "cmake" / "llvm-version.txt").read_text().strip()
_SCRIPTS = ("build_llvm.sh", "build_swage.sh")
_HOST_TOOLS = ("cat", "dirname")

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None
    or any(shutil.which(tool) is None for tool in _HOST_TOOLS),
    reason="bash, cat, and dirname are required to run the build scripts",
)


def _executable(path, body):
    """Write an executable bash script.

    Args:
        path: Destination of the script.
        body: Commands that follow the interpreter line.
    """
    path.write_text(f"#!{shutil.which('bash')}\n{body}\n", encoding="utf-8")
    path.chmod(0o755)


def _run(script, tmp_path, interpreters):
    """Run a build script with only stand-in tools on PATH.

    Args:
        script: File name of the script under `scripts/`.
        tmp_path: Scratch directory of the test.
        interpreters: Maps an interpreter name to "current" for a link to
            the running interpreter or "old" for one that fails the version
            probe.

    Returns:
        A tuple of the completed process, the directory that is the whole
        PATH, and the arguments `cmake` received, or None if it never ran.
    """
    tools = tmp_path / "bin"
    tools.mkdir()
    for tool in _HOST_TOOLS:
        (tools / tool).symlink_to(shutil.which(tool))
    record = tmp_path / "cmake-arguments.txt"
    # Only the configure call is recorded; `cmake --build` follows it.
    _executable(
        tools / "cmake",
        f'[ "$1" = --build ] || printf "%s\\n" "$@" > "{record}"',
    )
    _executable(tools / "ninja", "exit 0")
    for name, kind in interpreters.items():
        if kind == "current":
            (tools / name).symlink_to(sys.executable)
        else:
            _executable(tools / name, "exit 1")

    llvm_home = tmp_path / "llvm"
    (llvm_home / f"src-{_TAG}").mkdir(parents=True)
    package_dir = llvm_home / f"install-{_TAG}" / "lib" / "cmake" / "mlir"
    package_dir.mkdir(parents=True)
    result = subprocess.run(
        [shutil.which("bash"), str(_REPO_ROOT / "scripts" / script)],
        env={
            "PATH": str(tools),
            "HOME": str(tmp_path),
            "SWAGE_LLVM_HOME": str(llvm_home),
            "SWAGE_BUILD_DIR": str(tmp_path / "build"),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    arguments = None
    if record.exists():
        arguments = record.read_text(encoding="utf-8").splitlines()
    return result, tools, arguments


@pytest.mark.parametrize("script", _SCRIPTS)
def test_script_configures_bindings_for_python(script, tmp_path):
    """Prefer the `python` that the documented commands run."""
    result, tools, arguments = _run(
        script, tmp_path, {"python": "current", "python3": "current"}
    )

    assert result.returncode == 0, result.stderr
    assert f"-DPython3_EXECUTABLE={tools / 'python'}" in arguments
    assert str(tools / "python") in result.stdout


@pytest.mark.parametrize(
    "interpreters",
    [{"python3": "current"}, {"python": "old", "python3": "current"}],
    ids=["python-absent", "python-too-old"],
)
@pytest.mark.parametrize("script", _SCRIPTS)
def test_script_falls_back_to_python3(script, interpreters, tmp_path):
    """Use `python3` where `python` is absent or older than Python 3.10."""
    result, tools, arguments = _run(script, tmp_path, interpreters)

    assert result.returncode == 0, result.stderr
    assert f"-DPython3_EXECUTABLE={tools / 'python3'}" in arguments


@pytest.mark.parametrize("script", _SCRIPTS)
def test_script_stops_without_a_usable_interpreter(script, tmp_path):
    """Name both interpreter names and configure nothing."""
    result, _, arguments = _run(script, tmp_path, {"python": "old"})

    assert result.returncode != 0
    assert arguments is None
    assert "python" in result.stderr
    assert "python3" in result.stderr
    assert "3.10" in result.stderr
