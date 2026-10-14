# tests/python/test_qualify_installed_segments.py
"""Refusals of the installed-wheel segmented qualification script.

The qualification itself needs a CUDA GPU and an installed wheel, and runs
in the release and GPU workflows. These tests pin what the script refuses
before it copies or runs anything.
"""

import pathlib
import subprocess
import sys

import pytest

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[2]
    / "scripts"
    / "qualify_installed_segments.sh"
)


def _run(*arguments):
    return subprocess.run(
        ["bash", str(_SCRIPT), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )


def test_the_script_takes_three_arguments():
    """Print the usage for any other number of arguments."""
    result = _run(sys.executable)
    assert result.returncode == 2
    assert "usage:" in result.stderr


@pytest.mark.parametrize("missing", ["CMakeCache.txt", "bin/swage-opt"])
def test_an_oracle_build_without_its_tools_is_refused(tmp_path, missing):
    """Refuse a build directory that lacks what the CPU oracle runs."""
    build = tmp_path / "build"
    (build / "bin").mkdir(parents=True)
    for name in ("CMakeCache.txt", "bin/swage-opt"):
        if name != missing:
            (build / name).write_text("")
    work = tmp_path / "work"

    result = _run(sys.executable, str(build), str(work))

    assert result.returncode == 2
    assert f"lacks {missing}" in result.stderr
    assert not work.exists()


def test_an_existing_work_directory_is_refused(tmp_path):
    """Never write the copy or the evidence into a directory that exists."""
    build = tmp_path / "build"
    (build / "bin").mkdir(parents=True)
    (build / "CMakeCache.txt").write_text("")
    (build / "bin" / "swage-opt").write_text("")
    work = tmp_path / "work"
    work.mkdir()

    result = _run(sys.executable, str(build), str(work))

    assert result.returncode == 2
    assert "exists" in result.stderr
    assert list(work.iterdir()) == []
