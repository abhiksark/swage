# tests/python/test_packaging.py
"""Tests for the published Python distributions."""

import email
import subprocess
import sys
import tarfile
import textwrap
import venv
import zipfile
from pathlib import Path, PurePosixPath

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VERSION = "0.5.0"
_PACKAGE_FILES = {
    "swage/__init__.py",
    "swage/_frontend.py",
    "swage/_runtime.py",
    "swage/_segmented_qualification.py",
    "swage/env.py",
    "swage/language.py",
}
_SDIST_ROOT_ENTRIES = {
    ".gitignore",
    "LICENSE",
    "PKG-INFO",
    "README.md",
    "pyproject.toml",
    "python",
}
_FORBIDDEN_PARTS = {
    ".git",
    ".github",
    "CMakeFiles",
    "build",
    "docs",
    "mlir_swage",
    "test",
    "tests",
}


@pytest.fixture(scope="session")
def distributions(tmp_path_factory):
    """Build one wheel and sdist with the checked-in backend."""
    output = tmp_path_factory.mktemp("distributions")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--no-isolation",
            "--outdir",
            str(output),
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    wheels = list(output.glob("*.whl"))
    sdists = list(output.glob("*.tar.gz"))
    assert len(wheels) == 1
    assert len(sdists) == 1
    return wheels[0], sdists[0]


def test_wheel_contains_only_the_importable_python_package(distributions):
    """Exclude native bindings and repository debris from the wheel."""
    wheel, _ = distributions
    with zipfile.ZipFile(wheel) as archive:
        members = {PurePosixPath(name) for name in archive.namelist()}

    assert _PACKAGE_FILES <= {str(member) for member in members}
    assert {member.parts[0] for member in members} == {
        "swage",
        f"swage_compiler-{_VERSION}.dist-info",
    }
    assert not any(_FORBIDDEN_PARTS & set(member.parts) for member in members)
    assert not any(member.suffix in {".a", ".o", ".so"} for member in members)


def test_sdist_contains_only_distribution_sources(distributions):
    """Exclude build output, native sources, and repository debris."""
    _, sdist = distributions
    with tarfile.open(sdist) as archive:
        members = {
            PurePosixPath(*PurePosixPath(member.name).parts[1:])
            for member in archive.getmembers()
            if len(PurePosixPath(member.name).parts) > 1
        }

    assert {member.parts[0] for member in members} == _SDIST_ROOT_ENTRIES
    assert not any(_FORBIDDEN_PARTS & set(member.parts) for member in members)
    package_members = {
        str(PurePosixPath(*member.parts[1:]))
        for member in members
        if len(member.parts) > 2 and member.parts[:2] == ("python", "swage")
    }
    assert _PACKAGE_FILES == package_members


def test_distribution_metadata_identifies_v050(distributions):
    """Publish matching wheel and sdist metadata for version 0.5.0."""
    wheel, sdist = distributions
    with zipfile.ZipFile(wheel) as archive:
        metadata_name = next(
            name for name in archive.namelist() if name.endswith("/METADATA")
        )
        wheel_metadata = email.message_from_bytes(archive.read(metadata_name))
    with tarfile.open(sdist) as archive:
        metadata_member = next(
            member
            for member in archive.getmembers()
            if member.name.endswith("/PKG-INFO")
        )
        extracted = archive.extractfile(metadata_member)
        assert extracted is not None
        sdist_metadata = email.message_from_bytes(extracted.read())

    for metadata in (wheel_metadata, sdist_metadata):
        assert metadata["Name"] == "swage-compiler"
        assert metadata["Version"] == _VERSION
        assert metadata["Requires-Python"] == ">=3.10"
        assert metadata["Summary"] == (
            "Python-embedded MLIR/LLVM GPU compiler for variable-sized "
            "dense segments"
        )


def test_wheel_clean_install_has_the_expected_native_boundary(
    distributions, tmp_path
):
    """Import without optional dependencies and explain the native boundary."""
    wheel, _ = distributions
    environment = tmp_path / "environment"
    venv.EnvBuilder(with_pip=True).create(environment)
    python = environment / "bin" / "python"
    install = subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            str(wheel),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert install.returncode == 0, install.stderr

    smoke = tmp_path / "smoke.py"
    smoke.write_text(
        textwrap.dedent(
            """
            import sys

            import swage

            assert swage.__version__ == "0.5.0"
            assert "torch" not in sys.modules
            assert "mlir_swage" not in sys.modules

            @swage.jit
            def kernel():
                return

            try:
                kernel.emit_mlir(signature={}, constexprs={})
            except RuntimeError as error:
                assert str(error) == (
                    "Swage emit_mlir() requires the build-tree "
                    "mlir_swage bindings"
                )
            else:
                raise AssertionError("missing native bindings were accepted")
            """
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [str(python), str(smoke)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
