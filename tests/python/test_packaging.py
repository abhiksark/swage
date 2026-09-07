# tests/python/test_packaging.py
"""Check source artifacts without requiring LLVM or a native compiler."""

import email
import hashlib
import os
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VERSION = "0.5.2"
_SOURCE_DATE_EPOCH = "1700000000"


def _build_sdist(source, output):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--sdist",
            "--no-isolation",
            "--outdir",
            str(output),
        ],
        cwd=source,
        env={**os.environ, "SOURCE_DATE_EPOCH": _SOURCE_DATE_EPOCH},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    sdists = list(output.glob("*.tar.gz"))
    assert len(sdists) == 1
    assert not list(output.glob("*.whl"))
    return sdists[0]


@pytest.fixture(scope="session")
def sdist(tmp_path_factory):
    """Build only the sdist with the installed scikit-build-core backend."""
    return _build_sdist(_REPO_ROOT, tmp_path_factory.mktemp("sdist"))


@pytest.fixture(scope="session")
def source_members(sdist):
    """Read regular source files beneath a single safe archive root."""
    files = {}
    roots = set()
    with tarfile.open(sdist) as archive:
        for member in archive.getmembers():
            path = PurePosixPath(member.name)
            assert not path.is_absolute()
            assert ".." not in path.parts
            roots.add(path.parts[0])
            if member.isdir():
                continue
            assert member.isfile(), member.name
            assert len(path.parts) > 1, member.name
            relative = PurePosixPath(*path.parts[1:])
            assert relative not in files, member.name
            extracted = archive.extractfile(member)
            assert extracted is not None
            files[relative] = (extracted.read(), member.mode)
    assert len(roots) == 1
    return files


def test_sdist_release_metadata(source_members):
    """Advertise the supported release, Python range, platform, and extra."""
    metadata = email.message_from_bytes(
        source_members[PurePosixPath("PKG-INFO")][0]
    )
    assert metadata["Name"] == "swage-compiler"
    assert metadata["Version"] == _VERSION
    assert SpecifierSet(metadata["Requires-Python"]) == SpecifierSet(
        ">=3.10,<3.14"
    )
    classifiers = set(metadata.get_all("Classifier", []))
    assert "Development Status :: 4 - Beta" in classifiers
    assert {
        value for value in classifiers if value.startswith("Operating System")
    } == {"Operating System :: POSIX :: Linux"}
    for version in ("3.10", "3.11", "3.12", "3.13"):
        assert f"Programming Language :: Python :: {version}" in classifiers

    requirements = [
        Requirement(value) for value in metadata.get_all("Requires-Dist", [])
    ]
    torch_requirements = [item for item in requirements if item.name == "torch"]
    assert len(torch_requirements) == 1
    torch = torch_requirements[0]
    assert torch.specifier == SpecifierSet(">=2.6,<3")
    assert torch.marker is not None
    assert torch.marker.evaluate({"extra": "pytorch"})
    assert not torch.marker.evaluate({"extra": ""})
    assert "pytorch" in metadata.get_all("Provides-Extra", [])

    assert metadata["License-Expression"] == (
        "MIT AND Apache-2.0 WITH LLVM-exception"
    )
    assert {"LICENSE", "LICENSES/LLVM.txt"} <= set(
        metadata.get_all("License-File", [])
    )
    for license_path in ("LICENSE", "LICENSES/LLVM.txt"):
        assert source_members[PurePosixPath(license_path)][0] == (
            _REPO_ROOT / license_path
        ).read_bytes()


def test_sdist_preserves_source_build_resources(source_members):
    """Retain build inputs and qualification sources, not a frozen inventory."""
    # These are source-build entry points, not an exhaustive package file list.
    entry_points = (
        "CMakeLists.txt",
        "cmake/llvm-version.txt",
        "pyproject.toml",
        "README.md",
        "python/CMakeLists.txt",
        "python/mlir_swage/_build_info.json.in",
        "python/swage/py.typed",
        "python/swage/__init__.pyi",
        "python/swage/language.pyi",
        "scripts/build_llvm.sh",
        "scripts/build_swage.sh",
    )
    for name in entry_points:
        assert source_members[PurePosixPath(name)][0] == (
            _REPO_ROOT / name
        ).read_bytes()

    # Discover current build inputs so adding or renaming sources needs no
    # test inventory update. Missing a source category must still fail.
    resource_patterns = {
        "cmake": ("*.txt", "*.cmake"),
        "include": ("*.h", "*.td", "CMakeLists.txt"),
        "lib": ("*.cpp", "*.h", "CMakeLists.txt"),
        "python": ("*.cpp", "*.td", "*.py", "*.pyi", "CMakeLists.txt"),
        "tools": ("*.cpp", "CMakeLists.txt"),
        "test": ("*.mlir", "*.py", "*.in", "CMakeLists.txt"),
        "unittests": ("*.cpp", "CMakeLists.txt"),
        "tests": ("*.py",),
        "scripts": ("*.sh", "*.py"),
    }
    for directory, patterns in resource_patterns.items():
        sources = {
            path
            for pattern in patterns
            for path in (_REPO_ROOT / directory).rglob(pattern)
            if path.is_file()
        }
        assert sources, directory
        for source in sources:
            relative = PurePosixPath(source.relative_to(_REPO_ROOT).as_posix())
            assert relative in source_members, str(relative)
            assert source_members[relative][0] == source.read_bytes(), str(
                relative
            )


def test_sdist_excludes_generated_and_checkout_debris(source_members):
    """Do not distribute local binaries, caches, Git state, or paper assets."""
    forbidden_parts = {
        ".git",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".benchmarks",
        "__pycache__",
        "CMakeFiles",
    }
    for path in source_members:
        assert not forbidden_parts.intersection(path.parts), str(path)
        assert path.parts[0] not in {"build", "dist", "site", "paper"}
        assert not any(part.endswith(".egg-info") for part in path.parts)
        assert path.suffix not in {".pyc", ".pyo", ".o", ".a", ".so"}
        assert ".so." not in path.name


def test_sdist_rebuild_is_byte_reproducible(sdist, source_members, tmp_path):
    """Rebuild identically without Git or the original source timestamps."""
    source = tmp_path / "unpacked-source"
    source.mkdir()
    for relative, (content, mode) in source_members.items():
        destination = source / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        destination.chmod(mode)
        os.utime(destination, (946684800, 946684800))
    rebuilt = _build_sdist(source, tmp_path / "rebuilt")
    assert hashlib.sha256(sdist.read_bytes()).digest() == hashlib.sha256(
        rebuilt.read_bytes()
    ).digest()
