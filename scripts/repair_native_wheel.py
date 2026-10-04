# scripts/repair_native_wheel.py
"""Repair, gate and optionally reproduce one native release wheel."""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
from importlib.metadata import version
from pathlib import Path, PurePosixPath

if __package__:
    from .check_native_wheel import PLATFORM, check_wheel
else:
    from check_native_wheel import PLATFORM, check_wheel


def _run(command, *, cwd=None, env=None):
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=sys.stderr,
            check=False,
        )
    except OSError as error:
        raise ValueError(f"could not execute {command[0]}: {error}") from error
    if result.returncode:
        raise ValueError(
            f"command failed with exit {result.returncode}: {' '.join(command)}"
        )


def _repair_once(
    wheel,
    wheel_dir,
    *,
    expected_revision,
    expected_python=None,
    allow_dirty=False,
    forbidden_prefixes=(),
):
    wheel_dir.mkdir(parents=True)
    _run(
        [
            sys.executable,
            "-m",
            "auditwheel",
            "repair",
            str(wheel),
            "--plat",
            PLATFORM,
            "--only-plat",
            "--strip",
            "--wheel-dir",
            str(wheel_dir),
        ]
    )
    candidates = list(wheel_dir.glob("*.whl"))
    if len(candidates) != 1:
        raise ValueError("auditwheel repair must produce exactly one wheel")
    repaired = candidates[0]
    summary = check_wheel(
        repaired,
        expected_revision=expected_revision,
        expected_python=expected_python,
        allow_dirty=allow_dirty,
        forbidden_prefixes=forbidden_prefixes,
    )
    return repaired, summary


def _extract_sdist(sdist, destination):
    destination.mkdir()
    with tarfile.open(sdist, "r:gz") as archive:
        members = archive.getmembers()
        paths = set()
        roots = set()
        for member in members:
            path = PurePosixPath(member.name)
            if (
                path.is_absolute()
                or not path.parts
                or ".." in path.parts
                or "\\" in member.name
                or "\x00" in member.name
                or path.as_posix() != member.name.rstrip("/")
                or not (member.isfile() or member.isdir())
                or path in paths
            ):
                raise ValueError(f"unsafe sdist member: {member.name!r}")
            paths.add(path)
            roots.add(path.parts[0])
        if len(roots) != 1:
            raise ValueError("sdist must contain exactly one source root")
        # No extractall: even on Python 3.10, links/devices/path traversal never
        # reach the filesystem. Preserve ordinary executable bits, not setuid.
        for member in members:
            target = destination / member.name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"cannot read sdist member: {member.name}")
            with stream, target.open("xb") as output:
                shutil.copyfileobj(stream, output)
            target.chmod(member.mode & 0o777)
            os.utime(target, (member.mtime, member.mtime))
    source = destination / roots.pop()
    if not (source / "pyproject.toml").is_file():
        raise ValueError("sdist source root is missing pyproject.toml")
    return source.resolve()


def _cmake_roots(build_dir, source):
    cache = build_dir / "CMakeCache.txt"
    if not cache.is_file():
        raise ValueError(
            "rebuild did not record a native CMake build directory"
        )
    values = {}
    for line in cache.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith(("#", "//")):
            continue
        key, separator, value = line.partition("=")
        if separator:
            values[key.split(":", 1)[0]] = value
    roots = []
    for name, expected in (
        ("CMAKE_HOME_DIRECTORY", source),
        ("CMAKE_CACHEFILE_DIR", build_dir),
    ):
        value = values.get(name)
        if not value or Path(value).resolve() != expected.resolve():
            raise ValueError(
                f"rebuild {name} is not the fresh requested directory"
            )
        roots.extend((value, str(Path(value).resolve())))
    return tuple(dict.fromkeys(roots))


def _rebuild(sdist, work, summary, forbidden_prefixes):
    epoch = os.environ.get("SOURCE_DATE_EPOCH", "")
    if not re.fullmatch(r"[0-9]+", epoch):
        raise ValueError("--rebuild-sdist requires explicit SOURCE_DATE_EPOCH")
    python = f"cp{sys.version_info.major}{sys.version_info.minor}"
    if (
        sys.implementation.name != "cpython"
        or sysconfig.get_config_var("Py_GIL_DISABLED")
        or python != summary["python"]
    ):
        raise ValueError(
            "sdist rebuild must use the wheel's regular CPython ABI"
        )
    if version("scikit-build-core") != "1.0.3":
        raise ValueError("sdist rebuild requires scikit-build-core==1.0.3")
    source = _extract_sdist(sdist, work / "source")
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib
    project = tomllib.loads((source / "pyproject.toml").read_text("utf-8"))
    build_system = project.get("build-system", {})
    if (
        build_system.get("build-backend") != "scikit_build_core.build"
        or "scikit-build-core==1.0.3" not in build_system.get("requires", [])
    ):
        raise ValueError("sdist must use the pinned scikit-build-core backend")
    build_dir = (work / "native-build").resolve()
    output = work / "unrepaired"
    env = os.environ.copy()
    # Do not reuse the first wheel's cache, even if SKBUILD_BUILD_DIR was set.
    # Toolchain environment (including CMAKE_ARGS) and epoch are retained.
    env["SKBUILD_BUILD_DIR"] = str(build_dir)
    clean = str(summary["build_info"]["source_clean"]).lower()
    revision = summary["build_info"]["source_revision"]
    settings = {
        "build-dir": str(build_dir),
        "wheel.cmake": "true",
        "cmake.build-type": "Release",
        "build.targets": "SwagePythonModules",
        "install.components": "SwagePythonModules",
        "install.strip": "true",
        "cmake.define.SWAGE_WHEEL_BUILD": "ON",
        "cmake.define.SWAGE_PYTHON_BINDINGS": "ON",
        "cmake.define.SWAGE_SOURCE_REVISION": revision,
        "cmake.define.SWAGE_SOURCE_CLEAN": clean,
        "cmake.define.Python_EXECUTABLE": sys.executable,
        "cmake.define.Python3_EXECUTABLE": sys.executable,
    }
    for name in ("MLIR_DIR", "LLVM_DIR"):
        if env.get(name):
            settings[f"cmake.define.{name}"] = env[name]
    _run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(output),
            *(f"-C{key}={value}" for key, value in settings.items()),
            str(source),
        ],
        cwd=source,
        env=env,
    )
    roots = _cmake_roots(build_dir, source)
    candidates = list(output.glob("*.whl"))
    if len(candidates) != 1:
        raise ValueError("sdist rebuild must produce exactly one wheel")
    repaired, second = _repair_once(
        candidates[0],
        work / "repaired-sdist",
        expected_revision=revision,
        expected_python=summary["python"],
        allow_dirty=not summary["build_info"]["source_clean"],
        forbidden_prefixes=(*forbidden_prefixes, *roots),
    )
    if second["sha256"] != summary["sha256"]:
        raise ValueError(
            "sdist wheel is not byte-for-byte reproducible: "
            f"source-tree={summary['sha256']} sdist={second['sha256']}"
        )
    return {
        "sdist": str(sdist),
        "sha256": second["sha256"],
        "source_root": str(source),
        "build_root": str(build_dir),
        "source_date_epoch": int(epoch),
        "filename": repaired.name,
    }


def repair_wheel(
    path,
    *,
    wheel_dir,
    expected_revision,
    forbidden_prefixes=(),
    allow_dirty=False,
    rebuild_sdist=None,
):
    """Promote a repaired wheel only after every requested gate has passed."""
    path = Path(path).resolve()
    wheel_dir = Path(wheel_dir).resolve()
    prefixes = tuple(os.fspath(p) for p in forbidden_prefixes)
    original_build = os.environ.get("SKBUILD_BUILD_DIR")
    if original_build:
        prefixes = (
            *prefixes,
            original_build,
            str(Path(original_build).resolve()),
        )
    prefixes = tuple(dict.fromkeys(prefixes))
    try:
        with tempfile.TemporaryDirectory(prefix="swage-wheel-release-") as temp:
            work = Path(temp)
            repaired, summary = _repair_once(
                path,
                work / "repaired",
                expected_revision=expected_revision,
                allow_dirty=allow_dirty,
                forbidden_prefixes=prefixes,
            )
            if rebuild_sdist is not None:
                summary["reproducibility"] = _rebuild(
                    Path(rebuild_sdist).resolve(),
                    work,
                    summary,
                    prefixes,
                )
            summary["forbidden_prefixes"] = list(prefixes)
            wheel_dir.mkdir(parents=True, exist_ok=True)
            destination = wheel_dir / repaired.name
            # Do not overwrite a previously generated artifact or expose a
            # rejected first wheel when reproducibility fails.
            with (
                destination.open("xb") as output,
                repaired.open("rb") as source,
            ):
                shutil.copyfileobj(source, output)
            return summary
    except (OSError, tarfile.TarError, ImportError) as error:
        raise ValueError(f"native wheel repair failed: {error}") from error


def main(argv=None):
    """Repair through auditwheel and emit JSON after all gates pass."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--wheel-dir", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--forbid-prefix", action="append", default=[])
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--rebuild-sdist", type=Path)
    args = parser.parse_args(argv)
    try:
        summary = repair_wheel(
            args.wheel,
            wheel_dir=args.wheel_dir,
            expected_revision=args.expected_revision,
            forbidden_prefixes=args.forbid_prefix,
            allow_dirty=args.allow_dirty,
            rebuild_sdist=args.rebuild_sdist,
        )
    except ValueError as error:
        print(f"native wheel repair failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
