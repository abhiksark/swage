# scripts/check_native_wheel.py
"""Check a repaired native release wheel without importing its code."""

import argparse
import csv
import hashlib
import io
import json
import os
import posixpath
import re
import stat
import subprocess
import sys
import zipfile
from base64 import urlsafe_b64encode
from email.parser import BytesParser
from importlib.resources import files
from pathlib import Path, PurePosixPath

from elftools.common.exceptions import ELFError
from elftools.elf.elffile import ELFFile
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name, parse_wheel_filename

PLATFORM = "manylinux_2_28_x86_64"
PYTHONS = ("cp310", "cp311", "cp312", "cp313")
MAX_WHEEL_SIZE = 95_000_000
_BUILD_FIELDS = {
    "schema_version",
    "package_version",
    "source_revision",
    "source_clean",
    "frontend_digest",
    "llvm_version",
    "build_type",
}


def _one_header(message, name):
    values = message.get_all(name, [])
    if len(values) != 1:
        raise ValueError(f"metadata must contain exactly one {name}")
    return values[0]


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"build info has duplicate field: {key}")
        result[key] = value
    return result


def _build_info(payload, expected_revision, expected_version, allow_dirty):
    try:
        info = json.loads(payload, object_pairs_hook=_unique_json_object)
    except (json.JSONDecodeError, UnicodeError) as error:
        raise ValueError(f"malformed build info JSON: {error}") from error
    if not isinstance(info, dict) or set(info) != _BUILD_FIELDS:
        raise ValueError(
            "build info must have the exact schema_version 2 fields"
        )
    if type(info["schema_version"]) is not int or info["schema_version"] != 2:
        raise ValueError("build info schema_version must be integer 2")
    if info["package_version"] != expected_version:
        raise ValueError("build info package_version does not match release")
    revision = info["source_revision"]
    if not isinstance(revision, str) or not re.fullmatch(
        r"[0-9a-f]{40}", revision
    ):
        raise ValueError("build info source_revision must be 40 lowercase hex")
    if revision != expected_revision:
        raise ValueError(
            "build info source_revision does not match expected revision"
        )
    if type(info["source_clean"]) is not bool:
        raise ValueError("build info source_clean must be a boolean")
    if not info["source_clean"] and not allow_dirty:
        raise ValueError("official wheel requires source_clean=true")
    digest = info["frontend_digest"]
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("build info frontend_digest must be 64 lowercase hex")
    if info["llvm_version"] != "llvmorg-22.1.8":
        raise ValueError("build info llvm_version must be llvmorg-22.1.8")
    if info["build_type"] != "Release":
        raise ValueError("build info build_type must be Release")
    return info


def _origin_directory(member, value):
    match = re.fullmatch(r"(?:\$ORIGIN|\$\{ORIGIN\})(/[^${}:\\]*)?", value)
    if match is None:
        raise ValueError(
            f"{member}: RPATH/RUNPATH must be $ORIGIN-relative: {value!r}"
        )
    suffix = match[1] or ""
    if suffix.startswith("//"):
        raise ValueError(f"{member}: invalid $ORIGIN path: {value!r}")
    directory = posixpath.normpath(posixpath.dirname(member) + "/" + suffix[1:])
    if directory == ".." or directory.startswith("../"):
        raise ValueError(
            f"{member}: RPATH/RUNPATH escapes the wheel: {value!r}"
        )
    return directory


def _elf_info(member, payload):
    try:
        elf = ELFFile(io.BytesIO(payload))
        if (
            elf.elfclass != 64
            or not elf.little_endian
            or elf["e_machine"] != "EM_X86_64"
            or elf["e_type"] != "ET_DYN"
        ):
            raise ValueError(f"{member}: expected an ELF x86_64 shared object")
        # Inspect loader-visible segments as well as every dynamic section.
        tables = [s for s in elf.iter_segments() if s["p_type"] == "PT_DYNAMIC"]
        tables += [
            s for s in elf.iter_sections() if s["sh_type"] == "SHT_DYNAMIC"
        ]
        if not tables:
            raise ValueError(f"{member}: ELF has no dynamic section")
        needed = set()
        rpaths = []
        runpaths = []
        for table in tables:
            for tag in table.iter_tags():
                kind = tag.entry.d_tag
                if kind in {"DT_FILTER", "DT_AUXILIARY", "DT_SUNW_FILTER"}:
                    raise ValueError(
                        f"{member}: unsupported dynamic dependency: {kind}"
                    )
                if kind == "DT_NEEDED":
                    library = tag.needed
                    if re.match(r"libcuda(?:[.-]|$)", library):
                        raise ValueError(
                            f"{member}: forbidden libcuda {kind}: {library}"
                        )
                    if "/" in library or "$" in library or not library:
                        raise ValueError(
                            f"{member}: non-relocatable {kind}: {library!r}"
                        )
                    needed.add(library)
                elif kind in {"DT_RPATH", "DT_RUNPATH"}:
                    paths = getattr(tag, kind[3:].lower()).split(":")
                    directories = [_origin_directory(member, p) for p in paths]
                    (runpaths if kind == "DT_RUNPATH" else rpaths).extend(
                        directories
                    )
        return needed, set(runpaths or rpaths)
    except (ELFError, KeyError, IndexError, TypeError, AttributeError) as error:
        raise ValueError(f"{member}: malformed ELF: {error}") from error


def _system_libraries():
    try:
        policies = json.loads(
            files("auditwheel")
            .joinpath("policy/manylinux-policy.json")
            .read_text()
        )
        # auditwheel excludes the ELF loader separately from lib_whitelist.
        # This checker admits only its declared x86_64 glibc platform.
        return set(
            next(p for p in policies if p["name"] == "manylinux_2_28")[
                "lib_whitelist"
            ]
        ) | {"ld-linux-x86-64.so.2"}
    except (ImportError, OSError, KeyError, StopIteration) as error:
        raise ValueError(
            "auditwheel with the manylinux_2_28 policy is required"
        ) from error


def _auditwheel_show(path):
    try:
        result = subprocess.run(
            [sys.executable, "-m", "auditwheel", "show", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as error:
        raise ValueError(f"could not run auditwheel show: {error}") from error
    if result.returncode:
        raise ValueError(
            "auditwheel show failed: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    # The normal show format is supported by both stable and newer auditwheel.
    # Do not infer compatibility from the input filename echoed in its output.
    match = re.search(
        r'is\s+consistent with the following platform tag:\s*"([^"]+)"',
        result.stdout,
    )
    policy = match[1] if match else ""
    version = re.fullmatch(r"manylinux_(\d+)_(\d+)_x86_64", policy)
    aliases = {
        "manylinux1_x86_64": (2, 5),
        "manylinux2010_x86_64": (2, 12),
        "manylinux2014_x86_64": (2, 17),
    }
    minimum = (
        tuple(map(int, version.groups())) if version else aliases.get(policy)
    )
    if minimum is None or not (minimum[0] == 2 and minimum <= (2, 28)):
        raise ValueError(
            f"auditwheel policy is not manylinux <=2.28 x86_64: {policy!r}"
        )
    return policy


def _check_record(archive, names, dist_info):
    record = f"{dist_info}/RECORD"
    rows = list(csv.reader(io.StringIO(archive.read(record).decode("utf-8"))))
    entries = {}
    for row in rows:
        if len(row) != 3 or row[0] in entries:
            raise ValueError(
                "wheel RECORD contains malformed or duplicate entries"
            )
        entries[row[0]] = row[1:]
    if set(entries) != names:
        raise ValueError("wheel RECORD does not describe every packaged file")
    if entries[record] != ["", ""]:
        raise ValueError("wheel RECORD must not hash itself")
    return entries


def _check_archive(
    archive,
    python,
    expected_revision,
    expected_version,
    allow_dirty,
    forbidden_prefixes,
):
    infos = archive.infolist()
    names = set()
    for item in infos:
        path = PurePosixPath(item.filename)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in item.filename
            or path.as_posix() != item.filename.rstrip("/")
            or stat.S_ISLNK(item.external_attr >> 16)
        ):
            raise ValueError(f"unsafe wheel member: {item.filename!r}")
        if item.is_dir():
            continue
        if item.filename in names:
            raise ValueError(f"duplicate wheel member: {item.filename}")
        names.add(item.filename)
        if path.suffix in {".pyc", ".pyo"} or "__pycache__" in path.parts:
            raise ValueError(f"wheel contains bytecode: {item.filename}")
        if path.name.startswith("_segmented") and path.suffix in {
            ".py",
            ".pyi",
        }:
            raise ValueError(
                f"wheel contains private segmented Python: {item.filename}"
            )
        if ".cpython-" in path.name and not path.name.endswith(
            f".cpython-{python[2:]}-x86_64-linux-gnu.so"
        ):
            raise ValueError(
                f"{item.filename}: native extension ABI does not match {python}"
            )
    dist_info = f"swage_compiler-{expected_version}.dist-info"
    roots = {
        n.split("/")[0] for n in names if n.split("/")[0].endswith(".dist-info")
    }
    if roots != {dist_info}:
        raise ValueError(
            "wheel must contain exactly the matching project dist-info"
        )
    required = {
        "swage/__init__.py",
        "swage/bench.py",
        "swage/_benchmark.py",
        "swage/language.py",
        "swage/py.typed",
        "swage/__init__.pyi",
        "swage/language.pyi",
        "mlir_swage/ir.py",
        "mlir_swage/dialects/swage.py",
        "mlir_swage/_mlir_libs/__init__.py",
        "mlir_swage/_build_info.json",
        f"{dist_info}/METADATA",
        f"{dist_info}/WHEEL",
        f"{dist_info}/RECORD",
        f"{dist_info}/licenses/LICENSE",
        f"{dist_info}/licenses/LICENSES/LLVM.txt",
    }
    missing = required - names
    if missing:
        raise ValueError(
            "wheel is missing required package, typing or license files: "
            f"{sorted(missing)}"
        )
    wheel = BytesParser().parsebytes(archive.read(f"{dist_info}/WHEEL"))
    if _one_header(wheel, "Wheel-Version") != "1.0":
        raise ValueError("unsupported WHEEL version")
    if _one_header(wheel, "Root-Is-Purelib") != "false":
        raise ValueError("native WHEEL must have Root-Is-Purelib: false")
    if wheel.get_all("Tag", []) != [f"{python}-{python}-{PLATFORM}"]:
        raise ValueError(
            "internal WHEEL tags do not match the exact filename ABI/platform"
        )
    metadata = BytesParser().parsebytes(archive.read(f"{dist_info}/METADATA"))
    if canonicalize_name(_one_header(metadata, "Name")) != "swage-compiler":
        raise ValueError("METADATA project Name must be swage-compiler")
    if _one_header(metadata, "Version") != expected_version:
        raise ValueError("METADATA Version does not match release")
    if SpecifierSet(_one_header(metadata, "Requires-Python")) != SpecifierSet(
        ">=3.10,<3.14"
    ):
        raise ValueError("METADATA Requires-Python must be >=3.10,<3.14")
    if (
        _one_header(metadata, "License-Expression")
        != "MIT AND Apache-2.0 WITH LLVM-exception"
    ):
        raise ValueError(
            "METADATA must declare MIT and Apache-2.0 WITH LLVM-exception"
        )
    if not {"LICENSE", "LICENSES/LLVM.txt"} <= set(
        metadata.get_all("License-File", [])
    ):
        raise ValueError("METADATA must declare both license files")
    mit = archive.read(f"{dist_info}/licenses/LICENSE")
    llvm = archive.read(f"{dist_info}/licenses/LICENSES/LLVM.txt")
    if b"Permission is hereby granted, free of charge" not in mit:
        raise ValueError("MIT license text is absent or corrupt")
    if b"Apache License" not in llvm or b"LLVM Exceptions" not in llvm:
        raise ValueError("LLVM license and exception text is absent or corrupt")
    build_info = _build_info(
        archive.read("mlir_swage/_build_info.json"),
        expected_revision,
        expected_version,
        allow_dirty,
    )
    extensions = set()
    for module in ("_mlir", "_swageDialectsNanobind"):
        extension = (
            f"mlir_swage/_mlir_libs/{module}.cpython-{python[2:]}"
            "-x86_64-linux-gnu.so"
        )
        if extension not in names:
            raise ValueError(
                f"wheel is missing native extension for {python}: {module}"
            )
        extensions.add(extension)
    entries = _check_record(archive, names, dist_info)
    elfs = {}
    for member in sorted(names):
        payload = archive.read(member)
        for prefix in forbidden_prefixes:
            if prefix in payload:
                raise ValueError(
                    f"{member}: forbidden build-root prefix "
                    f"{os.fsdecode(prefix)!r}"
                )
        if member != f"{dist_info}/RECORD":
            digest = (
                urlsafe_b64encode(hashlib.sha256(payload).digest())
                .rstrip(b"=")
                .decode()
            )
            if entries[member] != [f"sha256={digest}", str(len(payload))]:
                raise ValueError(f"{member}: RECORD digest or size mismatch")
        if payload.startswith(b"\x7fELF"):
            elfs[member] = _elf_info(member, payload)
        elif member in extensions or re.search(r"\.so(?:\.\d+)*$", member):
            raise ValueError(f"{member}: native library is not ELF")
    for library in (
        "SwagePythonCAPI",
        "MLIRPythonSupport-mlir_swage",
        "nanobind-mlir_swage",
    ):
        pattern = rf"lib{re.escape(library)}(?:-[0-9a-f]+)?\.so(?:\.\d+)*"
        if not any(re.fullmatch(pattern, PurePosixPath(n).name) for n in elfs):
            raise ValueError(
                f"wheel is missing shared runtime library: {library}"
            )
    system = _system_libraries()
    for member, (needed, directories) in elfs.items():
        for library in sorted(needed - system):
            if not any(posixpath.join(d, library) in elfs for d in directories):
                raise ValueError(
                    f"{member}: unresolved non-system dependency: {library}"
                )
    return build_info


def check_wheel(
    path,
    *,
    expected_revision,
    expected_python=None,
    expected_version="0.5.2",
    allow_dirty=False,
    forbidden_prefixes=(),
):
    """Return artifact identity or raise ValueError on gate failure."""
    path = Path(path)
    if not isinstance(expected_revision, str) or not re.fullmatch(
        r"[0-9a-f]{40}", expected_revision
    ):
        raise ValueError(
            "expected revision must be 40 lowercase hex characters"
        )
    if expected_python is not None and expected_python not in PYTHONS:
        raise ValueError("expected Python must be cp310, cp311, cp312 or cp313")
    prefixes = tuple(os.fsencode(p) for p in forbidden_prefixes)
    if any(not p for p in prefixes):
        raise ValueError("forbidden build-root prefixes must not be empty")
    try:
        project, version, build, tags = parse_wheel_filename(path.name)
        if (
            project != "swage-compiler"
            or str(version) != expected_version
            or build
        ):
            raise ValueError(
                "wheel filename project/version/build does not match release"
            )
        if len(tags) != 1:
            raise ValueError(
                "wheel filename must have one exact ABI/platform tag"
            )
        tag = next(iter(tags))
        if tag.interpreter not in PYTHONS or tag.abi != tag.interpreter:
            raise ValueError(
                "wheel requires regular CPython cp310-cp313 with matching ABI"
            )
        if tag.platform != PLATFORM:
            raise ValueError(
                f"wheel filename platform must be exactly {PLATFORM}"
            )
        canonical_filename = (
            f"swage_compiler-{expected_version}-"
            f"{tag.interpreter}-{tag.interpreter}-{PLATFORM}.whl"
        )
        if path.name != canonical_filename:
            raise ValueError(
                "wheel filename must use one exact canonical ABI/platform tag"
            )
        if expected_python is not None and tag.interpreter != expected_python:
            raise ValueError("wheel Python does not match expected Python")
        size = path.stat().st_size
        if size >= MAX_WHEEL_SIZE:
            raise ValueError(
                f"wheel size {size} must be below {MAX_WHEEL_SIZE} bytes"
            )
        with zipfile.ZipFile(path) as archive:
            build_info = _check_archive(
                archive,
                tag.interpreter,
                expected_revision,
                expected_version,
                allow_dirty,
                prefixes,
            )
        _auditwheel_show(path.resolve())
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return {
            "filename": path.name,
            "size": size,
            "sha256": digest.hexdigest(),
            "python": tag.interpreter,
            "build_info": build_info,
        }
    except (
        OSError,
        zipfile.BadZipFile,
        KeyError,
        UnicodeError,
        csv.Error,
    ) as error:
        raise ValueError(f"invalid wheel {path.name}: {error}") from error


def main(argv=None):
    """Print JSON on success or a diagnostic with exit 1 on failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-python", choices=PYTHONS)
    parser.add_argument("--expected-version", default="0.5.2")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--forbid-prefix", action="append", default=[])
    args = parser.parse_args(argv)
    try:
        summary = check_wheel(
            args.wheel,
            expected_revision=args.expected_revision,
            expected_python=args.expected_python,
            expected_version=args.expected_version,
            allow_dirty=args.allow_dirty,
            forbidden_prefixes=args.forbid_prefix,
        )
    except ValueError as error:
        print(f"native wheel check failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
