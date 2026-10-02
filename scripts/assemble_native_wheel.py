# scripts/assemble_native_wheel.py
"""Pack an installed `mlir_swage` package into a wheel.

`scripts/build_native_wheel.sh` builds the native package, installs it into
a staging directory, and calls this script. The script checks that the
staged package can be moved to another machine directory, then writes the
wheel::

    python scripts/assemble_native_wheel.py --package STAGE/mlir_swage \
        --version 0.5.1 --revision REVISION --llvm-version 22.1.8 \
        --license LICENSE --notices THIRD_PARTY_NOTICES.md --output dist

A package is refused when it holds a symlink that is absolute or leaves the
package, a library whose run path is not relative to the library itself, or
a library that needs a shared library found neither in the package nor in
the short list of system libraries below. `readelf` reads the libraries.

The wheel is tagged for the interpreter and machine the extensions were
built for, as their file names state, with the plain `linux` platform tag:
nothing here checks the libraries against a `manylinux` policy.
"""

import argparse
import base64
import hashlib
import os
import pathlib
import re
import stat
import subprocess
import sys
import zipfile

DISTRIBUTION = "swage-compiler-native"
_PACKAGE = "mlir_swage"
_EXTENSIONS = ("_mlir", "_swageDialectsNanobind")
_EXTENSION_SUFFIX = re.compile(
    r"\.cpython-(?P<major>\d)(?P<minor>\d+)(?P<threads>t?)"
    r"-(?P<machine>[a-z0-9_]+)-linux-gnu\.so"
)
# Shared libraries a wheel may need from the system it is installed on: the
# C and C++ runtime of the build host, and the two compression libraries
# that LLVM links when its build finds them.
_SYSTEM_LIBRARIES = re.compile(
    r"(ld-linux-[a-z0-9-]+|libc|libdl|libgcc_s|libm|libpthread|librt"
    r"|libstdc\+\+|libutil|libz|libzstd)\.so\.\d+"
)
_ELF_MAGIC = b"\x7fELF"
_DYNAMIC_ENTRY = re.compile(
    r"\((?P<tag>NEEDED|RPATH|RUNPATH)\)\s+[^\[]*\[(?P<value>[^\]]*)\]"
)
# Zip archives cannot record a time before 1980.
_EARLIEST_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


class PackagingError(Exception):
    """The staged package cannot become a relocatable wheel."""


def _is_elf(path):
    """Return whether `path` starts with the ELF magic number."""
    with open(path, "rb") as file:
        return file.read(len(_ELF_MAGIC)) == _ELF_MAGIC


def _dynamic_entries(path):
    """Return the needed libraries and the run paths of an ELF file.

    Args:
        path: The shared library to read with `readelf`.

    Returns:
        `(needed, run_paths)`, two lists of strings.

    Raises:
        PackagingError: `readelf` is missing or cannot read the file.
    """
    try:
        listing = subprocess.run(
            ["readelf", "--dynamic", "--wide", str(path)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise PackagingError(
            f"readelf could not read {path}, so its run path and its "
            f"dependencies are unknown: {error}"
        ) from error
    needed, run_paths = [], []
    for entry in _DYNAMIC_ENTRY.finditer(listing):
        if entry["tag"] == "NEEDED":
            needed.append(entry["value"])
        else:
            run_paths.extend(part for part in entry["value"].split(":") if part)
    return needed, run_paths


def _collect(package):
    """Return the files of the staged package that belong in the wheel.

    A relative symlink to a file of the package is left out: a wheel holds
    no links, and a copy would double the file. Byte-code caches are left
    out too.

    Args:
        package: The staged `mlir_swage` directory.

    Returns:
        A sorted list of `(path in the wheel, file on disk)`.

    Raises:
        PackagingError: A symlink is absolute or leaves the package.
    """
    root = package.resolve()
    files = []
    for directory, names, file_names in os.walk(package):
        names[:] = sorted(name for name in names if name != "__pycache__")
        for name in sorted(names + file_names):
            path = pathlib.Path(directory, name)
            if path.is_symlink():
                target = os.readlink(path)
                inside = root in path.resolve().parents
                if os.path.isabs(target) or not inside:
                    raise PackagingError(
                        f"{path} is a symlink to {target}, which is "
                        "absolute or outside the package; a wheel must not "
                        "depend on the machine that built it"
                    )
                continue
            if path.is_file() and path.suffix != ".pyc":
                relative = path.relative_to(package.parent).as_posix()
                files.append((relative, path))
    return sorted(files)


def _wheel_tag(files):
    """Derive the wheel tag from the names of the two extension modules.

    Raises:
        PackagingError: An extension is missing, or the two were built for
            different interpreters.
    """
    tags = set()
    for extension in _EXTENSIONS:
        prefix = f"{_PACKAGE}/_mlir_libs/{extension}"
        suffixes = [
            _EXTENSION_SUFFIX.fullmatch(name[len(prefix) :])
            for name, _ in files
            if name.startswith(prefix + ".")
        ]
        if len(suffixes) != 1 or suffixes[0] is None:
            raise PackagingError(
                f"expected one {extension}.cpython-*-linux-gnu.so in "
                f"{_PACKAGE}/_mlir_libs; found {len(suffixes)} usable"
            )
        (suffix,) = suffixes
        python = f"cp{suffix['major']}{suffix['minor']}"
        tags.add(
            f"{python}-{python}{suffix['threads']}-linux_{suffix['machine']}"
        )
    if len(tags) != 1:
        raise PackagingError(
            "the extension modules were built for different interpreters: "
            f"{sorted(tags)}"
        )
    return tags.pop()


def _check_relocatable(files):
    """Check every library for run paths and dependencies that would break.

    Returns:
        The sorted names of the system libraries the wheel needs.

    Raises:
        PackagingError: A run path is not `$ORIGIN`, or a needed library is
            neither in the wheel nor an expected system library.
    """
    packaged = {pathlib.PurePosixPath(name).name for name, _ in files}
    external = set()
    for name, path in files:
        if not _is_elf(path):
            continue
        needed, run_paths = _dynamic_entries(path)
        for run_path in run_paths:
            if run_path != "$ORIGIN" and not run_path.startswith("$ORIGIN/"):
                raise PackagingError(
                    f"{name} searches {run_path} for its libraries; a "
                    "wheel may only search relative to $ORIGIN"
                )
        for library in needed:
            if library in packaged:
                continue
            if not _SYSTEM_LIBRARIES.fullmatch(library):
                raise PackagingError(
                    f"{name} needs {library}, which is not in the wheel and "
                    "is not one of the system libraries a wheel may expect"
                )
            external.add(library)
    return sorted(external)


def _metadata(version, revision, llvm_version, tag, external, licenses):
    """Return the core metadata of the native wheel.

    Args:
        version: The `swage` version the bindings were built for.
        revision: The source revision the bindings record.
        llvm_version: The LLVM version the bindings were linked against.
        tag: The wheel tag.
        external: Names of the system libraries the wheel needs.
        licenses: Names of the license file and of the notices file.
    """
    license_name, notices_name = licenses
    python = tag.split("-")[0]
    minor = python[3:]
    libraries = ", ".join(f"`{name}`" for name in external)
    return f"""\
Metadata-Version: 2.4
Name: {DISTRIBUTION}
Version: {version}
Summary: Native mlir_swage bindings and compiler library for swage-compiler
Author-email: Abhik Sarkar <abhiksark@gmail.com>
License: MIT for the Swage sources. The wheel also contains LLVM, MLIR, \
nanobind, and other third-party code under their own terms; see \
{notices_name} among the license files.
License-File: {license_name}
License-File: {notices_name}
Classifier: Development Status :: 2 - Pre-Alpha
Classifier: Operating System :: POSIX :: Linux
Classifier: Programming Language :: Python :: 3.{minor}
Classifier: Topic :: Software Development :: Compilers
Project-URL: Repository, https://github.com/abhiksark/swage
Requires-Python: ==3.{minor}.*
Requires-Dist: swage-compiler=={version}
Description-Content-Type: text/markdown

# {DISTRIBUTION}

The native `mlir_swage` package that `swage-compiler` {version} needs to
emit MLIR and to compile and launch kernels: the MLIR Python bindings, the
Swage dialect bindings, and the compiler library they share.

- Built for `swage-compiler` {version}. `swage` refuses bindings built for
  another version.
- Source revision: `{revision}`
- Linked LLVM: {llvm_version}
- Wheel tag: `{tag}`. The libraries were linked on the machine that built
  the wheel and are not checked against a `manylinux` policy. They need
  these shared libraries from the system: {libraries}.

`python -m swage.env` reports the version, the source revision, and the
linked LLVM of the installed bindings. The licenses of the third-party
code in the libraries are in `{notices_name}`.
"""


def _record_line(name, contents):
    """Return the RECORD line of one archive member."""
    digest = base64.urlsafe_b64encode(hashlib.sha256(contents).digest())
    return f"{name},sha256={digest.rstrip(b'=').decode()},{len(contents)}"


def assemble(
    package, *, version, revision, llvm_version, license_file, notices, output
):
    """Write the wheel of a staged `mlir_swage` package.

    The archive is the same for the same inputs: members are sorted and
    carry one fixed time.

    Args:
        package: The staged `mlir_swage` directory.
        version: The `swage` version the bindings were built for.
        revision: The source revision the bindings record.
        llvm_version: The LLVM version the bindings were linked against.
        license_file: The license of the Swage sources.
        notices: The third-party notices of the native build.
        output: Directory that receives the wheel.

    Returns:
        `(wheel path, wheel tag, names of the needed system libraries)`.

    Raises:
        PackagingError: The staged package is not a relocatable
            `mlir_swage` package.
    """
    package = pathlib.Path(package)
    if package.name != _PACKAGE or not package.is_dir():
        raise PackagingError(f"{package} is not a staged {_PACKAGE} directory")
    files = _collect(package)
    tag = _wheel_tag(files)
    external = _check_relocatable(files)

    name = DISTRIBUTION.replace("-", "_")
    information = f"{name}-{version}.dist-info"
    members = [
        (relative, path.read_bytes(), stat.S_IMODE(path.stat().st_mode))
        for relative, path in files
    ]
    license_file, notices = pathlib.Path(license_file), pathlib.Path(notices)
    generated = {
        "METADATA": _metadata(
            version,
            revision,
            llvm_version,
            tag,
            external,
            (license_file.name, notices.name),
        ),
        "WHEEL": (
            "Wheel-Version: 1.0\n"
            "Generator: swage scripts/assemble_native_wheel.py\n"
            "Root-Is-Purelib: false\n"
            f"Tag: {tag}\n"
        ),
        f"licenses/{license_file.name}": license_file.read_text("utf-8"),
        f"licenses/{notices.name}": notices.read_text("utf-8"),
    }
    members.extend(
        (f"{information}/{relative}", text.encode(), 0o644)
        for relative, text in generated.items()
    )
    record = "".join(
        f"{_record_line(relative, contents)}\n"
        for relative, contents, _ in members
    )
    record += f"{information}/RECORD,,\n"
    members.append((f"{information}/RECORD", record.encode(), 0o644))

    output = pathlib.Path(output)
    output.mkdir(parents=True, exist_ok=True)
    wheel = output / f"{name}-{version}-{tag}.whl"
    with zipfile.ZipFile(
        wheel, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for relative, contents, mode in members:
            member = zipfile.ZipInfo(relative, _EARLIEST_ZIP_TIME)
            member.compress_type = zipfile.ZIP_DEFLATED
            member.external_attr = (stat.S_IFREG | mode) << 16
            archive.writestr(member, contents)
    return wheel, tag, external


def main(arguments=None):
    """Assemble the wheel named by the command line and describe it."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--package", required=True, type=pathlib.Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--llvm-version", required=True)
    parser.add_argument("--license", required=True, type=pathlib.Path)
    parser.add_argument("--notices", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    options = parser.parse_args(arguments)
    try:
        wheel, tag, external = assemble(
            options.package,
            version=options.version,
            revision=options.revision,
            llvm_version=options.llvm_version,
            license_file=options.license,
            notices=options.notices,
            output=options.output,
        )
    except PackagingError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    print(f"wheel: {wheel}")
    print(f"tag: {tag}")
    print(f"size: {wheel.stat().st_size} bytes")
    print(f"sha256: {digest}")
    print(f"needs from the system: {', '.join(external)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
