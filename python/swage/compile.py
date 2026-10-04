# python/swage/compile.py
"""Compile the segmented kernels ahead of time into an artifact directory.

Run as a module on a host that has the native build::

    python -m swage.compile --target sm_86 --output /path/to/artifact

The command compiles every kernel that `swage.segment_reduce` and
`swage.segment_softmax` can launch for one NVPTX processor and writes the
PTX, the runtime library, and a manifest. It needs the `mlir_swage`
bindings and numpy. It needs no GPU and no PyTorch, and the target does not
have to be the processor of the build host.

A process that sets `SWAGE_ARTIFACT_DIR` to the directory runs the two
calls from it without `mlir_swage`; see `swage._artifact`. The module is a
command and has no public name.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import shutil
import sys
import tempfile

from . import __version__, _artifact, _runtime
from . import _segmented_plan as _plan
from . import _segmented_programs as _programs
from . import _segmented_runtime as _execution

_RUNTIME_LIBRARY = "libSwageRuntime.so"
# The `e_machine` values of an ELF header this command can name, as
# `platform.machine()` spells them on the host that loads the library.
_ELF_MACHINES = {62: "x86_64", 183: "aarch64"}
_PROGRAMS = (
    "sum",
    "max",
    "min",
    "mean",
    "sum_f64",
    "max_f64",
    "min_f64",
    "mean_f64",
    "sum_r2",
    "max_r2",
    "min_r2",
    "mean_r2",
    "sum_f64_r2",
    "max_f64_r2",
    "min_f64_r2",
    "mean_f64_r2",
    "softmax",
    "softmax_r2",
)


def _planning_limits():
    """Return the planning limits of the public reduction.

    They are the defaults of the planned path, which the call does not let
    a caller change: the warp limit and the CTA chunk limit of the native
    target description. Reading them needs the native bindings.
    """
    return _plan._planning_limits(None, None)


def _program(name):
    """Return the kernel name, module text, and planning of one program.

    Args:
        name: A name of `_PROGRAMS`: a kind of the reduction, which is its
            f32 program over rank-one values, followed by `_f64` for
            float64 values and by `_r2` for rank-two values, or
            `"softmax"`, which takes `_r2` as well.

    Returns:
        The name of the kernel function, the semantic module text, and
        whether the program runs through the planned path. A program over
        rank-two values has one kernel and is not planned.
    """
    kind, *suffixes = name.split("_")
    rank = 2 if "r2" in suffixes else 1
    if kind == "softmax":
        kernel = "ragged_softmax" + "_r2" * (rank == 2)
        return kernel, _programs._softmax_text(rank), False
    element = "f64" if "f64" in suffixes else "f32"
    return (
        _programs._reduction_kernel(kind, element, rank),
        _programs._semantic_module(kind, element, rank),
        rank == 1,
    )


def _packaged_runtime():
    """Return the runtime library that ships beside the native bindings."""
    try:
        spec = importlib.util.find_spec("mlir_swage._mlir_libs")
    except (ImportError, ValueError):
        spec = None
    for location in (spec.submodule_search_locations or ()) if spec else ():
        library = pathlib.Path(location) / _RUNTIME_LIBRARY
        if library.is_file():
            return library
    raise RuntimeError(
        f"the mlir_swage package holds no {_RUNTIME_LIBRARY}; rebuild the "
        "native package, or name a library with --runtime-library"
    )


def _machine(library, contents):
    """Return the machine an ELF shared library was built for.

    Raises:
        RuntimeError: The file is not a little-endian 64-bit ELF file for a
            machine in `_ELF_MACHINES`.
    """
    machine = None
    if contents[:6] == b"\x7fELF\x02\x01" and len(contents) >= 20:
        machine = _ELF_MACHINES.get(int.from_bytes(contents[18:20], "little"))
    if machine is None:
        raise RuntimeError(
            f"{library} is not an ELF library for one of "
            f"{', '.join(sorted(_ELF_MACHINES.values()))}"
        )
    return machine


def _compile_program(native, name, target):
    """Compile every kernel of one program and describe each.

    Returns:
        The manifest entry of the program, and a list of
        `(manifest entry, PTX text)` with one item per kernel.

    Raises:
        ValueError: The compiler rejects the program or the target.
        RuntimeError: A kernel does not have the entry name, the launch
            width, or the launch contract that the loader launches it with.
    """
    program, text, planned = _program(name)
    description = {
        "name": program,
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
    }
    if planned:
        description["small_element_program"] = _plan._admit_program(
            text, program, *_planning_limits()
        )
    kernels = []
    for kernel in _artifact._PROGRAMS[program]:
        compiled = _execution._compile_once(
            getattr(native, kernel.compiler),
            text,
            kernel_name=program,
            target=target,
            **dict(kernel.options),
        )
        ptx = compiled.image
        # The loader derives the contract of a kernel from its manifest
        # entry, so the compiler must have given it that contract.
        if compiled.contract_json != _artifact._contract_json(program, kernel):
            raise RuntimeError(
                f"the {kernel.role} kernel of {program} has the launch "
                f"contract {compiled.contract_json}, and the loader derives "
                f"{_artifact._contract_json(program, kernel)} from its "
                "manifest entry; the artifact was not written"
            )
        entry = program + kernel.entry_suffix
        width = re.search(r"^\s*\.reqntid (\d+)", ptx, re.MULTILINE)
        if f".entry {entry}(" not in ptx or width is None:
            raise RuntimeError(
                f"the {kernel.role} kernel of {program} has no entry "
                f"{entry!r} with a fixed launch width; the artifact was not "
                "written"
            )
        if int(width[1]) != kernel.block_size:
            raise RuntimeError(
                f"the {kernel.role} kernel of {program} was compiled for "
                f"{width[1]} threads per block, and the loader launches it "
                f"with {kernel.block_size}; the artifact was not written"
            )
        kernels.append(
            (
                {
                    "program": program,
                    "role": kernel.role,
                    "entry": entry,
                    "block_size": kernel.block_size,
                    "file": f"{program}.{kernel.role}.ptx",
                    "sha256": hashlib.sha256(ptx.encode()).hexdigest(),
                    "arguments": [
                        {"role": role, "type": kind}
                        for role, kind in kernel.arguments
                    ],
                },
                ptx,
            )
        )
    return description, kernels


def _write_file(directory, name, contents):
    """Create one artifact file that only its owner can write."""
    descriptor = os.open(
        directory / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644
    )
    with os.fdopen(descriptor, "wb") as file:
        file.write(contents)


def _write_artifact(output, target, programs, runtime_library):
    """Compile `programs` for `target` and write the artifact at `output`.

    The directory is staged beside `output` and renamed into place, so a
    failed run leaves nothing and a reader never sees part of an artifact.
    The directory and its files are created without write permission for
    the group and for others, which the loader requires.

    Args:
        output: Directory to create. It must not exist; its parent must.
        target: NVPTX processor to compile for, such as `"sm_86"`.
        programs: Names from `_PROGRAMS`, in manifest order.
        runtime_library: The runtime library to ship, or None for the one
            beside the native bindings.

    Returns:
        The manifest and the SHA-256 hex digest of its file.

    Raises:
        RuntimeError: The native bindings or the runtime library are
            missing, an artifact is selected, compiling is switched off,
            `output` exists, or a kernel does not fit the loader.
        ValueError: The compiler rejects the target.
        OSError: The artifact cannot be written.
    """
    if os.environ.get(_artifact._ENVIRONMENT):
        raise RuntimeError(
            f"{_artifact._ENVIRONMENT} is set, so this process would be "
            "served kernels instead of compiling them; unset it to write "
            "an artifact"
        )
    if os.path.lexists(output):
        raise RuntimeError(
            f"{output} exists; an artifact is written once, to a new directory"
        )
    try:
        native = _runtime._native_bindings()
    except ImportError as error:
        from ._frontend import _INSTALLATION

        raise RuntimeError(
            "writing an artifact requires the mlir_swage bindings, which "
            f"this installation does not have. See {_INSTALLATION} "
            "for the native build"
        ) from error
    library = pathlib.Path(runtime_library or _packaged_runtime())
    runtime = library.read_bytes()
    warp_max_elements, cta_chunk_elements = _planning_limits()
    description = _execution._target_description()
    manifest = {
        "format_version": _artifact._FORMAT_VERSION,
        "swage_version": __version__,
        "source_revision": native.__source_revision__,
        "llvm_version": native.__llvm_version__,
        "target": target,
        "target_description": {
            name: getattr(description, name)
            for name in ("subgroup_width", *_artifact._BLOCK_WIDTHS)
        },
        "planning": {
            "warp_max_elements": warp_max_elements,
            "cta_chunk_elements": cta_chunk_elements,
        },
        "runtime": {
            "file": _RUNTIME_LIBRARY,
            "sha256": hashlib.sha256(runtime).hexdigest(),
            "machine": _machine(library, runtime),
            "abi_version": _artifact._RUNTIME_ABI_VERSION,
        },
        "programs": [],
        "kernels": [],
    }
    files = {_RUNTIME_LIBRARY: runtime}
    for name in programs:
        description, kernels = _compile_program(native, name, target)
        manifest["programs"].append(description)
        for entry, ptx in kernels:
            manifest["kernels"].append(entry)
            files[entry["file"]] = ptx.encode()
    encoded = (json.dumps(manifest, indent=2) + "\n").encode()
    files[_artifact._MANIFEST] = encoded

    stage = pathlib.Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    try:
        for name, contents in files.items():
            _write_file(stage, name, contents)
        # mkdtemp creates the directory for its owner alone. An artifact is
        # read by other accounts, within what the umask of this process
        # allows, and is never writable by them.
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(stage, 0o755 & ~umask)
        os.rename(stage, output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return manifest, hashlib.sha256(encoded).hexdigest()


def _main(argv=None):
    """Write one artifact and print what it holds as key/value lines.

    Args:
        argv: Command-line arguments, or None for those of the process.

    Returns:
        The process exit status: 0 after the artifact was written, 1 when
        it was not, with the reason on standard error.
    """
    parser = argparse.ArgumentParser(
        prog="python -m swage.compile",
        description=(
            "Compile the kernels of swage.segment_reduce and "
            "swage.segment_softmax for one target and write them, the "
            "runtime library, and a manifest to a new directory."
        ),
    )
    parser.add_argument(
        "--target",
        required=True,
        help="NVPTX processor of the device that will run the kernels, "
        "such as sm_86",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=pathlib.Path,
        help="directory to create; it must not exist",
    )
    parser.add_argument(
        "--program",
        action="append",
        choices=_PROGRAMS,
        help="program to include; repeat for several (default: all)",
    )
    parser.add_argument(
        "--runtime-library",
        type=pathlib.Path,
        help=f"{_RUNTIME_LIBRARY} built for the host that will load the "
        "artifact (default: the library of this native build)",
    )
    arguments = parser.parse_args(argv)
    programs = [
        name for name in _PROGRAMS if name in (arguments.program or _PROGRAMS)
    ]
    output = pathlib.Path(os.path.abspath(arguments.output))
    try:
        manifest, digest = _write_artifact(
            output, arguments.target, programs, arguments.runtime_library
        )
    except (RuntimeError, ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    runtime = manifest["runtime"]
    report = {
        "artifact": output,
        "format_version": manifest["format_version"],
        "target": manifest["target"],
        "programs": ", ".join(
            program["name"] for program in manifest["programs"]
        ),
        "kernels": len(manifest["kernels"]),
        "runtime": f"{runtime['file']} ({runtime['machine']})",
        "manifest_sha256": digest,
    }
    for key, value in report.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
