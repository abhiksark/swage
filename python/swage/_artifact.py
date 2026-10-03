# python/swage/_artifact.py
"""Load segmented kernels that were compiled ahead of time.

`python -m swage.compile` writes an artifact directory on a build host: the
PTX of every kernel the public segmented calls can launch, a small runtime
library that classifies offsets and enqueues kernels, and a manifest. This
module reads such a directory, so that a process runs those calls without
the `mlir_swage` bindings and therefore without LLVM or MLIR.

`SWAGE_ARTIFACT_DIR` selects the directory. The artifact then stands in for
the native bindings of the private runner: it answers every kernel request
from its files, admits its programs from what the build host recorded, and
classifies offsets with its runtime library. Nothing is compiled.
"""

import ctypes
import hashlib
import json
import os
import pathlib
import platform
import stat
import types
from typing import NamedTuple

from . import _runtime

_ENVIRONMENT = "SWAGE_ARTIFACT_DIR"
_MANIFEST = "manifest.json"
_FORMAT_VERSION = 2
# SWAGE_RUNTIME_ABI_VERSION of include/swage-c/Runtime.h, which the calls
# into the runtime library below are written against.
_RUNTIME_ABI_VERSION = 1
# Results of swageRuntimeLaunch that are argument errors, not driver errors.
_ARGUMENT_ERRORS = (-2, -3, -4)


class _Kernel(NamedTuple):
    """One kernel of a program: how the runner asks for it and launches it.

    Attributes:
        role: The schedule the kernel serves, unique within its program.
        compiler: Name of the native compile function the runner names.
        options: The code generation options of the request besides the
            kernel name and the target, as sorted `(name, value)` pairs.
        block_size: Threads per block of the launch.
        entry_suffix: What the kernel name in the PTX adds to the program
            name.
        arguments: The launch arguments in order, as `(role, C type)`.
    """

    role: str
    compiler: str
    options: tuple
    block_size: int
    entry_suffix: str
    arguments: tuple


_VALUES = ("values", "const float*")
_OFFSETS = ("offsets", "const int32_t*")
_OUTPUT = ("output", "float*")
_TASK_IDS = ("task_ids", "const int32_t*")
_VALUE_COUNT = ("value_count", "int32_t")
_SEGMENT_COUNT = ("segment_count", "int32_t")
_PARTIAL_COUNT = ("partial_count", "int32_t")
_SEGMENTED = "_compile_segmented_reduction_ptx"
# The threads per block of the kernels below, by the field of the target
# description that the runner launches each with.
_CTA_BLOCK = 128
_SPLIT_BLOCK = 512
_BLOCK_WIDTHS = {
    "cta_block_threads": _CTA_BLOCK,
    "split_block_threads": _SPLIT_BLOCK,
}
def _reduction_kernels(scalar, reads_extent=False):
    """Return the kernels `segment_reduce` can request for one program.

    The pure warp kernel of the private prepared path is not among them: no
    public call launches it.

    Args:
        scalar: The C type of one element of the values, the scratch, and
            the output: `"float"` for an f32 program, `"double"` for f64.
        reads_extent: Whether the program divides by the extent of its
            segment, as a mean does. Its merge kernel then takes the
            partial range records as a fourth buffer.
    """
    values = ("values", f"const {scalar}*")
    output = ("output", f"{scalar}*")
    ranges = ("partial_ranges", "const int32_t*")
    return (
        _Kernel(
            "cta",
            _SEGMENTED,
            (("block_size", _CTA_BLOCK), ("use_task_ids", True)),
            _CTA_BLOCK,
            "",
            (
                values,
                _OFFSETS,
                output,
                _TASK_IDS,
                _VALUE_COUNT,
                ("task_count", "int32_t"),
                _SEGMENT_COUNT,
            ),
        ),
        _Kernel(
            "mixed",
            "_compile_fused_segmented_reduction_ptx",
            (),
            _CTA_BLOCK,
            "",
            (
                values,
                _OFFSETS,
                output,
                _TASK_IDS,
                _VALUE_COUNT,
                ("warp_task_count", "int32_t"),
                ("cta_task_count", "int32_t"),
                _SEGMENT_COUNT,
            ),
        ),
        _Kernel(
            "partial",
            "_compile_split_partial_reduction_ptx",
            (),
            _SPLIT_BLOCK,
            "__partial",
            (
                values,
                ranges,
                ("scratch", f"{scalar}*"),
                _VALUE_COUNT,
                _PARTIAL_COUNT,
            ),
        ),
        _Kernel(
            "merge",
            "_compile_split_merge_reduction_ptx",
            (),
            _SPLIT_BLOCK,
            "__merge",
            (
                ("scratch", f"const {scalar}*"),
                output,
                ("merge_records", "const int32_t*"),
                *((ranges,) if reads_extent else ()),
                _PARTIAL_COUNT,
                ("merge_count", "int32_t"),
                _SEGMENT_COUNT,
            ),
        ),
    )


_REDUCTION_KERNELS = _reduction_kernels("float")
_REDUCTION_KERNELS_F64 = _reduction_kernels("double")
_MEAN_KERNELS = _reduction_kernels("float", reads_extent=True)
_MEAN_KERNELS_F64 = _reduction_kernels("double", reads_extent=True)


def _column_kernels(scalar):
    """Return the one kernel of a program over rank-two values.

    The kernel is the direct schedule of the program: one block per
    segment, in which a thread takes a column. It takes the number of
    columns after the counts of the direct kernel.

    Args:
        scalar: The C type of one element of the values and the output.
    """
    return (
        _Kernel(
            "column",
            _SEGMENTED,
            (("block_size", _CTA_BLOCK),),
            _CTA_BLOCK,
            "",
            (
                ("values", f"const {scalar}*"),
                _OFFSETS,
                ("output", f"{scalar}*"),
                _VALUE_COUNT,
                _SEGMENT_COUNT,
                ("feature_count", "int32_t"),
            ),
        ),
    )


_COLUMN_KERNELS = _column_kernels("float")
_COLUMN_KERNELS_F64 = _column_kernels("double")
# The one kernel `segment_softmax` launches for rank-one values. Its value
# count is the length of the shorter of the values and output buffers.
_SOFTMAX_KERNELS = (
    _Kernel(
        "cta",
        _SEGMENTED,
        (("block_size", _CTA_BLOCK),),
        _CTA_BLOCK,
        "",
        (_VALUES, _OFFSETS, _OUTPUT, _VALUE_COUNT, _SEGMENT_COUNT),
    ),
)
# Every program an artifact can hold, by the name of its kernel function,
# and the kernels a call of that program can request. The writer compiles
# from this table and the loader checks a manifest against it.
_PROGRAMS = {
    "segmented_sum": _REDUCTION_KERNELS,
    "segmented_max": _REDUCTION_KERNELS,
    "segmented_min": _REDUCTION_KERNELS,
    "segmented_mean": _MEAN_KERNELS,
    "segmented_sum_f64": _REDUCTION_KERNELS_F64,
    "segmented_max_f64": _REDUCTION_KERNELS_F64,
    "segmented_min_f64": _REDUCTION_KERNELS_F64,
    "segmented_mean_f64": _MEAN_KERNELS_F64,
    **{
        f"segmented_{kind}{element}_r2": kernels
        for element, kernels in (
            ("", _COLUMN_KERNELS),
            ("_f64", _COLUMN_KERNELS_F64),
        )
        for kind in ("sum", "max", "min", "mean")
    },
    "ragged_softmax": _SOFTMAX_KERNELS,
    "ragged_softmax_r2": _COLUMN_KERNELS,
}

# The directory `SWAGE_ARTIFACT_DIR` named when an artifact was last loaded,
# and that artifact. One slot is enough: a process normally names one
# directory for its whole life.
_selected = (None, None)


def selected():
    """Return the artifact that `SWAGE_ARTIFACT_DIR` names, or None.

    The variable is read at every call. A directory is read and verified
    when it is first named, and kept for the process: a later change to its
    files is not seen. A directory that fails verification is read again at
    the next call.

    Raises:
        RuntimeError: The variable names a directory that is missing,
            unsafe, damaged, or written for another manifest format, host,
            or `swage`. Nothing is loaded from it.
    """
    global _selected
    named = os.environ.get(_ENVIRONMENT)
    if not named:
        return None
    directory, artifact = _selected
    if directory == named:
        return artifact
    with _runtime._compile_lock:
        directory, artifact = _selected
        if directory != named:
            artifact = _Artifact(pathlib.Path(os.path.abspath(named)))
            _selected = (named, artifact)
    return artifact


def _sha256(text):
    """Return the SHA-256 hex digest of a text."""
    return hashlib.sha256(text.encode()).hexdigest()


def _signature(arguments):
    """Render one argument list for an error message."""
    return "(" + ", ".join(f"{role}: {kind}" for role, kind in arguments) + ")"


class _CompileFunction:
    """Stands in for one native compile function of the bindings.

    The runner names a kernel by the compile function that produces it. An
    artifact compiles nothing, so its stand-in only carries that name, and
    its identity keeps the kernels of two artifacts apart in the runner's
    memo.
    """

    def __init__(self, name):
        """Name the native compile function this object stands in for."""
        self.__name__ = name


# The anonymous memory files that hold a loaded runtime library. They stay
# open for the life of the process, as the library stays loaded: the dynamic
# loader knows a library by the path it was opened by, and a closed file's
# number could name another library later.
_LOADED_LIBRARIES = []


def _verified_library_path(contents):
    """Return a path that opens exactly `contents`, for the dynamic loader.

    The bytes go to an anonymous memory file that nothing else can reach,
    and the loader opens it through `/proc/self/fd`. So the library that
    is loaded is the one whose digest was checked, whatever happens to the
    file of the artifact after that check.

    Raises:
        OSError: The memory file cannot be created or written.
    """
    descriptor = os.memfd_create("libSwageRuntime.so", os.MFD_CLOEXEC)
    try:
        view = memoryview(contents)
        while view:
            view = view[os.write(descriptor, view) :]
    except BaseException:
        os.close(descriptor)
        raise
    _LOADED_LIBRARIES.append(descriptor)
    return f"/proc/self/fd/{descriptor}"


class _Artifact:
    """A verified artifact directory, standing in for the native bindings.

    Attributes:
        directory: The absolute directory that was selected.
        manifest: The parsed manifest.
        target: The NVPTX processor the kernels were compiled for.
        target_description: The block widths and planning defaults the
            runner reads, in place of the native target description.
        programs: The names of the programs the artifact holds.
    """

    def __init__(self, directory):
        """Read and verify every file of the artifact at `directory`.

        Raises:
            RuntimeError: The directory is missing, unsafe, damaged, or
                not usable by this `swage` on this host.
        """
        self.directory = directory
        self._where = f"the artifact at {directory}"
        self._malformed = f"the manifest of {self._where} is malformed"
        self._check_directory()
        self.manifest = self._read_manifest()
        self.target = self._field(self.manifest, "target", str)
        planning = self._field(self.manifest, "planning", dict)
        self._planning = (
            self._field(planning, "warp_max_elements", int),
            self._field(planning, "cta_chunk_elements", int),
        )
        self.target_description = self._describe_target()
        self._programs = self._read_programs()
        self.programs = tuple(self._programs)
        self._digests = {
            digest: name for name, (digest, _) in self._programs.items()
        }
        self._kernels = self._read_kernels()
        self._library = self._load_runtime()
        self._compile_segmented_reduction_ptx = _CompileFunction(_SEGMENTED)
        self._compile_fused_segmented_reduction_ptx = _CompileFunction(
            "_compile_fused_segmented_reduction_ptx"
        )
        self._compile_split_partial_reduction_ptx = _CompileFunction(
            "_compile_split_partial_reduction_ptx"
        )
        self._compile_split_merge_reduction_ptx = _CompileFunction(
            "_compile_split_merge_reduction_ptx"
        )

    def _describe_target(self):
        """Return what the runner reads from the native target description.

        The widths are the ones the build host compiled the kernels for,
        which the manifest records, and the planning defaults are the limits
        it admitted the programs under. The two block widths must be the
        ones of the kernel table. The subgroup width is in no kernel of the
        table: the fused kernel serves one warp task per subgroup of a
        block, and the runner computes its grid from that width.

        Raises:
            RuntimeError: A width is missing, is not the one this `swage`
                launches with, or does not divide a block into subgroups.
        """
        recorded = self._field(self.manifest, "target_description", dict)
        widths = {
            name: self._field(recorded, name, int)
            for name in ("subgroup_width", *_BLOCK_WIDTHS)
        }
        for name, launched in _BLOCK_WIDTHS.items():
            if widths[name] != launched:
                raise RuntimeError(
                    f"{self._where} was written for {name}={widths[name]}; "
                    f"this swage launches those kernels with {launched} "
                    "threads per block"
                )
        subgroup_width = widths["subgroup_width"]
        if subgroup_width <= 0 or _CTA_BLOCK % subgroup_width:
            raise RuntimeError(
                f"{self._where} was written for subgroup_width="
                f"{subgroup_width}, which does not divide its "
                f"cta_block_threads={_CTA_BLOCK} into whole subgroups"
            )
        warp_max_elements, cta_chunk_elements = self._planning
        return types.SimpleNamespace(
            **widths,
            default_warp_max_elements=warp_max_elements,
            default_cta_chunk_elements=cta_chunk_elements,
        )

    def _check_directory(self):
        """Require a directory that only its owner and root can change."""
        try:
            details = os.stat(self.directory)
        except FileNotFoundError:
            raise RuntimeError(
                f"{_ENVIRONMENT} names {self.directory}, which does not "
                "exist"
            ) from None
        except OSError as error:
            raise RuntimeError(
                f"{_ENVIRONMENT} names {self.directory}, which cannot be "
                f"inspected: {error}"
            ) from error
        if not stat.S_ISDIR(details.st_mode):
            raise RuntimeError(
                f"{_ENVIRONMENT} names {self.directory}, which is not a "
                "directory"
            )
        self._require_private(self.directory, details)

    def _require_private(self, path, details):
        """Refuse a file or directory that its group or others can write.

        The owner is not compared with the current user: whoever starts the
        process names the directory, and an artifact is normally written by
        one account and read by another. What is refused is a file that an
        account other than its owner and root can replace.
        """
        if details.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise RuntimeError(
                f"{os.path.realpath(path)} of {self._where} is writable by "
                "its group or by other users (mode "
                f"{stat.filemode(details.st_mode)}); remove that permission "
                "with chmod go-w, because a kernel that another account can "
                "replace is not loaded"
            )

    def _read(self, name):
        """Return the bytes of one regular file of the directory.

        A symbolic link is followed, and the file it leads to is the one
        that is checked and read.
        """
        path = self.directory / name
        try:
            with open(path, "rb") as file:
                details = os.fstat(file.fileno())
                if not stat.S_ISREG(details.st_mode):
                    raise RuntimeError(
                        f"{path} of {self._where} is not a regular file"
                    )
                self._require_private(path, details)
                return file.read()
        except FileNotFoundError:
            if name == _MANIFEST:
                raise RuntimeError(
                    f"{self._where} has no {_MANIFEST}"
                ) from None
            raise RuntimeError(
                f"{self._where} lacks {name}, which its manifest lists"
            ) from None
        except OSError as error:
            raise RuntimeError(
                f"{path} of {self._where} cannot be read: {error}"
            ) from error

    def _read_verified(self, entry):
        """Return the bytes of the file a manifest entry names.

        Args:
            entry: A manifest object with a `file` and its `sha256`.

        Raises:
            RuntimeError: The name leaves the directory, or the file is
                missing, unsafe, or has another digest than the manifest
                states.
        """
        name = self._field(entry, "file", str)
        if not name or name != os.path.basename(name) or name in (".", ".."):
            raise RuntimeError(
                f"{self._malformed}: file must be a plain file name, found "
                f"{name!r}"
            )
        expected = self._field(entry, "sha256", str)
        contents = self._read(name)
        found = hashlib.sha256(contents).hexdigest()
        if found != expected:
            raise RuntimeError(
                f"{self._where} holds a {name} that does not match its "
                f"manifest: the SHA-256 is {found}, and the manifest states "
                f"{expected}. Nothing was loaded from the artifact"
            )
        return contents

    def _field(self, container, name, kind):
        """Return one manifest field, requiring its JSON type."""
        value = container.get(name)
        # A JSON boolean is a Python int, and no integer field takes one.
        if not isinstance(value, kind) or (
            kind is int and isinstance(value, bool)
        ):
            names = {
                str: "a string",
                int: "an integer",
                bool: "a boolean",
                dict: "an object",
                list: "a list",
            }
            raise RuntimeError(
                f"{self._malformed}: {name} must be {names[kind]}"
            )
        return value

    def _read_manifest(self):
        """Parse the manifest and require the format this loader reads."""
        try:
            manifest = json.loads(self._read(_MANIFEST))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"the manifest of {self._where} is not valid JSON: {error}"
            ) from error
        if not isinstance(manifest, dict):
            raise RuntimeError(
                f"{self._malformed}: found a "
                f"{type(manifest).__name__} where an object is required"
            )
        version = manifest.get("format_version")
        if version != _FORMAT_VERSION or isinstance(version, bool):
            raise RuntimeError(
                f"{self._where} has manifest format version {version}; this "
                f"swage reads format version {_FORMAT_VERSION}. Write the "
                "artifact again with the swage that loads it"
            )
        return manifest

    def _read_programs(self):
        """Return `{name: (text digest, small element program)}`.

        The second value is what the planning admission of the build host
        returned, or None for a program that is not planned.
        """
        programs = {}
        for entry in self._objects("programs"):
            name = self._field(entry, "name", str)
            if name not in _PROGRAMS:
                raise RuntimeError(
                    f"the manifest of {self._where} lists the program "
                    f"{name!r}, which this swage does not run"
                )
            small = None
            if "small_element_program" in entry:
                small = self._field(entry, "small_element_program", bool)
            programs[name] = (self._field(entry, "sha256", str), small)
        return programs

    def _objects(self, name):
        """Return one manifest list, requiring that it holds objects."""
        entries = self._field(self.manifest, name, list)
        for entry in entries:
            if not isinstance(entry, dict):
                raise RuntimeError(
                    f"{self._malformed}: {name} must hold objects"
                )
        return entries

    def _read_kernels(self):
        """Verify every kernel and return `{(program, role): PTX text}`.

        Every kernel the manifest lists must be one this `swage` requests,
        with the entry name, block size, and argument list it launches
        with, and every kernel a call of a listed program can request must
        be there. A missing kernel is therefore found when the artifact is
        loaded, not at the first batch that needs it.
        """
        kernels = {}
        for entry in self._objects("kernels"):
            program = self._field(entry, "program", str)
            role = self._field(entry, "role", str)
            known = next(
                (
                    kernel
                    for kernel in _PROGRAMS.get(program, ())
                    if kernel.role == role and program in self._programs
                ),
                None,
            )
            if known is None:
                raise RuntimeError(
                    f"the manifest of {self._where} lists a kernel that "
                    f"this swage does not know: program {program!r}, role "
                    f"{role!r}"
                )
            if (program, role) in kernels:
                raise RuntimeError(
                    f"the manifest of {self._where} lists the {role} "
                    f"kernel of {program} twice"
                )
            self._require_launchable(entry, program, known)
            try:
                kernels[program, role] = self._read_verified(entry).decode()
            except UnicodeError as error:
                raise RuntimeError(
                    f"{entry['file']} of {self._where} is not text: {error}"
                ) from error
        for program in self._programs:
            for kernel in _PROGRAMS[program]:
                if (program, kernel.role) not in kernels:
                    raise RuntimeError(
                        f"{self._where} lacks the {kernel.role} kernel of "
                        f"{program}, which a call can launch"
                    )
        return kernels

    def _require_launchable(self, entry, program, known):
        """Require that a kernel is described as this `swage` launches it."""
        subject = f"the {known.role} kernel of {program} in {self._where}"
        entry_name = self._field(entry, "entry", str)
        if entry_name != program + known.entry_suffix:
            raise RuntimeError(
                f"{subject} has entry {entry_name!r}; this swage loads "
                f"{program + known.entry_suffix!r}"
            )
        block_size = self._field(entry, "block_size", int)
        if block_size != known.block_size:
            raise RuntimeError(
                f"{subject} has block size {block_size}; this swage "
                f"launches it with {known.block_size}"
            )
        arguments = tuple(
            (argument.get("role"), argument.get("type"))
            if isinstance(argument, dict)
            else (argument, None)
            for argument in self._field(entry, "arguments", list)
        )
        if arguments != known.arguments:
            raise RuntimeError(
                f"{subject} takes {_signature(arguments)}; this swage "
                f"passes {_signature(known.arguments)}"
            )

    def _load_runtime(self):
        """Verify and load the runtime library, and bind its functions."""
        entry = self._field(self.manifest, "runtime", dict)
        subject = f"the runtime library of {self._where}"
        machine = self._field(entry, "machine", str)
        if machine != platform.machine():
            raise RuntimeError(
                f"{subject} was built for {machine}; this host is "
                f"{platform.machine()}"
            )
        version = self._field(entry, "abi_version", int)
        calls = f"this swage calls ABI version {_RUNTIME_ABI_VERSION}"
        if version != _RUNTIME_ABI_VERSION:
            raise RuntimeError(f"{subject} has ABI version {version}; {calls}")
        contents = self._read_verified(entry)
        try:
            # The functions run for microseconds and never call back into
            # Python, so the GIL stays held, as it is for the launcher of
            # the native bindings.
            library = ctypes.PyDLL(_verified_library_path(contents))
            reported = library.swageRuntimeAbiVersion
            count = library.swageRuntimeCountTasks
            write = library.swageRuntimeWriteTasks
            launch = library.swageRuntimeLaunch
            describe = library.swageRuntimeDescribe
        except (OSError, AttributeError) as error:
            raise RuntimeError(
                f"{subject}, {entry['file']}, cannot be loaded from its "
                f"verified bytes: {error}"
            ) from error
        reported.argtypes = []
        reported.restype = ctypes.c_int32
        if reported() != _RUNTIME_ABI_VERSION:
            raise RuntimeError(
                f"{subject} reports ABI version {reported()}; {calls}"
            )
        address, count64 = ctypes.c_void_p, ctypes.c_int64
        count.argtypes = [address, *[count64] * 5, address]
        count.restype = ctypes.c_char_p
        write.argtypes = [address, *[count64] * 3, address, address]
        write.restype = None
        launch.argtypes = [
            ctypes.c_uint64,
            count64,
            count64,
            ctypes.c_uint64,
            address,
            ctypes.c_int32,
            address,
            ctypes.c_int32,
        ]
        launch.restype = ctypes.c_int32
        describe.argtypes = [
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.POINTER(ctypes.c_char_p),
        ]
        describe.restype = None
        self._count_tasks = count
        self._write_tasks = write
        self._launch = launch
        self._describe = describe
        return library

    def kernel(self, compiler, module_text, options):
        """Return the PTX that answers one kernel request of the runner.

        The first kernel an artifact serves also gives the process driver
        the launcher of the runtime library, when the driver has no
        compiled launcher of its own.

        Args:
            compiler: Name of the native compile function the runner named.
            module_text: Semantic module text of the program.
            options: The code generation options of the request.

        Raises:
            RuntimeError: The artifact holds no such kernel, was compiled
                for another target, or was compiled from another program
                text. Nothing was compiled or launched.
        """
        program = options.get("kernel_name")
        requested = tuple(
            sorted(
                (name, value)
                for name, value in options.items()
                if name not in ("kernel_name", "target")
            )
        )
        known = next(
            (
                kernel
                for kernel in _PROGRAMS.get(program, ())
                if kernel.compiler == compiler and kernel.options == requested
            ),
            None,
        )
        if known is None or program not in self._programs:
            request = ", ".join(
                [compiler, *(f"{name}={value}" for name, value in requested)]
            )
            raise RuntimeError(
                f"{self._where} holds no kernel {program!r} ({request}); "
                f"it holds the programs {', '.join(self.programs)}. Nothing "
                "was compiled or launched"
            )
        target = options.get("target")
        if target != self.target:
            raise RuntimeError(
                f"{self._where} holds kernels for {self.target}, and the "
                f"current device needs {target}; nothing was compiled or "
                f"launched. Select an artifact that was written for {target}"
            )
        recorded = self._programs[program][0]
        found = _sha256(module_text)
        if found != recorded:
            raise RuntimeError(
                f"{self._where} was compiled from another {program!r} "
                f"program than this swage runs (SHA-256 {recorded} in the "
                f"manifest, {found} here); nothing was compiled or launched. "
                "Write the artifact again with the swage that loads it"
            )
        driver = _runtime._get_driver()
        if driver._native_launch is None:
            driver._native_launch = self._launch_kernel
        return self._kernels[program, known.role]

    def admit(self, module_text, warp_max_elements, cta_chunk_elements):
        """Admit one program for planning from what the build host recorded.

        The build host ran planning admission on each program with the
        planning limits of the manifest. This returns its answer for the
        same program and limits, and refuses any other.

        Returns:
            Whether the element program is small enough for the direct-CTA
            schedule selection, as the build host found it.

        Raises:
            RuntimeError: The artifact was planned for other limits, or
                holds no planned program with this text.
        """
        limits = (warp_max_elements, cta_chunk_elements)
        if limits != self._planning:
            raise RuntimeError(
                f"{self._where} was planned for warp_max_elements="
                f"{self._planning[0]} and cta_chunk_elements="
                f"{self._planning[1]}, and this call plans with "
                f"warp_max_elements={warp_max_elements} and "
                f"cta_chunk_elements={cta_chunk_elements}; nothing was "
                "compiled or launched"
            )
        found = _sha256(module_text)
        program = self._digests.get(found)
        small = None if program is None else self._programs[program][1]
        if small is None:
            planned = [
                name
                for name, (_, admitted) in self._programs.items()
                if admitted is not None
            ]
            raise RuntimeError(
                f"{self._where} holds no planned program with the text of "
                f"this call (SHA-256 {found}); it holds the planned "
                f"programs {', '.join(planned) or 'none'}. The artifact was "
                "written without this program, or from another program text "
                "than this swage runs. Nothing was compiled or launched"
            )
        return small

    def _classify_segments(
        self,
        offsets,
        *,
        value_count,
        segment_count,
        warp_max_elements=32,
        cta_chunk_elements=4096,
    ):
        """Classify host offsets as `_classify_segments` of the bindings.

        Args:
            offsets: A contiguous rank-one host int32 array. It must not
                change during the call, which reads it twice.
            value_count: Number of values the offsets index.
            segment_count: Number of segments, one less than the offsets.
            warp_max_elements: Largest segment assigned to direct warp work.
            cta_chunk_elements: Largest input range of one CTA task.

        Returns:
            The records in one int32 array, laid out as
            `swageRuntimeWriteTasks` states, then the warp, CTA, partial,
            and merge counts.

        Raises:
            TypeError: `offsets` is not such an array. Nothing is converted.
            ValueError: The classifier refuses the input; the message is
                its reason.
        """
        import numpy

        if not (
            type(offsets) is numpy.ndarray
            and offsets.dtype == numpy.int32
            and offsets.ndim == 1
            and offsets.flags.c_contiguous
        ):
            raise TypeError(
                "offsets must be a contiguous rank-one host int32 array"
            )
        counts = (ctypes.c_int64 * 4)()
        address = offsets.__array_interface__["data"][0]
        problem = self._count_tasks(
            address,
            len(offsets),
            value_count,
            segment_count,
            warp_max_elements,
            cta_chunk_elements,
            counts,
        )
        if problem is not None:
            raise ValueError(problem.decode())
        warp_count, cta_count, partial_count, merge_count = counts
        records = numpy.empty(
            warp_count + cta_count + 3 * partial_count + 3 * merge_count,
            dtype=numpy.int32,
        )
        self._write_tasks(
            address,
            segment_count,
            warp_max_elements,
            cta_chunk_elements,
            counts,
            records.__array_interface__["data"][0],
        )
        return records, warp_count, cta_count, partial_count, merge_count

    def _launch_kernel(
        self, function, grid_x, block_x, stream, pointers, scalars
    ):
        """Enqueue one kernel as `_launch_kernel` of the bindings does.

        Raises:
            ValueError: The grid, the block, or the argument count is
                outside what a launch takes.
            RuntimeError: The driver is unavailable or refused the launch.
        """
        pointer_count = len(pointers)
        scalar_count = len(scalars)
        result = self._launch(
            function,
            grid_x,
            block_x,
            stream,
            (ctypes.c_uint64 * pointer_count)(*pointers),
            pointer_count,
            (ctypes.c_int32 * scalar_count)(*scalars),
            scalar_count,
        )
        if result:
            raise self._launch_error(result)

    def _launch_error(self, result):
        """Return the error for a nonzero result of the launch function."""
        name = ctypes.c_char_p()
        text = ctypes.c_char_p()
        self._describe(result, ctypes.byref(name), ctypes.byref(text))
        stable_name = name.value.decode() if name.value else "unknown"
        stable_text = text.value.decode() if text.value else "unknown"
        if result in _ARGUMENT_ERRORS:
            return ValueError(stable_text)
        if result < 0:
            return RuntimeError(stable_text)
        return RuntimeError(
            f"CUDA Driver cuLaunchKernel failed: {stable_name} ({result}): "
            f"{stable_text}"
        )
