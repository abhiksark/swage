# tests/python/test_artifact.py
"""LLVM-free tests for the artifact loader.

These run without the native bindings, without a GPU, and without the
runtime library: a stand-in answers for the library, and the kernels are
placeholder text. They pin how an artifact is selected, what is verified
before it is used, the trust rule for its directory, and every refusal.
The native tier runs the same loader on artifacts that the compiler wrote.
"""

import ctypes
import hashlib
import json
import os
import pathlib
import platform
import re
import stat
import subprocess
import sys
import types

import pytest
import swage
from swage import _artifact, env
from swage import _segmented_qualification as qualification

PROGRAM_TEXTS = {
    "segmented_sum": qualification._semantic_module("sum"),
    "segmented_max": qualification._semantic_module("max"),
    "ragged_softmax": qualification._SOFTMAX_MODULE,
}
RUNTIME_BYTES = b"placeholder for the runtime library\n"


def _digest(data):
    """Return the SHA-256 hex digest of bytes or text."""
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()


class _FakeLibrary:
    """What `ctypes.PyDLL` returns here: the functions the loader binds."""

    abi_version = 1

    def __init__(self, path):
        self.path = path
        for name in (
            "swageRuntimeCountTasks",
            "swageRuntimeWriteTasks",
            "swageRuntimeLaunch",
            "swageRuntimeDescribe",
        ):
            setattr(self, name, types.SimpleNamespace())
        self.swageRuntimeAbiVersion = _AbiVersion(self.abi_version)


class _AbiVersion:
    """The version function of the stand-in library."""

    def __init__(self, version):
        self._version = version

    def __call__(self):
        return self._version


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    """Start every test with no artifact selected and none remembered.

    The native bindings are unimportable, a stand-in answers for the
    runtime library, and the process driver is a bare object, so no test
    needs a CUDA driver library.
    """
    monkeypatch.setitem(sys.modules, "mlir_swage", None)
    monkeypatch.delenv("SWAGE_ARTIFACT_DIR", raising=False)
    monkeypatch.setattr(_artifact, "_selected", (None, None))
    monkeypatch.setattr(ctypes, "PyDLL", _FakeLibrary)
    driver = types.SimpleNamespace(_native_launch=None)
    monkeypatch.setattr(_artifact._runtime, "_get_driver", lambda: driver)


def _manifest(programs=tuple(PROGRAM_TEXTS), target="sm_86"):
    """Return the manifest of an artifact that holds `programs`."""
    kernels = []
    for program in programs:
        for kernel in _artifact._PROGRAMS[program]:
            ptx = f"// {program} {kernel.role}\n"
            kernels.append(
                {
                    "program": program,
                    "role": kernel.role,
                    "entry": program + kernel.entry_suffix,
                    "block_size": kernel.block_size,
                    "file": f"{program}.{kernel.role}.ptx",
                    "sha256": _digest(ptx),
                    "arguments": [
                        {"role": role, "type": kind}
                        for role, kind in kernel.arguments
                    ],
                }
            )
    return {
        "format_version": 1,
        "swage_version": swage.__version__,
        "source_revision": "0123456789abcdef0123456789abcdef01234567",
        "llvm_version": "22.1.8",
        "target": target,
        "planning": {"warp_max_elements": 32, "cta_chunk_elements": 4096},
        "runtime": {
            "file": "libSwageRuntime.so",
            "sha256": _digest(RUNTIME_BYTES),
            "machine": platform.machine(),
            "abi_version": 1,
        },
        "programs": [
            {
                "name": program,
                "sha256": _digest(PROGRAM_TEXTS[program]),
                # The softmax is not planned, so nothing was admitted.
                **(
                    {}
                    if program == "ragged_softmax"
                    else {"small_element_program": program == "segmented_sum"}
                ),
            }
            for program in programs
        ],
        "kernels": kernels,
    }


def _write(root, manifest):
    """Write an artifact directory that no group or other user can write."""
    root.mkdir(mode=0o755)
    files = {
        "manifest.json": json.dumps(manifest).encode(),
        "libSwageRuntime.so": RUNTIME_BYTES,
    }
    for kernel in manifest.get("kernels", ()):
        if isinstance(kernel, dict) and "/" not in str(kernel.get("file")):
            files[kernel["file"]] = (
                f"// {kernel['program']} {kernel['role']}\n".encode()
            )
    for name, contents in files.items():
        (root / name).write_bytes(contents)
        (root / name).chmod(0o644)
    root.chmod(0o755)
    return root


@pytest.fixture
def artifact_dir(tmp_path):
    """Return a complete artifact directory for `sm_86`."""
    return _write(tmp_path / "artifact", _manifest())


def _select(monkeypatch, directory):
    """Name `directory` as the artifact of this process."""
    monkeypatch.setenv("SWAGE_ARTIFACT_DIR", str(directory))


def _load(monkeypatch, directory):
    """Select `directory` and return the loaded artifact."""
    _select(monkeypatch, directory)
    return _artifact.selected()


def _request(program="segmented_sum", role="warp", target="sm_86"):
    """Return what the runner passes for one kernel of one program."""
    kernel = next(
        kernel
        for kernel in _artifact._PROGRAMS[program]
        if kernel.role == role
    )
    return (
        kernel.compiler,
        PROGRAM_TEXTS[program],
        {"kernel_name": program, "target": target, **dict(kernel.options)},
    )


def test_importing_the_loader_stays_light():
    """Import no PyTorch, numpy, or native bindings with the loader."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "import swage\n"
            "from swage import _artifact\n"
            "assert _artifact.selected() is None\n"
            "for name in ('torch', 'numpy', 'mlir_swage'):\n"
            "    assert name not in sys.modules, name",
        ],
        capture_output=True,
        text=True,
        check=False,
        env={
            key: value
            for key, value in os.environ.items()
            if key != "SWAGE_ARTIFACT_DIR"
        },
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("value", [None, ""])
def test_no_artifact_is_selected_without_the_variable(value, monkeypatch):
    """Treat an unset and an empty variable alike, and read nothing."""
    if value is not None:
        monkeypatch.setenv("SWAGE_ARTIFACT_DIR", value)
    monkeypatch.setattr(
        _artifact, "_Artifact", lambda directory: pytest.fail("loaded")
    )

    assert _artifact.selected() is None


def test_the_kernel_table_names_every_kernel_a_public_call_requests():
    """Pin the requests and launch ABIs that writer and loader share."""
    task = (
        ("values", "const float*"),
        ("offsets", "const int32_t*"),
        ("output", "float*"),
        ("task_ids", "const int32_t*"),
        ("value_count", "int32_t"),
        ("task_count", "int32_t"),
        ("segment_count", "int32_t"),
    )
    segmented = "_compile_segmented_reduction_ptx"
    reduction = [
        (
            "warp",
            segmented,
            (("block_size", 32), ("use_task_ids", True)),
            32,
            "",
            task,
        ),
        (
            "cta",
            segmented,
            (("block_size", 128), ("use_task_ids", True)),
            128,
            "",
            task,
        ),
        (
            "mixed",
            "_compile_fused_segmented_reduction_ptx",
            (),
            128,
            "",
            (
                *task[:5],
                ("warp_task_count", "int32_t"),
                ("cta_task_count", "int32_t"),
                ("segment_count", "int32_t"),
            ),
        ),
        (
            "partial",
            "_compile_split_partial_reduction_ptx",
            (),
            512,
            "__partial",
            (
                ("values", "const float*"),
                ("partial_ranges", "const int32_t*"),
                ("scratch", "float*"),
                ("value_count", "int32_t"),
                ("partial_count", "int32_t"),
            ),
        ),
        (
            "merge",
            "_compile_split_merge_reduction_ptx",
            (),
            512,
            "__merge",
            (
                ("scratch", "const float*"),
                ("output", "float*"),
                ("merge_records", "const int32_t*"),
                ("partial_count", "int32_t"),
                ("merge_count", "int32_t"),
                ("segment_count", "int32_t"),
            ),
        ),
    ]
    softmax = [
        (
            "cta",
            segmented,
            (("block_size", 128),),
            128,
            "",
            (
                *task[:3],
                ("value_count", "int32_t"),
                ("segment_count", "int32_t"),
            ),
        )
    ]

    assert {
        name: [tuple(kernel) for kernel in kernels]
        for name, kernels in _artifact._PROGRAMS.items()
    } == {
        "segmented_sum": reduction,
        "segmented_max": reduction,
        "ragged_softmax": softmax,
    }


def _compiler_target_description():
    """Return the integer fields of the target description in the source.

    The runner reads its block widths from the native target description,
    which this tier cannot import, so the test reads the same record where
    the compiler defines it.
    """
    source = (
        pathlib.Path(__file__).parents[2] / "lib/Target/NVIDIATarget.cpp"
    ).read_text()
    record = source[source.index("TargetDescription description = {"):]
    return {
        name: int(value)
        for name, value in re.findall(r"/\*(\w+)=\*/(\d+),", record)
    }


def test_the_kernel_table_uses_the_block_sizes_of_the_runner():
    """Keep the table equal to the widths the runner launches with."""
    description = _compiler_target_description()
    widths = {
        "warp": description["subgroupWidth"],
        "cta": description["ctaBlockThreads"],
        "mixed": description["ctaBlockThreads"],
        "partial": description["splitBlockThreads"],
        "merge": description["splitBlockThreads"],
    }
    assert set(widths.values()) == {32, 128, 512}

    for kernels in _artifact._PROGRAMS.values():
        for kernel in kernels:
            assert kernel.block_size == widths[kernel.role]
            requested = dict(kernel.options).get("block_size")
            assert requested in (None, kernel.block_size)


def _compiler_kernel_layouts():
    """Return the parameter names of each kernel layout in the source.

    The lowerings find every kernel parameter through the layout tables of
    `KernelLayout.h`, and the names of the five arguments a segment function
    declares are the `swage.role` names.
    """
    root = pathlib.Path(__file__).parents[2] / "include/swage/Dialect"
    header = (root / "SwagePlan/IR/KernelLayout.h").read_text()
    names = dict(
        re.findall(r'case KernelArgument::(\w+):\s+return "(\w+)";', header)
    )
    layouts = {
        table: [
            names[argument]
            for argument in re.findall(r"KernelArgument::(\w+)", body)
        ]
        for table, body in re.findall(
            r"KernelArgument (\w+)Arguments\[\] = \{(.*?)\};", header, re.S
        )
    }
    roles = re.findall(
        r'Swage_ArgumentRole\w+\s*: I32EnumAttrCase<"\w+", \d+,\s*"(\w+)">',
        (root / "Swage/IR/SwageOps.td").read_text(),
    )
    return layouts, roles


def test_the_kernel_table_passes_the_arguments_of_the_kernel_layouts():
    """Name and order every launch argument as the lowering emits it."""
    layouts, roles = _compiler_kernel_layouts()
    layout_of = {
        "warp": "taskId",
        "cta": "taskId",
        "mixed": "fusedMixed",
        "partial": "splitPartial",
        "merge": "splitMerge",
    }
    assert sorted(roles) == sorted(layouts["direct"])
    assert len(roles) == 5

    for program, kernels in _artifact._PROGRAMS.items():
        for kernel in kernels:
            # The softmax kernel is the direct kernel: one block per segment.
            layout = (
                "direct"
                if program == "ragged_softmax"
                else layout_of[kernel.role]
            )
            assert [role for role, _ in kernel.arguments] == layouts[layout]


def test_an_artifact_gives_the_runner_its_block_widths_and_limits(
    artifact_dir, monkeypatch
):
    """Answer for the native target description without the bindings."""
    description = _compiler_target_description()
    _select(monkeypatch, artifact_dir)

    runner = qualification._target_description()

    assert runner.subgroup_width == description["subgroupWidth"]
    assert runner.cta_block_threads == description["ctaBlockThreads"]
    assert runner.split_block_threads == description["splitBlockThreads"]
    assert qualification._planning_limits(None, None) == (
        description["defaultWarpMaxElements"],
        description["defaultCtaChunkElements"],
    )
    assert qualification._planning_limits(8, None) == (8, 4096)
    assert sys.modules["mlir_swage"] is None


def test_a_selected_artifact_is_read_and_verified_once(
    artifact_dir, monkeypatch
):
    """Serve later calls from the process, without reading the directory."""
    artifact = _load(monkeypatch, artifact_dir)
    for path in artifact_dir.iterdir():
        path.unlink()

    assert _artifact.selected() is artifact
    assert artifact.target == "sm_86"
    assert artifact.directory == artifact_dir
    assert artifact.kernel(*_request()) == "// segmented_sum warp\n"


def test_an_artifact_serves_every_kernel_of_its_programs(
    artifact_dir, monkeypatch
):
    """Answer each request of the table with the PTX of its file."""
    artifact = _load(monkeypatch, artifact_dir)

    for program, kernels in _artifact._PROGRAMS.items():
        for kernel in kernels:
            assert artifact.kernel(*_request(program, kernel.role)) == (
                f"// {program} {kernel.role}\n"
            )


def test_the_artifact_stands_in_for_the_native_compile_functions(
    artifact_dir, monkeypatch
):
    """Offer the names the runner looks up, stable within one artifact."""
    artifact = _load(monkeypatch, artifact_dir)

    for name in (
        "_compile_segmented_reduction_ptx",
        "_compile_fused_segmented_reduction_ptx",
        "_compile_split_partial_reduction_ptx",
        "_compile_split_merge_reduction_ptx",
    ):
        compiler = getattr(artifact, name)
        assert compiler.__name__ == name
        assert getattr(artifact, name) is compiler
        assert hash(compiler) == hash(getattr(artifact, name))


def test_selecting_another_directory_loads_another_artifact(
    tmp_path, monkeypatch
):
    """Follow the variable, so a kernel memo never crosses two artifacts."""
    first = _load(monkeypatch, _write(tmp_path / "a", _manifest()))
    second = _load(
        monkeypatch, _write(tmp_path / "b", _manifest(target="sm_87"))
    )

    assert second is not first
    assert second.target == "sm_87"
    assert (
        second._compile_segmented_reduction_ptx
        is not first._compile_segmented_reduction_ptx
    )


def test_a_kernel_lends_the_launcher_to_a_driver_without_one(
    artifact_dir, monkeypatch
):
    """Give the driver the runtime library launcher, and keep a better one."""
    artifact = _load(monkeypatch, artifact_dir)
    bare = types.SimpleNamespace(_native_launch=None)
    monkeypatch.setattr(_artifact._runtime, "_get_driver", lambda: bare)
    artifact.kernel(*_request())

    assert bare._native_launch == artifact._launch_kernel

    compiled = types.SimpleNamespace(_native_launch=print)
    monkeypatch.setattr(_artifact._runtime, "_get_driver", lambda: compiled)
    artifact.kernel(*_request())

    assert compiled._native_launch is print


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            lambda root: _remove_tree(root),
            "SWAGE_ARTIFACT_DIR names {root}, which does not exist",
        ),
        (
            lambda root: (root / "manifest.json").unlink(),
            "the artifact at {root} has no manifest.json",
        ),
        (
            lambda root: _rewrite(root / "manifest.json", b"{not json"),
            "the manifest of the artifact at {root} is not valid JSON",
        ),
        (
            lambda root: _rewrite(root / "manifest.json", b"[1, 2]"),
            "the manifest of the artifact at {root} is malformed: found a "
            "list where an object is required",
        ),
        (
            lambda root: (root / "segmented_max.merge.ptx").unlink(),
            "the artifact at {root} lacks segmented_max.merge.ptx, which "
            "its manifest lists",
        ),
        (
            lambda root: _rewrite(
                root / "segmented_sum.mixed.ptx", b"// tampered\n"
            ),
            "the artifact at {root} holds a segmented_sum.mixed.ptx that "
            "does not match its manifest: the SHA-256 is ",
        ),
        (
            lambda root: _rewrite(root / "libSwageRuntime.so", b"other"),
            "the artifact at {root} holds a libSwageRuntime.so that does "
            "not match its manifest: the SHA-256 is ",
        ),
    ],
    ids=[
        "no-directory",
        "no-manifest",
        "not-json",
        "not-an-object",
        "kernel-file-missing",
        "kernel-digest",
        "runtime-digest",
    ],
)
def test_a_damaged_artifact_is_refused(
    change, message, artifact_dir, monkeypatch
):
    """Name the directory and the damage, and load nothing."""
    change(artifact_dir)
    _select(monkeypatch, artifact_dir)

    with pytest.raises(RuntimeError) as error:
        _artifact.selected()

    assert str(error.value).startswith(message.format(root=artifact_dir))
    assert _artifact._selected == (None, None)


def _remove_tree(root):
    """Delete an artifact directory and everything in it."""
    for path in root.iterdir():
        path.unlink()
    root.rmdir()


def _rewrite(path, contents):
    """Replace the contents of one artifact file and keep its mode."""
    path.write_bytes(contents)
    path.chmod(0o644)


def _drop(manifest, key):
    del manifest[key]


def _set(manifest, path, value):
    target = manifest
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            lambda manifest: _set(manifest, ["format_version"], 2),
            "the artifact at {root} has manifest format version 2; this "
            "swage reads format version 1. Write the artifact again with "
            "the swage that loads it",
        ),
        (
            lambda manifest: _drop(manifest, "format_version"),
            "the artifact at {root} has manifest format version None; this "
            "swage reads format version 1.",
        ),
        (
            lambda manifest: _drop(manifest, "target"),
            "the manifest of the artifact at {root} is malformed: target "
            "must be a string",
        ),
        (
            lambda manifest: _set(manifest, ["planning"], [32, 4096]),
            "the manifest of the artifact at {root} is malformed: planning "
            "must be an object",
        ),
        (
            lambda manifest: _set(
                manifest, ["planning", "warp_max_elements"], "32"
            ),
            "the manifest of the artifact at {root} is malformed: "
            "warp_max_elements must be an integer",
        ),
        (
            lambda manifest: _set(
                manifest, ["programs", 0, "small_element_program"], 1
            ),
            "the manifest of the artifact at {root} is malformed: "
            "small_element_program must be a boolean",
        ),
        (
            lambda manifest: _set(manifest, ["kernels"], {}),
            "the manifest of the artifact at {root} is malformed: kernels "
            "must be a list",
        ),
        (
            lambda manifest: _set(
                manifest, ["kernels", 0, "file"], "../outside.ptx"
            ),
            "the manifest of the artifact at {root} is malformed: file "
            "must be a plain file name, found '../outside.ptx'",
        ),
        (
            lambda manifest: _set(
                manifest, ["runtime", "file"], "/lib/libc.so.6"
            ),
            "the manifest of the artifact at {root} is malformed: file "
            "must be a plain file name, found '/lib/libc.so.6'",
        ),
        (
            lambda manifest: _set(manifest, ["kernels", 0, "role"], "warped"),
            "the manifest of the artifact at {root} lists a kernel that "
            "this swage does not know: program 'segmented_sum', role "
            "'warped'",
        ),
        (
            lambda manifest: manifest["kernels"].append(manifest["kernels"][0]),
            "the manifest of the artifact at {root} lists the warp kernel "
            "of segmented_sum twice",
        ),
        (
            lambda manifest: manifest["kernels"].pop(3),
            "the artifact at {root} lacks the partial kernel of "
            "segmented_sum, which a call can launch",
        ),
        (
            lambda manifest: _set(
                manifest, ["programs", 0, "name"], "segmented_min"
            ),
            "the manifest of the artifact at {root} lists the program "
            "'segmented_min', which this swage does not run",
        ),
        (
            lambda manifest: _set(manifest, ["kernels", 2, "block_size"], 256),
            "the mixed kernel of segmented_sum in the artifact at {root} "
            "has block size 256; this swage launches it with 128",
        ),
        (
            lambda manifest: manifest["kernels"][1]["arguments"].pop(),
            "the cta kernel of segmented_sum in the artifact at {root} "
            "takes (values: const float*, offsets: const int32_t*, output: "
            "float*, task_ids: const int32_t*, value_count: int32_t, "
            "task_count: int32_t); this swage passes (values: const float*, "
            "offsets: const int32_t*, output: float*, task_ids: const "
            "int32_t*, value_count: int32_t, task_count: int32_t, "
            "segment_count: int32_t)",
        ),
        (
            lambda manifest: _set(
                manifest, ["kernels", 4, "entry"], "segmented_sum"
            ),
            "the merge kernel of segmented_sum in the artifact at {root} "
            "has entry 'segmented_sum'; this swage loads "
            "'segmented_sum__merge'",
        ),
        (
            lambda manifest: _set(manifest, ["runtime", "machine"], "riscv64"),
            "the runtime library of the artifact at {root} was built for "
            f"riscv64; this host is {platform.machine()}",
        ),
        (
            lambda manifest: _set(manifest, ["runtime", "abi_version"], 2),
            "the runtime library of the artifact at {root} has ABI version "
            "2; this swage calls ABI version 1",
        ),
    ],
    ids=[
        "format-version",
        "no-format-version",
        "no-target",
        "planning-type",
        "limit-type",
        "admission-type",
        "kernels-type",
        "kernel-path",
        "runtime-path",
        "unknown-role",
        "duplicate-kernel",
        "missing-kernel",
        "unknown-program",
        "block-size",
        "arguments",
        "entry",
        "machine",
        "abi-version",
    ],
)
def test_a_manifest_this_swage_cannot_use_is_refused(
    change, message, tmp_path, monkeypatch
):
    """Refuse the whole artifact for one statement that does not fit."""
    manifest = _manifest()
    change(manifest)
    root = _write(tmp_path / "artifact", manifest)
    _select(monkeypatch, root)

    with pytest.raises(RuntimeError) as error:
        _artifact.selected()

    assert str(error.value).startswith(message.format(root=root))


def test_a_library_with_another_abi_version_is_refused(
    artifact_dir, monkeypatch
):
    """Ask the loaded library itself, not only the manifest."""
    monkeypatch.setattr(_FakeLibrary, "abi_version", 3)
    _select(monkeypatch, artifact_dir)

    with pytest.raises(
        RuntimeError,
        match=(
            "^the runtime library of the artifact at .* reports ABI version "
            "3; this swage calls ABI version 1$"
        ),
    ):
        _artifact.selected()


def test_a_library_that_does_not_load_is_refused(artifact_dir, monkeypatch):
    """Report the loader error with the library it concerns."""

    def refuse(path):
        raise OSError(f"{path}: invalid ELF header")

    monkeypatch.setattr(ctypes, "PyDLL", refuse)
    _select(monkeypatch, artifact_dir)

    with pytest.raises(
        RuntimeError,
        match=(
            "^the runtime library of the artifact at .* cannot be loaded: "
            ".*libSwageRuntime.so: invalid ELF header$"
        ),
    ):
        _artifact.selected()


def test_a_read_only_artifact_loads(artifact_dir, monkeypatch):
    """Need no write permission anywhere in the directory."""
    for path in artifact_dir.iterdir():
        path.chmod(0o444)
    artifact_dir.chmod(0o555)
    try:
        artifact = _load(monkeypatch, artifact_dir)
    finally:
        artifact_dir.chmod(0o755)

    assert artifact.target == "sm_86"


def test_an_artifact_owned_by_another_account_loads(artifact_dir, monkeypatch):
    """Never compare the owner of the directory with the current user.

    The directory is named by whoever starts the process, which is the
    trust decision. The other owner is simulated in both ways the reviews
    probed: another effective user, and files that report another owner.
    """
    real_stat, real_fstat = os.stat, os.fstat

    def owned_by_root(details):
        return os.stat_result((*details[:4], 0, 0, *details[6:]))

    monkeypatch.setattr(os, "geteuid", lambda: 54321)
    monkeypatch.setattr(os, "getuid", lambda: 54321)
    monkeypatch.setattr(
        os, "stat", lambda *a, **k: owned_by_root(real_stat(*a, **k))
    )
    monkeypatch.setattr(
        os, "fstat", lambda *a, **k: owned_by_root(real_fstat(*a, **k))
    )

    assert _load(monkeypatch, artifact_dir).target == "sm_86"


@pytest.mark.parametrize("name", [".", "manifest.json", "segmented_sum.cta.ptx",
                                  "libSwageRuntime.so"])
@pytest.mark.parametrize("bit", [stat.S_IWGRP, stat.S_IWOTH])
def test_an_artifact_that_others_can_write_is_refused(
    name, bit, artifact_dir, monkeypatch
):
    """Refuse a directory or file that its group or anyone may change."""
    path = artifact_dir / name
    path.chmod(stat.S_IMODE(path.stat().st_mode) | bit)
    _select(monkeypatch, artifact_dir)

    with pytest.raises(RuntimeError) as error:
        _artifact.selected()

    assert str(error.value) == (
        f"{path.resolve()} of the artifact at {artifact_dir} is writable by "
        f"its group or by other users (mode "
        f"{stat.filemode(path.stat().st_mode)}); remove that permission "
        "with chmod go-w, because a kernel that another account can "
        "replace is not loaded"
    )


def test_symbolic_links_are_followed_and_judged_by_their_target(
    artifact_dir, tmp_path, monkeypatch
):
    """Admit a linked directory and a linked file, and check what they are."""
    store = tmp_path / "store"
    store.mkdir(mode=0o755)
    kernel = artifact_dir / "segmented_sum.warp.ptx"
    stored = store / "warp.ptx"
    stored.write_bytes(kernel.read_bytes())
    stored.chmod(0o644)
    kernel.unlink()
    kernel.symlink_to(stored)
    link = tmp_path / "current"
    link.symlink_to(artifact_dir)

    artifact = _load(monkeypatch, link)

    assert artifact.kernel(*_request()) == "// segmented_sum warp\n"

    stored.chmod(0o664)
    monkeypatch.setattr(_artifact, "_selected", (None, None))
    with pytest.raises(RuntimeError, match="is writable by its group"):
        _artifact.selected()


def test_a_file_in_place_of_the_directory_is_refused(tmp_path, monkeypatch):
    """Require a directory under the name the variable gives."""
    path = tmp_path / "artifact"
    path.write_text("not a directory")
    path.chmod(0o644)
    _select(monkeypatch, path)

    with pytest.raises(
        RuntimeError,
        match=f"^SWAGE_ARTIFACT_DIR names {path}, which is not a directory$",
    ):
        _artifact.selected()


def test_a_kernel_for_another_target_is_refused(artifact_dir, monkeypatch):
    """Never load PTX of one processor on a device of another."""
    artifact = _load(monkeypatch, artifact_dir)

    with pytest.raises(RuntimeError) as error:
        artifact.kernel(*_request(target="sm_87"))

    assert str(error.value) == (
        f"the artifact at {artifact_dir} holds kernels for sm_86, and the "
        "current device needs sm_87; nothing was compiled or launched. "
        "Select an artifact that was written for sm_87"
    )


def test_a_program_the_artifact_does_not_hold_is_refused(
    tmp_path, monkeypatch
):
    """Name the missing kernel and what the artifact holds instead."""
    root = _write(tmp_path / "artifact", _manifest(programs=["segmented_sum"]))
    artifact = _load(monkeypatch, root)

    with pytest.raises(RuntimeError) as error:
        artifact.kernel(*_request("segmented_max"))

    assert str(error.value) == (
        f"the artifact at {root} holds no kernel 'segmented_max' "
        "(_compile_segmented_reduction_ptx, block_size=32, "
        "use_task_ids=True); it holds the programs segmented_sum. Nothing "
        "was compiled or launched"
    )


def test_a_request_outside_the_table_is_refused(artifact_dir, monkeypatch):
    """Serve only the kernels of the public calls, at their block sizes."""
    artifact = _load(monkeypatch, artifact_dir)
    compiler, text, options = _request()

    with pytest.raises(
        RuntimeError,
        match=(
            r"holds no kernel 'segmented_sum' \(_compile_segmented_reduction"
            r"_ptx, block_size=64, use_task_ids=True\); it holds the "
            "programs segmented_sum, segmented_max, ragged_softmax"
        ),
    ):
        artifact.kernel(compiler, text, {**options, "block_size": 64})
    with pytest.raises(RuntimeError, match="holds no kernel 'segmented_sum'"):
        artifact.kernel(
            "_compile_persistent_segmented_reduction_ptx",
            text,
            {"kernel_name": "segmented_sum", "target": "sm_86"},
        )


def test_a_program_text_other_than_the_compiled_one_is_refused(
    artifact_dir, monkeypatch
):
    """Tie each kernel to the program text it was compiled from."""
    artifact = _load(monkeypatch, artifact_dir)
    compiler, text, options = _request()

    with pytest.raises(RuntimeError) as error:
        artifact.kernel(compiler, text + "\n", options)

    assert str(error.value) == (
        f"the artifact at {artifact_dir} was compiled from another "
        "'segmented_sum' program than this swage runs (SHA-256 "
        f"{_digest(PROGRAM_TEXTS['segmented_sum'])} in the manifest, "
        f"{_digest(text + chr(10))} here); nothing was compiled or "
        "launched. Write the artifact again with the swage that loads it"
    )


def test_admission_returns_what_the_build_host_recorded(
    artifact_dir, monkeypatch
):
    """Answer the planning admission without the native bindings."""
    artifact = _load(monkeypatch, artifact_dir)

    assert artifact.admit(PROGRAM_TEXTS["segmented_sum"], 32, 4096) is True
    assert artifact.admit(PROGRAM_TEXTS["segmented_max"], 32, 4096) is False


def test_admission_refuses_other_planning_limits(artifact_dir, monkeypatch):
    """Serve only the limits the kernels were admitted under."""
    artifact = _load(monkeypatch, artifact_dir)

    with pytest.raises(RuntimeError) as error:
        artifact.admit(PROGRAM_TEXTS["segmented_sum"], 16, 4096)

    assert str(error.value) == (
        f"the artifact at {artifact_dir} was planned for "
        "warp_max_elements=32 and cta_chunk_elements=4096, and this call "
        "plans with warp_max_elements=16 and cta_chunk_elements=4096; "
        "nothing was compiled or launched"
    )


@pytest.mark.parametrize(
    "text", ["module {}", PROGRAM_TEXTS["ragged_softmax"]]
)
def test_admission_refuses_a_program_the_build_host_did_not_plan(
    text, artifact_dir, monkeypatch
):
    """Admit no program by default, and none that was not planned."""
    artifact = _load(monkeypatch, artifact_dir)

    with pytest.raises(RuntimeError) as error:
        artifact.admit(text, 32, 4096)

    assert str(error.value) == (
        f"the artifact at {artifact_dir} holds no planned program with the "
        f"text of this call (SHA-256 {_digest(text)}); it holds the planned "
        "programs segmented_sum, segmented_max. The artifact was written "
        "without this program, or from another program text than this "
        "swage runs. Nothing was compiled or launched"
    )


def test_the_loader_never_imports_the_native_bindings(
    artifact_dir, monkeypatch
):
    """Load, serve, and admit with `mlir_swage` unimportable."""
    artifact = _load(monkeypatch, artifact_dir)
    artifact.kernel(*_request("ragged_softmax", "cta"))
    artifact.admit(PROGRAM_TEXTS["segmented_sum"], 32, 4096)

    assert sys.modules["mlir_swage"] is None


def test_the_manifest_is_kept_for_reports(artifact_dir, monkeypatch):
    """Expose what the artifact says about itself."""
    artifact = _load(monkeypatch, artifact_dir)

    assert artifact.manifest == json.loads(
        (artifact_dir / "manifest.json").read_text()
    )
    assert artifact.programs == (
        "segmented_sum",
        "segmented_max",
        "ragged_softmax",
    )


def test_a_relative_directory_is_resolved_when_it_is_selected(
    artifact_dir, monkeypatch
):
    """Name the absolute directory in every later message."""
    monkeypatch.chdir(artifact_dir.parent)
    artifact = _load(monkeypatch, artifact_dir.name)

    assert artifact.directory == pathlib.Path(artifact_dir)


@pytest.fixture
def empty_kernel_memo(monkeypatch):
    """Give the test a process that holds no compiled segmented kernel."""
    monkeypatch.setattr(
        qualification,
        "_ptx_memo",
        _artifact._runtime._BoundedCache(_artifact._runtime._CACHE_LIMIT),
    )


def test_the_runner_takes_the_artifact_for_the_native_bindings(
    artifact_dir, monkeypatch
):
    """Hand the runner the artifact, and the bindings only without one."""
    with pytest.raises(ImportError):
        qualification._native_swage()

    artifact = _load(monkeypatch, artifact_dir)

    assert qualification._native_swage() is artifact


def test_the_runner_serves_a_kernel_from_the_artifact(
    artifact_dir, empty_kernel_memo, monkeypatch
):
    """Answer a kernel request without parsing or compiling anything."""
    artifact = _load(monkeypatch, artifact_dir)

    def compile_from(artifact):
        return qualification._compile_once(
            artifact._compile_segmented_reduction_ptx,
            PROGRAM_TEXTS["segmented_sum"],
            kernel_name="segmented_sum",
            block_size=128,
            target="sm_86",
            use_task_ids=True,
        )

    ptx = compile_from(artifact)
    for path in artifact_dir.iterdir():
        path.unlink()

    assert ptx == "// segmented_sum cta\n"
    assert compile_from(artifact) is ptx
    assert sys.modules["mlir_swage"] is None


def test_an_artifact_serves_kernels_while_compiling_is_switched_off(
    artifact_dir, empty_kernel_memo, monkeypatch
):
    """Count a kernel of the artifact as held, not as compiled."""
    artifact = _load(monkeypatch, artifact_dir)
    monkeypatch.setenv("SWAGE_NO_COMPILE", "1")

    assert qualification._compile_once(
        artifact._compile_fused_segmented_reduction_ptx,
        PROGRAM_TEXTS["segmented_max"],
        kernel_name="segmented_max",
        target="sm_86",
    ) == "// segmented_max mixed\n"


def test_the_runner_compiles_nothing_while_an_artifact_is_selected(
    artifact_dir, empty_kernel_memo, monkeypatch
):
    """Refuse a kernel the artifact lacks, whatever names the compiler."""
    _select(monkeypatch, artifact_dir)

    def native_compile(module, **options):
        pytest.fail("a kernel was compiled")

    native_compile.__name__ = "_compile_persistent_segmented_reduction_ptx"

    with pytest.raises(RuntimeError, match="holds no kernel 'segmented_sum'"):
        qualification._compile_once(
            native_compile,
            PROGRAM_TEXTS["segmented_sum"],
            kernel_name="segmented_sum",
            target="sm_86",
        )
    native_compile.__name__ = "_compile_fused_segmented_reduction_ptx"
    with pytest.raises(RuntimeError, match="the current device needs sm_120"):
        qualification._compile_once(
            native_compile,
            PROGRAM_TEXTS["segmented_sum"],
            kernel_name="segmented_sum",
            target="sm_120",
        )


def test_the_runner_admits_a_program_from_the_artifact(
    artifact_dir, monkeypatch
):
    """Skip planning admission, which needs the native bindings."""
    _select(monkeypatch, artifact_dir)
    monkeypatch.setattr(
        qualification,
        "_admitted",
        _artifact._runtime._BoundedCache(_artifact._runtime._CACHE_LIMIT),
    )

    for _ in range(2):
        assert (
            qualification._admit_program(
                PROGRAM_TEXTS["segmented_max"], "segmented_max", 32, 4096
            )
            is False
        )
    with pytest.raises(RuntimeError, match="was planned for"):
        qualification._admit_program(
            PROGRAM_TEXTS["segmented_max"], "segmented_max", 8, 64
        )
    assert sys.modules["mlir_swage"] is None


class _Device:
    """The device of a stand-in tensor."""

    type = "cuda"
    index = 0


class _Tensor:
    """The metadata of one rank-one tensor, as the public checks read it."""

    device = _Device()
    requires_grad = False

    def __init__(self, torch, count, *, integer=False, pointer=0x1000):
        self.dtype = torch.int32 if integer else torch.float32
        self._count = count
        self._pointer = pointer

    def dim(self):
        return 1

    def numel(self):
        return self._count

    def is_contiguous(self):
        return True

    def is_neg(self):
        return False

    def is_conj(self):
        return False

    def is_inference(self):
        return False

    def element_size(self):
        return 4

    def data_ptr(self):
        return self._pointer

    def record_stream(self, stream):
        raise AssertionError("a tensor was retained without a launch")


class _Reached(Exception):
    """Raised by the stand-in for PyTorch at a step a test watches for."""


def _reached(step):
    """Return a function that raises `_Reached` naming `step`."""

    def raise_reached(*arguments, **keywords):
        raise _Reached(step)

    return raise_reached


def _fake_torch(monkeypatch):
    """Install a PyTorch stand-in that passes the launch requirements."""
    torch = types.ModuleType("torch")
    torch.__version__ = "2.6.0"
    torch.float32 = object()
    torch.int32 = object()
    torch.Tensor = _Tensor
    torch.autograd = types.SimpleNamespace(
        graph=types.SimpleNamespace(increment_version=lambda tensor: None)
    )
    torch.cuda = types.SimpleNamespace(
        is_current_stream_capturing=_reached("capture check")
    )
    torch.empty = _reached("result allocation")
    monkeypatch.setitem(sys.modules, "torch", torch)
    # The calls also require numpy; a stand-in keeps this tier free of it.
    monkeypatch.setitem(sys.modules, "numpy", types.ModuleType("numpy"))
    return torch


CALLS = [
    lambda values, offsets, **keywords: swage.segment_reduce(
        values, offsets, "sum", **keywords
    ),
    lambda values, offsets, **keywords: swage.segment_softmax(
        values, offsets, **keywords
    ),
]


@pytest.mark.parametrize("call", CALLS, ids=["reduce", "softmax"])
def test_a_public_call_verifies_the_artifact_where_it_needs_the_bindings(
    call, artifact_dir, monkeypatch
):
    """Refuse a damaged artifact after the argument checks and before CUDA."""
    torch = _fake_torch(monkeypatch)
    values = _Tensor(torch, 6)
    offsets = _Tensor(torch, 5, integer=True, pointer=0x2000)
    (artifact_dir / "ragged_softmax.cta.ptx").write_text("// other\n")
    _select(monkeypatch, artifact_dir)

    with pytest.raises(TypeError, match="^out must be a torch.Tensor or None"):
        call(values, offsets, out=[0.0])
    with pytest.raises(
        RuntimeError, match="holds a ragged_softmax.cta.ptx that does not match"
    ):
        call(values, offsets)


@pytest.mark.parametrize("call", CALLS, ids=["reduce", "softmax"])
def test_a_public_call_needs_no_bindings_with_an_artifact(
    call, artifact_dir, monkeypatch
):
    """Pass the bindings check on a wheel-only install with an artifact."""
    torch = _fake_torch(monkeypatch)
    values = _Tensor(torch, 6)
    offsets = _Tensor(torch, 5, integer=True, pointer=0x2000)
    _select(monkeypatch, artifact_dir)

    with pytest.raises(_Reached, match="^capture check$"):
        call(values, offsets)

    assert sys.modules["mlir_swage"] is None


@pytest.mark.parametrize("call", CALLS, ids=["reduce", "softmax"])
def test_a_public_call_from_an_artifact_names_numpy_when_it_is_missing(
    call, artifact_dir, monkeypatch
):
    """Require numpy with an artifact too, which classifies through it."""
    torch = _fake_torch(monkeypatch)
    values = _Tensor(torch, 6)
    offsets = _Tensor(torch, 5, integer=True, pointer=0x2000)
    _select(monkeypatch, artifact_dir)
    monkeypatch.setitem(sys.modules, "numpy", None)

    with pytest.raises(
        RuntimeError,
        match=(
            r"^Swage segment_(reduce|softmax)\(\) requires numpy, which "
            "cannot be imported; nothing was launched"
        ),
    ):
        call(values, offsets)

    assert sys.modules["mlir_swage"] is None


def test_the_compile_command_stays_light_and_needs_the_bindings(tmp_path):
    """Import nothing heavy, and name what a wheel-only install lacks."""
    output = tmp_path / "artifact"
    script = (
        "import sys\n"
        "sys.modules['mlir_swage'] = None\n"
        "from swage import compile\n"
        "for name in ('torch', 'numpy'):\n"
        "    assert name not in sys.modules, name\n"
        "sys.exit(compile._main(sys.argv[1:]))\n"
    )

    completed = subprocess.run(
        [sys.executable, "-c", script, "--target", "sm_86", "--output",
         str(output)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    assert completed.stdout == ""
    assert completed.stderr == (
        "error: writing an artifact requires the mlir_swage bindings, which "
        "the swage-compiler wheel does not include. See "
        "docs/getting-started/installation.md in "
        "https://github.com/abhiksark/swage for the native build\n"
    )
    assert not output.exists()


def test_the_environment_report_names_no_artifact_without_the_variable():
    """Say that segmented calls compile in the reporting process."""
    assert env.report()["artifact"] == "none (SWAGE_ARTIFACT_DIR is unset)"


def test_the_environment_report_describes_the_selected_artifact(
    artifact_dir, monkeypatch
):
    """Name the directory, what it was written for, and what it holds."""
    _select(monkeypatch, artifact_dir)

    assert env.report()["artifact"] == (
        f"{artifact_dir} (format 1, target sm_86, 11 kernels of "
        "segmented_sum, segmented_max, ragged_softmax, written by swage "
        f"{swage.__version__} at revision "
        "0123456789abcdef0123456789abcdef01234567)"
    )


def test_the_environment_report_says_why_an_artifact_is_rejected(
    artifact_dir, monkeypatch
):
    """Report the refusal a segmented call would raise, without raising."""
    (artifact_dir / "manifest.json").unlink()
    _select(monkeypatch, artifact_dir)

    assert env.report()["artifact"] == (
        f"rejected (the artifact at {artifact_dir} has no manifest.json)"
    )


def test_the_environment_command_reports_an_artifact_it_cannot_load(
    artifact_dir,
):
    """Exit cleanly on an artifact whose library is not a library."""
    completed = subprocess.run(
        [sys.executable, "-m", "swage.env"],
        capture_output=True,
        text=True,
        check=False,
        env=dict(os.environ, SWAGE_ARTIFACT_DIR=str(artifact_dir)),
    )

    assert completed.returncode == 0, completed.stderr
    assert (
        f"artifact: rejected (the runtime library of the artifact at "
        f"{artifact_dir} cannot be loaded: "
    ) in completed.stdout


def test_the_compile_module_defines_no_public_name():
    """Offer the command and nothing to call."""
    from swage import compile

    defined = [
        name
        for name, value in vars(compile).items()
        if not name.startswith("_")
        and getattr(value, "__module__", None) == compile.__name__
    ]

    assert defined == []
    assert swage.__all__ == [
        "CompilationError",
        "jit",
        "segment_reduce",
        "segment_softmax",
    ]
