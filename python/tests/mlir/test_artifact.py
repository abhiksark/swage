# python/tests/mlir/test_artifact.py
"""Artifacts that the compiler writes, and calls that run from them.

`python -m swage.compile` writes the kernels of the public segmented calls
for one target, and `SWAGE_ARTIFACT_DIR` makes a process run those calls
from the directory. This file pins three things:

- What the command writes. These tests need the native bindings and no GPU.
- That a written artifact loads and refuses what it cannot serve.
- On the GPU, that a process without `mlir_swage`, and with no LLVM or
  MLIR library mapped, returns results that agree with PyTorch and with a
  float64 reference and that equal the compiled path bit for bit.
"""

import contextlib
import hashlib
import inspect
import io
import json
import os
import pathlib
import platform
import re
import shutil
import stat
import subprocess
import sys

import pytest
import swage
import torch
from artifact_child import COMPILER_LIBRARY
from mlir_swage import ir
from mlir_swage._mlir_libs._swageDialectsNanobind import swage as native_swage
from mlir_swage.dialects import swage as swage_dialect
from swage import _artifact, _runtime, compile
from swage import _segmented_qualification as qualification
from test_public_segments import (
    DISTRIBUTIONS,
    _assert_reduction_matches,
    _assert_softmax_matches,
    _distributions,
    _host_case,
)
from test_segmented_runtime import _bits, _offsets

_needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"
)
_SENTINEL = -5.0
_CHILD = pathlib.Path(__file__).with_name("artifact_child.py")
_PROGRAMS = {
    "segmented_sum": qualification._semantic_module("sum"),
    "segmented_max": qualification._semantic_module("max"),
    "ragged_softmax": qualification._SOFTMAX_MODULE,
}


def _run(*arguments):
    """Run the command in this process and return its status and output."""
    output, errors = io.StringIO(), io.StringIO()
    with (
        contextlib.redirect_stdout(output),
        contextlib.redirect_stderr(errors),
    ):
        status = compile.main([str(argument) for argument in arguments])
    return status, output.getvalue(), errors.getvalue()


def _written(directory, target, *arguments):
    """Write an artifact for `target` at `directory` and return it."""
    status, _, errors = _run(
        "--target", target, "--output", directory, *arguments
    )
    assert status == 0, errors
    return directory


def _device_target():
    """Return the NVPTX processor of the current CUDA device."""
    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}"


@pytest.fixture(autouse=True)
def _no_artifact_selected(monkeypatch):
    """Start every test without a selected artifact."""
    monkeypatch.delenv("SWAGE_ARTIFACT_DIR", raising=False)


@pytest.fixture(scope="module")
def written(tmp_path_factory):
    """Return one artifact for `sm_87`, a target no test executes on."""
    return _written(tmp_path_factory.mktemp("written") / "artifact", "sm_87")


@pytest.fixture(scope="module")
def manifest(written):
    """Return the manifest of the `sm_87` artifact."""
    return json.loads((written / "manifest.json").read_text())


def _kernel_ids():
    """Return `(program, role)` for every kernel an artifact holds."""
    return [
        (program, kernel.role)
        for program, kernels in _artifact._PROGRAMS.items()
        for kernel in kernels
    ]


def test_the_command_writes_the_kernels_the_library_and_a_manifest(
    written, manifest
):
    """Write eleven kernels, the runtime library, and what describes them."""
    names = sorted(path.name for path in written.iterdir())

    assert names == sorted(
        [
            "manifest.json",
            "libSwageRuntime.so",
            *[f"{program}.{role}.ptx" for program, role in _kernel_ids()],
        ]
    )
    assert len(_kernel_ids()) == 11
    assert [key for key in manifest] == [
        "format_version",
        "swage_version",
        "source_revision",
        "llvm_version",
        "target",
        "planning",
        "runtime",
        "programs",
        "kernels",
    ]
    assert manifest["format_version"] == 1
    assert manifest["swage_version"] == swage.__version__
    assert manifest["source_revision"] == native_swage.__source_revision__
    assert manifest["llvm_version"] == native_swage.__llvm_version__
    assert manifest["target"] == "sm_87"
    assert manifest["planning"] == {
        "warp_max_elements": 32,
        "cta_chunk_elements": 4096,
    }


def test_the_manifest_states_the_planning_limits_of_the_public_call():
    """Admit the programs under the limits `segment_reduce` plans with."""
    defaults = inspect.signature(
        qualification._prepare_planned_reduction
    ).parameters

    assert compile._WARP_MAX_ELEMENTS == defaults["warp_max_elements"].default
    assert (
        compile._CTA_CHUNK_ELEMENTS == defaults["cta_chunk_elements"].default
    )


def test_the_manifest_identifies_each_program(manifest):
    """Record the text digest and the planning admission of each program."""
    assert manifest["programs"] == [
        {
            "name": "segmented_sum",
            "sha256": hashlib.sha256(
                _PROGRAMS["segmented_sum"].encode()
            ).hexdigest(),
            "small_element_program": True,
        },
        {
            "name": "segmented_max",
            "sha256": hashlib.sha256(
                _PROGRAMS["segmented_max"].encode()
            ).hexdigest(),
            "small_element_program": True,
        },
        {
            "name": "ragged_softmax",
            "sha256": hashlib.sha256(
                _PROGRAMS["ragged_softmax"].encode()
            ).hexdigest(),
        },
    ]
    for name in ("segmented_sum", "segmented_max"):
        assert qualification._admit_program(_PROGRAMS[name], 32, 4096) is True


def test_the_manifest_identifies_the_runtime_library(written, manifest):
    """Record the digest, the machine, and the ABI of the shipped library."""
    library = written / "libSwageRuntime.so"
    packaged = compile._packaged_runtime()

    assert manifest["runtime"] == {
        "file": "libSwageRuntime.so",
        "sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "machine": platform.machine(),
        "abi_version": 1,
    }
    assert library.read_bytes() == packaged.read_bytes()
    assert packaged.parent.name == "_mlir_libs"


@pytest.mark.parametrize(("program", "role"), _kernel_ids())
def test_each_kernel_is_what_the_runner_compiles(
    program, role, written, manifest
):
    """Ship the PTX that the compiled path would load for the same request."""
    known = next(
        kernel
        for kernel in _artifact._PROGRAMS[program]
        if kernel.role == role
    )
    entry = next(
        kernel
        for kernel in manifest["kernels"]
        if (kernel["program"], kernel["role"]) == (program, role)
    )
    # A compile of its own, not the memo the command filled.
    with ir.Context() as context:
        swage_dialect.register_dialects(context)
        _, compiled = getattr(native_swage, known.compiler)(
            ir.Module.parse(_PROGRAMS[program]),
            kernel_name=program,
            target="sm_87",
            **dict(known.options),
        )
    shipped = (written / entry["file"]).read_bytes()

    assert shipped == compiled.encode()
    assert entry["sha256"] == hashlib.sha256(shipped).hexdigest()
    assert entry["file"] == f"{program}.{role}.ptx"


@pytest.mark.parametrize(("program", "role"), _kernel_ids())
def test_the_manifest_describes_each_kernel_as_its_ptx_declares_it(
    program, role, written, manifest
):
    """State the entry, the width, and the parameters that the PTX has."""
    entry = next(
        kernel
        for kernel in manifest["kernels"]
        if (kernel["program"], kernel["role"]) == (program, role)
    )
    ptx = (written / entry["file"]).read_text()
    declaration = re.search(
        rf"\.entry {re.escape(entry['entry'])}\((.*?)\)", ptx, re.DOTALL
    )
    parameters = [
        "pointer" if ".ptr" in parameter else parameter.split()[1]
        for parameter in declaration[1].split(",")
    ]
    stated = [
        "pointer" if argument["type"].endswith("*") else ".u32"
        for argument in entry["arguments"]
    ]

    assert ".target sm_87" in ptx
    assert f".reqntid {entry['block_size']}, 1, 1" in ptx
    assert len(re.findall(r"\.entry ", ptx)) == 1
    assert parameters == stated
    assert {argument["type"] for argument in entry["arguments"]} <= {
        "const float*",
        "float*",
        "const int32_t*",
        "int32_t",
    }
    roles = [argument["role"] for argument in entry["arguments"]]
    assert len(set(roles)) == len(roles)


def test_the_command_reports_what_it_wrote(tmp_path):
    """Print the directory, the contents, and the digest of the manifest."""
    output = tmp_path / "artifact"

    status, printed, errors = _run("--target", "sm_86", "--output", output)

    assert (status, errors) == (0, "")
    assert printed.splitlines() == [
        f"artifact: {output}",
        "format_version: 1",
        "target: sm_86",
        "programs: segmented_sum, segmented_max, ragged_softmax",
        "kernels: 11",
        f"runtime: libSwageRuntime.so ({platform.machine()})",
        "manifest_sha256: "
        + hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest(),
    ]


def test_the_command_needs_no_gpu_and_no_pytorch(tmp_path):
    """Compile for another target with no visible device and no PyTorch."""
    output = tmp_path / "artifact"
    script = (
        "import sys\n"
        "sys.modules['torch'] = None\n"
        "from swage import compile\n"
        "status = compile.main(sys.argv[1:])\n"
        "assert sys.modules['torch'] is None\n"
        "sys.exit(status)\n"
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            "--target",
            "sm_120",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "target: sm_120" in completed.stdout
    assert ".target sm_120" in (output / "segmented_sum.mixed.ptx").read_text()


def test_the_command_runs_as_a_module(tmp_path):
    """Offer the entry point as `python -m swage.compile`."""
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "swage.compile",
            "--target",
            "sm_86",
            "--output",
            str(tmp_path / "artifact"),
            "--program",
            "softmax",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "programs: ragged_softmax\nkernels: 1\n" in completed.stdout


def test_the_command_writes_only_the_programs_it_is_given(tmp_path):
    """Leave out the kernels of a program that was not asked for."""
    output = _written(
        tmp_path / "artifact", "sm_86", "--program", "max", "--program", "sum"
    )
    written = json.loads((output / "manifest.json").read_text())

    assert [program["name"] for program in written["programs"]] == [
        "segmented_sum",
        "segmented_max",
    ]
    assert len(written["kernels"]) == 10
    assert not (output / "ragged_softmax.cta.ptx").exists()


def test_an_artifact_can_be_read_and_not_changed_by_other_accounts(tmp_path):
    """Create the directory and its files without group or other write.

    The loader refuses an artifact that its group or others can write, so
    the command must not depend on the umask to meet its own rule.
    """
    previous = os.umask(0)
    try:
        output = _written(tmp_path / "artifact", "sm_86")
    finally:
        os.umask(previous)

    assert stat.S_IMODE(output.stat().st_mode) == 0o755
    assert {
        stat.S_IMODE(path.stat().st_mode) for path in output.iterdir()
    } == {0o644}


def test_the_command_follows_a_restrictive_umask(tmp_path):
    """Give other accounts no more than the umask of the process allows."""
    previous = os.umask(0o077)
    try:
        output = _written(tmp_path / "artifact", "sm_86")
    finally:
        os.umask(previous)

    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    assert {
        stat.S_IMODE(path.stat().st_mode) for path in output.iterdir()
    } == {0o600}


def _refused(tmp_path, *arguments):
    """Run a command that must fail, and return what it printed."""
    status, printed, errors = _run(*arguments)

    assert status == 1
    assert printed == ""
    assert not [path for path in tmp_path.iterdir() if "staging" in path.name]
    return errors


def test_the_command_refuses_an_existing_directory(tmp_path):
    """Write an artifact once, never into or over a directory."""
    output = tmp_path / "artifact"
    output.mkdir()

    errors = _refused(tmp_path, "--target", "sm_86", "--output", output)

    assert errors == (
        f"error: {output} exists; an artifact is written once, to a new "
        "directory\n"
    )
    assert list(output.iterdir()) == []


@pytest.mark.parametrize("target", ["sm_72", "sm_85", "cpu"])
def test_the_command_refuses_a_target_the_compiler_rejects(target, tmp_path):
    """Report the diagnostic of the compiler and write nothing."""
    output = tmp_path / "artifact"

    errors = _refused(tmp_path, "--target", target, "--output", output)

    assert errors.startswith("error: ")
    assert target in errors
    assert not output.exists()


def test_the_command_refuses_to_run_while_an_artifact_is_selected(
    written, tmp_path, monkeypatch
):
    """Compile nothing in a process that would be served kernels."""
    monkeypatch.setenv("SWAGE_ARTIFACT_DIR", str(written))
    output = tmp_path / "artifact"

    errors = _refused(tmp_path, "--target", "sm_86", "--output", output)

    assert errors == (
        "error: SWAGE_ARTIFACT_DIR is set, so this process would be served "
        "kernels instead of compiling them; unset it to write an artifact\n"
    )
    assert not output.exists()


def test_the_command_obeys_the_no_compile_switch(tmp_path, monkeypatch):
    """Refuse to compile on a host whose operator switched compiling off."""
    monkeypatch.setattr(
        qualification,
        "_ptx_memo",
        _runtime._BoundedCache(_runtime._CACHE_LIMIT),
    )
    monkeypatch.setenv("SWAGE_NO_COMPILE", "1")
    output = tmp_path / "artifact"

    errors = _refused(tmp_path, "--target", "sm_86", "--output", output)

    assert errors.startswith(
        "error: SWAGE_NO_COMPILE=1 refuses to compile kernel 'segmented_sum'"
    )
    assert not output.exists()


def _foreign_library(directory):
    """Write the packaged library with the machine field of AArch64.

    Only the two bytes of `e_machine` differ, so the file names another
    machine and holds no code for it. It is never loaded: the loader refuses
    an artifact for the machine of its library before it loads anything.
    """
    packaged = compile._packaged_runtime().read_bytes()
    assert int.from_bytes(packaged[18:20], "little") == 62
    foreign = directory / "libSwageRuntime.so"
    foreign.write_bytes(
        packaged[:18] + (183).to_bytes(2, "little") + packaged[20:]
    )
    return foreign


@pytest.mark.skipif(
    platform.machine() != "x86_64", reason="written for an x86-64 host"
)
def test_the_command_ships_the_runtime_library_it_is_given(tmp_path):
    """Take the library of another host, and record its machine."""
    foreign = _foreign_library(tmp_path)

    output = _written(
        tmp_path / "artifact", "sm_87", "--runtime-library", foreign
    )
    written = json.loads((output / "manifest.json").read_text())

    assert written["runtime"] == {
        "file": "libSwageRuntime.so",
        "sha256": hashlib.sha256(foreign.read_bytes()).hexdigest(),
        "machine": "aarch64",
        "abi_version": 1,
    }
    assert (output / "libSwageRuntime.so").read_bytes() == foreign.read_bytes()


@pytest.mark.skipif(
    platform.machine() != "x86_64", reason="written for an x86-64 host"
)
def test_a_runtime_library_of_another_machine_is_refused_at_load(
    tmp_path, monkeypatch
):
    """Refuse the artifact before the library is handed to the loader."""
    foreign = _foreign_library(tmp_path)
    output = _written(
        tmp_path / "artifact", "sm_87", "--runtime-library", foreign
    )
    monkeypatch.setenv("SWAGE_ARTIFACT_DIR", str(output))
    monkeypatch.setattr(_artifact, "_selected", (None, None))

    with pytest.raises(
        RuntimeError,
        match=(
            "^the runtime library of the artifact at .* was built for "
            f"aarch64; this host is {platform.machine()}$"
        ),
    ):
        _artifact.selected()


def test_the_command_refuses_a_file_that_is_not_a_library(tmp_path):
    """Name the file that cannot be the runtime library."""
    other = tmp_path / "notes.txt"
    other.write_text("not a library")

    errors = _refused(
        tmp_path,
        "--target",
        "sm_86",
        "--output",
        tmp_path / "artifact",
        "--runtime-library",
        other,
    )

    assert errors == (
        f"error: {other} is not an ELF library for one of aarch64, x86_64\n"
    )


@pytest.fixture
def selected(written, monkeypatch):
    """Select the `sm_87` artifact and return it loaded."""
    monkeypatch.setenv("SWAGE_ARTIFACT_DIR", str(written))
    monkeypatch.setattr(_artifact, "_selected", (None, None))
    return _artifact.selected()


def test_a_written_artifact_loads(selected, written, manifest):
    """Pass every check of the loader on what the command wrote."""
    assert selected.directory == written
    assert selected.target == "sm_87"
    assert selected.manifest == manifest
    assert selected.programs == (
        "segmented_sum",
        "segmented_max",
        "ragged_softmax",
    )


def test_a_written_artifact_survives_a_copy_that_keeps_its_modes(
    written, tmp_path, monkeypatch
):
    """Load from another place, as after a transfer to another host."""
    copied = tmp_path / "copied"
    shutil.copytree(written, copied)
    monkeypatch.setenv("SWAGE_ARTIFACT_DIR", str(copied))
    monkeypatch.setattr(_artifact, "_selected", (None, None))

    assert _artifact.selected().directory == copied


_BAD_GEOMETRY = [
    ((0, 0, 32, 0, (0,), (0,)), "grid_x must be a positive u32"),
    ((0, 1 << 32, 32, 0, (0,), (0,)), "grid_x must be a positive u32"),
    ((0, 1, 0, 0, (0,), (0,)), "block_x must be in 1..1024"),
    ((0, 1, 1025, 0, (0,), (0,)), "block_x must be in 1..1024"),
    ((0, 1, 32, 0, (0,) * 10, (0,) * 7), "too many kernel arguments"),
]


@pytest.mark.parametrize(("arguments", "message"), _BAD_GEOMETRY)
def test_the_runtime_library_launcher_checks_its_geometry(
    arguments, message, selected
):
    """Refuse a launch the driver cannot take, without asking the driver."""
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        selected._launch_kernel(*arguments)


@_needs_cuda
@pytest.mark.parametrize(("arguments", "message"), _BAD_GEOMETRY)
def test_both_launchers_refuse_a_geometry_in_the_same_words(
    arguments, message, selected
):
    """Raise what the launcher of the bindings raises for the same call."""
    with pytest.raises(ValueError) as bindings:
        native_swage._launch_kernel(*arguments)
    with pytest.raises(ValueError) as library:
        selected._launch_kernel(*arguments)

    assert str(library.value) == str(bindings.value) == message


@_needs_cuda
def test_the_runtime_library_launcher_surfaces_driver_errors(selected):
    """Report a refused launch in the words of the other launchers."""
    torch.cuda.init()
    torch.zeros(1, device="cuda")
    arguments = (0, 1, 32, 0, (0, 0, 0), (0, 0))

    with pytest.raises(RuntimeError) as shim:
        selected._launch_kernel(*arguments)
    with pytest.raises(RuntimeError) as bindings:
        native_swage._launch_kernel(*arguments)

    assert str(shim.value) == str(bindings.value)
    assert str(shim.value).startswith("CUDA Driver cuLaunchKernel failed: ")


@pytest.fixture(scope="module")
def device_artifact(tmp_path_factory):
    """Return an artifact for the target of the current CUDA device."""
    return _written(
        tmp_path_factory.mktemp("device") / "artifact", _device_target()
    )


def _cases():
    """Return the inputs the child process runs, by name.

    The batches are those of the public differential suite, one seed each:
    every benchmark distribution at a small size, which reaches warp, CTA,
    and split work and empty segments. Two more batches reach what those do
    not: the direct-CTA selection, which needs at least as many segments of
    4097 to 8192 elements as the device has SMs, and one long segment.
    """
    cases = {}
    for name in DISTRIBUTIONS:
        count = 2048 if name == "power-law" else 257
        lengths = _distributions.generate_lengths(name, count, 0)
        generator = torch.Generator().manual_seed(0)
        values, offsets = _host_case(lengths, generator)
        for kind in ("sum", "max"):
            cases[f"{kind}/{name}"] = (kind, values, offsets)
        logits = 4 * torch.randn(sum(lengths), generator=generator)
        cases[f"softmax/{name}"] = ("softmax", logits, offsets)
    generator = torch.Generator().manual_seed(1)
    blocks = torch.cuda.get_device_properties(0).multi_processor_count
    selected = torch.randint(4097, 8193, (blocks + 3,), generator=generator)
    values, offsets = _host_case(selected.tolist(), generator)
    long_values, long_offsets = _host_case([100_003], generator)
    empty = torch.zeros(1, dtype=torch.int32)
    for kind in ("sum", "max"):
        cases[f"{kind}/direct-cta"] = (kind, values, offsets)
        cases[f"{kind}/one-long"] = (kind, long_values, long_offsets)
        cases[f"{kind}/no-segment"] = (kind, torch.empty(0), empty)
    cases["softmax/one-long"] = (
        "softmax",
        4 * torch.randn(100_003, generator=generator),
        long_offsets,
    )
    return cases


def _case_names():
    """Return the names `_cases` produces, without a device."""
    names = [
        f"{kind}/{name}"
        for name in DISTRIBUTIONS
        for kind in ("sum", "max", "softmax")
    ]
    for kind in ("sum", "max"):
        names += [
            f"{kind}/direct-cta",
            f"{kind}/one-long",
            f"{kind}/no-segment",
        ]
    return [*names, "softmax/one-long"]


@pytest.fixture(scope="module")
def child(device_artifact, tmp_path_factory):
    """Run every case in a process that cannot import `mlir_swage`.

    The process finds `swage` in a copy of the pure package and nowhere
    else, and `SWAGE_ARTIFACT_DIR` names the artifact.

    Returns:
        The cases and the report the process saved.
    """
    root = tmp_path_factory.mktemp("child")
    package = pathlib.Path(swage.__file__).parent
    shutil.copytree(
        package,
        root / "site" / "swage",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    cases = _cases()
    torch.save(cases, root / "cases.pt")
    environment = {
        name: value
        for name, value in os.environ.items()
        if name not in ("SWAGE_NO_COMPILE", "SWAGE_CACHE_DIR")
    }
    environment["PYTHONPATH"] = str(root / "site")
    environment["SWAGE_ARTIFACT_DIR"] = str(device_artifact)
    environment["SWAGE_CACHE_DIR"] = str(root / "cache")
    completed = subprocess.run(
        [sys.executable, str(_CHILD), "cases.pt", "report.pt"],
        cwd=root,
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    report = torch.load(root / "report.pt")
    report["site"] = str(root / "site")
    report["cache"] = root / "cache"
    return cases, report


@_needs_cuda
def test_the_cases_are_the_ones_the_tests_name(child):
    """Keep the parametrized names equal to the cases the process ran."""
    cases, report = child

    assert sorted(cases) == sorted(_case_names())
    assert sorted(report["results"]) == sorted(cases)


@_needs_cuda
def test_the_child_process_held_no_compiler(child, device_artifact):
    """Run both calls with no `mlir_swage` and no LLVM or MLIR library."""
    _, report = child
    mapped = report["mapped"]

    assert report["modules"] == []
    assert not [
        path
        for path in mapped
        if COMPILER_LIBRARY.search(os.path.basename(path))
    ]
    assert report["swage_file"] == os.path.join(
        report["site"], "swage", "__init__.py"
    )
    # The process did load CUDA and PyTorch, and it classified and launched
    # through the library of the artifact.
    names = [os.path.basename(path) for path in mapped]
    assert any(name.startswith("libcuda.so") for name in names)
    assert any("libtorch" in name for name in names)
    assert str(device_artifact / "libSwageRuntime.so") in mapped
    assert report["launches_with_the_runtime_library"] is True
    assert not report["cache"].exists()


def test_the_parent_process_does_hold_the_compiler():
    """Show that the check for mapped libraries can fail.

    This process imported the bindings, so the names the child was checked
    for are mapped here.
    """
    with open("/proc/self/maps", encoding="utf-8") as maps:
        mapped = {line.split(None, 5)[-1].strip() for line in maps}

    assert any(
        COMPILER_LIBRARY.search(os.path.basename(path)) for path in mapped
    )


@_needs_cuda
@pytest.mark.parametrize("name", _case_names())
def test_artifact_results_match_pytorch_and_float64(name, child):
    """Compare what the child returned with the references of the suite."""
    cases, report = child
    kind, values, offsets = cases[name]
    actual = report["results"][name]

    if offsets.numel() == 1:
        # `torch.segment_reduce` raises for a batch without a segment.
        assert actual.shape == (0,)
        assert actual.dtype == torch.float32
    elif kind == "softmax":
        _assert_softmax_matches(values, offsets, actual)
    else:
        _assert_reduction_matches(kind, values, offsets, actual)


@_needs_cuda
@pytest.mark.parametrize("name", _case_names())
def test_artifact_results_equal_the_compiled_path_bit_for_bit(name, child):
    """Return the bits of the compiled path for the same input and device.

    This process has no artifact selected and compiles its kernels. The
    child took them from the artifact. Both run on the same device, so both
    select the same schedule.
    """
    cases, report = child
    kind, values, offsets = cases[name]

    if kind == "softmax":
        compiled = swage.segment_softmax(values.cuda(), offsets.cuda())
    else:
        compiled = swage.segment_reduce(values.cuda(), offsets.cuda(), kind)

    assert _bits(report["results"][name]) == _bits(compiled.cpu())
    assert os.environ.get("SWAGE_ARTIFACT_DIR") is None


def _select(monkeypatch, directory):
    """Select an artifact in this process, whose kernel memo starts empty."""
    monkeypatch.setenv("SWAGE_ARTIFACT_DIR", str(directory))
    monkeypatch.setattr(_artifact, "_selected", (None, None))
    monkeypatch.setattr(
        qualification,
        "_ptx_memo",
        _runtime._BoundedCache(_runtime._CACHE_LIMIT),
    )


def _device_batch():
    """Return a batch on the device with warp, CTA, and split work."""
    generator = torch.Generator().manual_seed(11)
    values, offsets = _host_case([3, 40, 0, 4100, 9000, 33, 32], generator)
    return values.cuda(), offsets.cuda()


@_needs_cuda
def test_an_artifact_serves_calls_while_compiling_is_switched_off(
    device_artifact, monkeypatch
):
    """Run from an artifact in a process that may not compile."""
    values, offsets = _device_batch()
    expected = swage.segment_reduce(values, offsets, "sum").cpu()
    weights = swage.segment_softmax(values, offsets).cpu()
    _select(monkeypatch, device_artifact)
    monkeypatch.setenv("SWAGE_NO_COMPILE", "1")

    assert _bits(swage.segment_reduce(values, offsets, "sum").cpu()) == _bits(
        expected
    )
    assert _bits(swage.segment_softmax(values, offsets).cpu()) == _bits(
        weights
    )


@_needs_cuda
def test_a_call_passes_each_kernel_the_arguments_its_manifest_states(
    device_artifact, monkeypatch
):
    """Launch every kernel with the pointers and counts of its manifest.

    The batches reach the fused, partial, and merge kernels, the task-ID
    CTA kernel through the direct-CTA selection, and the softmax kernel.
    The pure warp kernel is prepared and never launched by a public call.
    """
    _select(monkeypatch, device_artifact)
    artifact = _artifact.selected()
    driver = _runtime._get_driver()
    launched = []

    def record(function, grid_x, block_x, stream, pointers, scalars):
        launched.append((block_x, len(pointers), len(scalars)))
        artifact._launch_kernel(
            function, grid_x, block_x, stream, pointers, scalars
        )

    monkeypatch.setattr(driver, "_native_launch", record)
    values, offsets = _device_batch()
    swage.segment_reduce(values, offsets, "sum")
    swage.segment_softmax(values, offsets)
    blocks = torch.cuda.get_device_properties(0).multi_processor_count
    lengths = [4097 + index for index in range(blocks)]
    selected = torch.tensor(_offsets(lengths), dtype=torch.int32).cuda()
    swage.segment_reduce(torch.ones(sum(lengths)).cuda(), selected, "max")
    torch.cuda.synchronize()

    def stated(program, role):
        entry = next(
            kernel
            for kernel in artifact.manifest["kernels"]
            if (kernel["program"], kernel["role"]) == (program, role)
        )
        pointers = sum(
            argument["type"].endswith("*") for argument in entry["arguments"]
        )
        return (
            entry["block_size"],
            pointers,
            len(entry["arguments"]) - pointers,
        )

    assert launched == [
        stated("segmented_sum", "mixed"),
        stated("segmented_sum", "partial"),
        stated("segmented_sum", "merge"),
        stated("ragged_softmax", "cta"),
        stated("segmented_max", "cta"),
    ]


@_needs_cuda
@pytest.mark.parametrize("kind", ["sum", "softmax"])
def test_an_artifact_for_another_target_is_refused_before_a_launch(
    kind, written, monkeypatch
):
    """Raise for the `sm_87` artifact on this device and write nothing."""
    assert _device_target() != "sm_87"
    values, offsets = _device_batch()
    count = values.numel() if kind == "softmax" else offsets.numel() - 1
    out = torch.full((count,), _SENTINEL, device="cuda")
    _select(monkeypatch, written)

    with pytest.raises(
        RuntimeError,
        match=(
            "^the artifact at .* holds kernels for sm_87, and the current "
            f"device needs {_device_target()}; nothing was compiled or "
            "launched"
        ),
    ):
        if kind == "softmax":
            swage.segment_softmax(values, offsets, out=out)
        else:
            swage.segment_reduce(values, offsets, kind, out=out)
    torch.cuda.synchronize()

    assert torch.all(out == _SENTINEL)


@_needs_cuda
def test_a_program_missing_from_the_artifact_is_refused_before_a_launch(
    tmp_path, monkeypatch
):
    """Raise for a kind the artifact was not written with."""
    artifact = _written(
        tmp_path / "artifact", _device_target(), "--program", "sum"
    )
    values, offsets = _device_batch()
    out = torch.full((offsets.numel() - 1,), _SENTINEL, device="cuda")
    weights = torch.full((values.numel(),), _SENTINEL, device="cuda")
    _select(monkeypatch, artifact)

    swage.segment_reduce(values, offsets, "sum")
    with pytest.raises(
        RuntimeError,
        match=(
            "holds no planned program with the text of this call .* it "
            "holds the planned programs segmented_sum. The artifact was "
            "written without this program"
        ),
    ):
        swage.segment_reduce(values, offsets, "max", out=out)
    with pytest.raises(RuntimeError, match="holds no kernel 'ragged_softmax'"):
        swage.segment_softmax(values, offsets, out=weights)
    torch.cuda.synchronize()

    assert torch.all(out == _SENTINEL)
    assert torch.all(weights == _SENTINEL)


@_needs_cuda
def test_a_damaged_artifact_is_refused_before_the_offsets_are_copied(
    device_artifact, tmp_path, monkeypatch
):
    """Verify the artifact where a call would need the bindings.

    The refusal comes before the host copy of the offsets and before the
    result is allocated, so a call on a damaged artifact touches no device
    memory.
    """
    damaged = tmp_path / "damaged"
    shutil.copytree(device_artifact, damaged)
    kernel = damaged / "segmented_sum.merge.ptx"
    kernel.chmod(0o644)
    kernel.write_text(kernel.read_text() + "\n")
    values, offsets = _device_batch()
    _select(monkeypatch, damaged)
    monkeypatch.setattr(
        torch, "empty", lambda *a, **k: pytest.fail("a result was allocated")
    )

    for call in (
        lambda: swage.segment_reduce(values, offsets, "max"),
        lambda: swage.segment_softmax(values, offsets),
    ):
        with pytest.raises(
            RuntimeError,
            match="holds a segmented_sum.merge.ptx that does not match",
        ):
            call()


@_needs_cuda
def test_the_private_helpers_compile_nothing_while_an_artifact_is_selected(
    device_artifact, monkeypatch
):
    """Refuse a kernel outside the artifact instead of compiling it."""
    values, offsets = _device_batch()
    output = torch.full((offsets.numel() - 1,), _SENTINEL, device="cuda")
    _select(monkeypatch, device_artifact)

    with pytest.raises(RuntimeError, match="holds no kernel 'segmented_sum'"):
        qualification.launch_gpu(values, offsets, output, "sum")
    with pytest.raises(RuntimeError, match="holds no kernel 'segmented_sum'"):
        qualification._prepare_persistent_sum(values, offsets, output)
    torch.cuda.synchronize()

    assert torch.all(output == _SENTINEL)
