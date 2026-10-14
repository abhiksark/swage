# python/tests/mlir/test_racecheck.py
"""Opt-in shared-memory hazard check of the private segmented kernels.

NVIDIA Compute Sanitizer instruments every kernel a process launches. Its
racecheck tool reports two threads of a block that access one shared-memory
address with no barrier between them, when at least one access is a write.
The segmented kernels reduce through shared memory, and a hazard there can
return correct results on every launch, so an exactness test does not see
it.

The check is off by default: it needs the sanitizer, which the CUDA toolkit
ships and no workflow installs, and it is slower than the tests it sits
beside, about half a minute on an RTX A6000. Set SWAGE_RACECHECK=1 to run
it. Without that variable, without CUDA, or without the tool every test
here is skipped.

Two programs run under the tool, and this file is both of them:

- `kernels` launches every segmented kernel family and prints, as JSON,
  whether each result is exact.
- `control` launches a hand-written kernel in which every thread of a block
  stores to one shared word. The tool must report it, or a clean report on
  the first program would mean nothing.
"""

import importlib.util
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
from itertools import accumulate, pairwise

import pytest
import swage
import torch
from swage import _abi, _cuda_backend, _segmented_runtime
from swage._segmented_programs import (
    _parameter_roles,
    _reduction_kernel,
    _semantic_module,
    _softmax_text,
)
from swage._segmented_qualification import (
    _prepare_persistent_sum,
    _prepare_planned_reduction,
    launch_gpu,
    launch_softmax_gpu,
)

_SUMMARY = re.compile(
    r"RACECHECK SUMMARY: (\d+) hazards? displayed "
    r"\((\d+) errors?, (\d+) warnings?\)"
)
# The exit code the tool returns when it reports an error, chosen to differ
# from the exit code of a Python failure.
_HAZARD_EXIT = 86

# Every thread of the block stores its index to one shared word, with no
# barrier: a write-after-write hazard between each pair of threads. The
# parameters are the three pointers and two counts of the segmented ABI.
_RACY_PTX = """
.version 7.1
.target sm_80
.address_size 64

.visible .entry racy_control(
	.param .u64 racy_control_param_0,
	.param .u64 racy_control_param_1,
	.param .u64 racy_control_param_2,
	.param .u32 racy_control_param_3,
	.param .u32 racy_control_param_4
)
{
	.reg .b32 	%r<2>;
	.shared .align 4 .b8 racy_control_word[4];
	mov.u32 	%r1, %tid.x;
	st.shared.b32 	[racy_control_word], %r1;
	ret;
}
"""

# Lengths on both sides of the warp limit (32) and the chunk limit (4096),
# an empty segment, and segments of two, three, and four chunks.
_LENGTHS = [1, 32, 33, 513, 4097, 8193, 0, 12289, 31, 4096]
# Limits that send every segment longer than one element through the
# partial and merge kernels, in 16-element chunks.
_SPLIT_LIMITS = {"warp_max_elements": 1, "cta_chunk_elements": 16}


def _find_sanitizer():
    """Return the path of `compute-sanitizer`, or None when it is absent.

    The CUDA toolkit installs the tool in a `compute-sanitizer` directory
    of its own, which is usually not on PATH.
    """
    found = shutil.which("compute-sanitizer")
    if found is not None:
        return found
    roots = (os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH"))
    for root in (*filter(None, roots), "/usr/local/cuda"):
        tool = pathlib.Path(root, "compute-sanitizer", "compute-sanitizer")
        if tool.is_file() and os.access(tool, os.X_OK):
            return str(tool)
    return None


def _skip_reason():
    """Return why the check does not run here, or None when it does."""
    if os.environ.get("SWAGE_RACECHECK") != "1":
        return "racecheck is opt-in; set SWAGE_RACECHECK=1 to run it"
    if not torch.cuda.is_available():
        return "CUDA unavailable"
    if _find_sanitizer() is None:
        return "NVIDIA Compute Sanitizer (compute-sanitizer) not found"
    return None


pytestmark = pytest.mark.skipif(
    _skip_reason() is not None, reason=str(_skip_reason())
)


def _case():
    """Build exactly summable, position-dependent values and references."""
    offsets = [0]
    for length in _LENGTHS:
        offsets.append(offsets[-1] + length)
    index = torch.arange(offsets[-1])
    values = (2 * (index % 67) - 65).to(torch.float32) / 4
    lengths = torch.tensor(_LENGTHS)
    expected = {
        "sum": torch.segment_reduce(values.double(), "sum", lengths=lengths),
        "max": torch.segment_reduce(
            values.double(), "max", lengths=lengths, initial=float("-inf")
        ),
        "min": torch.segment_reduce(
            values.double(), "min", lengths=lengths, initial=float("inf")
        ),
    }
    return (
        values.cuda(),
        torch.tensor(offsets, dtype=torch.int32, device="cuda"),
        {kind: result.float() for kind, result in expected.items()},
    )


def _exact(launch, output, expected):
    """Launch once on a poisoned output and compare without a tolerance.

    Where `expected` is NaN, which is the mean of an empty segment, the
    output must be NaN too. Everywhere else it must hold the same value.
    """
    output.fill_(float("nan"))
    launch()
    torch.cuda.synchronize()
    actual = output.cpu()
    stored = ~expected.isnan()
    return bool(
        torch.equal(actual[stored], expected[stored])
        and actual[~stored].isnan().all()
    )


def _run_static_kernels(results, element, values, offsets, expected):
    """Launch the static kernel families of one element type.

    Args:
        results: Receives whether each launch was exact, by name.
        element: The element type, `"f32"` or `"f64"`.
        values: The values, of the dtype of `element`.
        offsets: The int32 offsets.
        expected: The exact results of each kind, of the dtype of `element`.
    """
    output = torch.empty(len(_LENGTHS), dtype=values.dtype, device="cuda")
    # The sums are exact, so a mean is its sum divided once by the length.
    lengths = torch.tensor(_LENGTHS, dtype=values.dtype)
    expected = dict(expected, mean=expected["sum"] / lengths)
    for kind in ("sum", "max", "min", "mean"):
        label = f"{kind} {element}"
        for block_size in (33, 100, 128, 512):
            results[f"direct {label} block {block_size}"] = _exact(
                lambda: launch_gpu(values, offsets, output, kind, block_size),
                output,
                expected[kind],
            )
        for name, limits in (("default", {}), ("split", _SPLIT_LIMITS)):
            prepared = _prepare_planned_reduction(
                values,
                offsets,
                output,
                module_text=_semantic_module(kind, element),
                kernel_name=_reduction_kernel(kind, element),
                select_schedule=False,
                **limits,
            )
            results[f"task-id warp {label} {name}"] = _exact(
                prepared.warp, output, expected[kind]
            )
            results[f"task-id cta {label} {name}"] = _exact(
                prepared.cta, output, expected[kind]
            )
            results[f"fused mixed and split {label} {name}"] = _exact(
                prepared.mixed, output, expected[kind]
            )


def _run_column_kernels(results, element, dtype, offsets):
    """Launch the column kernel of each kind over rank-two values.

    The kernel holds no shared memory, no barrier, and no shuffle, so the
    tool has nothing to report on it. It runs here all the same: a hazard
    check that skipped a kernel family could not say so. The rows hold
    three and 130 columns, fewer than a block has threads and more.

    Args:
        results: Receives whether each launch was exact, by name.
        element: The element type, `"f32"` or `"f64"`.
        dtype: Its tensor dtype.
        offsets: The int32 row offsets, on the device.
    """
    lengths = torch.tensor(_LENGTHS)
    row = torch.arange(int(lengths.sum()))[:, None]
    for columns in (3, 130):
        # Small integers that depend on the row and on the column, so that
        # every column sum is exact in both element types.
        column = torch.arange(columns)[None, :]
        host = ((row * 31 + column * 17) % 127 - 63).to(dtype)
        values = host.cuda()
        output = torch.empty(len(_LENGTHS), columns, dtype=dtype, device="cuda")
        for kind in ("sum", "max", "min", "mean"):
            expected = torch.segment_reduce(host, kind, lengths=lengths, axis=0)
            for block_size in (32, 128):
                name = f"columns {kind} {element} {columns} block {block_size}"
                results[name] = _exact(
                    lambda: launch_gpu(
                        values, offsets, output, kind, block_size
                    ),
                    output,
                    expected,
                )


def _launcher(compiler, text, kernel_name, **options):
    """Compile one kernel and return its raw launch, bound by its contract.

    Returns:
        A function of the block count of a one-dimensional grid, the plan
        and scratch buffers and derived counts by contract key, and the
        user values by parameter role. It enqueues on the current stream,
        below the validation of the calls.
    """
    major, minor = torch.cuda.get_device_capability()
    kernel = _segmented_runtime._compile_once(
        getattr(_segmented_runtime._native_swage(), compiler),
        text,
        kernel_name=kernel_name,
        target=f"sm_{major}{minor}",
        **options,
    )

    def launch(blocks, named, **by_role):
        user = dict.fromkeys(_parameter_roles(text))
        user.update(by_role)
        arguments = _segmented_runtime._bind(
            kernel, _segmented_runtime._user_arguments(text, **user), named
        )
        lease = _segmented_runtime._lease(torch, kernel)
        try:
            _segmented_runtime._enqueue(
                torch,
                lease,
                kernel,
                arguments,
                blocks,
                torch.cuda.current_stream(),
            )
        finally:
            lease.release()

    return launch


def _row_tile(text, kernel_name, block_size):
    """Return the launch of the row-stripe kernel of one rank-two program."""
    return _launcher(
        "_compile_segmented_reduction_ptx",
        text,
        kernel_name,
        block_size=block_size,
        use_task_ids=True,
    )


def _launch_row_tile(launch, values, offsets, output, columns):
    """Launch a row-stripe kernel with one task per segment, in order."""
    segment_count = len(_LENGTHS)
    width = 1
    while width < 32 and width < columns:
        width <<= 1
    ids = torch.arange(segment_count, dtype=torch.int32, device="cuda")
    launch(
        segment_count * -(-columns // width),
        {"task_ids": ids, "task_count": segment_count},
        values=values,
        offsets=offsets,
        output=output,
        value_count=values.shape[0],
        segment_count=segment_count,
        feature_count=columns,
    )


def _run_row_kernels(results, element, dtype, offsets):
    """Launch the row-stripe kernel of each kind over rank-two values.

    The stripes of one column combine through a shared buffer of one
    element per thread, between two barriers per reduction. The rows hold
    3, 33, and 130 columns: one group of four columns, two groups of 32
    with one column in the second, and five groups. The values are those
    of the column kernels, so every result is exact.

    Args:
        results: Receives whether each launch was exact, by name.
        element: The element type, `"f32"` or `"f64"`.
        dtype: Its tensor dtype.
        offsets: The int32 row offsets, on the device.
    """
    lengths = torch.tensor(_LENGTHS)
    row = torch.arange(int(lengths.sum()))[:, None]
    for columns in (3, 33, 130):
        column = torch.arange(columns)[None, :]
        host = ((row * 31 + column * 17) % 127 - 63).to(dtype)
        values = host.cuda()
        output = torch.empty(len(_LENGTHS), columns, dtype=dtype, device="cuda")
        for kind in ("sum", "max", "min", "mean"):
            expected = torch.segment_reduce(host, kind, lengths=lengths, axis=0)
            name = _reduction_kernel(kind, element, 2)
            for block_size in (32, 128):
                function = _row_tile(
                    _semantic_module(kind, element, 2), name, block_size
                )
                key = f"rows {kind} {element} {columns} block {block_size}"
                results[key] = _exact(
                    lambda: _launch_row_tile(
                        function, values, offsets, output, columns
                    ),
                    output,
                    expected,
                )


def _split_kernel(compile_name, text, kernel_name):
    """Return the launch of one split kernel of a rank-two program."""
    return _launcher(compile_name, text, kernel_name)


def _launch_row_split(partial, merge, values, output, columns):
    """Run the partial and merge kernels over every segment of `_LENGTHS`.

    Each segment is cut into chunks of `4096 / W` rows, the chunk size of
    the public split, and every segment, the empty one too, is merged.
    """
    width = 1
    while width < 32 and width < columns:
        width <<= 1
    groups = -(-columns // width)
    chunk = 4096 // width
    ranges, records = [], []
    bounds = list(accumulate(_LENGTHS, initial=0))
    for segment, (begin, end) in enumerate(pairwise(bounds)):
        first = len(ranges) // 2
        for start in range(begin, end, chunk):
            ranges += [start, min(start + chunk, end)]
        records += [segment, first, len(ranges) // 2]
    partial_count = len(ranges) // 2
    ranges = torch.tensor(ranges, dtype=torch.int32, device="cuda")
    records = torch.tensor(records, dtype=torch.int32, device="cuda")
    scratch = torch.empty(
        partial_count, columns, dtype=values.dtype, device="cuda"
    )
    partial(
        partial_count * groups,
        {
            "partial_ranges": ranges,
            "scratch": scratch,
            "partial_count": partial_count,
        },
        values=values,
        value_count=values.shape[0],
        feature_count=columns,
    )
    # The merge of a mean also reads the range records; any other merge
    # takes no such argument, and its contract leaves the key unread.
    merge(
        len(_LENGTHS) * groups,
        {
            "scratch": scratch,
            "merge_records": records,
            "partial_ranges": ranges,
            "partial_count": partial_count,
            "merge_count": len(_LENGTHS),
        },
        output=output,
        segment_count=len(_LENGTHS),
        feature_count=columns,
    )


def _run_row_split_kernels(results, element, dtype):
    """Launch the partial and merge kernels of each kind over rank-two values.

    Both are the row-stripe tile at 512 threads, so their stripes combine
    through a shared buffer of 512 elements. The values are those of the
    row-stripe kernels, so every result is exact.

    Args:
        results: Receives whether each launch was exact, by name.
        element: The element type, `"f32"` or `"f64"`.
        dtype: Its tensor dtype.
    """
    lengths = torch.tensor(_LENGTHS)
    row = torch.arange(int(lengths.sum()))[:, None]
    for columns in (3, 33, 130):
        column = torch.arange(columns)[None, :]
        host = ((row * 31 + column * 17) % 127 - 63).to(dtype)
        values = host.cuda()
        output = torch.empty(len(_LENGTHS), columns, dtype=dtype, device="cuda")
        for kind in ("sum", "max", "min", "mean"):
            expected = torch.segment_reduce(host, kind, lengths=lengths, axis=0)
            name = _reduction_kernel(kind, element, 2)
            text = _semantic_module(kind, element, 2)
            partial = _split_kernel(
                "_compile_split_partial_reduction_ptx", text, name
            )
            merge = _split_kernel(
                "_compile_split_merge_reduction_ptx", text, name
            )
            key = f"split rows {kind} {element} {columns}"
            results[key] = _exact(
                lambda: _launch_row_split(
                    partial, merge, values, output, columns
                ),
                output,
                expected,
            )


def _run_softmax_rows(results, offsets):
    """Launch the row-stripe kernel of the softmax over rank-two logits.

    Its two reductions exchange through one shared buffer, so the trailing
    barrier of the first is what keeps the store of the second from
    overwriting a result another thread has not read. A launch is reported
    as correct as in `_run_softmax_columns`.

    Args:
        results: Receives whether each launch was correct, by name.
        offsets: The int32 row offsets, on the device.
    """
    bounds = list(accumulate(_LENGTHS, initial=0))
    row = torch.arange(bounds[-1])[:, None]
    for columns in (3, 130):
        column = torch.arange(columns)[None, :]
        host = ((row * 31 + column * 17) % 127 - 63).to(torch.float32) / 8
        expected = torch.cat(
            [
                torch.softmax(host[begin:end].double(), 0)
                for begin, end in pairwise(bounds)
            ]
        )
        values = host.cuda()
        for block_size in (32, 128):
            function = _row_tile(
                _softmax_text(2), "ragged_softmax_r2", block_size
            )
            output = torch.full_like(values, float("nan"))
            _launch_row_tile(function, values, offsets, output, columns)
            torch.cuda.synchronize()
            results[f"softmax rows {columns} block {block_size}"] = bool(
                torch.allclose(
                    output.cpu().double(), expected, rtol=1e-3, atol=0
                )
            )


def _run_softmax_columns(results, offsets):
    """Launch the column kernel of the softmax over rank-two logits.

    Like the column kernel of a reduction it holds no shared memory, and
    it runs here for the same reason. A launch is reported as correct when
    every output is within a relative `1e-3` of float64 `torch.softmax`
    along the rows of its segment, which is above the bound of
    docs/internals/ragged-softmax.md for the longest segment here.

    Args:
        results: Receives whether each launch was correct, by name.
        offsets: The int32 row offsets, on the device.
    """
    bounds = list(accumulate(_LENGTHS, initial=0))
    row = torch.arange(bounds[-1])[:, None]
    for columns in (3, 130):
        column = torch.arange(columns)[None, :]
        host = ((row * 31 + column * 17) % 127 - 63).to(torch.float32) / 8
        expected = torch.cat(
            [
                torch.softmax(host[begin:end].double(), 0)
                for begin, end in pairwise(bounds)
            ]
        )
        values = host.cuda()
        for block_size in (32, 128):
            output = torch.full_like(values, float("nan"))
            launch_softmax_gpu(values, offsets, output, block_size)
            torch.cuda.synchronize()
            results[f"softmax columns {columns} block {block_size}"] = bool(
                torch.allclose(
                    output.cpu().double(), expected, rtol=1e-3, atol=0
                )
            )


def _run_kernels():
    """Launch every segmented kernel family and report exactness by name.

    - direct: one CTA per segment, at block sizes of one, two, four, and
      sixteen warps, with a partly filled last warp at 33 and 100.
    - task-id: the warp and CTA kernels that load segment IDs.
    - fused mixed: warp and CTA work in one kernel.
    - split: the partial and merge kernels, under the default limits and
      under limits that split every segment into 16-element chunks.
    - persistent: the resident queue kernel at several residencies, with
      the default limits and with the splitting ones.
    - softmax: the multi-phase map-store kernel.
    - columns: the kernel of rank-two values, in which a thread reduces a
      column on its own, and the one in which a thread normalizes it.
    - rows: the row-stripe kernel of rank-two values, in which the threads
      of one column combine through shared memory, for each reduction and
      for the softmax, and the partial and merge kernels of its split.

    The direct, task-id, fused, and split families run once per element
    type. The f64 kernels reduce through shared slots of eight bytes, so
    their shared accesses are not those of the f32 kernels.
    """
    values, offsets, expected = _case()
    output = torch.empty(len(_LENGTHS), device="cuda")
    results = {}
    for element, dtype in (("f32", torch.float32), ("f64", torch.float64)):
        _run_static_kernels(
            results,
            element,
            values.to(dtype),
            offsets,
            {kind: result.to(dtype) for kind, result in expected.items()},
        )
        _run_column_kernels(results, element, dtype, offsets)
        _run_row_kernels(results, element, dtype, offsets)
        _run_row_split_kernels(results, element, dtype)
    for name, limits in (("default", {}), ("split", _SPLIT_LIMITS)):
        for resident_blocks in (1, 2, 5):
            persistent = _prepare_persistent_sum(
                values,
                offsets,
                output,
                resident_blocks=resident_blocks,
                **limits,
            )
            for launch in range(2):
                key = f"persistent {name} resident {resident_blocks} #{launch}"
                results[key] = _exact(
                    persistent.launch, output, expected["sum"]
                )
    softmax = torch.empty(values.numel(), device="cuda")
    launch_softmax_gpu(values, offsets, softmax)
    torch.cuda.synchronize()
    results["softmax"] = bool(torch.isfinite(softmax).all())
    _run_softmax_columns(results, offsets)
    _run_softmax_rows(results, offsets)
    return results


def _run_control():
    """Launch the racy kernel on one block of 64 threads.

    The kernel has no compiler, so its launch contract is written here: the
    five parameters of the segmented ABI, all bound as user values.
    """
    user = [
        _abi.KernelArgument("ptr", "user", source_index=index, access=access)
        for index, access in enumerate(("read", "read", "write"))
    ]
    user += [
        _abi.KernelArgument("i32", "user", source_index=index)
        for index in (3, 4)
    ]
    contract = _abi.KernelContract(
        version=2,
        backend="cuda",
        entry="racy_control",
        launch=_abi.KernelLaunch("spmd-grid", (64, 1, 1)),
        arguments=tuple(user),
    )
    # The tensor comes first: creating it makes the CUDA context current.
    unused = torch.zeros(1, device="cuda")
    pointer = unused.data_ptr()
    driver = _cuda_backend._get_driver()
    _, function = driver.load(_RACY_PTX, contract.entry)
    driver.launch_entry(
        function,
        contract,
        _abi.materialize_launch_arguments(
            _abi.bind_kernel_contract(
                contract, user=(pointer, pointer, pointer, 0, 0)
            )
        ),
        (1, 1, 1),
        torch.cuda.current_stream().cuda_stream,
    )
    torch.cuda.synchronize()
    return {"control": True}


def _racecheck(program, log):
    """Run one program of this file under racecheck.

    Args:
        program: `kernels` or `control`.
        log: File that receives the report of the tool, so the standard
            output of the program holds only its own JSON line.

    Returns:
        The completed process, the report text, and the hazard, error, and
        warning counts of the summary line.
    """
    bindings = importlib.util.find_spec("mlir_swage._mlir_libs")
    bindings_parent = pathlib.Path(
        list(bindings.submodule_search_locations)[0]
    ).parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(pathlib.Path(swage.__file__).parents[1]), str(bindings_parent)]
    )
    completed = subprocess.run(
        [
            _find_sanitizer(),
            "--tool",
            "racecheck",
            "--racecheck-report",
            "all",
            "--error-exitcode",
            str(_HAZARD_EXIT),
            "--log-file",
            str(log),
            sys.executable,
            __file__,
            program,
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=1800,
    )
    report = log.read_text() if log.is_file() else ""
    summary = _SUMMARY.search(report)
    assert summary is not None, (
        f"racecheck printed no summary: exit {completed.returncode}\n"
        f"{report}\n{completed.stderr}"
    )
    return completed, report, tuple(int(count) for count in summary.groups())


def test_racecheck_reports_the_hazard_of_a_racy_kernel(tmp_path):
    """Require the tool to see a write-after-write hazard it must see."""
    completed, report, (hazards, errors, _) = _racecheck(
        "control", tmp_path / "control.log"
    )

    assert "hazard detected at __shared__" in report, report
    assert "racy_control" in report
    assert hazards > 0 and errors > 0
    assert completed.returncode == _HAZARD_EXIT, completed.stderr


def test_segmented_kernels_have_no_shared_memory_hazard(tmp_path):
    """Run every segmented kernel family under racecheck, with exact sums."""
    completed, report, counts = _racecheck("kernels", tmp_path / "kernels.log")

    assert completed.returncode == 0, f"{report}\n{completed.stderr}"
    assert counts == (0, 0, 0), report
    results = json.loads(completed.stdout.splitlines()[-1])
    assert len(results) > 120
    # Ten static launches of each of four kinds over f64 values, four
    # launches of the column kernel of each kind, six of the row-stripe
    # kernel of each kind, and three of its split.
    assert sum(" f64" in name for name in results) == 40 + 16 + 24 + 12
    assert any(" mean " in name for name in results)
    assert sum(name.startswith("columns ") for name in results) == 32
    assert sum(name.startswith("softmax columns ") for name in results) == 4
    assert sum(name.startswith("rows ") for name in results) == 48
    assert sum(name.startswith("split rows ") for name in results) == 24
    assert sum(name.startswith("softmax rows ") for name in results) == 4
    assert all(results.values()), results


if __name__ == "__main__":
    program = {"kernels": _run_kernels, "control": _run_control}[sys.argv[1]]
    print(json.dumps(program()))
