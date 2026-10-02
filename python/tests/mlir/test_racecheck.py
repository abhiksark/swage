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

import pytest
import swage
import torch
from swage import _runtime
from swage._segmented_qualification import (
    _prepare_persistent_sum,
    _prepare_planned_reduction,
    _reduction_kernel,
    _semantic_module,
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
    """Launch once on a poisoned output and compare without a tolerance."""
    output.fill_(float("nan"))
    launch()
    torch.cuda.synchronize()
    return torch.equal(output.cpu(), expected)


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
    for kind in ("sum", "max", "min"):
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
    return results


def _run_control():
    """Launch the racy kernel on one block of 64 threads."""
    # The tensor comes first: creating it makes the CUDA context current.
    unused = torch.zeros(1, device="cuda")
    pointer = unused.data_ptr()
    driver = _runtime._get_driver()
    _, function = driver.load(_RACY_PTX, "racy_control")
    driver.launch_segmented(
        function,
        (1,),
        64,
        torch.cuda.current_stream().cuda_stream,
        (pointer, pointer, pointer, 0, 0),
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
    assert len(results) > 70
    # Ten static launches of each of three kinds over f64 values.
    assert sum(" f64" in name for name in results) == 30
    assert all(results.values()), results


if __name__ == "__main__":
    program = {"kernels": _run_kernels, "control": _run_control}[sys.argv[1]]
    print(json.dumps(program()))
