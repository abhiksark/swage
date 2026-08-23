# benchmarks/benchmark_m7_mixed_sum.py
"""Run the predeclared M7 mixed-policy segmented-sum benchmark."""

import argparse
import json
import pathlib
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone

_COUNT = 32_768
_SEED = 7
_WARP_MAX_ELEMENTS = 32
_WARMUPS = 25
_SAMPLES = 100
_GATE_RATIO = 1.05


def _arguments():
    """Parse the output path without exposing policy tuning controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args()


def _git_metadata(root):
    """Return the exact source revision, rejecting dirty evidence."""
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty:
        raise RuntimeError("M7 benchmark requires a clean source worktree")
    return {"revision": revision, "worktree_clean": True}


def _evaluate_gate(medians):
    """Return the fixed mixed-to-best-pure ratio and gate result."""
    ratio = medians["mixed"] / min(medians["warp"], medians["cta"])
    return ratio, ratio <= _GATE_RATIO


def _check_results(launches, output, expected):
    """Require exact all-one sums before collecting timing evidence."""
    import torch

    for name, launch in launches.items():
        output.fill_(float("nan"))
        launch()
        torch.testing.assert_close(
            output.cpu(),
            expected,
            rtol=0,
            atol=0,
            msg=lambda message: (
                f"{name} policy failed exact correctness:\n{message}"
            ),
        )


def _measure(launches):
    """Collect interleaved CUDA-event samples after fixed warmups."""
    import torch

    for _ in range(_WARMUPS):
        for launch in launches.values():
            launch()
    torch.cuda.synchronize()

    policies = tuple(launches)
    events = {
        name: (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        for name in policies
    }
    samples = {name: [] for name in policies}
    for sample_index in range(_SAMPLES):
        shift = sample_index % len(policies)
        order = policies[shift:] + policies[:shift]
        for name in order:
            start, end = events[name]
            start.record()
            launches[name]()
            end.record()
        for name in order:
            start, end = events[name]
            end.synchronize()
            samples[name].append(start.elapsed_time(end))
    return samples


def main():
    """Run the fixed A6000 benchmark and commit-ready JSON report."""
    import torch
    from distributions import generate_lengths, summarize_lengths
    from swage import _runtime
    from swage._segmented_qualification import (
        _CTA_BLOCK,
        _WARP_BLOCK,
        _prepare_planned_sum,
    )

    arguments = _arguments()
    root = pathlib.Path(__file__).resolve().parents[1]
    source = _git_metadata(root)
    if not torch.cuda.is_available():
        raise RuntimeError("M7 benchmark requires CUDA-enabled PyTorch")
    device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(device)
    capability = torch.cuda.get_device_capability(device)
    if gpu_name != "NVIDIA RTX A6000" or capability != (8, 6):
        raise RuntimeError(
            "M7 evidence must run on NVIDIA RTX A6000 at sm_86; found "
            f"{gpu_name} at sm_{capability[0]}{capability[1]}"
        )

    lengths = generate_lengths("bimodal", _COUNT, _SEED)
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    values = torch.ones(offsets[-1], device="cuda", dtype=torch.float32)
    device_offsets = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    output = torch.empty(_COUNT, device="cuda", dtype=torch.float32)
    prepared = _prepare_planned_sum(
        values,
        device_offsets,
        output,
        warp_max_elements=_WARP_MAX_ELEMENTS,
    )
    launches = prepared._asdict()
    torch.cuda.synchronize()
    _check_results(launches, output, torch.tensor(lengths, dtype=torch.float32))
    samples = _measure(launches)
    medians = {
        name: statistics.median(policy_samples)
        for name, policy_samples in samples.items()
    }
    ratio, passed = _evaluate_gate(medians)
    properties = torch.cuda.get_device_properties(device)
    result = {
        "benchmark": "m7-minimal-mixed-policy-segmented-sum",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "pytorch": torch.__version__,
            "pytorch_cuda": torch.version.cuda,
            "cuda_driver": _runtime.driver_version(),
            "gpu": gpu_name,
            "compute_capability": f"sm_{capability[0]}{capability[1]}",
            "multiprocessors": properties.multi_processor_count,
            "total_memory_bytes": properties.total_memory,
        },
        "configuration": {
            "distribution": "bimodal",
            "seed": _SEED,
            "segment_count": _COUNT,
            "warp_max_elements": _WARP_MAX_ELEMENTS,
            "warp_block": _WARP_BLOCK,
            "cta_block": _CTA_BLOCK,
            "warmups_per_policy": _WARMUPS,
            "interleaved_samples_per_policy": _SAMPLES,
            "values": "f32 ones",
            "excluded": [
                "compilation",
                "classification",
                "allocation",
                "module loading",
            ],
        },
        "distribution_statistics": summarize_lengths(lengths),
        "raw_samples_ms": samples,
        "medians_ms": medians,
        "mixed_to_best_pure_ratio": ratio,
        "gate": {
            "maximum_ratio": _GATE_RATIO,
            "passed": passed,
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "medians_ms": medians,
                "mixed_to_best_pure_ratio": ratio,
                "passed": result["gate"]["passed"],
            },
            sort_keys=True,
        )
    )
    if not result["gate"]["passed"]:
        raise SystemExit("M7 performance gate failed")


if __name__ == "__main__":
    main()
