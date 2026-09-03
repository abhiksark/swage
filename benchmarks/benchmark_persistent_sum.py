# benchmarks/benchmark_persistent_sum.py
"""Run the frozen static-versus-persistent tail-skew experiment."""

import argparse
import json
import pathlib
import platform
import random
import statistics
import subprocess
import sys
from datetime import datetime, timezone

_SEGMENT_COUNT = 32_768
_SHORT_COUNT = _SEGMENT_COUNT - 1
_OUTLIER_LENGTH = 16_777_216
_SEED = 7
_WARP_MAX_ELEMENTS = 32
_CTA_CHUNK_ELEMENTS = 4096
_PERSISTENT_BLOCK = 512
_RESIDENT_BLOCKS = 168
_WARMUPS = 25
_SAMPLES = 100
_GATE_RATIO = 0.95


def _arguments():
    """Parse the output path without exposing frozen tuning controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args()


def _generate_lengths() -> list[int]:
    """Generate the predeclared tail-skew lengths."""
    rng = random.Random(_SEED)
    lengths = [rng.randint(1, 32) for _ in range(_SHORT_COUNT)]
    lengths.append(_OUTLIER_LENGTH)
    return lengths


def _configuration() -> dict[str, object]:
    """Return the immutable experiment contract."""
    return {
        "distribution": "persistent-tail-skew",
        "seed": _SEED,
        "segment_count": _SEGMENT_COUNT,
        "short_lengths": "32767 uniform integers in [1, 32]",
        "outlier_length": _OUTLIER_LENGTH,
        "outlier_position": _SEGMENT_COUNT - 1,
        "warp_max_elements": _WARP_MAX_ELEMENTS,
        "cta_chunk_elements": _CTA_CHUNK_ELEMENTS,
        "persistent_block_threads": _PERSISTENT_BLOCK,
        "persistent_resident_blocks": _RESIDENT_BLOCKS,
        "warmups_per_policy": _WARMUPS,
        "interleaved_samples_per_policy": _SAMPLES,
        "values": "f32 ones",
        "timed_static_sequence": [
            "fused direct",
            "split partial",
            "split merge",
        ],
        "timed_persistent_sequence": ["counter reset", "resident kernel"],
        "excluded": [
            "compilation",
            "classification",
            "allocation",
            "module loading",
        ],
    }


def _git_metadata(root: pathlib.Path) -> dict[str, object]:
    """Return exact source provenance, rejecting dirty evidence."""
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
        raise RuntimeError(
            "persistent benchmark requires a clean source worktree"
        )
    return {"revision": revision, "worktree_clean": True}


def _evaluate_gate(medians: dict[str, float]) -> tuple[float, bool]:
    """Return the persistent-to-static ratio and fixed gate result."""
    ratio = medians["persistent"] / medians["static_mixed"]
    return ratio, ratio <= _GATE_RATIO


def _check_results(torch, launches, outputs, expected):
    """Require exact sums before timing either schedule."""
    for name, launch in launches.items():
        outputs[name].fill_(float("nan"))
        launch()
        torch.testing.assert_close(
            outputs[name].cpu(),
            expected,
            rtol=0,
            atol=0,
            msg=lambda message, policy=name: f"{policy}: {message}",
        )


def _measure(torch, launches):
    """Collect fixed warmups and interleaved CUDA-event samples."""
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
    """Run the frozen A6000 persistent-scheduling gate."""
    import torch
    from distributions import summarize_lengths
    from swage import _runtime
    from swage._segmented_qualification import (
        _prepare_persistent_sum,
        _prepare_planned_sum,
    )

    arguments = _arguments()
    root = pathlib.Path(__file__).resolve().parents[1]
    source = _git_metadata(root)
    if not torch.cuda.is_available():
        raise RuntimeError("persistent benchmark requires CUDA-enabled PyTorch")
    device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(device)
    capability = torch.cuda.get_device_capability(device)
    properties = torch.cuda.get_device_properties(device)
    if (
        gpu_name != "NVIDIA RTX A6000"
        or capability != (8, 6)
        or properties.multi_processor_count != 84
    ):
        raise RuntimeError(
            "persistent evidence requires NVIDIA RTX A6000 with 84 SMs at "
            f"sm_86; found {gpu_name} with "
            f"{properties.multi_processor_count} SMs at "
            f"sm_{capability[0]}{capability[1]}"
        )

    lengths = _generate_lengths()
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    values = torch.ones(offsets[-1], device="cuda", dtype=torch.float32)
    device_offsets = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    outputs = {
        "static_mixed": torch.empty(
            _SEGMENT_COUNT, device="cuda", dtype=torch.float32
        ),
        "persistent": torch.empty(
            _SEGMENT_COUNT, device="cuda", dtype=torch.float32
        ),
    }
    static = _prepare_planned_sum(
        values,
        device_offsets,
        outputs["static_mixed"],
        warp_max_elements=_WARP_MAX_ELEMENTS,
        cta_chunk_elements=_CTA_CHUNK_ELEMENTS,
    )
    persistent = _prepare_persistent_sum(
        values,
        device_offsets,
        outputs["persistent"],
        warp_max_elements=_WARP_MAX_ELEMENTS,
        cta_chunk_elements=_CTA_CHUNK_ELEMENTS,
        resident_blocks=_RESIDENT_BLOCKS,
    )
    launches = {
        "static_mixed": static.mixed,
        "persistent": persistent.launch,
    }
    torch.cuda.synchronize()
    expected = torch.tensor(lengths, dtype=torch.float32)
    _check_results(torch, launches, outputs, expected)
    samples = _measure(torch, launches)
    medians = {
        name: statistics.median(policy_samples)
        for name, policy_samples in samples.items()
    }
    ratio, passed = _evaluate_gate(medians)
    result = {
        "benchmark": "persistent-tail-skew-segmented-sum",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "pytorch": torch.__version__,
            "pytorch_cuda": torch.version.cuda,
            "cuda_driver": _runtime.driver_version(),
            "gpu": gpu_name,
            "compute_capability": "sm_86",
            "multiprocessors": properties.multi_processor_count,
            "total_memory_bytes": properties.total_memory,
        },
        "configuration": _configuration(),
        "distribution_statistics": summarize_lengths(lengths),
        "materialized_tasks": {
            "warp": persistent.warp_tasks,
            "cta": persistent.cta_tasks,
            "partial": persistent.partial_tasks,
            "merge": persistent.merge_tasks,
            "resident_blocks": persistent.resident_blocks,
        },
        "raw_samples_ms": samples,
        "medians_ms": medians,
        "persistent_to_static_ratio": ratio,
        "gate": {"maximum_ratio": _GATE_RATIO, "passed": passed},
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "medians_ms": medians,
                "persistent_to_static_ratio": ratio,
                "passed": passed,
            },
            sort_keys=True,
        )
    )
    if not passed:
        raise SystemExit("persistent scheduling performance gate failed")


if __name__ == "__main__":
    main()
