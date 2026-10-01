# benchmarks/benchmark_persistent_sum.py
"""Run the frozen static-versus-persistent tail-skew experiment.

The inputs, the timing loop, and the gate are frozen. The gate is decided
on an NVIDIA RTX A6000 with 84 SMs at sm_86. With --any-device the same
measurement runs on another GPU with two resident blocks per SM of that
GPU; the record is labelled as a rerun and is not gate evidence.
"""

import argparse
import json
import pathlib
import platform
import random
import statistics
import subprocess
import sys
from datetime import datetime, timezone

import benchmark_provenance
from benchmark_mixed_sum import _position_medians, _status, _timer
from benchmark_triton_comparison import _gb_per_s, _useful_bytes

_GATE_GPU = "NVIDIA RTX A6000"
_GATE_CAPABILITY = (8, 6)
_GATE_MULTIPROCESSORS = 84
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


def _arguments(argv=None):
    """Parse the output path without exposing frozen tuning controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument(
        "--any-device",
        action="store_true",
        help=(
            "Run the same measurement on a GPU other than the gate device, "
            "with two resident blocks per SM. The record is labelled as a "
            "rerun, not gate evidence, and the exit status does not depend "
            "on the ratio."
        ),
    )
    return parser.parse_args(argv)


def _gate_device(gpu_name, capability, multiprocessors, *, any_device):
    """Return whether this is the device the gate was declared on.

    Args:
        gpu_name: Device name reported by PyTorch.
        capability: Its compute capability as a major and minor pair.
        multiprocessors: Its SM count.
        any_device: Whether another device may rerun the measurement.

    Returns:
        True on the gate device, False on another device with the option.

    Raises:
        RuntimeError: If the device is not the gate device and the option
            was not given.
    """
    if (
        gpu_name == _GATE_GPU
        and tuple(capability) == _GATE_CAPABILITY
        and multiprocessors == _GATE_MULTIPROCESSORS
    ):
        return True
    if not any_device:
        raise RuntimeError(
            "persistent evidence requires NVIDIA RTX A6000 with 84 SMs at "
            f"sm_86; found {gpu_name} with {multiprocessors} SMs at "
            f"sm_{capability[0]}{capability[1]}; pass --any-device to "
            "rerun the measurement here without gate evidence"
        )
    return False


def _resident_blocks(gate_device, multiprocessors):
    """Return the resident block count to request from the preparation.

    The gate declares 168 blocks, two per SM of its device. Another device
    requests two per SM of its own.
    """
    return _RESIDENT_BLOCKS if gate_device else 2 * multiprocessors


def _generate_lengths() -> list[int]:
    """Generate the predeclared tail-skew lengths."""
    rng = random.Random(_SEED)
    lengths = [rng.randint(1, 32) for _ in range(_SHORT_COUNT)]
    lengths.append(_OUTLIER_LENGTH)
    return lengths


def _configuration(resident_blocks=_RESIDENT_BLOCKS) -> dict[str, object]:
    """Return the experiment contract with the resident blocks used."""
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
        "persistent_resident_blocks": resident_blocks,
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
    gate_device = _gate_device(
        gpu_name,
        capability,
        properties.multi_processor_count,
        any_device=arguments.any_device,
    )
    resident_blocks = _resident_blocks(
        gate_device, properties.multi_processor_count
    )
    torch.ones(1, device="cuda").sum().item()
    provenance = benchmark_provenance.start(
        torch, benchmark_provenance.swage_build()
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
        resident_blocks=resident_blocks,
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
    position_medians = _position_medians(samples)
    useful_bytes = _useful_bytes(offsets[-1], _SEGMENT_COUNT)
    result = {
        "benchmark": "persistent-tail-skew-segmented-sum",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "status": _status(gate_device),
        "gate_device": gate_device,
        "source": source,
        "provenance": benchmark_provenance.finish(provenance),
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
        "configuration": _configuration(resident_blocks),
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
        # The fields below describe the frozen samples; the gate above is
        # decided on the medians alone, as it was declared.
        "position_matched": {
            "places": (
                "place 0 is launched on an idle stream and includes host "
                "dispatch; place 1 queues behind a running kernel"
            ),
            "medians_ms": position_medians,
            "persistent_to_static_ratio": [
                _evaluate_gate(
                    {
                        name: places[place]
                        for name, places in position_medians.items()
                    }
                )[0]
                for place in range(len(position_medians))
            ],
        },
        "timer": _timer(samples, medians, "persistent", "static_mixed"),
        "useful_bytes": useful_bytes,
        "effective_gb_per_s": {
            name: _gb_per_s(useful_bytes, median * 1_000.0)
            for name, median in medians.items()
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
                "persistent_to_static_ratio": ratio,
                "passed": passed,
                "status": result["status"],
            },
            sort_keys=True,
        )
    )
    if gate_device and not passed:
        raise SystemExit("persistent scheduling performance gate failed")


if __name__ == "__main__":
    main()
