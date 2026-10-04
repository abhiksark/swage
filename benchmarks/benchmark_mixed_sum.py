# benchmarks/benchmark_mixed_sum.py
"""Run the frozen mixed-policy segmented-sum benchmark.

The inputs, the timing loop, and the gate are frozen. The gate is decided
on an NVIDIA RTX A6000 at sm_86. With --any-device the same measurement
runs on another GPU; that record is labelled as a rerun and is not gate
evidence.
"""

import argparse
import json
import pathlib
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone

import benchmark_provenance
from benchmark_triton_comparison import _gb_per_s, _useful_bytes

_GATE_GPU = "NVIDIA RTX A6000"
_GATE_CAPABILITY = (8, 6)
_COUNT = 32_768
_SEED = 7
_WARP_MAX_ELEMENTS = 32
_WARMUPS = 25
_SAMPLES = 100
_GATE_RATIO = 1.05


def _arguments(argv=None):
    """Parse the output path without exposing policy tuning controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument(
        "--any-device",
        action="store_true",
        help=(
            "Run the same measurement on a GPU other than the gate device. "
            "The record is labelled as a rerun, not gate evidence, and the "
            "exit status does not depend on the ratio."
        ),
    )
    return parser.parse_args(argv)


def _gate_device(gpu_name, capability, *, any_device):
    """Return whether this is the device the gate was declared on.

    Args:
        gpu_name: Device name reported by PyTorch.
        capability: Its compute capability as a major and minor pair.
        any_device: Whether another device may rerun the measurement.

    Returns:
        True on the gate device, False on another device with the option.

    Raises:
        RuntimeError: If the device is not the gate device and the option
            was not given.
    """
    if gpu_name == _GATE_GPU and tuple(capability) == _GATE_CAPABILITY:
        return True
    if not any_device:
        raise RuntimeError(
            "mixed-policy evidence must run on NVIDIA RTX A6000 at "
            "sm_86; found "
            f"{gpu_name} at sm_{capability[0]}{capability[1]}; pass "
            "--any-device to rerun the measurement here without gate "
            "evidence"
        )
    return False


def _status(gate_device):
    """Return the label that separates gate evidence from a rerun."""
    if gate_device:
        return "frozen gate run"
    return "rerun on another device; not gate evidence"


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
        raise RuntimeError(
            "mixed-policy benchmark requires a clean source worktree"
        )
    return {"revision": revision, "worktree_clean": True}


def _evaluate_gate(medians):
    """Return the fixed mixed-to-best-pure ratio and gate result."""
    ratio = medians["mixed"] / min(medians["warp"], medians["cta"])
    return ratio, ratio <= _GATE_RATIO


def _configuration():
    """Return the fixed benchmark and fused schedule contract."""
    return {
        "distribution": "bimodal",
        "seed": _SEED,
        "segment_count": _COUNT,
        "warp_max_elements": _WARP_MAX_ELEMENTS,
        "warp_block": 32,
        "cta_block": 128,
        "mixed_schedule": {
            "kind": "fused",
            "kernel_launches": 1,
            "block_threads": 128,
            "warp_slots_per_block": 4,
        },
        "warmups_per_policy": _WARMUPS,
        "interleaved_samples_per_policy": _SAMPLES,
        "values": "f32 ones",
        "excluded": [
            "compilation",
            "classification",
            "allocation",
            "module loading",
        ],
    }


def _positions(policies, sample_count):
    """Return each policy's place in the rotation, sample by sample.

    ``_measure`` rotates the launch order by one policy per sample and
    synchronizes only after the last launch of a sample. The policy in
    place 0 starts on an idle stream, so its interval includes host
    dispatch; the later places queue behind a running kernel.
    """
    count = len(policies)
    return {
        name: [(index - sample) % count for sample in range(sample_count)]
        for index, name in enumerate(policies)
    }


def _position_medians(samples):
    """Return the median of each policy at each place in the rotation.

    Args:
        samples: The samples of each policy in run order, as ``_measure``
            returns them.

    Returns:
        For each policy, one median per place, place 0 first.
    """
    policies = tuple(samples)
    positions = _positions(policies, len(samples[policies[0]]))
    return {
        name: [
            statistics.median(
                value
                for value, position in zip(
                    samples[name], positions[name], strict=True
                )
                if position == place
            )
            for place in range(len(policies))
        ]
        for name in policies
    }


def _ratio_tick_interval(numerator, denominator, tick):
    """Return the ratios half a timer tick on each median allows.

    A median that is a few dozen ticks long is known to about one tick, so
    the ratio of two such medians carries no more digits than this range.
    None when no tick was observed.
    """
    if tick is None:
        return None
    half = tick / 2
    return [
        (numerator - half) / (denominator + half),
        (numerator + half) / (denominator - half),
    ]


def _timer(samples, medians, numerator, denominator):
    """Return the observed timer tick and what it means for the ratio.

    Args:
        samples: The samples of each policy.
        medians: The median of each policy.
        numerator: Policy on top of the gate ratio.
        denominator: Policy under it.

    Returns:
        The step the samples favour as the tick, each median in ticks, and
        the tick-limited range of the gate ratio.
    """
    tick = benchmark_provenance.timer_tick(
        [value for policy in samples.values() for value in policy]
    )
    return {
        "tick_ms": tick,
        "ticks_per_median": {
            name: None if tick is None else median / tick
            for name, median in medians.items()
        },
        "ratio_tick_interval": _ratio_tick_interval(
            medians[numerator], medians[denominator], tick
        ),
    }


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
    from swage import _cuda_backend
    from swage._segmented_qualification import (
        _prepare_planned_sum,
    )

    arguments = _arguments()
    root = pathlib.Path(__file__).resolve().parents[1]
    source = _git_metadata(root)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "mixed-policy benchmark requires CUDA-enabled PyTorch"
        )
    device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(device)
    capability = torch.cuda.get_device_capability(device)
    gate_device = _gate_device(
        gpu_name, capability, any_device=arguments.any_device
    )
    torch.ones(1, device="cuda").sum().item()
    provenance = benchmark_provenance.start(
        torch, benchmark_provenance.swage_build()
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
    position_medians = _position_medians(samples)
    useful_bytes = _useful_bytes(offsets[-1], _COUNT)
    properties = torch.cuda.get_device_properties(device)
    result = {
        "benchmark": "frozen-mixed-policy-segmented-sum",
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
            "cuda_driver": _cuda_backend.driver_version(),
            "gpu": gpu_name,
            "compute_capability": f"sm_{capability[0]}{capability[1]}",
            "multiprocessors": properties.multi_processor_count,
            "total_memory_bytes": properties.total_memory,
        },
        "configuration": _configuration(),
        "distribution_statistics": summarize_lengths(lengths),
        "raw_samples_ms": samples,
        "medians_ms": medians,
        "mixed_to_best_pure_ratio": ratio,
        "gate": {
            "maximum_ratio": _GATE_RATIO,
            "passed": passed,
        },
        # The fields below describe the frozen samples; the gate above is
        # decided on the medians alone, as it was declared.
        "position_matched": {
            "places": (
                "place 0 is launched on an idle stream and includes host "
                "dispatch; later places queue behind a running kernel"
            ),
            "medians_ms": position_medians,
            "mixed_to_best_pure_ratio": [
                _evaluate_gate(
                    {
                        name: places[place]
                        for name, places in position_medians.items()
                    }
                )[0]
                for place in range(len(position_medians))
            ],
        },
        "timer": _timer(
            samples,
            medians,
            "mixed",
            min(("warp", "cta"), key=medians.get),
        ),
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
                "mixed_to_best_pure_ratio": ratio,
                "passed": result["gate"]["passed"],
                "status": result["status"],
            },
            sort_keys=True,
        )
    )
    if gate_device and not result["gate"]["passed"]:
        raise SystemExit("mixed-policy performance gate failed")


if __name__ == "__main__":
    main()
