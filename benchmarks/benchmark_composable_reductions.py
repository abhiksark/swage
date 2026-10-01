# benchmarks/benchmark_composable_reductions.py
"""Measure private composable reductions on a possibly dirty worktree.

Exploratory evidence only; this does not alter or run the frozen sum gate.
Run with PYTHONPATH=python:build/python_packages and --output result.json.
"""

import argparse
import hashlib
import itertools
import json
import pathlib
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone

from benchmark_triton_comparison import _call_us, _git_metadata, _graph_us
from distributions import generate_lengths, summarize_lengths


def _workloads():
    """Return fixed tiny, bimodal, split, and mixed workloads."""
    rng = random.Random(7)
    mixed = [rng.randint(0, 32) for _ in range(3600)]
    mixed += [rng.randint(33, 4096) for _ in range(456)]
    mixed += [65536] * 40
    rng.shuffle(mixed)
    return {
        "many-tiny": generate_lengths("many-tiny", 32768, 7),
        "bimodal": generate_lengths("bimodal", 32768, 7),
        "split-only": [8192] * 256,
        "mixed-with-splits": mixed,
    }


def _transform(values, name):
    """Evaluate the same element program for CPU and eager CUDA references."""
    if name == "square":
        return values * values
    if name == "maps":
        return 2 * (values + 1)
    if name == "exp2":
        return values.exp2()
    if name in ("exp2_chain", "exp2_pair"):
        for _ in range(8 if name == "exp2_chain" else 2):
            values = (-0.5 * values).exp2()
    if name in ("rational8", "rational4"):
        for _ in range(8 if name == "rational8" else 4):
            values = (values + 0.125) / (1 + 0.25 * values * values)
    if name in ("affine4", "affine16", "affine32"):
        for _ in range({"affine4": 2, "affine16": 8, "affine32": 16}[name]):
            values = -0.5 * values + 0.125
    return values


def _held_out_workloads(validation=False):
    """Exercise new lengths, ordering, and selection boundaries, seed 23."""
    if validation:
        rng = random.Random(91)
        return {
            "near-uniform-192": [rng.randint(7168, 8192) for _ in range(192)],
            "varied-384": [rng.randint(4097, 8192) for _ in range(384)],
            "sparse-varied-48": [rng.randint(4097, 8192) for _ in range(48)],
            "long-varied-192": [rng.randint(8193, 12288) for _ in range(192)],
        }
    rng = random.Random(23)
    varied = [rng.randint(4097, 8192) for _ in range(256)]
    tail = [4097] * 230 + [8192] * 26
    rng.shuffle(tail)
    return {
        "varied-84": varied[:84],
        "varied-170": varied[:170],
        "moderate-tail-256": tail,
        "alternating-512": [4097, 8192] * 256,
        "sparse-varied-32": varied[:32],
        "boundary-outlier-256": [*varied[:-1], 8193],
    }


def main():
    """Check every policy, then record preparation and warm execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--warmups", type=int, default=25)
    parser.add_argument(
        "--suite", choices=("baseline", "held-out", "validation"),
        default="baseline",
    )
    parser.add_argument(
        "--fixed-schedule", action="store_true",
        help="Disable preparation-time selection to reproduce the baseline.",
    )
    args = parser.parse_args()
    if args.samples < 2 or args.warmups < 1:
        parser.error("samples must be >= 2 and warmups must be >= 1")
    held_out = args.suite != "baseline"
    validation = args.suite == "validation"
    tolerance = {"rtol": 1e-4, "atol": 1e-5} if held_out else {
        "rtol": 0, "atol": 0
    }

    import swage
    import torch
    from mlir_swage._mlir_libs import _swageDialectsNanobind
    from swage._segmented_qualification import _prepare_planned_reduction

    root = pathlib.Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "python/tests/mlir"))
    from reduction_programs import reduction_module

    device = torch.cuda.current_device()
    torch.ones(1, device="cuda").sum().item()
    native_path = pathlib.Path(_swageDialectsNanobind.__file__)
    paths = [
        pathlib.Path(__file__).resolve(),
        root / "benchmarks/benchmark_triton_comparison.py",
        root / "benchmarks/distributions.py",
        root / "python/tests/mlir/reduction_programs.py",
        root / "python/swage/_segmented_qualification.py",
        root / "lib/Conversion/SegmentedReduction/SegmentedReduction.cpp",
        native_path,
    ]
    telemetry = subprocess.run(
        ["nvidia-smi"], capture_output=True, text=True, check=False
    )
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "provisional engineering run; no performance gate",
        "source": _git_metadata(root),
        "sha256": {
            str(path.relative_to(root)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in paths
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "capability": torch.cuda.get_device_capability(device),
            "swage_module": swage.__file__,
            "native_module": str(native_path),
            "nvidia_smi_returncode": telemetry.returncode,
            "nvidia_smi": telemetry.stdout + telemetry.stderr,
        },
        "configuration": {
            "suite": args.suite,
            "samples": args.samples,
            "warmups": args.warmups,
            "graph_batch": 32,
            "seed": 7,
            "warp_max_elements": 32,
            "cta_chunk_elements": 4096,
            "select_schedule": not args.fixed_schedule,
            "values": "CPU seeded randint(-4, 4) / 4, float32",
            "correctness": (
                "CPU float32 transform, float64 segment_reduce reference"
                if held_out else "exact CPU segment_reduce reference"
            ),
            "tolerance": tolerance,
            "length_seed": 91 if validation else (23 if held_out else 7),
            "timed_policies": (
                ["cta", "mixed"] if held_out
                else ["warp", "cta", "mixed", "torch"]
            ),
            "prepare_ms": (
                "One synchronized wall-time sample for the whole policy "
                "bundle: validation, offset transfer, planning, "
                "metadata/scratch allocation, and compilation and module "
                "loading only for kernels the process has not already "
                "compiled and loaded. Input/output allocation and CUDA "
                "initialization excluded. First case includes native "
                "initialization; no cold-cache claim."
            ),
            "call": "Synchronized Python call latency, preparation excluded.",
            "graph": "CUDA events around 32 captured calls, divided by 32.",
            "order": "Rotate policy order across cases; sequential samples.",
            "torch": (
                "Eager segment_reduce with transform and output allocations "
                "inside call; Swage writes preallocated output. Graph replay "
                "reuses captured allocations for both."
            ),
        },
        "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    workloads = _held_out_workloads(validation) if held_out else _workloads()
    transforms = (
        ("identity", "exp2", "exp2_chain", "rational8", "affine4", "affine32")
        if held_out else ("identity", "square", "maps")
    )
    if validation:
        transforms = ("identity", "exp2_pair", "rational4", "affine16")
    for workload, lengths in workloads.items():
        host_offsets = torch.tensor(
            [0, *itertools.accumulate(lengths)], dtype=torch.int32
        )
        host_values = torch.randint(
            -4, 4, (sum(lengths),), generator=torch.Generator().manual_seed(7)
        ).float() / 4
        values = host_values.cuda()
        offsets = host_offsets.cuda()
        output = torch.empty(len(lengths), device="cuda")
        for kind, transform in itertools.product(
            ("sum", "max"), transforms
        ):
            reference_values = _transform(host_values, transform)
            if held_out:
                reference_values = reference_values.double()
            expected = torch.segment_reduce(
                reference_values, kind, offsets=host_offsets
            ).float().cuda()
            module_text = reduction_module(kind, transform)
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            prepared = _prepare_planned_reduction(
                values, offsets, output, module_text=module_text,
                kernel_name=f"segmented_{kind}",
                select_schedule=not args.fixed_schedule,
            )
            torch.cuda.synchronize()
            prepare_ms = (time.perf_counter_ns() - start) / 1e6

            def launch_torch():
                return torch.segment_reduce(
                    _transform(values, transform), kind, offsets=offsets
                )

            launches = dict(zip(prepared._fields, prepared))
            launches["torch"] = launch_torch
            if held_out:
                launches = {"cta": prepared.cta, "mixed": prepared.mixed}
            row = {
                "workload": workload,
                "lengths": summarize_lengths(lengths),
                "kind": kind,
                "transform": transform,
                "prepare_ms": prepare_ms,
                "mixed_schedule": (
                    "cta" if prepared.mixed is prepared.cta else "mixed"
                ),
                "policies": {},
            }
            names = list(launches)
            rotation = len(report["results"]) % len(names)
            for name in names[rotation:] + names[:rotation]:
                launch = launches[name]
                output.fill_(float("nan"))
                result = launch()
                torch.testing.assert_close(
                    result if name == "torch" else output,
                    expected, **tolerance,
                )
                row["policies"][name] = {
                    "call": _call_us(
                        torch, launch, args.warmups, args.samples
                    ),
                    "graph": _graph_us(
                        torch, launch, args.warmups, args.samples
                    ),
                }
                if name != "torch":
                    torch.testing.assert_close(output, expected, **tolerance)
            row["correctness_passed"] = True
            report["results"].append(row)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            medians = {
                name: round(timing["graph"]["summary_us"]["median"], 2)
                for name, timing in row["policies"].items()
                if timing["graph"]["available"]
            }
            print(
                f"{workload} {kind}/{transform}: prep={prepare_ms:.1f}ms "
                f"graph_us={medians}", flush=True,
            )


if __name__ == "__main__":
    main()
