# benchmarks/benchmark_fixed_runtime.py
"""Measure the frozen installed-wheel CUDA release gates; always retain JSON."""

import argparse
import json
import math
import os
import pathlib
import platform
import statistics
import subprocess
import sys
import sysconfig
import tempfile
import time
from datetime import datetime, timezone

_CONFIG = {
    "cold": {"processes": 5, "n": 129, "block": 128},
    "warm": {
        "warmups": 200,
        "batches": 20,
        "launches": 500,
        "n": 129,
        "block": 128,
    },
    "throughput": {
        "warmups": 25,
        "samples": 100,
        "launches": 32,
        "sizes": [1 << 18, 1 << 20],
        "block": 256,
        "order": "alternating swage/torch, then torch/swage",
    },
    "memory": {"source": "/proc/self/status VmRSS", "unit": "bytes"},
}


def _kernel():
    import swage as sw
    import swage.language as sl

    @sw.jit
    def add_kernel(x_ptr, y_ptr, output_ptr, n, BLOCK: sl.constexpr):
        pid = sl.program_id(0)
        offsets = pid * BLOCK + sl.arange(0, BLOCK)
        mask = offsets < n
        x = sl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = sl.load(y_ptr + offsets, mask=mask, other=0.0)
        sl.store(output_ptr + offsets, x + y, mask=mask)

    return add_kernel


def _installed():
    import swage

    if os.environ.get("PYTHONPATH"):
        raise ValueError(
            "installed-wheel measurement requires PYTHONPATH unset"
        )
    sites = {
        pathlib.Path(sysconfig.get_path(k)).resolve()
        for k in ("purelib", "platlib")
    }
    if not any(
        pathlib.Path(swage.__file__).resolve().is_relative_to(p) for p in sites
    ):
        raise ValueError("benchmark must import swage from the installed wheel")
    return swage


def _rss():
    for line in pathlib.Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    raise ValueError("Linux VmRSS is unavailable")


def _inputs(torch, kernel, n, block):
    x = torch.arange(n, device="cuda", dtype=torch.float32)
    y = x * 0.5
    output = torch.empty_like(x)
    reference = torch.add(x, y)
    arguments = {"x_ptr": x, "y_ptr": y, "output_ptr": output, "n": n}
    constexprs = {"BLOCK": block}
    grid = ((n + block - 1) // block,)

    def launch():
        kernel.launch(
            arguments=arguments,
            constexprs=constexprs,
            grid=grid,
            backend="cuda",
        )

    return launch, output, reference, x, y


def _correct(torch, launch, output, expected):
    output.fill_(float("nan"))
    launch()
    torch.cuda.synchronize()
    if not torch.equal(output, expected):
        raise ValueError("fixed vector-add correctness mismatch")


def _cold_child():
    _installed()
    import torch

    kernel = _kernel()
    launch, output, expected, _, _ = _inputs(torch, kernel, 129, 128)
    torch.cuda.synchronize()
    before = _rss()
    start = time.perf_counter_ns()
    launch()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter_ns() - start) / 1e6
    after = _rss()
    if not torch.equal(output, expected):
        raise ValueError("cold launch correctness mismatch")
    return {
        "elapsed_ms": elapsed,
        "rss_before_bytes": before,
        "rss_after_bytes": after,
        "rss_delta_bytes": after - before,
        "correct": True,
    }


def _summary(samples):
    if not samples or any(not math.isfinite(x) for x in samples):
        raise ValueError("timing samples must be nonempty and finite")
    return {
        "median": statistics.median(samples),
        "maximum": max(samples),
        "minimum": min(samples),
        "p95": statistics.quantiles(samples, n=100, method="inclusive")[94]
        if len(samples) > 1
        else samples[0],
    }


def _gate(samples, thresholds):
    summary = _summary(samples)
    return {
        "statistics": summary,
        "thresholds": thresholds,
        "passed": all(summary[k] <= v for k, v in thresholds.items()),
    }


def evaluate(record):
    """Validate evidence completeness and apply the frozen inclusive gates."""
    raw = record["measurements"]
    if len(raw["cold"]) != 5 or not all(s["correct"] for s in raw["cold"]):
        raise ValueError("five correct fresh-process cold samples are required")
    if len(raw["warm_host_us"]) != 20:
        raise ValueError("twenty warm dispatch batches are required")
    if not all(
        record["correctness"].get(k) is True
        for k in ("cold", "warm", "throughput_262144", "throughput_1048576")
    ):
        raise ValueError("every section requires a correctness preflight")
    gates = {
        "cold_ms": _gate(
            [s["elapsed_ms"] for s in raw["cold"]],
            {"median": 250.0, "maximum": 400.0},
        ),
        "warm_host_us": _gate(
            raw["warm_host_us"], {"median": 15.0, "p95": 20.0}
        ),
        "native_rss_bytes": _gate(
            [s["rss_delta_bytes"] for s in raw["cold"]],
            {"maximum": 512 * 1024 * 1024},
        ),
        "throughput": {},
    }
    for n in (1 << 18, 1 << 20):
        section = raw["throughput"][str(n)]
        if any(len(section[k]) != 100 for k in ("swage_us", "torch_us")):
            raise ValueError(
                "one hundred throughput samples per method required"
            )
        swage = _summary(section["swage_us"])
        torch = _summary(section["torch_us"])
        if torch["median"] <= 0:
            raise ValueError("throughput reference duration must be positive")
        ratio = swage["median"] / torch["median"]
        gates["throughput"][str(n)] = {
            "swage_us": swage,
            "torch_us": torch,
            "median_ratio": ratio,
            "maximum_ratio": 1.50,
            "passed": ratio <= 1.50,
        }
    record["gates"] = gates
    record["valid"] = True
    record["passed"] = all(
        gates[k]["passed"]
        for k in ("cold_ms", "warm_host_us", "native_rss_bytes")
    ) and all(g["passed"] for g in gates["throughput"].values())
    return record["passed"]


def require_qualified_hardware(name, target):
    """Refuse enforcement on any device other than the frozen release runner."""
    if (name, target) != ("NVIDIA RTX A6000", "sm_86"):
        raise ValueError("--enforce requires NVIDIA RTX A6000 / sm_86 exactly")


def _measure(record, enforce):
    swage = _installed()
    import torch
    from swage import env

    record["package"] = swage.__version__
    record["python"] = platform.python_version()
    record["implementation"] = platform.python_implementation()
    record["platform"] = platform.platform()
    record["glibc"] = platform.libc_ver()
    record["torch"] = torch.__version__
    record["torch_cuda_build"] = torch.version.cuda
    if not torch.cuda.is_available():
        raise ValueError(
            "CUDA-enabled PyTorch and an accessible GPU are required"
        )
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    major, minor = torch.cuda.get_device_capability()
    target = f"sm_{major}{minor}"
    name = torch.cuda.get_device_name()
    record["hardware"] = {
        "name": name,
        "target": target,
        "total_memory": properties.total_memory,
        "multiprocessors": properties.multi_processor_count,
        "cpu": platform.processor(),
        "cpu_count": os.cpu_count(),
    }
    record["qualified"] = (name, target) == ("NVIDIA RTX A6000", "sm_86")
    record["environment"] = env.report()
    if enforce:
        require_qualified_hardware(name, target)
    if not record["environment"]["backends"]["cuda"]["available"]:
        raise ValueError("installed CUDA backend health failed")
    native = record["environment"]["native"]
    if native["error"] or native["package_version"] != swage.__version__:
        raise ValueError("installed native package identity failed")
    kernel = _kernel()
    launch, output, expected, _, _ = _inputs(torch, kernel, 129, 128)
    # Preflight is separate so measured child processes still start cold.
    _correct(torch, launch, output, expected)
    record["correctness"]["cold"] = True
    with tempfile.TemporaryDirectory(prefix="swage-slo-cold-") as temp:
        for index in range(5):
            cache = pathlib.Path(temp) / str(index)
            cache.mkdir()
            child = subprocess.run(
                [
                    sys.executable,
                    str(pathlib.Path(__file__).resolve()),
                    "--cold-child",
                ],
                env={**os.environ, "SWAGE_CACHE_DIR": str(cache)},
                capture_output=True,
                text=True,
                check=False,
                timeout=60,
            )
            if child.returncode:
                raise ValueError(f"cold child {index} failed")
            sample = json.loads(child.stdout)
            sample["empty_unique_cache"] = True
            record["measurements"]["cold"].append(sample)
    _correct(torch, launch, output, expected)
    record["correctness"]["warm"] = True
    for _ in range(200):
        launch()
    for _ in range(20):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        for _ in range(500):
            launch()
        torch.cuda.synchronize()
        elapsed = (time.perf_counter_ns() - start) / 1000 / 500
        record["measurements"]["warm_host_us"].append(elapsed)
    for n in (1 << 18, 1 << 20):
        launch, output, expected, x, y = _inputs(torch, kernel, n, 256)
        torch_output = torch.empty_like(x)
        launches = {
            "swage": launch,
            "torch": lambda: torch.add(x, y, out=torch_output),
        }
        for method, run in launches.items():
            _correct(
                torch,
                run,
                output if method == "swage" else torch_output,
                expected,
            )
        record["correctness"][f"throughput_{n}"] = True
        section = {"swage_us": [], "torch_us": [], "orders": []}
        record["measurements"]["throughput"][str(n)] = section
        for index in range(25):
            for method in (
                ("swage", "torch") if index % 2 == 0 else ("torch", "swage")
            ):
                launches[method]()
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        for index in range(100):
            order = ("swage", "torch") if index % 2 == 0 else ("torch", "swage")
            section["orders"].append(order)
            for method in order:
                start_event.record()
                for _ in range(32):
                    launches[method]()
                end_event.record()
                end_event.synchronize()
                section[f"{method}_us"].append(
                    start_event.elapsed_time(end_event) * 1000 / 32
                )


def main(argv=None):
    """Write raw evidence on invalid runs; never waive correctness failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--enforce", action="store_true")
    parser.add_argument(
        "--cold-child", action="store_true", help=argparse.SUPPRESS
    )
    args = parser.parse_args(argv)
    if args.cold_child:
        print(json.dumps(_cold_child(), sort_keys=True))
        return 0
    if args.output is None:
        parser.error("--output is required")
    record = {
        "schema_version": 1,
        "benchmark": "fixed-runtime",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "configuration": _CONFIG,
        "enforced": args.enforce,
        "qualified": False,
        "valid": False,
        "passed": False,
        "correctness": {},
        "gates": {},
        "measurements": {"cold": [], "warm_host_us": [], "throughput": {}},
    }
    try:
        _measure(record, args.enforce)
        evaluate(record)
    except Exception as error:
        record["error"] = {
            "type": type(error).__name__,
            "reason": str(error)
            if isinstance(error, ValueError)
            else "measurement failed; record is invalid",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "valid": record["valid"],
                "passed": record["passed"],
            },
            sort_keys=True,
        )
    )
    return int(not record["valid"] or (args.enforce and not record["passed"]))


if __name__ == "__main__":
    raise SystemExit(main())
