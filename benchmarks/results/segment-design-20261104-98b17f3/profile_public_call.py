# benchmarks/results/segment-design-20261104-98b17f3/profile_public_call.py
"""Diagnose warm public-call host work without changing compiler sources."""

import argparse
import cProfile
import json
import pathlib
import pstats
import statistics
import subprocess
import time

import swage
import torch


def main():
    """Record exactness, unprofiled latency, and diagnostic CPU call stacks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    destination = parser.parse_args().output_dir
    destination.mkdir(parents=True, exist_ok=False)
    repository = pathlib.Path(__file__).resolve().parents[3]
    segments = 8192
    length = 16
    values = (torch.arange(segments * length) % 17 - 8).float().cuda() / 4
    offsets = torch.arange(segments + 1, device="cuda", dtype=torch.int32)
    offsets *= length
    expected = values.reshape(segments, length).sum(dim=1)

    def call():
        return swage.segment_reduce(values, offsets, "sum")

    for _ in range(25):
        call()
    torch.cuda.synchronize()
    assert torch.equal(call(), expected)
    latencies = []
    for _ in range(50):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        call()
        torch.cuda.synchronize()
        latencies.append((time.perf_counter_ns() - start) / 1000)

    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(100):
        result = call()
    profiler.disable()
    torch.cuda.synchronize()
    assert torch.equal(result, expected)
    profiler.dump_stats(str(destination / "public-call.prof"))
    stats = pstats.Stats(profiler)
    functions = []
    for (filename, line, function), entry in stats.stats.items():
        primitive, total, self_seconds, cumulative, _ = entry
        functions.append({
            "file": filename,
            "line": line,
            "function": function,
            "primitive_calls": primitive,
            "total_calls": total,
            "self_ms": self_seconds * 1000,
            "cumulative_ms": cumulative * 1000,
        })
    functions.sort(key=lambda item: item["cumulative_ms"], reverse=True)
    record = {
        "purpose": "Diagnostic host attribution, not benchmark qualification",
        "source_revision": subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
        ).strip(),
        "gpu": torch.cuda.get_device_name(),
        "pytorch": str(torch.__version__),
        "segments": segments,
        "segment_length": length,
        "profiled_calls": 100,
        "correctness_passed": True,
        "unprofiled_synchronized_median_us": statistics.median(latencies),
        "unprofiled_synchronized_samples_us": latencies,
        "profiled_total_ms": stats.total_tt * 1000,
        "caveat": (
            "cProfile adds overhead; cumulative spans overlap and include waits"
        ),
        "functions": functions,
    }
    (destination / "public-call-profile.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    excluded = ("functions", "unprofiled_synchronized_samples_us")
    print(json.dumps({key: value for key, value in record.items()
                      if key not in excluded}))
    for function in functions[:18]:
        print(json.dumps(function))


if __name__ == "__main__":
    main()
