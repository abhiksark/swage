# benchmarks/results/fixed-dispatch-a6000-sm86-3d0fe92/analyze.py
"""Verify all frozen process records and recompute the paired decisions."""

import hashlib
import json
import lzma
import random
import statistics
from pathlib import Path


def _metrics(record):
    """Read the recorded gate statistics without altering their samples."""
    gates = record["gates"]
    return {
        "warm_us": gates["warm_host_us"]["statistics"]["median"],
        "warm_p95_us": gates["warm_host_us"]["statistics"]["p95"],
        "cold_median_ms": gates["cold_ms"]["statistics"]["median"],
        "cold_max_ms": gates["cold_ms"]["statistics"]["maximum"],
        "rss_max_bytes": gates["native_rss_bytes"]["statistics"]["maximum"],
        "ratio_262144": gates["throughput"]["262144"]["median_ratio"],
        "ratio_1048576": gates["throughput"]["1048576"]["median_ratio"],
    }


def _summary(directory):
    """Hash and validate every record, then summarize all twenty pairs."""
    manifest = json.loads((directory / "manifest.json").read_text())
    if not manifest["complete"] or len(manifest["records"]) != 40:
        raise ValueError("Expected one complete forty-process campaign")
    records = {}
    all_passed = True
    for entry in manifest["records"]:
        raw = lzma.decompress((directory / entry["record"]).read_bytes())
        if hashlib.sha256(raw).hexdigest() != entry["uncompressed_sha256"]:
            raise ValueError(f"Record checksum differs: {entry['record']}")
        record = json.loads(raw)
        native = record["environment"]["native"]
        identity = (
            native["source_clean"]
            and native["source_revision"]
            == manifest["source_revisions"][entry["side"]]
        )
        if not identity or not record["valid"] or not record["qualified"]:
            raise ValueError(f"Invalid record: {entry['record']}")
        if not all(record["correctness"].values()):
            raise ValueError(f"Incorrect record: {entry['record']}")
        key = (entry["pair"], entry["side"])
        if key in records:
            raise ValueError("Repeated pair/side record")
        records[key] = _metrics(record)
        all_passed &= entry["exit_code"] == 0 and record["passed"]
    expected = {
        (pair, side)
        for pair in range(1, 21)
        for side in ("baseline", "candidate")
    }
    if set(records) != expected:
        raise ValueError("A pair/side record is missing")
    gains = {}
    for metric in records[(1, "baseline")]:
        gains[metric] = [
            100
            * (
                1
                - records[(pair, "candidate")][metric]
                / records[(pair, "baseline")][metric]
            )
            for pair in range(1, 21)
        ]
    warm = gains["warm_us"]
    rng = random.Random(20261014)
    bootstrap = sorted(
        statistics.median(rng.choices(warm, k=20)) for _ in range(20000)
    )
    interval = [bootstrap[499], bootstrap[19499]]
    regressions = {
        metric: statistics.median(values) < -5
        for metric, values in gains.items()
        if metric != "warm_us"
    }
    result = {
        "warm_gain_percent": statistics.median(warm),
        "warm_paired_bootstrap_95_percent_interval": interval,
        "all_records_within_frozen_gates_and_identity_valid": all_passed,
        "material_regressions": regressions,
        "retain": (
            all_passed
            and statistics.median(warm) >= 5
            and interval[0] > 0
            and not any(regressions.values())
        ),
    }
    archived = json.loads((directory / "decision.json").read_text())
    if any(archived[key] != value for key, value in result.items()):
        raise ValueError("Recomputed decision differs from archived result")
    return result


def main():
    """Print decisions only after every raw record has been verified."""
    root = Path(__file__).resolve().parent
    print(
        json.dumps(
            {name: _summary(root / name) for name in ("method", "slots")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
