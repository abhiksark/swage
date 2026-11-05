# benchmarks/results/segment-design-20261104-98b17f3/artifact-subgroup-repro.py
"""Reproduce accepted inconsistent subgroup metadata in a generated artifact."""

import json
import os
import pathlib
import subprocess
import sys
import tempfile

root = pathlib.Path(tempfile.mkdtemp(prefix="swage-review-subgroup-"))
artifact = root / "artifact"
subprocess.run(
    [
        sys.executable, "-m", "swage.compile", "--target", "sm_86",
        "--program", "sum", "--output", str(artifact),
    ],
    check=True,
)
child = """
# Embedded child for the artifact subgroup reproducer.
import json
import sys

import torch
import swage
values = torch.arange(1, 9, dtype=torch.float32, device='cuda')
offsets = torch.arange(9, dtype=torch.int32, device='cuda')
out = torch.full_like(values, -999)
swage.segment_reduce(values, offsets, 'sum', out=out)
actual = out.tolist()
assert actual == json.loads(sys.argv[1]), actual
print(actual)
"""
environment = dict(os.environ, SWAGE_ARTIFACT_DIR=str(artifact))
print("Original artifact:", flush=True)
subprocess.run(
    [sys.executable, "-c", child, json.dumps(list(range(1, 9)))],
    check=True, env=environment,
)
manifest_path = artifact / "manifest.json"
manifest = json.loads(manifest_path.read_text())
assert manifest["target_description"]["subgroup_width"] == 32
manifest["target_description"]["subgroup_width"] = 16
manifest_path.write_text(json.dumps(manifest))
print("Only subgroup metadata changed to 16:", flush=True)
subprocess.run(
    [sys.executable, "-c", child, json.dumps([1, 2, 3, 4] + [-999] * 4)],
    check=True, env=environment,
)
print("Reproduction artifact:", artifact)
