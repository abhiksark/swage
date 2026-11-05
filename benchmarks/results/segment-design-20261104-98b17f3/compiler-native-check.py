# benchmarks/results/segment-design-20261104-98b17f3/compiler-native-check.py
"""Check existing native support for proposed caller-written programs.

Run from the repository root:
    record=benchmarks/results/segment-design-20261104-98b17f3
    PYTHONPATH=build/python_packages:python:python/tests/mlir python \
        "$record/compiler-native-check.py" --output-dir /tmp/swage-recheck

This executes the CPU oracle and compiles PTX. It launches no GPU kernel.
"""

import argparse
import json
import pathlib
import subprocess

import torch
from mlir_swage import ir
from mlir_swage._mlir_libs._swageDialectsNanobind import swage as native
from mlir_swage.dialects import swage
from reduction_programs import reduction_module
from swage._segmented_oracle import _execute_guarded
from swage._segmented_programs import _semantic_module

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", type=pathlib.Path, required=True)
destination = parser.parse_args().output_dir
destination.mkdir(parents=True, exist_ok=False)

values = torch.tensor([1.0, 3.0, 2.0, 6.0], dtype=torch.float32)
offsets = torch.tensor([0, 2, 2, 4], dtype=torch.int32)
energy = reduction_module("sum", "square").replace(
    "@segmented_sum(", "@caller_energy("
)
normalize = (
    _semantic_module("sum")
    .replace("@segmented_sum(", "@caller_normalize(")
    .replace(
        "memref.store %result, %output[%sid] : memref<?xf32>",
        """swage.map_store %segment, %output captures(%result : f32)
        : !swage.segment<f32>, memref<?xf32> {
    ^bb0(%value: f32, %denominator: f32):
      %normalized = arith.divf %value, %denominator : f32
      swage.yield %normalized : f32
    }""",
    )
)
reports = []
for text, name, expected, live, modes in [
    (
        energy,
        "caller_energy",
        [10.0, 0.0, 40.0],
        3,
        ["direct", "partial", "merge"],
    ),
    (
        normalize,
        "caller_normalize",
        [0.25, 0.75, 0.25, 0.75],
        4,
        ["direct"],
    ),
]:
    result = _execute_guarded(values, offsets, text, name, live)
    assert result == expected, (name, result)
    compiled = []
    with ir.Context() as context:
        swage.register_dialects(context)
        module = ir.Module.parse(text)
        original = str(module)
        for mode in modes:
            compiler = {
                "direct": native._compile_segmented_reduction_ptx,
                "partial": native._compile_split_partial_reduction_ptx,
                "merge": native._compile_split_merge_reduction_ptx,
            }[mode]
            extra = {"block_size": 128} if mode == "direct" else {}
            lowered, ptx, contract = compiler(
                module, kernel_name=name, target="sm_86", **extra
            )
            assert ".entry " in ptx and name in ptx
            assert str(module) == original
            compiled.append(
                {
                    "mode": mode,
                    "ptx_bytes": len(ptx),
                    "contract_entry": json.loads(contract)["entry"],
                }
            )
        rejected = None
        if name == "caller_normalize":
            try:
                native._compile_split_partial_reduction_ptx(
                    module, kernel_name=name, target="sm_86"
                )
            except ValueError as error:
                rejected = "planning requires memref.store" in str(error)
                assert rejected, str(error)
            else:
                raise AssertionError("Captured map-store unexpectedly split")
            assert str(module) == original
    reports.append(
        {
            "program": name,
            "cpu_result": result,
            "compiled": compiled,
            "split_rejected": rejected,
            "source_preserved": True,
        }
    )

repository = pathlib.Path(__file__).resolve().parents[3]
record = {
    "source_revision": subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip(),
    "values": values.tolist(),
    "offsets": offsets.tolist(),
    "ptx_target": "sm_86",
    "gpu_kernels_launched": False,
    "programs": reports,
}
serialized = json.dumps(record, indent=2) + "\n"
(destination / "compiler-native-results.json").write_text(
    serialized, encoding="utf-8"
)
print(serialized, end="")
