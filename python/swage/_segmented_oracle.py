# python/swage/_segmented_oracle.py
"""Subprocess-backed CPU oracles for segmented qualification."""

import os
import pathlib
import re
import shutil
import struct
import subprocess

from . import _runtime
from ._segmented_programs import _SOFTMAX_MODULE, _semantic_module
from ._segmented_validation import (
    _validate_softmax_tensors,
    _validate_tensors,
)

_LOWERING_PIPELINE = (
    "builtin.module(func.func(convert-scf-to-cf,convert-math-to-llvm,"
    "convert-arith-to-llvm),"
    "finalize-memref-to-llvm,convert-func-to-llvm,convert-cf-to-llvm,"
    "reconcile-unrealized-casts)"
)
_SENTINEL = -1.0


def _float_literal(value):
    """Emit an exact f32 bit pattern accepted by the MLIR parser."""
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return f"0x{bits:08X}"


def _runner_module(values, offsets, semantic, kernel_name, output_length):
    """Add a no-argument executable wrapper around the semantic kernel."""
    value_count = values.numel()
    segment_count = offsets.numel() - 1
    values_type = f"memref<{value_count}xf32>"
    offsets_type = f"memref<{segment_count + 1}xi32>"
    output_type = f"memref<{output_length}xf32>"
    lines = [
        semantic.rstrip()[:-1],
        "",
        "  func.func @main() {",
        f"    %values_storage = memref.alloc() : {values_type}",
        f"    %offsets_storage = memref.alloc() : {offsets_type}",
        f"    %output_storage = memref.alloc() : {output_type}",
        (
            f"    %values = memref.cast %values_storage : {values_type} "
            "to memref<?xf32>"
        ),
        (
            f"    %offsets = memref.cast %offsets_storage : {offsets_type} "
            "to memref<?xi32>"
        ),
        (
            f"    %output = memref.cast %output_storage : {output_type} "
            "to memref<?xf32>"
        ),
        f"    %sentinel = arith.constant {_float_literal(_SENTINEL)} : f32",
        "    %prefill_from = arith.constant 0 : index",
        "    %prefill_step = arith.constant 1 : index",
        f"    %prefill_to = arith.constant {output_length} : index",
        "    scf.for %pi = %prefill_from to %prefill_to step %prefill_step {",
        "      memref.store %sentinel, %output[%pi] : memref<?xf32>",
        "    }",
    ]
    for index, value in enumerate(values.tolist()):
        lines.extend(
            [
                f"    %vi{index} = arith.constant {index} : index",
                (
                    f"    %vv{index} = arith.constant "
                    f"{_float_literal(value)} : f32"
                ),
                (
                    f"    memref.store %vv{index}, %values[%vi{index}] "
                    ": memref<?xf32>"
                ),
            ]
        )
    for index, offset in enumerate(offsets.tolist()):
        lines.extend(
            [
                f"    %oi{index} = arith.constant {index} : index",
                f"    %ov{index} = arith.constant {offset} : i32",
                (
                    f"    memref.store %ov{index}, %offsets[%oi{index}] "
                    ": memref<?xi32>"
                ),
            ]
        )
    lines.extend(
        [
            (
                f"    call @{kernel_name}(%values, %offsets, %output) : "
                "(memref<?xf32>, memref<?xi32>, memref<?xf32>) -> ()"
            ),
            (
                "    %unranked = memref.cast %output : memref<?xf32> "
                "to memref<*xf32>"
            ),
            "    call @printMemrefF32(%unranked) : (memref<*xf32>) -> ()",
            f"    memref.dealloc %values_storage : {values_type}",
            f"    memref.dealloc %offsets_storage : {offsets_type}",
            f"    memref.dealloc %output_storage : {output_type}",
            "    return",
            "  }",
            "",
            (
                "  func.func private @printMemrefF32(memref<*xf32>) "
                "attributes {llvm.emit_c_interface}"
            ),
            "}",
        ]
    )
    return "\n".join(lines)


def _llvm_root(build):
    """Find the pinned install used to configure the current build."""
    cache = build / "CMakeCache.txt"
    match = re.search(r"^MLIR_DIR:[^=]*=(.+)$", cache.read_text(), re.MULTILINE)
    if not match:
        raise RuntimeError(f"{cache} does not identify MLIR_DIR")
    return pathlib.Path(match.group(1)).parents[2]


def _run(command, source):
    """Run one compiler stage and return its text output."""
    result = subprocess.run(
        [str(argument) for argument in command],
        input=source,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"{' '.join(map(str, command))} failed:\n{result.stderr}"
        )
    return result.stdout


def _execute(module_text):
    """Lower and run one executable module, returning its printed values."""
    root = pathlib.Path(__file__).resolve().parents[2]
    build = pathlib.Path(os.environ.get("SWAGE_BUILD_DIR", root / "build"))
    llvm_root = _llvm_root(build)
    swage_opt = build / "bin" / "swage-opt"
    mlir_opt = shutil.which("mlir-opt") or llvm_root / "bin" / "mlir-opt"
    mlir_runner = (
        shutil.which("mlir-runner") or llvm_root / "bin" / "mlir-runner"
    )
    lowered = _run(
        [swage_opt, "--swage-segmented-reduction-to-scf"], module_text
    )
    llvm = _run([mlir_opt, f"--pass-pipeline={_LOWERING_PIPELINE}"], lowered)
    runner_utils = llvm_root / "lib" / "libmlir_runner_utils.so"
    c_runner_utils = llvm_root / "lib" / "libmlir_c_runner_utils.so"
    printed = _run(
        [
            mlir_runner,
            "-e",
            "main",
            "-entry-point-result=void",
            f"-shared-libs={runner_utils}",
            f"-shared-libs={c_runner_utils}",
        ],
        llvm,
    )
    match = re.search(r"data =\s*\n\[(.*?)\]", printed, re.DOTALL)
    if not match:
        raise RuntimeError(
            f"mlir-runner returned an unreadable result:\n{printed}"
        )
    tokens = [token.strip() for token in match.group(1).split(",")]
    return [float(token) for token in tokens if token]


def cpu_oracle(values, offsets, kind):
    """Execute the sequential reduction lowering with the MLIR runner."""
    torch = _runtime._import_torch()
    segment_count = max(offsets.numel() - 1, 0)
    output = torch.empty(segment_count, dtype=torch.float32)
    _validate_tensors(values, offsets, output, require_cuda=False)
    printed = _execute(
        _runner_module(
            values,
            offsets,
            _semantic_module(kind),
            f"segmented_{kind}",
            segment_count,
        )
    )
    return torch.tensor(printed, dtype=torch.float32)


def cpu_softmax_oracle(values, offsets):
    """Execute the sequential softmax lowering with the MLIR runner."""
    torch = _runtime._import_torch()
    covered = int(offsets[-1]) if offsets.numel() else 0
    output = torch.empty(covered, dtype=torch.float32)
    _validate_softmax_tensors(values, offsets, output, require_cuda=False)
    printed = _execute(
        _runner_module(
            values, offsets, _SOFTMAX_MODULE, "ragged_softmax", covered + 1
        )
    )
    if len(printed) != covered + 1:
        raise RuntimeError(
            f"oracle printed {len(printed)} values for {covered + 1} slots"
        )
    if printed[-1] != _SENTINEL:
        raise RuntimeError(
            f"map_store wrote past the final offset: {printed[-1]}"
        )
    return torch.tensor(printed[:-1], dtype=torch.float32)
