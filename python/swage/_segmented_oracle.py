# python/swage/_segmented_oracle.py
"""CPU oracles of the segment programs over `--swage-plan-to-scf`.

An oracle wraps the semantic program in an executable module, lowers it
with the sequential schedule of `swage-opt`, and runs it with the MLIR
runner. Its results are the exact values of a left-to-right evaluation,
which the GPU tests compare against.
"""

import os
import pathlib
import re
import shutil
import struct
import subprocess

from . import _runtime
from . import _segmented_programs as _programs
from . import _segmented_validation as _validation

_LOWERING_PIPELINE = (
    "builtin.module(func.func(convert-scf-to-cf,convert-math-to-llvm,"
    "convert-arith-to-llvm),"
    "finalize-memref-to-llvm,convert-func-to-llvm,convert-cf-to-llvm,"
    "reconcile-unrealized-casts)"
)
_SENTINEL = -1.0


# How an element of each type travels through the oracle: the struct codes
# of its value and of its bit pattern, the integer type of the pattern, and
# the runner function that prints a memref of that integer type.
_TRANSPORT = {
    "f32": ("f", "I", "i32", "printMemrefI32"),
    "f64": ("d", "Q", "i64", "printMemrefI64"),
}


def _float_literal(value, element="f32"):
    """Emit an exact bit pattern of `element` that the MLIR parser accepts."""
    code, pattern, bits_type, _ = _TRANSPORT[element]
    bits = struct.unpack(f"<{pattern}", struct.pack(f"<{code}", value))[0]
    return f"0x{bits:0{int(bits_type[1:]) // 4}X}"


def _dense_literal(code, numbers):
    """Emit the raw little-endian bytes of a dense MLIR initializer."""
    payload = struct.pack(f"<{len(numbers)}{code}", *numbers)
    return f'dense<"0x{payload.hex()}">'


def _runner_module(values, offsets, semantic, kernel_name, output_length):
    """Add a no-argument executable wrapper around the semantic kernel.

    The inputs become constant globals initialized from their exact bytes,
    one operation per buffer instead of three per element. The result
    leaves as bit patterns: each output is bitcast to the integer of its
    width and printed by the integer memref printer, because the float
    printer keeps only six significant digits. The element type is the one
    the program declares for its values.
    """
    if values.dim() == 2:
        return _column_runner_module(
            values, offsets, semantic, kernel_name, output_length
        )
    element = _programs._program_element(semantic)
    code, _, word, printer = _TRANSPORT[element]
    value_count = values.numel()
    segment_count = offsets.numel() - 1
    values_type = f"memref<{value_count}x{element}>"
    offsets_type = f"memref<{segment_count + 1}xi32>"
    output_type = f"memref<{output_length}x{element}>"
    bits_type = f"memref<{output_length}x{word}>"
    lines = [
        semantic.rstrip()[:-1],
        "",
        (
            f'  memref.global "private" constant @runner_offsets : '
            f"{offsets_type} = {_dense_literal('i', offsets.tolist())}"
        ),
    ]
    # A zero-element dense initializer has no bytes to parse, so an empty
    # values buffer stays a plain allocation that nothing reads.
    values_storage = f"memref.alloc() : {values_type}"
    if value_count:
        lines.append(
            f'  memref.global "private" constant @runner_values : '
            f"{values_type} = {_dense_literal(code, values.tolist())}"
        )
        values_storage = f"memref.get_global @runner_values : {values_type}"
    lines.extend(
        [
            "",
            "  func.func @main() {",
            f"    %values_storage = {values_storage}",
            (
                f"    %offsets_storage = memref.get_global @runner_offsets : "
                f"{offsets_type}"
            ),
            f"    %output_storage = memref.alloc() : {output_type}",
            f"    %bits = memref.alloc() : {bits_type}",
            (
                f"    %values = memref.cast %values_storage : {values_type} "
                f"to memref<?x{element}>"
            ),
            (
                f"    %offsets = memref.cast %offsets_storage : "
                f"{offsets_type} to memref<?xi32>"
            ),
            (
                f"    %output = memref.cast %output_storage : {output_type} "
                f"to memref<?x{element}>"
            ),
            # Prefill so that a store past the live range is visible in the
            # parsed result instead of being garbage.
            (
                f"    %sentinel = arith.constant "
                f"{_float_literal(_SENTINEL, element)} : {element}"
            ),
            "    %from = arith.constant 0 : index",
            "    %step = arith.constant 1 : index",
            f"    %to = arith.constant {output_length} : index",
            "    scf.for %pi = %from to %to step %step {",
            (
                "      memref.store %sentinel, %output[%pi] : "
                f"memref<?x{element}>"
            ),
            "    }",
            f"    %value_count = arith.constant {value_count} : i32",
            f"    %segment_count = arith.constant {segment_count} : i32",
            (
                f"    call @{kernel_name}(%values, %offsets, %output, "
                f"%value_count, %segment_count) : (memref<?x{element}>, "
                f"memref<?xi32>, memref<?x{element}>, i32, i32) -> ()"
            ),
            "    scf.for %bi = %from to %to step %step {",
            (f"      %result = memref.load %output[%bi] : memref<?x{element}>"),
            f"      %pattern = arith.bitcast %result : {element} to {word}",
            f"      memref.store %pattern, %bits[%bi] : {bits_type}",
            "    }",
            (
                f"    %unranked = memref.cast %bits : {bits_type} "
                f"to memref<*x{word}>"
            ),
            f"    call @{printer}(%unranked) : (memref<*x{word}>) -> ()",
        ]
    )
    if not value_count:
        lines.append(f"    memref.dealloc %values_storage : {values_type}")
    lines.extend(
        [
            f"    memref.dealloc %output_storage : {output_type}",
            f"    memref.dealloc %bits : {bits_type}",
            "    return",
            "  }",
            "",
            (
                f"  func.func private @{printer}(memref<*x{word}>) "
                "attributes {llvm.emit_c_interface}"
            ),
            "}",
        ]
    )
    return "\n".join(lines)


def _column_runner_module(values, offsets, semantic, kernel_name, output_rows):
    """Add an executable wrapper around a kernel over rank-two values.

    The module is the one of `_runner_module` with rows in place of
    elements: the values are a constant `[rows, columns]` global, the output
    has `output_rows` rows of the same columns, and its bit patterns are
    printed in row order.
    """
    element = _programs._program_element(semantic)
    code, _, word, printer = _TRANSPORT[element]
    row_count, columns = values.shape
    segment_count = offsets.numel() - 1
    values_type = f"memref<{row_count}x{columns}x{element}>"
    offsets_type = f"memref<{segment_count + 1}xi32>"
    output_type = f"memref<{output_rows}x{columns}x{element}>"
    rows_type = f"memref<?x?x{element}>"
    bits_type = f"memref<{output_rows * columns}x{word}>"
    lines = [
        semantic.rstrip()[:-1],
        "",
        (
            f'  memref.global "private" constant @runner_offsets : '
            f"{offsets_type} = {_dense_literal('i', offsets.tolist())}"
        ),
    ]
    values_storage = f"memref.alloc() : {values_type}"
    if values.numel():
        lines.append(
            f'  memref.global "private" constant @runner_values : '
            f"{values_type} = "
            f"{_dense_literal(code, values.reshape(-1).tolist())}"
        )
        values_storage = f"memref.get_global @runner_values : {values_type}"
    lines.extend(
        [
            "",
            "  func.func @main() {",
            f"    %values_storage = {values_storage}",
            (
                f"    %offsets_storage = memref.get_global @runner_offsets : "
                f"{offsets_type}"
            ),
            f"    %output_storage = memref.alloc() : {output_type}",
            f"    %bits = memref.alloc() : {bits_type}",
            (
                f"    %values = memref.cast %values_storage : {values_type} "
                f"to {rows_type}"
            ),
            (
                f"    %offsets = memref.cast %offsets_storage : "
                f"{offsets_type} to memref<?xi32>"
            ),
            (
                f"    %output = memref.cast %output_storage : {output_type} "
                f"to {rows_type}"
            ),
            (
                f"    %sentinel = arith.constant "
                f"{_float_literal(_SENTINEL, element)} : {element}"
            ),
            "    %from = arith.constant 0 : index",
            "    %step = arith.constant 1 : index",
            f"    %rows = arith.constant {output_rows} : index",
            f"    %columns = arith.constant {columns} : index",
            "    scf.for %pr = %from to %rows step %step {",
            "      scf.for %pc = %from to %columns step %step {",
            f"        memref.store %sentinel, %output[%pr, %pc] : {rows_type}",
            "      }",
            "    }",
            f"    %value_count = arith.constant {row_count} : i32",
            f"    %segment_count = arith.constant {segment_count} : i32",
            f"    %feature_count = arith.constant {columns} : i32",
            (
                f"    call @{kernel_name}(%values, %offsets, %output, "
                f"%value_count, %segment_count, %feature_count) : "
                f"({rows_type}, memref<?xi32>, {rows_type}, i32, i32, i32) "
                "-> ()"
            ),
            "    scf.for %br = %from to %rows step %step {",
            "      %row = arith.muli %br, %columns : index",
            "      scf.for %bc = %from to %columns step %step {",
            f"        %result = memref.load %output[%br, %bc] : {rows_type}",
            f"        %pattern = arith.bitcast %result : {element} to {word}",
            "        %slot = arith.addi %row, %bc : index",
            f"        memref.store %pattern, %bits[%slot] : {bits_type}",
            "      }",
            "    }",
            (
                f"    %unranked = memref.cast %bits : {bits_type} "
                f"to memref<*x{word}>"
            ),
            f"    call @{printer}(%unranked) : (memref<*x{word}>) -> ()",
        ]
    )
    if not values.numel():
        lines.append(f"    memref.dealloc %values_storage : {values_type}")
    lines.extend(
        [
            f"    memref.dealloc %output_storage : {output_type}",
            f"    memref.dealloc %bits : {bits_type}",
            "    return",
            "  }",
            "",
            (
                f"  func.func private @{printer}(memref<*x{word}>) "
                "attributes {llvm.emit_c_interface}"
            ),
            "}",
        ]
    )
    return "\n".join(lines)


# Names the Swage build directory the CPU oracle takes its tools from. It is
# a test and development setting: a `swage` that was installed from a wheel
# has no build directory beside it.
_ORACLE_BUILD = "SWAGE_ORACLE_BUILD_DIR"
_ORACLE_BUILD_FILES = ("CMakeCache.txt", "bin/swage-opt")


def _oracle_build():
    """Return the Swage build directory the CPU oracle takes its tools from.

    The oracle runs `bin/swage-opt` of a build directory and reads its
    `CMakeCache.txt` to find the LLVM install that build was configured
    with. `SWAGE_ORACLE_BUILD_DIR` names the directory. Without it, the
    directory is `build` in the checkout this module was imported from.
    The variable is read at every call.

    Raises:
        RuntimeError: The directory lacks one of the two files.
    """
    named = os.environ.get(_ORACLE_BUILD)
    if named:
        build = pathlib.Path(named)
        subject, remedy = f"{_ORACLE_BUILD} names {build}, which", ""
    else:
        build = pathlib.Path(__file__).resolve().parents[2] / "build"
        subject, remedy = str(build), f". Set {_ORACLE_BUILD} to one"
    missing = [
        name for name in _ORACLE_BUILD_FILES if not (build / name).is_file()
    ]
    if missing:
        raise RuntimeError(
            f"{subject} lacks {' and '.join(missing)}; the CPU oracle needs "
            f"a Swage build directory{remedy}"
        )
    return build


def _llvm_root(build):
    """Find the pinned install that a Swage build was configured with."""
    cache = build / "CMakeCache.txt"
    match = re.search(r"^MLIR_DIR:[^=]*=(.+)$", cache.read_text(), re.MULTILINE)
    if not match:
        raise RuntimeError(f"{cache} does not identify MLIR_DIR")
    return pathlib.Path(match.group(1)).parents[2]


def _llvm_tool(llvm_root, name):
    """Return one LLVM tool, preferring the pinned install over PATH.

    The runner libraries always come from the pinned install. A tool found
    on PATH may belong to a different LLVM, so it is used only when the
    install does not provide that tool.
    """
    pinned = llvm_root / "bin" / name
    if pinned.is_file():
        return pinned
    found = shutil.which(name)
    if found is None:
        raise RuntimeError(
            f"{name} was found neither at {pinned} nor on PATH; the CPU "
            "oracle requires it from the pinned LLVM install"
        )
    return pathlib.Path(found)


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


def _execute(module_text, element="f32"):
    """Lower and run one executable module, returning its exact results.

    The module prints one signed integer bit pattern per output element, of
    the width of `element`. Each is reinterpreted as the float it encodes,
    so the returned Python floats hold the computed values with no decimal
    rounding in between.
    """
    build = _oracle_build()
    llvm_root = _llvm_root(build)
    swage_opt = build / "bin" / "swage-opt"
    mlir_opt = _llvm_tool(llvm_root, "mlir-opt")
    mlir_runner = _llvm_tool(llvm_root, "mlir-runner")
    lowered = _run(
        [
            swage_opt,
            "--swage-to-plan=schedule=sequential",
            "--swage-plan-to-scf",
        ],
        module_text,
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
    code, pattern, word, _ = _TRANSPORT[element]
    mask = (1 << int(word[1:])) - 1
    patterns = [
        int(token) & mask
        for token in match.group(1).split(",")
        if token.strip()
    ]
    payload = struct.pack(f"<{len(patterns)}{pattern}", *patterns)
    return list(struct.unpack(f"<{len(patterns)}{code}", payload))


def _execute_guarded(values, offsets, semantic, kernel_name, live):
    """Run a kernel with one guard slot after its live output range.

    The extra slot keeps the prefilled sentinel. That makes a zero-length
    result printable and turns "the kernel never writes past its live
    range" into a checked invariant of every oracle call.
    """
    results = _execute(
        _runner_module(values, offsets, semantic, kernel_name, live + 1),
        _programs._program_element(semantic),
    )
    # For rank-two values a slot is a row, and the guard is a whole row.
    width = values.shape[1] if values.dim() == 2 else 1
    if len(results) != (live + 1) * width:
        raise RuntimeError(
            f"oracle printed {len(results)} values for "
            f"{(live + 1) * width} slots"
        )
    guard = results[live * width :]
    if any(value != _SENTINEL for value in guard):
        raise RuntimeError(
            f"{kernel_name} wrote past its live output range: {guard}"
        )
    return results[: live * width]


def cpu_oracle(values, offsets, kind):
    """Execute the sequential reduction lowering with the MLIR runner.

    Returns the exact result of each segment, accumulated left to right, in
    the dtype of `values`: float32 or float64. Rank-two values give one row
    of results per segment, each column accumulated in row order.
    """
    torch = _runtime._import_torch()
    element = _programs._element_of(torch, values) or "f32"
    dtype = _programs._element_dtype(torch, element)
    rank = 2 if getattr(values, "ndim", 1) == 2 else 1
    segment_count = max(offsets.numel() - 1, 0)
    shape = (segment_count,) + tuple(values.shape[1:rank])
    output = torch.empty(shape, dtype=dtype)
    _validation._validate_shapes(
        values,
        offsets,
        output,
        _validation._validate_offsets,
        require_cuda=False,
        element=element,
        rank=rank,
    )
    # A result without a column holds nothing the runner could print.
    if output.numel() == 0 and rank == 2:
        return output
    results = _execute_guarded(
        values,
        offsets,
        _programs._semantic_module(kind, element, rank),
        _programs._reduction_kernel(kind, element, rank),
        segment_count,
    )
    return torch.tensor(results, dtype=dtype).reshape(shape)


def cpu_softmax_oracle(values, offsets):
    """Execute the sequential softmax lowering with the MLIR runner.

    Returns the exact f32 value the lowering stored for each covered element,
    or for each covered row of rank-two values.
    """
    torch = _runtime._import_torch()
    covered = int(offsets[-1]) if offsets.numel() else 0
    rank = 2 if getattr(values, "ndim", 1) == 2 else 1
    shape = (covered,) + tuple(values.shape[1:rank])
    output = torch.empty(shape, dtype=torch.float32)
    _validation._validate_shapes(
        values,
        offsets,
        output,
        _validation._validate_softmax_offsets,
        require_cuda=False,
        rank=rank,
    )
    # A result without a column holds nothing the runner could print.
    if output.numel() == 0 and rank == 2:
        return output
    results = _execute_guarded(
        values,
        offsets,
        _programs._softmax_text(rank),
        "ragged_softmax" if rank == 1 else "ragged_softmax_r2",
        covered,
    )
    return torch.tensor(results, dtype=torch.float32).reshape(shape)
