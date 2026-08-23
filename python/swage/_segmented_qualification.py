# python/swage/_segmented_qualification.py
"""Internal M4 qualification runner for native segmented reductions."""

import pathlib
import re
import shutil
import struct
import subprocess

from . import _runtime

_I32_LIMIT = 1 << 31
_LOWERING_PIPELINE = (
    "builtin.module(func.func(convert-scf-to-cf,convert-math-to-llvm,"
    "convert-arith-to-llvm),"
    "finalize-memref-to-llvm,convert-func-to-llvm,convert-cf-to-llvm,"
    "reconcile-unrealized-casts)"
)


def _validate_counts(value_count, segment_count):
    """Validate the two explicit signed-i32 CUDA ABI counts."""
    for name, count in (
        ("value count", value_count),
        ("segment count", segment_count),
    ):
        if type(count) is not int or not 0 <= count < _I32_LIMIT:
            raise ValueError(f"{name} must be a nonnegative i32")


def _validate_offset_sequence(offsets, value_count):
    """Validate the offset array itself and return the segment count."""
    if not offsets:
        raise ValueError("offsets must contain at least the initial zero")
    segment_count = len(offsets) - 1
    _validate_counts(value_count, segment_count)
    if offsets[0] != 0:
        raise ValueError("offsets must start at zero")
    previous = 0
    for offset in offsets:
        if type(offset) is not int or not -(1 << 31) <= offset < _I32_LIMIT:
            raise ValueError("offsets must contain signed i32 values")
        if offset < 0:
            raise ValueError("offsets must not be negative")
        if offset < previous:
            raise ValueError("offsets must be nondecreasing")
        previous = offset
    if offsets[-1] > value_count:
        raise ValueError(
            f"final offset {offsets[-1]} exceeds value count {value_count}"
        )
    return segment_count


def _validate_offsets(offsets, value_count, output_count):
    """Require one output element per segment, the reduction ABI."""
    segment_count = _validate_offset_sequence(offsets, value_count)
    if type(output_count) is not int or output_count < segment_count:
        raise ValueError(
            f"output has {output_count} elements for {segment_count} segments"
        )
    return segment_count


def _validate_softmax_offsets(offsets, value_count, output_count):
    """Require one output element per covered value, the map_store ABI.

    The bound is the final offset rather than the value count, because
    offsets may cover fewer values than the buffer holds and binding to the
    value count would reject a correctly sized output.
    """
    segment_count = _validate_offset_sequence(offsets, value_count)
    required = offsets[-1]
    if type(output_count) is not int or output_count < required:
        raise ValueError(
            f"output has {output_count} elements for {required} values"
        )
    return segment_count


def _validate_disjoint(name, buffer, output):
    """Reject an output that overlaps a buffer the kernel reads.

    ADR-0008 names only the values buffer, but the offsets buffer needs the
    same treatment: the kernel re-reads offsets from device memory after
    other CTAs have already stored through an aliased output, which voids
    the host-side offset walk entirely.

    Both tensors are known contiguous and rank one by this point, so the
    byte extent is exact and a half-open intersection is exact. It cannot
    see two virtual mappings of one physical allocation, nor aliasing
    created after this returns.
    """
    buffer_start = buffer.data_ptr()
    buffer_end = buffer_start + buffer.numel() * buffer.element_size()
    output_start = output.data_ptr()
    output_end = output_start + output.numel() * output.element_size()
    if buffer_start < output_end and output_start < buffer_end:
        raise ValueError(f"output must not overlap the {name} buffer")


def _validate_shapes(
    values, offsets, output, validate_offsets, *, require_cuda=True
):
    """Validate tensor shapes against one of the two output ABIs."""
    torch = _runtime._import_torch()
    for name, tensor in (("values", values), ("offsets", offsets),
                         ("output", output)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
    for name, tensor in (("values", values), ("output", output)):
        if tensor.dtype != torch.float32:
            raise TypeError(f"{name} must have dtype torch.float32")
    if offsets.dtype != torch.int32:
        raise TypeError("offsets must have dtype torch.int32")
    for name, tensor in (("values", values), ("offsets", offsets),
                         ("output", output)):
        if tensor.dim() != 1:
            raise TypeError(f"{name} must have rank one")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    value_count = values.numel()
    host_offsets = offsets.detach().cpu().tolist()
    segment_count = validate_offsets(
        host_offsets, value_count, output.numel()
    )
    if not require_cuda:
        return value_count, segment_count
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in PyTorch")
    current_device = torch.cuda.current_device()
    for name, tensor in (("values", values), ("offsets", offsets),
                         ("output", output)):
        if tensor.device.type != "cuda":
            raise TypeError(f"{name} must be a CUDA tensor")
        if tensor.device.index != current_device:
            raise ValueError(f"{name} must be on the current CUDA device")
    return value_count, segment_count


def _validate_tensors(values, offsets, output, *, require_cuda=True):
    """Validate the reduction tensors and return their explicit counts."""
    return _validate_shapes(
        values, offsets, output, _validate_offsets, require_cuda=require_cuda
    )


def _validate_softmax_tensors(values, offsets, output, *, require_cuda=True):
    """Validate the softmax tensors, including the aliasing obligation."""
    counts = _validate_shapes(
        values,
        offsets,
        output,
        _validate_softmax_offsets,
        require_cuda=require_cuda,
    )
    # On the CUDA path both tensors are now known to sit on one device,
    # which is what makes comparing their raw addresses meaningful.
    _validate_disjoint("values", values, output)
    _validate_disjoint("offsets", offsets, output)
    return counts


def _semantic_module(kind):
    """Return the canonical private qualification module."""
    if kind not in {"sum", "max"}:
        raise ValueError("reduction kind must be 'sum' or 'max'")
    return f"""
module {{
  func.func @segmented_{kind}(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32) {{
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %result = swage.reduce %segment kind<{kind}>
        : !swage.segment<f32> -> f32 {{
    ^bb0(%value: f32):
      swage.yield %value : f32
    }}
    memref.store %result, %output[%sid] : memref<?xf32>
    return
  }}
}}
"""


_SOFTMAX_MODULE = """
module {
  func.func @ragged_softmax(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %max = swage.reduce %segment kind<max> : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    %shifted = swage.map %segment captures(%max : f32)
        : !swage.segment<f32> -> !swage.segment<f32> {
    ^bb0(%value: f32, %m: f32):
      %log2e = arith.constant 1.44269502 : f32
      %centered = arith.subf %value, %m : f32
      %scaled = arith.mulf %centered, %log2e : f32
      %exponential = math.exp2 %scaled : f32
      swage.yield %exponential : f32
    }
    %total = swage.reduce %shifted kind<sum> : !swage.segment<f32> -> f32 {
    ^bb0(%element: f32):
      swage.yield %element : f32
    }
    swage.map_store %segment, %output captures(%max, %total : f32, f32)
        : !swage.segment<f32>, memref<?xf32> {
    ^bb0(%value: f32, %m: f32, %t: f32):
      %log2e = arith.constant 1.44269502 : f32
      %centered = arith.subf %value, %m : f32
      %scaled = arith.mulf %centered, %log2e : f32
      %exponential = math.exp2 %scaled : f32
      %normalized = arith.divf %exponential, %t : f32
      swage.yield %normalized : f32
    }
    return
  }
}
"""

_SENTINEL = -1.0


def launch_gpu(values, offsets, output, kind, block_size=128):
    """Compile and launch one internally qualified segmented reduction."""
    torch = _runtime._import_torch()
    value_count, segment_count = _validate_tensors(values, offsets, output)
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("block size must be a positive integer")
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    if block_size > properties.max_threads_per_block:
        raise ValueError(
            f"block size {block_size} exceeds device limit "
            f"{properties.max_threads_per_block}"
        )
    if segment_count == 0:
        return None

    from mlir_swage import ir
    from mlir_swage._mlir_libs._swageDialectsNanobind import (
        swage as native_swage,
    )
    from mlir_swage.dialects import swage

    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    target = f"sm_{major}{minor}"
    kernel_name = f"segmented_{kind}"
    with ir.Context() as context:
        swage.register_dialects(context)
        module = ir.Module.parse(_semantic_module(kind))
        _, ptx = native_swage._compile_segmented_reduction_ptx(
            module,
            kernel_name=kernel_name,
            block_size=block_size,
            target=target,
        )

    driver = _runtime._get_driver()
    _, function = driver.load(ptx, kernel_name)
    stream = torch.cuda.current_stream()
    driver.launch_segmented(
        function,
        (segment_count,),
        block_size,
        stream.cuda_stream,
        (
            values.data_ptr(),
            offsets.data_ptr(),
            output.data_ptr(),
            value_count,
            segment_count,
        ),
    )
    for tensor in (values, offsets, output):
        tensor.record_stream(stream)
    return None


def _launch_segmented_sum_tasks(
    values, offsets, output, task_ids, *, block_size
):
    """Launch one internal identity-sum task list with a warp or CTA block."""
    torch = _runtime._import_torch()
    value_count, segment_count = _validate_tensors(values, offsets, output)
    if not isinstance(task_ids, torch.Tensor):
        raise TypeError("task_ids must be a torch.Tensor")
    if task_ids.dtype != torch.int32:
        raise TypeError("task_ids must have dtype torch.int32")
    if task_ids.dim() != 1:
        raise TypeError("task_ids must have rank one")
    if not task_ids.is_contiguous():
        raise ValueError("task_ids must be contiguous")
    if task_ids.device.type != "cuda":
        raise TypeError("task_ids must be a CUDA tensor")
    if task_ids.device.index != torch.cuda.current_device():
        raise ValueError("task_ids must be on the current CUDA device")
    host_task_ids = task_ids.detach().cpu().tolist()
    task_count = len(host_task_ids)
    _validate_counts(value_count, task_count)
    if any(type(task_id) is not int or not 0 <= task_id < segment_count
           for task_id in host_task_ids):
        raise ValueError("task_ids must contain valid segment IDs")
    if block_size not in {32, 128}:
        raise ValueError("task block size must be 32 or 128")
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    if block_size > properties.max_threads_per_block:
        raise ValueError(
            f"block size {block_size} exceeds device limit "
            f"{properties.max_threads_per_block}"
        )
    if task_count == 0:
        return None

    from mlir_swage import ir
    from mlir_swage._mlir_libs._swageDialectsNanobind import (
        swage as native_swage,
    )
    from mlir_swage.dialects import swage

    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    target = f"sm_{major}{minor}"
    kernel_name = "segmented_sum"
    with ir.Context() as context:
        swage.register_dialects(context)
        module = ir.Module.parse(_semantic_module("sum"))
        _, ptx = native_swage._compile_segmented_reduction_ptx(
            module,
            kernel_name=kernel_name,
            block_size=block_size,
            target=target,
            use_task_ids=True,
        )

    driver = _runtime._get_driver()
    _, function = driver.load(ptx, kernel_name)
    stream = torch.cuda.current_stream()
    driver.launch_segmented_tasks(
        function,
        (task_count,),
        block_size,
        stream.cuda_stream,
        (
            values.data_ptr(),
            offsets.data_ptr(),
            output.data_ptr(),
            task_ids.data_ptr(),
            value_count,
            task_count,
        ),
    )
    for tensor in (values, offsets, output, task_ids):
        tensor.record_stream(stream)
    return None


def launch_softmax_gpu(values, offsets, output, block_size=128):
    """Compile and launch the internally qualified ragged softmax."""
    torch = _runtime._import_torch()
    value_count, segment_count = _validate_softmax_tensors(
        values, offsets, output
    )
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("block size must be a positive integer")
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    if block_size > properties.max_threads_per_block:
        raise ValueError(
            f"block size {block_size} exceeds device limit "
            f"{properties.max_threads_per_block}"
        )
    if segment_count == 0:
        return None

    from mlir_swage import ir
    from mlir_swage._mlir_libs._swageDialectsNanobind import (
        swage as native_swage,
    )
    from mlir_swage.dialects import swage

    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    target = f"sm_{major}{minor}"
    kernel_name = "ragged_softmax"
    with ir.Context() as context:
        swage.register_dialects(context)
        module = ir.Module.parse(_SOFTMAX_MODULE)
        _, ptx = native_swage._compile_segmented_reduction_ptx(
            module,
            kernel_name=kernel_name,
            block_size=block_size,
            target=target,
        )

    driver = _runtime._get_driver()
    _, function = driver.load(ptx, kernel_name)
    stream = torch.cuda.current_stream()
    driver.launch_segmented(
        function,
        (segment_count,),
        block_size,
        stream.cuda_stream,
        (
            values.data_ptr(),
            offsets.data_ptr(),
            output.data_ptr(),
            value_count,
            segment_count,
        ),
    )
    for tensor in (values, offsets, output):
        tensor.record_stream(stream)
    return None


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
        # Prefill so that a store past the live range, or a zero-length
        # print, is visible in the parsed result instead of being garbage.
        (
            f"    %sentinel = arith.constant {_float_literal(_SENTINEL)} : f32"
        ),
        "    %prefill_from = arith.constant 0 : index",
        "    %prefill_step = arith.constant 1 : index",
        f"    %prefill_to = arith.constant {output_length} : index",
        (
            "    scf.for %pi = %prefill_from to %prefill_to "
            "step %prefill_step {"
        ),
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
            f"    %value_count = arith.constant {value_count} : i32",
            f"    %segment_count = arith.constant {segment_count} : i32",
            (
                f"    call @{kernel_name}(%values, %offsets, %output, "
                "%value_count, %segment_count) : (memref<?xf32>, "
                "memref<?xi32>, memref<?xf32>, i32, i32) -> ()"
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


def _llvm_root(root):
    """Find the pinned install used to configure the current build."""
    cache = root / "build" / "CMakeCache.txt"
    match = re.search(r"^MLIR_DIR:[^=]*=(.+)$", cache.read_text(), re.MULTILINE)
    if not match:
        raise RuntimeError("build/CMakeCache.txt does not identify MLIR_DIR")
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
    llvm_root = _llvm_root(root)
    swage_opt = root / "build" / "bin" / "swage-opt"
    mlir_opt = shutil.which("mlir-opt") or llvm_root / "bin" / "mlir-opt"
    mlir_runner = (
        shutil.which("mlir-runner") or llvm_root / "bin" / "mlir-runner"
    )
    lowered = _run(
        [swage_opt, "--swage-segmented-reduction-to-scf"], module_text
    )
    llvm = _run(
        [mlir_opt, f"--pass-pipeline={_LOWERING_PIPELINE}"], lowered
    )
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
    segment_count = offsets.numel() - 1 if offsets.dim() == 1 else 0
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
    # One slot beyond the covered range keeps the sentinel, which both makes
    # a zero-length result printable and turns "map_store never writes past
    # the final offset" into a checked invariant of every oracle call.
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
