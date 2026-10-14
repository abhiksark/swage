# python/tests/mlir/test_bindings.py
"""Integration tests for constructing Swage IR with Python bindings."""

import pathlib
import sys
import threading
import time

import numpy
import pytest
from mlir_swage import ir
from mlir_swage._mlir_libs._swageDialectsNanobind import swage as native_swage
from mlir_swage.dialects import arith, builtin, func, math, swage
from reduction_programs import reduction_module

_FIXED_VECTOR_ADD = """
module {
  func.func @add_kernel(
      %x: memref<?xf32>, %y: memref<?xf32>, %output: memref<?xf32>, %n: i32) {
    %pid = swage.program_id 0
    %block = arith.constant 128 : index
    %base = arith.muli %pid, %block : index
    %lane = vector.step : vector<128xindex>
    %base_vector = vector.broadcast %base : index to vector<128xindex>
    %offsets = arith.addi %base_vector, %lane : vector<128xindex>
    %n_index = arith.index_cast %n : i32 to index
    %n_vector = vector.broadcast %n_index : index to vector<128xindex>
    %mask = arith.cmpi slt, %offsets, %n_vector : vector<128xindex>
    %zero = arith.constant 0.0 : f32
    %passthrough = vector.broadcast %zero : f32 to vector<128xf32>
    %c0 = arith.constant 0 : index
    %lhs = vector.gather %x[%c0] [%offsets], %mask, %passthrough
        : memref<?xf32>, vector<128xindex>, vector<128xi1>, vector<128xf32>
          into vector<128xf32>
    %rhs = vector.gather %y[%c0] [%offsets], %mask, %passthrough
        : memref<?xf32>, vector<128xindex>, vector<128xi1>, vector<128xf32>
          into vector<128xf32>
    %sum = arith.addf %lhs, %rhs : vector<128xf32>
    vector.scatter %output[%c0] [%offsets], %mask, %sum
        : memref<?xf32>, vector<128xindex>, vector<128xi1>, vector<128xf32>
    return
  }
}
"""

# Every native compile entry point: the function, the module text it
# compiles, and its keyword arguments.
_COMPILES = (
    (
        "_compile_ptx",
        _FIXED_VECTOR_ADD,
        {"kernel_name": "add_kernel", "block_size": 128},
    ),
    (
        "_compile_segmented_reduction_ptx",
        reduction_module("sum", "identity"),
        {"kernel_name": "segmented_sum", "block_size": 128},
    ),
    (
        "_compile_segmented_reduction_ptx",
        reduction_module("sum", "identity"),
        {
            "kernel_name": "segmented_sum",
            "block_size": 32,
            "use_task_ids": True,
        },
    ),
    (
        "_compile_fused_segmented_reduction_ptx",
        reduction_module("sum", "identity"),
        {"kernel_name": "segmented_sum"},
    ),
    (
        "_compile_persistent_segmented_reduction_ptx",
        reduction_module("sum", "identity"),
        {"kernel_name": "segmented_sum"},
    ),
    (
        "_compile_split_partial_reduction_ptx",
        reduction_module("sum", "identity"),
        {"kernel_name": "segmented_sum"},
    ),
    (
        "_compile_split_merge_reduction_ptx",
        reduction_module("sum", "identity"),
        {"kernel_name": "segmented_sum"},
    ),
)


def _compile_all(modules=None):
    """Compile every entry point once and return the results in order.

    Args:
        modules: Parsed modules to compile, one per entry of `_COMPILES`.
            When omitted, each text is parsed into a context that only the
            calling thread uses.

    Returns:
        The `(lowered, ptx)` pair of every entry point.
    """
    if modules is not None:
        return [
            getattr(native_swage, name)(module, target="sm_86", **options)
            for (name, _, options), module in zip(
                _COMPILES, modules, strict=True
            )
        ]
    with ir.Context() as context:
        swage.register_dialects(context)
        return [
            getattr(native_swage, name)(
                ir.Module.parse(text), target="sm_86", **options
            )
            for name, text, options in _COMPILES
        ]


def _run_each(works):
    """Run each callable on its own thread, all starting at the same moment.

    Args:
        works: Zero-argument callables, one per thread.

    Returns:
        What each callable returned, or the exception it raised, in order.
    """
    results = [None] * len(works)
    barrier = threading.Barrier(len(works))

    def run(index):
        barrier.wait()
        try:
            results[index] = works[index]()
        except BaseException as error:  # noqa: BLE001
            results[index] = error

    threads = [
        threading.Thread(target=run, args=(index,))
        for index in range(len(works))
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return results


def _run_together(count, work):
    """Run `work` on `count` threads that start at the same moment.

    Args:
        count: Number of threads.
        work: Zero-argument callable each thread runs once.

    Returns:
        What each thread returned, or the exception it raised.
    """
    return _run_each([work] * count)


def test_native_module_reports_the_pinned_llvm_version():
    """Expose the linked LLVM version and keep it equal to the pin."""
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    pin = (repo_root / "cmake" / "llvm-version.txt").read_text().strip()

    assert native_swage.__llvm_version__ == pin.removeprefix("llvmorg-")


def test_builds_and_round_trips_every_swage_operation():
    """Build, verify, print, and reparse all Swage operations."""
    with ir.Context() as context, ir.Location.unknown():
        swage.register_dialects(context)
        f32 = ir.F32Type.get()
        index = ir.IndexType.get()
        segment = swage.SegmentType.get(f32)
        dynamic = ir.ShapedType.get_dynamic_size()
        values = ir.MemRefType.get([dynamic], f32)
        offsets = ir.MemRefType.get([dynamic], ir.IntegerType.get_signless(32))
        module = builtin.ModuleOp()

        with ir.InsertionPoint(module.body):
            kernel = func.FuncOp("kernel", ([values, offsets, values, f32], []))
        with ir.InsertionPoint(kernel.add_entry_block()):
            values_arg, offsets_arg, output, scale = kernel.arguments
            swage.ProgramIdOp(index, 0)
            segment_id = swage.SegmentIdOp(index, 0).result
            input_segment = swage.MakeSegmentOp(
                segment, values_arg, offsets_arg, segment_id
            ).result
            swage.ExtentOp(index, input_segment)

            mapped = swage.MapOp(segment, input_segment, [scale])
            mapped.body.blocks.append(f32, f32)
            with ir.InsertionPoint(mapped.body.blocks[0]):
                scaled = arith.MulFOp(
                    mapped.body.blocks[0].arguments[0],
                    mapped.body.blocks[0].arguments[1],
                )
                swage.YieldOp(math.ExpOp(scaled.result).result)

            reductions = []
            for kind in ("sum", "max", "min"):
                reduction = swage.ReduceOp(
                    f32,
                    mapped.result,
                    [scale],
                    ir.Attribute.parse(f"#swage.reduction_kind<{kind}>"),
                )
                reduction.body.blocks.append(f32, f32)
                with ir.InsertionPoint(reduction.body.blocks[0]):
                    scaled = arith.MulFOp(
                        reduction.body.blocks[0].arguments[0],
                        reduction.body.blocks[0].arguments[1],
                    )
                    swage.YieldOp(scaled.result)
                reductions.append(reduction.result)

            store = swage.MapStoreOp(mapped.result, output, reductions)
            store.body.blocks.append(f32, f32, f32, f32)
            with ir.InsertionPoint(store.body.blocks[0]):
                numerator = math.ExpOp(store.body.blocks[0].arguments[0])
                denominator = arith.AddFOp(
                    store.body.blocks[0].arguments[1],
                    store.body.blocks[0].arguments[2],
                )
                denominator = arith.AddFOp(
                    denominator.result, store.body.blocks[0].arguments[3]
                )
                quotient = arith.DivFOp(numerator.result, denominator.result)
                swage.YieldOp(quotient.result)
            func.ReturnOp([])

        assert module.operation.verify()
        reparsed = ir.Module.parse(str(module))
        assert reparsed.operation.verify()


def test_embeds_memref_and_vector_dialects():
    """Build upstream memref and vector operations from the pinned package."""
    from mlir_swage.dialects import memref, vector

    with ir.Context() as context, ir.Location.unknown():
        swage.register_dialects(context)
        f32 = ir.F32Type.get()
        buffer = ir.MemRefType.get([4], f32)
        tile = ir.VectorType.get([4], f32)
        module = builtin.ModuleOp()

        with ir.InsertionPoint(module.body):
            kernel = func.FuncOp("kernel", ([], []))
        with ir.InsertionPoint(kernel.add_entry_block()):
            memref.AllocOp(buffer, [], [])
            scalar = arith.ConstantOp(f32, 0.0)
            vector.BroadcastOp(tile, scalar.result)
            func.ReturnOp([])

        assert module.operation.verify()


def test_rejects_make_segment_element_type_mismatch():
    """Reject a segment whose element type differs from the values buffer."""
    with ir.Context() as context, ir.Location.unknown():
        swage.register_dialects(context)
        with pytest.raises(
            ir.MLIRError,
            match=(
                "values element type 'f32' does not match segment element type "
                "'i32'"
            ),
        ):
            ir.Module.parse(
                """
                module {
                  func.func @bad(
                      %values: memref<?xf32>, %offsets: memref<?xi32>,
                      %segment_id: index) {
                    %segment = swage.make_segment %values, %offsets, %segment_id
                        : memref<?xf32>, memref<?xi32>, index
                        -> !swage.segment<i32>
                    return
                  }
                }
                """
            )


def test_segment_type_constructor_builds_the_parsed_type():
    """Build `!swage.segment<T>` without going through the parser."""
    with ir.Context() as context:
        swage.register_dialects(context)
        f32 = ir.F32Type.get()
        parsed = ir.Type.parse("!swage.segment<f32>")

        built = swage.SegmentType.get(f32)

        assert built == parsed
        assert str(built) == "!swage.segment<f32>"
        assert built.element_type == f32
        assert isinstance(built, swage.SegmentType)
        # A parsed type arrives as the same class, through its type ID.
        assert isinstance(parsed, swage.SegmentType)
        assert swage.SegmentType.isinstance(parsed)
        assert not swage.SegmentType.isinstance(f32)
        with pytest.raises(ValueError, match="Cannot cast type to SegmentType"):
            swage.SegmentType(f32)


def test_segment_type_constructor_rejects_a_non_scalar_element():
    """Report the dialect's own diagnostic instead of asserting."""
    with ir.Context() as context:
        swage.register_dialects(context)
        buffer = ir.Type.parse("memref<2xf32>")

        with pytest.raises(
            ValueError,
            match="segment element type must be an integer or float type, "
            "got 'memref<2xf32>'",
        ):
            swage.SegmentType.get(buffer)


def test_segment_type_constructor_needs_the_dialect_loaded():
    """Fail with a message where MLIR itself would abort the process."""
    with ir.Context():
        with pytest.raises(
            ValueError,
            match="the swage dialect is not loaded in this context",
        ):
            swage.SegmentType.get(ir.F32Type.get())


def test_two_threads_compile_on_separate_contexts():
    """Compile every kernel kind at once from two threads.

    Each thread parses into its own context, which is what the code
    generation header requires of concurrent calls. The compile releases
    the GIL, so the two pipelines overlap.
    """
    expected = _compile_all()

    def work():
        return [_compile_all() for _ in range(4)]

    for result in _run_together(2, work):
        assert result == [expected] * 4


def test_two_threads_compile_one_module():
    """Serialize compiles that share a context instead of racing them.

    Python callers could always compile one module from two threads, because
    the GIL kept the calls apart. The binding keeps that true after it
    releases the GIL.
    """
    expected = _compile_all()
    with ir.Context() as context:
        swage.register_dialects(context)
        modules = [ir.Module.parse(text) for _, text, _ in _COMPILES]

        def work():
            return [_compile_all(modules) for _ in range(4)]

        results = _run_together(2, work)

    for result in results:
        assert result == [expected] * 4


def test_a_plan_call_keeps_off_a_context_that_is_compiling():
    """Plan a module while another thread compiles it for the first time.

    A compile gives up the GIL and the plan call keeps it, so the plan call
    can start while the compile is still loading dialects and running its
    pipeline in the same context. MLIR refuses that with a failed assertion
    that ends the process, so the plan call takes the guard the compiles
    take and waits. Each round uses a context neither call has touched, and
    the plan call starts only after the compiling thread has signalled that
    its next step is the native call.
    """
    name, text, options = _COMPILES[4]
    assert name == "_compile_persistent_segmented_reduction_ptx"
    compile_ptx = getattr(native_swage, name)
    offsets = numpy.asarray([0, 32, 132, 8325], dtype=numpy.int32)
    with ir.Context() as context:
        swage.register_dialects(context)
        expected_ptx = compile_ptx(
            ir.Module.parse(text), target="sm_86", **options
        )
    expected_plan = ([0], [1], [132, 4228, 4228, 8324, 8324, 8325], [2, 0, 3])

    for _ in range(10):
        with ir.Context() as context:
            swage.register_dialects(context)
            module = ir.Module.parse(text)
            compiling = threading.Event()
            compiled = []

            def compile_once(
                module=module, compiling=compiling, compiled=compiled
            ):
                compiling.set()
                compiled.append(compile_ptx(module, target="sm_86", **options))

            thread = threading.Thread(target=compile_once)
            thread.start()
            compiling.wait()
            records = native_swage._materialize_segmented_plan(
                module,
                options["kernel_name"],
                offsets=offsets,
                value_count=8325,
                segment_count=3,
            )
            thread.join()

        assert tuple(array.tolist() for array in records) == expected_plan
        assert compiled == [expected_ptx]


def test_a_failed_compile_on_another_thread_reports_its_diagnostic():
    """Keep the diagnostic text when the compile ran without the GIL."""

    def work():
        with ir.Context() as context:
            swage.register_dialects(context)
            with pytest.raises(ValueError) as caught:
                native_swage._compile_ptx(
                    ir.Module.parse(_FIXED_VECTOR_ADD),
                    kernel_name="add_kernel",
                    block_size=64,
                    target="sm_86",
                )
            return str(caught.value)

    for message in _run_together(2, work):
        assert "vector width 128 does not match requested block size 64" in (
            message
        )


def test_a_compile_does_not_stall_other_python_threads():
    """Run Python on one thread while another compiles.

    The main thread spins and adds up every pause longer than a millisecond.
    A compile that held the GIL would pause it for the whole compile, about
    18 ms for the persistent kernel, so nearly all of the run would count as
    paused. The bound is half of the run, far from both outcomes.
    """
    name, text, options = _COMPILES[4]
    assert name == "_compile_persistent_segmented_reduction_ptx"
    finished = threading.Event()

    def compile_repeatedly():
        try:
            with ir.Context() as context:
                swage.register_dialects(context)
                module = ir.Module.parse(text)
                for _ in range(20):
                    getattr(native_swage, name)(
                        module, target="sm_86", **options
                    )
        finally:
            finished.set()

    interval = sys.getswitchinterval()
    # A short switch interval hands the GIL back to the compiling thread as
    # soon as it asks, so a compile that holds the GIL dominates the run.
    sys.setswitchinterval(1e-4)
    try:
        thread = threading.Thread(target=compile_repeatedly)
        paused = 0.0
        start = previous = time.perf_counter()
        thread.start()
        while not finished.is_set():
            now = time.perf_counter()
            if now - previous > 1e-3:
                paused += now - previous
            previous = now
        elapsed = time.perf_counter() - start
        thread.join()
    finally:
        sys.setswitchinterval(interval)

    assert paused < 0.5 * elapsed, (paused, elapsed)


def test_target_description_holds_the_values_the_lowerings_read():
    """Expose the compiler's target description to the host as plain values."""
    assert native_swage._target_description() == {
        "name": "nvidia",
        "triple": "nvptx64-nvidia-cuda",
        "processor_prefix": "sm_",
        "processors": (80, 86, 87, 88, 89, 90, 100, 101, 103, 110, 120, 121),
        "subgroup_width": 32,
        "max_block_threads": 1024,
        "cta_block_threads": 128,
        "split_block_threads": 512,
        "persistent_block_threads": 512,
        "persistent_partial_claim": 4,
        "persistent_warp_claim": 8,
        "default_warp_max_elements": 32,
        "default_cta_chunk_elements": 4096,
    }


def test_the_runner_takes_its_block_widths_from_the_target_description(
    monkeypatch,
):
    """Read block widths and planning defaults from the compiler, lazily."""
    from swage import _segmented_plan as _plan
    from swage import _segmented_runtime as _execution
    from swage import _segmented_validation as _validation

    description = _execution._target_description()
    assert vars(description) == native_swage._target_description()
    assert _execution._target_description() is description
    assert _plan._planning_limits(None, None) == (32, 4096)
    assert _plan._planning_limits(8, None) == (8, 4096)
    assert _plan._planning_limits(None, 64) == (32, 64)

    # The runner holds no copy of its own: another description changes what
    # an omitted limit and the warp rule resolve to.
    narrow = dict(native_swage._target_description())
    narrow.update(subgroup_width=16, default_cta_chunk_elements=2048)
    monkeypatch.setattr(
        _execution, "_target_record", type(description)(**narrow)
    )
    assert _plan._planning_limits(None, None) == (32, 2048)
    _validation._validate_warp_count(16)
    with pytest.raises(ValueError, match="power-of-two warp count, got 48"):
        _validation._validate_warp_count(48)
