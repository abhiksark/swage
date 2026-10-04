# python/tests/mlir/test_segmented_codegen.py
"""Native tests for segmented-reduction NVPTX compilation."""

import re

import numpy
import pytest
from mlir_swage import ir
from mlir_swage._mlir_libs._swageDialectsNanobind import swage as native_swage
from mlir_swage.dialects import swage
from reduction_programs import reduction_module
from swage._segmented_qualification import (
    _has_small_element_program,
    _reduction_kernel,
    _semantic_module,
    _softmax_text,
)


def _plan(module, kernel_name, offsets, **arguments):
    """Classify host offsets through the native plan and return lists.

    Args:
        module: Parsed semantic module to plan.
        kernel_name: Name of the segment function in the module to admit.
        offsets: Offsets as Python integers. They cross into native code as
            one host int32 buffer, the only form the binding accepts.
        **arguments: The value count, the segment count, and any limit.

    Returns:
        The warp ids, CTA ids, partial records, and merge records, each as
        a list.
    """
    records = native_swage._materialize_segmented_plan(
        module,
        kernel_name,
        offsets=numpy.asarray(offsets, dtype=numpy.int32),
        **arguments,
    )
    return tuple(array.tolist() for array in records)


@pytest.mark.parametrize(
    ("transform", "eligible"),
    [
        ("identity", True), ("square", True), ("maps", True),
        ("affine4", True), ("affine32", True), ("exp2", True),
        ("exp2_chain", False), ("rational8", False),
    ],
)
def test_direct_cta_element_work_guard(transform, eligible):
    """Inspect typed native element operations instead of matching text."""
    with ir.Context() as context:
        swage.register_dialects(context)
        module = ir.Module.parse(reduction_module("sum", transform))
        assert _has_small_element_program(module) is eligible


def test_the_division_of_a_mean_is_no_element_work():
    """A mean has the element work of its sum and stays a small program.

    The division runs once per segment, outside every element region, so
    the schedule selection treats a mean as it treats the sum.
    """
    for transform, work in (("identity", 0), ("square", 1)):
        with ir.Context() as context:
            swage.register_dialects(context)
            module = ir.Module.parse(reduction_module("mean", transform))
            assert native_swage._element_work(module) == work
            assert _has_small_element_program(module) is True


@pytest.mark.parametrize(
    ("transform", "work"),
    [
        ("identity", 0), ("square", 1), ("affine4", 4), ("affine32", 32),
        ("exp2", 8), ("exp2_chain", 72), ("rational8", 160),
    ],
)
def test_native_element_work_estimate(transform, work):
    """Count one unit per cheap operation, eight per exp2, 16 per divide."""
    with ir.Context() as context:
        swage.register_dialects(context)
        module = ir.Module.parse(reduction_module("sum", transform))
        assert native_swage._element_work(module) == work


def test_native_element_work_has_no_estimate_for_an_unweighted_operation():
    """Give no estimate, and no small program, for an unknown operation."""
    text = reduction_module("sum", "identity").replace(
        "swage.yield %value : f32",
        "%root = math.sqrt %value : f32\n      swage.yield %root : f32",
    )
    with ir.Context() as context:
        swage.register_dialects(context)
        module = ir.Module.parse(text)
        assert native_swage._element_work(module) is None
        assert not _has_small_element_program(module)


def test_direct_cta_work_budget_spans_maps_and_reduction():
    """Individually small regions must share the per-element work budget."""
    operations = []
    value = "%value"
    for index in range(31):
        operations.append(f"%v{index} = arith.mulf {value}, %value : f32")
        value = f"%v{index}"
    operations.append(f"swage.yield {value} : f32")
    text = reduction_module("sum", "maps").replace(
        "swage.yield %value : f32", "\n      ".join(operations)
    )
    with ir.Context() as context:
        swage.register_dialects(context)
        assert not _has_small_element_program(ir.Module.parse(text))

SEGMENTED_SUM = """
module {
  func.func @segmented_sum(
      %values: memref<?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %sum = swage.reduce %segment kind<sum>
        : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    memref.store %sum, %output[%sid] : memref<?xf32>
    return
  }
}
"""


@pytest.mark.parametrize("element", ["f32", "f64"])
@pytest.mark.parametrize("kind", ["sum", "max", "min", "mean"])
@pytest.mark.parametrize("transform", ["identity", "square", "maps"])
def test_static_schedules_share_reduction_program(kind, transform, element):
    """One admitted program compiles unchanged through every static path.

    The plan of a program does not depend on its element type or on its
    kind: an f64 program and a mean have the task records of the f32 sum.
    A mean divides in every kernel but the partial one, whose chunks yield
    raw sums, and no other kind divides.
    """
    kernel_name = _reduction_kernel(kind, element)
    with ir.Context() as context:
        swage.register_dialects(context)
        module = ir.Module.parse(reduction_module(kind, transform, element))
        original = module.operation.get_asm(enable_debug_info=False)
        assert _plan(
            module,
            kernel_name,
            [0, 1, 34, 4131],
            value_count=4131,
            segment_count=3,
        ) == ([0], [1], [34, 4130, 4130, 4131], [2, 0, 2])
        options = [
            (
                "_compile_segmented_reduction_ptx",
                {"block_size": 32, "use_task_ids": True},
            ),
            (
                "_compile_segmented_reduction_ptx",
                {"block_size": 128, "use_task_ids": True},
            ),
            ("_compile_fused_segmented_reduction_ptx", {}),
            ("_compile_split_partial_reduction_ptx", {}),
            ("_compile_split_merge_reduction_ptx", {}),
        ]
        for name, arguments in options:
            compiler = getattr(native_swage, name)
            first = compiler(
                module,
                kernel_name=kernel_name,
                target="sm_86",
                **arguments,
            )
            assert first == compiler(
                module,
                kernel_name=kernel_name,
                target="sm_86",
                **arguments,
            )
            lowered, ptx, _ = first
            merge = name == "_compile_split_merge_reduction_ptx"
            partial = name == "_compile_split_partial_reduction_ptx"
            assert ("llvm.fmul" in lowered) == (
                transform != "identity" and not merge
            )
            assert ("llvm.fdiv" in lowered) == (kind == "mean" and not partial)
            assert "swage." not in lowered
            assert ".entry segmented_" in ptx
            other = "f32" if element == "f64" else "f64"
            assert not re.search(rf"\b{other}\b", lowered)
            assert module.operation.get_asm(enable_debug_info=False) == original


_RANK_ONE_SCHEDULES = [
    (
        "_compile_segmented_reduction_ptx",
        {"block_size": 32, "use_task_ids": True},
    ),
    (
        "_compile_segmented_reduction_ptx",
        {"block_size": 128, "use_task_ids": True},
    ),
    ("_compile_fused_segmented_reduction_ptx", {}),
    ("_compile_split_partial_reduction_ptx", {}),
    ("_compile_split_merge_reduction_ptx", {}),
    ("_compile_persistent_segmented_reduction_ptx", {}),
]


@pytest.mark.parametrize("element", ["f32", "f64"])
@pytest.mark.parametrize("kind", ["sum", "max", "min", "mean"])
def test_rank_two_programs_have_one_kernel_and_no_task_buffer(kind, element):
    """A program over rank-two values compiles on the direct schedule only.

    Its kernel takes the three buffers and three counts, the last of which
    is the feature count, holds nothing that synchronizes threads, and has
    one loop accumulator per reduction: a thread never holds one value per
    column. Every schedule that reads a task buffer refuses the program
    with a diagnostic, and so does host classification.
    """
    kernel_name = _reduction_kernel(kind, element, 2)
    with ir.Context() as context:
        swage.register_dialects(context)
        module = ir.Module.parse(_semantic_module(kind, element, 2))
        original = module.operation.get_asm(enable_debug_info=False)

        lowered, ptx, _ = native_swage._compile_segmented_reduction_ptx(
            module, kernel_name=kernel_name, block_size=128, target="sm_86"
        )

        assert f".entry {kernel_name}(" in ptx
        declaration = re.search(
            rf"\.entry {kernel_name}\((.*?)\)", ptx, re.DOTALL
        )
        parameters = [
            "pointer" if ".ptr" in parameter else parameter.split()[1]
            for parameter in declaration[1].split(",")
        ]
        assert parameters == ["pointer"] * 3 + [".u32"] * 3
        assert "shfl.sync" not in ptx and "bar.sync" not in ptx
        assert ".shared" not in ptx
        # One accumulator: of the blocks that take arguments, the header of
        # the row loop takes one element and no other block takes any.
        headers = re.findall(r"^\s*\^bb\d+\(([^)]*)\):", lowered, re.M)
        assert sorted(header.count(f": {element}") for header in headers) == [
            0,
            1,
        ]
        for name, arguments in _RANK_ONE_SCHEDULES:
            with pytest.raises(
                ValueError, match="planning requires rank-one values"
            ):
                getattr(native_swage, name)(
                    module, kernel_name=kernel_name, target="sm_86", **arguments
                )
        with pytest.raises(
            ValueError, match="planning requires rank-one values"
        ):
            _plan(
                module, kernel_name, [0, 1, 34], value_count=34, segment_count=2
            )
        assert module.operation.get_asm(enable_debug_info=False) == original


def test_the_rank_two_softmax_has_one_kernel_and_one_scalar_per_stage():
    """The softmax over rank-two values compiles on the direct schedule only.

    A thread runs the three stages of its column one after the other, so
    the kernel needs none of the workgroup buffers, barriers, and shuffles
    of the rank-one softmax. The two reduction loops carry one accumulator
    each, and the store loop and the column loop carry none: a thread
    never holds one value per column or per row.
    """
    with ir.Context() as context:
        swage.register_dialects(context)
        module = ir.Module.parse(_softmax_text(2))
        original = module.operation.get_asm(enable_debug_info=False)

        lowered, ptx, _ = native_swage._compile_segmented_reduction_ptx(
            module,
            kernel_name="ragged_softmax_r2",
            block_size=128,
            target="sm_86",
        )

        declaration = re.search(
            r"\.entry ragged_softmax_r2\((.*?)\)", ptx, re.DOTALL
        )
        parameters = [
            "pointer" if ".ptr" in parameter else parameter.split()[1]
            for parameter in declaration[1].split(",")
        ]
        assert parameters == ["pointer"] * 3 + [".u32"] * 3
        assert "shfl.sync" not in ptx and "bar.sync" not in ptx
        assert ".shared" not in ptx and "__wg_" not in lowered
        assert ".extern .func" not in ptx
        assert ptx.count("ex2.approx.f32") == 2
        headers = re.findall(r"^\s*\^bb\d+\(([^)]*)\):", lowered, re.M)
        assert sorted(header.count(": f32") for header in headers) == [
            0,
            0,
            1,
            1,
        ]
        for name, arguments in _RANK_ONE_SCHEDULES:
            with pytest.raises(
                ValueError, match="planning requires rank-one values"
            ):
                getattr(native_swage, name)(
                    module,
                    kernel_name="ragged_softmax_r2",
                    target="sm_86",
                    **arguments,
                )
        with pytest.raises(
            ValueError, match="planning requires rank-one values"
        ):
            _plan(
                module,
                "ragged_softmax_r2",
                [0, 1, 34],
                value_count=34,
                segment_count=2,
            )
        assert module.operation.get_asm(enable_debug_info=False) == original


def test_persistent_execution_refuses_f64_values():
    """The persistent kernel is an f32 kernel, and f64 values are refused.

    The identity sum is the one program persistent execution admits, so
    the element type is the only reason this program is refused.
    """
    with ir.Context() as context:
        swage.register_dialects(context)
        module = ir.Module.parse(reduction_module("sum", "identity", "f64"))
        original = module.operation.get_asm(enable_debug_info=False)
        with pytest.raises(
            ValueError,
            match="persistent execution requires f32 values, got f64",
        ):
            native_swage._compile_persistent_segmented_reduction_ptx(
                module, kernel_name="segmented_sum_f64", target="sm_86"
            )
        assert module.operation.get_asm(enable_debug_info=False) == original


@pytest.mark.parametrize(
    ("kind", "transform"),
    [
        ("sum", "square"),
        ("sum", "maps"),
        ("max", "identity"),
        ("max", "square"),
        ("max", "maps"),
        ("min", "identity"),
        ("mean", "identity"),
    ],
)
def test_persistent_admission_remains_identity_sum(kind, transform):
    """Broadening static planning must not widen persistent execution."""
    with ir.Context() as context:
        swage.register_dialects(context)
        module = ir.Module.parse(reduction_module(kind, transform))
        original = module.operation.get_asm(enable_debug_info=False)
        with pytest.raises(ValueError, match="persistent execution"):
            native_swage._compile_persistent_segmented_reduction_ptx(
                module, kernel_name=f"segmented_{kind}", target="sm_86"
            )
        assert module.operation.get_asm(enable_debug_info=False) == original


@pytest.mark.parametrize(
    "compiler",
    [
        "_compile_segmented_reduction_ptx",
        "_compile_fused_segmented_reduction_ptx",
        "_compile_split_partial_reduction_ptx",
        "_compile_split_merge_reduction_ptx",
    ],
)
def test_static_schedules_reject_stage_dependencies(compiler):
    """Keep captured softmax stages out of every generalized static path."""
    with ir.Context() as context:
        swage.register_dialects(context)
        module = ir.Module.parse(RAGGED_SOFTMAX)
        original = module.operation.get_asm(enable_debug_info=False)
        arguments = (
            {"block_size": 32, "use_task_ids": True}
            if compiler == "_compile_segmented_reduction_ptx"
            else {}
        )
        with pytest.raises(ValueError, match="capture-free maps"):
            getattr(native_swage, compiler)(
                module,
                kernel_name="ragged_softmax",
                target="sm_86",
                **arguments,
            )
        assert module.operation.get_asm(enable_debug_info=False) == original


def test_compiles_segmented_sum_to_deterministic_ptx():
    """Preserve the source module while emitting one CTA reduction."""
    with ir.Context() as context:
        swage.register_dialects(context)
        module = ir.Module.parse(SEGMENTED_SUM)
        original = module.operation.get_asm(enable_debug_info=False)

        first = native_swage._compile_segmented_reduction_ptx(
            module,
            kernel_name="segmented_sum",
            block_size=128,
            target="sm_80",
        )
        second = native_swage._compile_segmented_reduction_ptx(
            module,
            kernel_name="segmented_sum",
            block_size=128,
            target="sm_80",
        )

        assert first == second
        lowered, ptx, _ = first
        assert "swage." not in lowered
        assert "gpu.all_reduce" not in lowered
        assert "llvm.func @segmented_sum" in lowered
        assert '#nvvm.target<chip = "sm_80">' in lowered
        assert ".target sm_80" in ptx
        assert ".entry segmented_sum" in ptx
        assert "ld.global.b32" in ptx
        assert "st.global.b32" in ptx
        assert module.operation.get_asm(enable_debug_info=False) == original


def test_compiles_identity_sum_with_task_id_indirection():
    """Load original segment IDs through the private task-ID ABI."""
    with ir.Context() as context:
        swage.register_dialects(context)
        module = ir.Module.parse(SEGMENTED_SUM)
        original = module.operation.get_asm(enable_debug_info=False)

        lowered, ptx, _ = native_swage._compile_segmented_reduction_ptx(
            module,
            kernel_name="segmented_sum",
            block_size=32,
            target="sm_80",
            use_task_ids=True,
        )

        signature = re.search(r"llvm.func @segmented_sum\(([^)]*)\)", lowered)
        assert signature is not None
        assert signature.group(1).count("!llvm.ptr") == 4
        assert signature.group(1).count("i32") == 3
        assert "ld.global.b32" in ptx
        assert ".entry segmented_sum" in ptx
        assert "shfl.sync.bfly" in ptx
        assert ".shared" not in ptx
        assert "bar.sync" not in ptx
        assert module.operation.get_asm(enable_debug_info=False) == original


def test_compiles_fused_mixed_identity_sum_to_deterministic_ptx():
    """Map four warp tasks then CTA tasks through one private kernel ABI."""
    with ir.Context() as context:
        swage.register_dialects(context)
        module = ir.Module.parse(SEGMENTED_SUM)
        original = module.operation.get_asm(enable_debug_info=False)

        first = native_swage._compile_fused_segmented_reduction_ptx(
            module,
            kernel_name="segmented_sum",
            target="sm_80",
        )
        second = native_swage._compile_fused_segmented_reduction_ptx(
            module,
            kernel_name="segmented_sum",
            target="sm_80",
        )

        assert first == second
        lowered, ptx, _ = first
        signature = re.search(r"llvm.func @segmented_sum\(([^)]*)\)", lowered)
        assert signature is not None
        assert signature.group(1).count("!llvm.ptr") == 4
        assert signature.group(1).count("i32") == 4
        assert "llvm.udiv" in lowered
        assert "llvm.urem" in lowered
        assert "llvm.mlir.constant(128 : index)" in lowered
        assert "llvm.mlir.constant(32 : index)" in lowered
        assert lowered.count("nvvm.shfl.sync  bfly") >= 15
        assert "nvvm.barrier0" in lowered
        assert "addr_space = 3" in lowered
        assert ptx.count(".param .u64") == 4
        assert ptx.count(".param .u32") == 4
        assert "shfl.sync.bfly" in ptx
        assert "bar.sync" in ptx
        assert ".entry segmented_sum" in ptx
        assert module.operation.get_asm(enable_debug_info=False) == original

        definitions = {
            name: expression
            for line in lowered.splitlines()
            if (match := re.match(r"\s*(%\d+) = (.*)", line))
            for name, expression in [match.groups()]
        }

        def result_of(expression):
            matches = [
                name
                for name, actual in definitions.items()
                if actual == expression
            ]
            assert len(matches) == 1, (expression, matches)
            return matches[0]

        block_id_i32 = result_of("nvvm.read.ptx.sreg.ctaid.x : i32")
        block_id = result_of(f"llvm.sext {block_id_i32} : i32 to i64")
        thread_id_i32 = next(
            name
            for name, expression in definitions.items()
            if expression == "nvvm.read.ptx.sreg.tid.x : i32"
        )
        thread_id = result_of(f"llvm.sext {thread_id_i32} : i32 to i64")
        warp_count = result_of("llvm.sext %arg5 : i32 to i64")
        cta_count = result_of("llvm.sext %arg6 : i32 to i64")
        three = result_of("llvm.mlir.constant(3 : index) : i64")
        four = result_of("llvm.mlir.constant(4 : index) : i64")
        warp_width = next(
            name
            for name, expression in reversed(definitions.items())
            if expression == "llvm.mlir.constant(32 : index) : i64"
        )
        rounded_warp_count = result_of(f"llvm.add {warp_count}, {three} : i64")
        warp_blocks = result_of(
            f"llvm.udiv {rounded_warp_count}, {four} : i64"
        )
        result_of(f'llvm.icmp "ult" {block_id}, {warp_blocks} : i64')

        physical_warp = result_of(
            f"llvm.udiv {thread_id}, {warp_width} : i64"
        )
        lane = result_of(f"llvm.urem {thread_id}, {warp_width} : i64")
        first_warp_task = result_of(f"llvm.mul {block_id}, {four} : i64")
        warp_task = result_of(
            f"llvm.add {first_warp_task}, {physical_warp} : i64"
        )
        warp_guard = result_of(
            f'llvm.icmp "ult" {warp_task}, {warp_count} : i64'
        )
        assert f"llvm.cond_br {warp_guard}" in lowered
        assert f"llvm.getelementptr %arg3[{warp_task}]" in lowered
        assert any(
            re.fullmatch(rf"llvm.add %\d+, {lane} : i64", expression)
            for expression in definitions.values()
        )
        assert any(
            re.fullmatch(
                rf"llvm.add %\d+, {warp_width} : i64", expression
            )
            for expression in definitions.values()
        )

        cta_task = result_of(f"llvm.sub {block_id}, {warp_blocks} : i64")
        cta_guard = result_of(f'llvm.icmp "ult" {cta_task}, {cta_count} : i64')
        mixed_task = result_of(f"llvm.add {warp_count}, {cta_task} : i64")
        assert f"llvm.cond_br {cta_guard}" in lowered
        assert f"llvm.getelementptr %arg3[{mixed_task}]" in lowered
        cta_width = result_of("llvm.mlir.constant(128 : index) : i64")
        assert any(
            re.fullmatch(rf"llvm.add %\d+, {cta_width} : i64", expression)
            for expression in definitions.values()
        )


def test_compiles_persistent_identity_sum_to_deterministic_ptx():
    """Emit resident workers with direct, partial, and dependency claims."""
    with ir.Context() as context:
        swage.register_dialects(context)
        module = ir.Module.parse(SEGMENTED_SUM)
        original = module.operation.get_asm(enable_debug_info=False)

        first = native_swage._compile_persistent_segmented_reduction_ptx(
            module,
            kernel_name="segmented_sum",
            target="sm_80",
        )
        second = native_swage._compile_persistent_segmented_reduction_ptx(
            module,
            kernel_name="segmented_sum",
            target="sm_80",
        )

        assert first == second
        lowered, ptx, _ = first
        signature = re.search(r"llvm.func @segmented_sum\(([^)]*)\)", lowered)
        assert signature is not None
        assert signature.group(1).count("!llvm.ptr") == 10
        assert signature.group(1).count("i32") == 6
        assert lowered.count("llvm.atomicrmw add") == 4
        assert lowered.count("nvvm.memory.barrier <gpu>") == 2
        assert "nvvm.shfl.sync  idx" in lowered
        assert "nvvm.shfl.sync  bfly" in lowered
        # Includes the CTA-to-partial phase barrier that protects reuse of
        # the shared queue-claim broadcast slot.
        assert lowered.count("nvvm.barrier0") == 12
        assert ptx.count(".param .u64") == 10
        assert ptx.count(".param .u32") == 6
        assert ptx.count("atom") >= 4
        assert ptx.count("membar.gl") == 2
        assert "shfl.sync.idx" in ptx
        assert "shfl.sync.bfly" in ptx
        assert ptx.count("bar.sync") == 12
        assert ".entry segmented_sum" in ptx
        assert module.operation.get_asm(enable_debug_info=False) == original


def test_persistent_lowering_rejects_non_planning_program():
    """Fail before mutation when persistent execution cannot preserve it."""
    with ir.Context() as context:
        swage.register_dialects(context)
        module = ir.Module.parse(SEGMENTED_EXPONENTIAL_SUM)
        original = module.operation.get_asm(enable_debug_info=False)

        with pytest.raises(ValueError, match="identity reduction region"):
            native_swage._compile_persistent_segmented_reduction_ptx(
                module,
                kernel_name="segmented_sum",
                target="sm_80",
            )

        assert module.operation.get_asm(enable_debug_info=False) == original


def test_fused_mixed_lowering_accepts_element_program():
    """Compile the element expression in both branches of the fused kernel."""
    with ir.Context() as context:
        swage.register_dialects(context)
        module = ir.Module.parse(SEGMENTED_EXPONENTIAL_SUM)
        original = module.operation.get_asm(enable_debug_info=False)
        lowered, ptx, _ = native_swage._compile_fused_segmented_reduction_ptx(
            module, kernel_name="segmented_sum", target="sm_80"
        )
        assert lowered.count("llvm.intr.exp2") == 2
        assert "ex2.approx.f32" in ptx
        assert module.operation.get_asm(enable_debug_info=False) == original


@pytest.mark.parametrize(
    ("compiler", "entry", "counts"),
    [
        ("_compile_split_partial_reduction_ptx", "segmented_sum__partial", 2),
        ("_compile_split_merge_reduction_ptx", "segmented_sum__merge", 3),
    ],
)
def test_compiles_deterministic_split_cta_kernels(compiler, entry, counts):
    """Emit each private three-pointer split ABI at 512 threads.

    The partial ABI ends with two i32 counts. The merge ABI adds the segment
    count that bounds the output segment its records name.
    """
    with ir.Context() as context:
        swage.register_dialects(context)
        module = ir.Module.parse(SEGMENTED_SUM)
        original = module.operation.get_asm(enable_debug_info=False)
        compile_split = getattr(native_swage, compiler)

        first = compile_split(
            module,
            kernel_name="segmented_sum",
            target="sm_80",
        )
        second = compile_split(
            module,
            kernel_name="segmented_sum",
            target="sm_80",
        )

        assert first == second
        lowered, ptx, _ = first
        signature = re.search(rf"llvm.func @{entry}\(([^)]*)\)", lowered)
        assert signature is not None
        assert signature.group(1).count("!llvm.ptr") == 3
        assert signature.group(1).count("i32") == counts
        assert "llvm.mlir.constant(512 : index)" in lowered
        assert "nvvm.barrier0" in lowered
        assert ptx.count(".param .u64") == 3
        assert ptx.count(".param .u32") == counts
        assert f".entry {entry}" in ptx
        assert "bar.sync" in ptx
        assert module.operation.get_asm(enable_debug_info=False) == original

        definitions = {
            name: expression
            for line in lowered.splitlines()
            if (match := re.match(r"\s*(%\d+) = (.*)", line))
            for name, expression in [match.groups()]
        }
        block_id_i32 = next(
            name
            for name, expression in definitions.items()
            if expression == "nvvm.read.ptx.sreg.ctaid.x : i32"
        )
        block_id = next(
            name
            for name, expression in definitions.items()
            if expression == f"llvm.sext {block_id_i32} : i32 to i64"
        )
        if compiler == "_compile_split_partial_reduction_ptx":
            assert f"llvm.getelementptr %arg2[{block_id}]" in lowered
        else:
            segment_pointer = next(
                name
                for name, expression in definitions.items()
                if expression.startswith("llvm.getelementptr %arg2[")
            )
            segment_id = next(
                name
                for name, expression in definitions.items()
                if expression
                == f"llvm.load {segment_pointer} : !llvm.ptr -> i32"
            )
            segment_index = next(
                name
                for name, expression in definitions.items()
                if expression == f"llvm.sext {segment_id} : i32 to i64"
            )
            assert f"llvm.getelementptr %arg1[{segment_index}]" in lowered


@pytest.mark.parametrize(
    "compiler",
    [
        "_compile_split_partial_reduction_ptx",
        "_compile_split_merge_reduction_ptx",
    ],
)
def test_split_lowering_transforms_only_input_elements(compiler):
    """Partial tasks evaluate the region; merges only combine scratch."""
    with ir.Context() as context:
        swage.register_dialects(context)
        module = ir.Module.parse(SEGMENTED_EXPONENTIAL_SUM)
        original = module.operation.get_asm(enable_debug_info=False)
        lowered, ptx, _ = getattr(native_swage, compiler)(
            module, kernel_name="segmented_sum", target="sm_80"
        )
        partial = compiler == "_compile_split_partial_reduction_ptx"
        assert ("llvm.intr.exp2" in lowered) == partial
        assert ("ex2.approx.f32" in ptx) == partial
        assert module.operation.get_asm(enable_debug_info=False) == original


@pytest.mark.parametrize(
    "compiler",
    [
        "_compile_split_partial_reduction_ptx",
        "_compile_split_merge_reduction_ptx",
    ],
)
def test_split_lowering_rejects_an_unsupported_target(compiler):
    """Keep unsupported targets outside split lowering."""
    with ir.Context() as context:
        swage.register_dialects(context)
        total = ir.Module.parse(SEGMENTED_SUM)
        with pytest.raises(ValueError, match="target must match"):
            getattr(native_swage, compiler)(
                total,
                kernel_name="segmented_sum",
                target="sm_79",
            )


def test_materializes_stable_policy_segment_ids_without_mutating_source():
    """Connect the private plan operation to the existing host classifier."""
    with ir.Context() as context:
        swage.register_dialects(context)
        module = ir.Module.parse(SEGMENTED_SUM)
        original = module.operation.get_asm(enable_debug_info=False)

        warp, cta, partial, merge = _plan(
            module,
            "segmented_sum",
            [0, 0, 32, 65, 65, 66],
            value_count=66,
            segment_count=5,
            warp_max_elements=32,
        )

        assert warp == [0, 1, 3, 4]
        assert cta == [2]
        assert partial == []
        assert merge == []
        assert module.operation.get_asm(enable_debug_info=False) == original


def test_materializes_split_ranges_and_compact_merge_records():
    """Return direct IDs and ordered flat records from one private plan."""
    with ir.Context() as context:
        swage.register_dialects(context)
        module = ir.Module.parse(SEGMENTED_SUM)
        original = module.operation.get_asm(enable_debug_info=False)

        warp, cta, partial, merge = _plan(
            module,
            "segmented_sum",
            [0, 32, 65, 4162, 12354],
            value_count=12354,
            segment_count=4,
            warp_max_elements=32,
            cta_chunk_elements=4096,
        )

        assert warp == [0]
        assert cta == [1]
        assert partial == [65, 4161, 4161, 4162, 4162, 8258, 8258, 12354]
        assert merge == [2, 0, 2, 3, 2, 4]
        assert module.operation.get_asm(enable_debug_info=False) == original


def test_materialized_plan_rejects_invalid_metadata_and_semantics():
    """Fail closed through the native planning and classification chain."""
    with ir.Context() as context:
        swage.register_dialects(context)
        module = ir.Module.parse(SEGMENTED_SUM)
        with pytest.raises(ValueError, match="offsets must be nondecreasing"):
            _plan(
                module,
                "segmented_sum",
                [0, 2, 1],
                value_count=2,
                segment_count=2,
                warp_max_elements=32,
            )

        with pytest.raises(ValueError, match="planning limits must satisfy"):
            _plan(
                module,
                "segmented_sum",
                [0, 1],
                value_count=1,
                segment_count=1,
                warp_max_elements=33,
                cta_chunk_elements=32,
            )

        transformed = ir.Module.parse(RAGGED_SOFTMAX)
        original = transformed.operation.get_asm(enable_debug_info=False)
        with pytest.raises(ValueError, match="capture-free maps"):
            _plan(
                transformed,
                "ragged_softmax",
                [0, 1],
                value_count=1,
                segment_count=1,
                warp_max_elements=32,
            )
        assert (
            transformed.operation.get_asm(enable_debug_info=False) == original
        )


SEGMENTED_EXPONENTIAL_SUM = """
module {
  func.func @segmented_sum(
      %values: memref<?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %sum = swage.reduce %segment kind<sum>
        : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      %log2e = arith.constant 1.44269504 : f32
      %scaled = arith.mulf %value, %log2e : f32
      %exponential = math.exp2 %scaled : f32
      swage.yield %exponential : f32
    }
    memref.store %sum, %output[%sid] : memref<?xf32>
    return
  }
}
"""


_EVERY_COMPILER = [
    ("_compile_segmented_reduction_ptx", {"block_size": 128}),
    (
        "_compile_segmented_reduction_ptx",
        {"block_size": 32, "use_task_ids": True},
    ),
    (
        "_compile_segmented_reduction_ptx",
        {"block_size": 128, "use_task_ids": True},
    ),
    ("_compile_fused_segmented_reduction_ptx", {}),
    ("_compile_split_partial_reduction_ptx", {}),
    ("_compile_split_merge_reduction_ptx", {}),
    ("_compile_persistent_segmented_reduction_ptx", {}),
]


@pytest.mark.parametrize(("compiler", "arguments"), _EVERY_COMPILER)
def test_an_f64_exponential_is_refused_before_the_backend(
    compiler, arguments
):
    """An f64 `math.exp2` raises a diagnostic from every compile function.

    The device has an f32 `ex2.approx` and no f64 one. The pinned NVPTX
    backend aborts the process on an f64 exp2 intrinsic, and without the
    planner's rule the libdevice guard would refuse the program late, as an
    unresolved `__nv_exp2`. The planner refuses the operation by name
    first, before any lowering runs. This test runs in the test process:
    an abort would end the run instead of failing one test.
    """
    with ir.Context() as context:
        swage.register_dialects(context)
        module = ir.Module.parse(
            SEGMENTED_EXPONENTIAL_SUM.replace("f32", "f64")
        )
        original = module.operation.get_asm(enable_debug_info=False)

        with pytest.raises(
            ValueError,
            match="operation 'math.exp2' is admitted for f32 values only: "
            "the device has no f64 exp2",
        ):
            getattr(native_swage, compiler)(
                module, kernel_name="segmented_sum", target="sm_80", **arguments
            )

        assert module.operation.get_asm(enable_debug_info=False) == original


def test_an_f64_softmax_is_refused_before_the_backend():
    """The softmax program is refused over f64 for its exponential."""
    with ir.Context() as context:
        swage.register_dialects(context)
        module = ir.Module.parse(RAGGED_SOFTMAX.replace("f32", "f64"))

        with pytest.raises(
            ValueError,
            match="operation 'math.exp2' is admitted for f32 values only",
        ):
            native_swage._compile_segmented_reduction_ptx(
                module,
                kernel_name="ragged_softmax",
                block_size=128,
                target="sm_80",
            )


def test_region_exponential_compiles_without_libdevice():
    """Emit a native ex2 rather than an unresolvable libdevice call."""
    with ir.Context() as context:
        swage.register_dialects(context)
        module = ir.Module.parse(SEGMENTED_EXPONENTIAL_SUM)

        lowered, ptx, _ = native_swage._compile_segmented_reduction_ptx(
            module,
            kernel_name="segmented_sum",
            block_size=128,
            target="sm_80",
        )

        assert "llvm.intr.exp2" in lowered
        assert "__nv_exp2f" not in lowered
        assert ".extern .func" not in ptx
        assert "ex2.approx.f32" in ptx


RAGGED_SOFTMAX = """
module {
  func.func @ragged_softmax(
      %values: memref<?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
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


def test_ragged_softmax_reductions_use_disjoint_workgroup_buffers():
    """Guard the barrier-free three-phase schedule at its assumption."""
    with ir.Context() as context:
        swage.register_dialects(context)
        module = ir.Module.parse(RAGGED_SOFTMAX)

        lowered, ptx, _ = native_swage._compile_segmented_reduction_ptx(
            module,
            kernel_name="ragged_softmax",
            block_size=128,
            target="sm_80",
        )

        # One workgroup buffer per all-reduce is what lets the two phases run
        # without a barrier of the emitter's own. If a future LLVM pooled the
        # buffers, the second phase would overwrite the first phase's
        # broadcast slot and produce plausible wrong denominators.
        buffers = set(re.findall(r"__wg_\w+", lowered))
        assert len(buffers) == 2, buffers
        assert ptx.count(".shared .align") == 2
        assert ".extern .func" not in ptx
        assert ptx.count("ex2.approx.f32") == 2


@pytest.mark.parametrize("block_size", [32, 128])
def test_segmented_kernels_pin_their_launch_width_with_reqntid(block_size):
    """Make a mismatched blockDim a launch error, not a wrong sum."""
    with ir.Context() as context:
        swage.register_dialects(context)
        module = ir.Module.parse(SEGMENTED_SUM)
        _, ptx, _ = native_swage._compile_segmented_reduction_ptx(
            module,
            kernel_name="segmented_sum",
            block_size=block_size,
            target="sm_80",
        )
        assert f".reqntid {block_size}, 1, 1" in ptx


@pytest.mark.parametrize(
    ("compiler", "width"),
    [
        ("_compile_split_partial_reduction_ptx", 512),
        ("_compile_split_merge_reduction_ptx", 512),
        ("_compile_fused_segmented_reduction_ptx", 128),
    ],
)
def test_fixed_width_kernels_carry_their_reqntid(compiler, width):
    """The split and fused ABIs hardcode widths; the PTX must agree."""
    with ir.Context() as context:
        swage.register_dialects(context)
        module = ir.Module.parse(SEGMENTED_SUM)
        _, ptx, _ = getattr(native_swage, compiler)(
            module,
            kernel_name="segmented_sum",
            target="sm_80",
        )
        assert f".reqntid {width}, 1, 1" in ptx


BYSTANDER_MODULE = SEGMENTED_SUM.rstrip().removesuffix("}") + """
  func.func @bystander(%x: i32) -> i32 {
    return %x : i32
  }
}
"""


def test_kernel_name_must_name_a_segment_function():
    """Reject a function without Swage operations at compile time."""
    with ir.Context() as context:
        swage.register_dialects(context)
        module = ir.Module.parse(BYSTANDER_MODULE)
        with pytest.raises(
            ValueError,
            match="function names @bystander, which holds no Swage segment",
        ):
            native_swage._compile_segmented_reduction_ptx(
                module,
                kernel_name="bystander",
                block_size=128,
                target="sm_80",
            )


def test_kernel_name_matching_survives_bystander_functions():
    """Compile the swage function when kernel_name names it correctly."""
    with ir.Context() as context:
        swage.register_dialects(context)
        module = ir.Module.parse(BYSTANDER_MODULE)
        _, ptx, _ = native_swage._compile_segmented_reduction_ptx(
            module,
            kernel_name="segmented_sum",
            block_size=128,
            target="sm_80",
        )
        assert ".entry segmented_sum" in ptx


def test_rejects_unsupported_sm_values_before_the_backend_runs():
    """Refuse unknown chips instead of aborting in instruction selection."""
    with ir.Context() as context:
        swage.register_dialects(context)
        module = ir.Module.parse(SEGMENTED_SUM)
        with pytest.raises(
            ValueError, match="not a processor supported by the pinned LLVM"
        ):
            native_swage._compile_segmented_reduction_ptx(
                module,
                kernel_name="segmented_sum",
                block_size=128,
                target="sm_99",
            )


def test_rejects_an_unverified_module_with_a_diagnostic():
    """Surface verifier errors instead of walking malformed regions."""
    from mlir_swage.dialects import func as func_dialect
    from mlir_swage.dialects import memref as memref_dialect

    with ir.Context() as context, ir.Location.unknown():
        swage.register_dialects(context)
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            f32 = ir.F32Type.get()
            i32 = ir.IntegerType.get_signless(32)
            dynamic = ir.ShapedType.get_dynamic_size()
            values_type = ir.MemRefType.get([dynamic], f32)
            offsets_type = ir.MemRefType.get([dynamic], i32)
            function = func_dialect.FuncOp(
                "segmented_sum",
                ([values_type, offsets_type, values_type, i32, i32], []),
            )
            with ir.InsertionPoint(function.add_entry_block()):
                segment_id = swage.SegmentIdOp(ir.IndexType.get(), 0)
                segment = swage.MakeSegmentOp(
                    ir.Type.parse("!swage.segment<f32>"),
                    function.arguments[0],
                    function.arguments[1],
                    segment_id,
                )
                kind = ir.Attribute.parse("#swage.reduction_kind<sum>")
                # The region is left empty, so the module never verifies.
                reduce = swage.ReduceOp(f32, segment, [], kind)
                memref_dialect.StoreOp(
                    reduce.result, function.arguments[2], [segment_id]
                )
                func_dialect.ReturnOp([])

        with pytest.raises(ValueError, match="region"):
            native_swage._compile_segmented_reduction_ptx(
                module,
                kernel_name="segmented_sum",
                block_size=128,
                target="sm_80",
            )
