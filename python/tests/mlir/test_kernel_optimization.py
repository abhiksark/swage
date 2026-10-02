# python/tests/mlir/test_kernel_optimization.py
"""PTX-level contract of the LLVM passes that run before code generation.

Every kernel goes through a short, curated list of LLVM passes between the
translation to LLVM IR and the NVPTX backend. The passes may simplify
arithmetic and control flow. They may not change how the threads of a block
synchronize: a barrier, a warp shuffle, a memory fence, or an atomic that
the lowering emitted must reach the PTX exactly once, and none may appear
that the lowering did not emit.

No test here launches a kernel, so the file needs the native bindings and no
GPU. The numerical side of the same contract, round-to-nearest arithmetic
with no contraction and results that are bit-identical between launches, is
in test_segmented_numerics.py, and the device-side bounds are in
test_segmented_bounds.py.
"""

import re

import pytest
import swage as sw
import swage.language as sl
from mlir_swage import ir
from mlir_swage._mlir_libs._swageDialectsNanobind import swage as native_swage
from mlir_swage.dialects import swage
from reduction_programs import reduction_module
from swage._segmented_qualification import _SOFTMAX_MODULE, _semantic_module

# The oldest admitted processor, the qualified one, and the newest admitted.
_TARGETS = ("sm_80", "sm_86", "sm_121")

# One pattern per kind of synchronization instruction in PTX.
_SYNCHRONIZATION = {
    "barriers": re.compile(r"^\s*bar\.sync\b", re.MULTILINE),
    "shuffles": re.compile(r"^\s*shfl\.sync\.", re.MULTILINE),
    "fences": re.compile(r"^\s*membar\.", re.MULTILINE),
    "atomics": re.compile(r"^\s*atom\.", re.MULTILINE),
}

_LABEL = re.compile(r"(\$L__\w+):")
_UNCONDITIONAL_BRANCH = re.compile(r"bra\.uni\s+(\$L__\w+);")
_CONDITIONAL_BRANCH = re.compile(r"@!?%p\d+\s+bra\s+(\$L__\w+);")


def _counts(barriers, shuffles, fences=0, atomics=0):
    return {
        "barriers": barriers,
        "shuffles": shuffles,
        "fences": fences,
        "atomics": atomics,
    }


# What each kernel family carries as lowered, counted in the PTX of the
# kernels before any LLVM pass ran on them. A block reduction is two
# barriers and twenty shuffles: a full-warp and a partial-warp path of five
# shuffles each, within a warp and then across the warps.
_ONE_BLOCK_REDUCTION = _counts(barriers=2, shuffles=20)
# The five shuffles of one warp reduction, with no shared memory.
_ONE_WARP_REDUCTION = _counts(barriers=0, shuffles=5)
# A warp branch and a CTA branch in one kernel.
_FUSED = _counts(barriers=2, shuffles=25)
# Softmax reduces twice, a maximum and then a sum.
_TWO_BLOCK_REDUCTIONS = _counts(barriers=4, shuffles=40)
# Three block reductions, the queue claims, and the split completion
# protocol: two fences and four atomic updates.
_PERSISTENT = _counts(barriers=12, shuffles=66, fences=2, atomics=4)
_NONE = _counts(barriers=0, shuffles=0)

_DIRECT = "_compile_segmented_reduction_ptx"
_FUSED_COMPILER = "_compile_fused_segmented_reduction_ptx"
_PARTIAL = "_compile_split_partial_reduction_ptx"
_MERGE = "_compile_split_merge_reduction_ptx"
_PERSISTENT_COMPILER = "_compile_persistent_segmented_reduction_ptx"


def _segmented_kernels():
    """List every segmented kernel family with its expected counts.

    Yields:
        A pytest parameter of the module text, the native compiler, the
        kernel name, the compiler options, and the expected counts.
    """
    for kind in ("sum", "max"):
        text = _semantic_module(kind)
        name = f"segmented_{kind}"
        # Block size 1 is where a pipeline that knew the launch width would
        # delete the full-warp path, and 40 and 100 end in a partial warp.
        for block_size in (1, 32, 40, 100, 128, 256, 512, 1024):
            yield pytest.param(
                text, _DIRECT, name, {"block_size": block_size},
                _ONE_BLOCK_REDUCTION, id=f"direct-{kind}-{block_size}",
            )
        yield pytest.param(
            text, _DIRECT, name, {"block_size": 32, "use_task_ids": True},
            _ONE_WARP_REDUCTION, id=f"task-ids-warp-{kind}",
        )
        yield pytest.param(
            text, _DIRECT, name, {"block_size": 128, "use_task_ids": True},
            _ONE_BLOCK_REDUCTION, id=f"task-ids-cta-{kind}",
        )
        yield pytest.param(
            text, _FUSED_COMPILER, name, {}, _FUSED, id=f"fused-{kind}"
        )
        yield pytest.param(
            text, _PARTIAL, name, {}, _ONE_BLOCK_REDUCTION,
            id=f"split-partial-{kind}",
        )
        yield pytest.param(
            text, _MERGE, name, {}, _ONE_BLOCK_REDUCTION,
            id=f"split-merge-{kind}",
        )
    # Element programs add arithmetic to the loops and must add nothing to
    # the synchronization.
    for kind, transform in (
        ("sum", "square"), ("max", "maps"), ("sum", "exp2_pair"),
        ("sum", "rational8"), ("sum", "affine32"),
    ):
        text = reduction_module(kind, transform)
        name = f"segmented_{kind}"
        yield pytest.param(
            text, _FUSED_COMPILER, name, {}, _FUSED,
            id=f"fused-{kind}-{transform}",
        )
        yield pytest.param(
            text, _PARTIAL, name, {}, _ONE_BLOCK_REDUCTION,
            id=f"split-partial-{kind}-{transform}",
        )
    yield pytest.param(
        _semantic_module("sum"), _PERSISTENT_COMPILER, "segmented_sum", {},
        _PERSISTENT, id="persistent",
    )
    for block_size in (32, 128, 512):
        yield pytest.param(
            _SOFTMAX_MODULE, _DIRECT, "ragged_softmax",
            {"block_size": block_size}, _TWO_BLOCK_REDUCTIONS,
            id=f"softmax-{block_size}",
        )


# The kernels whose only loops are reduction or store loops over a segment.
# The persistent kernel is left out: its queue loops hold barriers and
# shuffles, and those keep the shape the lowering gave them.
_REDUCTION_LOOP_KERNELS = [
    parameter
    for parameter in _segmented_kernels()
    if parameter.id != "persistent"
]


@sw.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n, BLOCK: sl.constexpr):  # noqa: D103
    pid = sl.program_id(0)
    offsets = pid * BLOCK + sl.arange(0, BLOCK)
    mask = offsets < n
    x = sl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = sl.load(y_ptr + offsets, mask=mask, other=0.0)
    sl.store(output_ptr + offsets, x + y, mask=mask)


def _compile(text, compiler, kernel_name, options, target):
    with ir.Context() as context:
        swage.register_dialects(context)
        module = ir.Module.parse(text)
        _, ptx = getattr(native_swage, compiler)(
            module, kernel_name=kernel_name, target=target, **options
        )
    return ptx


def _synchronization(ptx):
    return {
        name: len(pattern.findall(ptx))
        for name, pattern in _SYNCHRONIZATION.items()
    }


def _backward_branches(ptx):
    """Count the branches of a PTX text that jump to an earlier label.

    Returns:
        The number of unconditional and of conditional backward branches.
    """
    seen = set()
    unconditional = 0
    conditional = 0
    for line in ptx.splitlines():
        line = line.strip()
        if label := _LABEL.fullmatch(line):
            seen.add(label[1])
        elif (branch := _UNCONDITIONAL_BRANCH.fullmatch(line)) and (
            branch[1] in seen
        ):
            unconditional += 1
        elif (branch := _CONDITIONAL_BRANCH.fullmatch(line)) and (
            branch[1] in seen
        ):
            conditional += 1
    return unconditional, conditional


def test_synchronization_scan_counts_each_instruction_once():
    """The scan sees every spelling it must count and nothing else.

    Without this check a pattern that matched nothing would let a kernel
    that lost its barriers pass with the kernels that have none.
    """
    ptx = """
\tbar.sync \t0;
\tshfl.sync.bfly.b32 \t%r1, %r2, 1, 31, -1;
\tshfl.sync.idx.b32 \t%r3|%p1, %r4, 0, 31, -1;
\tmembar.gl;
\tatom.global.add.u32 \t%r5, [%rd1+4], 1;
\tst.shared.b32 \t[%rd2], %r6;
\t// bar.sync in a comment
"""

    assert _synchronization(ptx) == _counts(
        barriers=1, shuffles=2, fences=1, atomics=1
    )


def test_backward_branch_scan_tells_the_two_loop_shapes_apart():
    """The scan separates a jump back to a test from a test at the bottom."""
    tested_at_the_top = """
$L__BB0_2:
\tsetp.ge.s64 \t%p2, %rd23, %rd6;
\t@%p2 bra \t$L__BB0_4;
\tadd.rn.f32 \t%r1, %r1, %r71;
\tbra.uni \t$L__BB0_2;
$L__BB0_4:
"""
    tested_at_the_bottom = """
\t@%p2 bra \t$L__BB0_4;
$L__BB0_3:
\tadd.rn.f32 \t%r83, %r83, %r17;
\tsetp.lt.s64 \t%p3, %rd21, %rd5;
\t@%p3 bra \t$L__BB0_3;
$L__BB0_4:
"""

    assert _backward_branches(tested_at_the_top) == (1, 0)
    assert _backward_branches(tested_at_the_bottom) == (0, 1)


@pytest.mark.parametrize("target", _TARGETS)
@pytest.mark.parametrize(
    ("text", "compiler", "kernel_name", "options", "expected"),
    _segmented_kernels(),
)
def test_segmented_kernels_keep_their_synchronization(
    text, compiler, kernel_name, options, expected, target
):
    """Emit each barrier, shuffle, fence, and atomic of the lowering once.

    The expected counts are those of the kernels as lowered, taken from
    their PTX before any LLVM pass ran on them. A pass that unrolled a queue
    loop, merged two shuffles, or deleted a path it proved dead would change
    a count. The counts do not depend on the block size: no pass may use the
    launch width to remove a shuffle path.
    """
    ptx = _compile(text, compiler, kernel_name, options, target)

    assert _synchronization(ptx) == expected


@pytest.mark.parametrize("target", _TARGETS)
@pytest.mark.parametrize("block_size", [32, 128, 1024])
def test_public_kernel_has_no_synchronization(block_size, target):
    """The fixed vector add stays a straight line of loads and stores."""
    module = add_kernel.emit_mlir(
        signature={
            "x_ptr": sl.pointer(sl.float32),
            "y_ptr": sl.pointer(sl.float32),
            "output_ptr": sl.pointer(sl.float32),
            "n": sl.int32,
        },
        constexprs={"BLOCK": block_size},
    )

    _, ptx = native_swage._compile_ptx(
        module, kernel_name="add_kernel", block_size=block_size, target=target
    )

    assert _synchronization(ptx) == _NONE
    assert _backward_branches(ptx) == (0, 0)
    assert f".reqntid {block_size}, 1, 1" in ptx
    assert ".entry add_kernel" in ptx


@pytest.mark.parametrize(
    ("text", "compiler", "kernel_name", "options", "expected"),
    _REDUCTION_LOOP_KERNELS,
)
def test_segment_loops_test_their_bound_once_per_iteration(
    text, compiler, kernel_name, options, expected
):
    """Close every segment loop with one conditional branch.

    A loop as lowered tests its bound at the top and jumps back from the
    bottom, two branches for each element. The pass pipeline rotates it, so
    the only backward branch is the conditional one at the bottom. The
    reduction itself is untouched: one load and one accumulator update per
    element, in the same order.
    """
    del expected
    ptx = _compile(text, compiler, kernel_name, options, "sm_86")

    unconditional, conditional = _backward_branches(ptx)

    assert unconditional == 0
    assert conditional > 0
