# python/tests/mlir/test_segmented_runtime.py
"""Differential qualification for native segmented reductions."""

import gc
import threading
import weakref
from itertools import pairwise

import pytest
import torch
from reduction_programs import reduction_module
from swage._segmented_qualification import (
    _execute,
    _launch_segmented_sum_tasks,
    _prepare_persistent_sum,
    _prepare_planned_reduction,
    _prepare_planned_sum,
    _runner_module,
    _validate_counts,
    _validate_offsets,
    _validate_tensors,
    cpu_oracle,
    cpu_softmax_oracle,
    launch_gpu,
    launch_softmax_gpu,
)


def _offsets(lengths):
    """Return the host offset list for consecutive segment lengths."""
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    return offsets


def _exact_values(first, last):
    """Return the exactly summable test pattern on an index range.

    Element ``index`` is ``(2 * (index % 67) - 65) / 4``, an odd number of
    quarters that climbs by one half from -16.25 to 16.75 and then wraps.

    - Every value is a multiple of 0.25 and the magnitudes of a segment
      sum to less than 2**20, so a sum is the same f32 under any
      association and tests can compare with no tolerance.
    - No value is zero, so dropping or repeating one element changes a sum.
    - The 67 values of a period are distinct and 67 exceeds the largest
      shift the tests guard, so a window moved by 1 to 64 elements reads
      different values. One period sums to 16.75 rather than zero, so
      whole periods do not cancel.
    - Squares are multiples of 1/16 and their sum over the longest exactly
      compared segment (8193 elements) stays below 2**20.

    A segment whose length is a multiple of 67 would still read the same
    multiset after any shift. No pattern with bounded values avoids every
    such coincidence, so test_exact_inputs_are_informative_for_every_case
    checks the properties for the shapes the suite uses.
    """
    index = torch.arange(first, last)
    return (2 * (index % 67) - 65).to(torch.float32) / 4


def _case(lengths):
    """Build deterministic values and offsets for segment lengths."""
    offsets = _offsets(lengths)
    return _exact_values(0, offsets[-1]), torch.tensor(
        offsets, dtype=torch.int32
    )


CASES = [
    pytest.param([0], id="empty"),
    pytest.param([1], id="singleton"),
    pytest.param([128], id="block-boundary"),
    pytest.param([129], id="non-multiple"),
    pytest.param([4097], id="large"),
    pytest.param([33, 33, 33, 33], id="uniform"),
    pytest.param([1, 257, 2, 3837], id="skewed"),
    pytest.param([0, 2, 0, 0, 3], id="repeated-empty"),
]


TASK_CASES = [
    pytest.param([], id="no-segments"),
    pytest.param([0], id="empty"),
    pytest.param([32, 33], id="policy-boundary"),
    pytest.param([0, 2, 0, 0, 3], id="repeated-empty"),
    pytest.param([1, 257, 2, 3837], id="skewed"),
    pytest.param([4095], id="chunk-minus-one"),
    pytest.param([4096], id="chunk-boundary"),
    pytest.param([4097], id="chunk-plus-one"),
    pytest.param([8192], id="exact-two-chunks"),
    pytest.param([8193], id="two-chunks-plus-one"),
    pytest.param([1, 8193, 2], id="one-huge-outlier"),
    pytest.param([4097, 4097, 4097], id="many-huge"),
]


_SHIFT_LIMIT = 64


@pytest.mark.parametrize("lengths", [*CASES, *TASK_CASES])
def test_exact_inputs_are_informative_for_every_case(lengths):
    """Each non-empty segment has a nonzero sum that depends on its window.

    A kernel that reads the right number of elements from the wrong place,
    or drops or repeats one, must change the expected value. For every
    segment this checks that the sum is nonzero, that moving both window
    ends by any 1 to 64 elements in either direction changes it, and that
    no element is zero, so dropping or duplicating any one changes it too.

    It also checks what makes an exact comparison valid: every value is a
    multiple of 0.25, and the per-segment sums of magnitudes and of squares
    stay below 2**20, so the identity and squared sums are exact in f32
    under any association.
    """
    values, offsets = _case(lengths)
    count = values.numel()
    quarters = values.double() * 4
    assert torch.equal(quarters, quarters.round())
    assert (quarters != 0).all()

    extended = _exact_values(-_SHIFT_LIMIT, count + _SHIFT_LIMIT)
    assert torch.equal(extended[_SHIFT_LIMIT : _SHIFT_LIMIT + count], values)
    prefix = torch.cat(
        [
            torch.zeros(1, dtype=torch.int64),
            (extended.double() * 4).to(torch.int64).cumsum(0),
        ]
    )
    shifts = torch.tensor(
        [shift for shift in range(-_SHIFT_LIMIT, _SHIFT_LIMIT + 1) if shift]
    )
    for begin, end in pairwise(offsets.tolist()):
        if begin == end:
            continue
        total = prefix[end + _SHIFT_LIMIT] - prefix[begin + _SHIFT_LIMIT]
        assert total != 0
        shifted = (
            prefix[end + _SHIFT_LIMIT + shifts]
            - prefix[begin + _SHIFT_LIMIT + shifts]
        )
        assert (shifted != total).all()
        segment = values[begin:end].double()
        assert segment.abs().sum() < 2**20
        assert segment.square().sum() < 2**20


def test_exact_values_stay_exact_for_the_longest_compared_segments():
    """Bound the two largest exact comparisons at every pattern phase.

    Squared values are multiples of 1/16, exact in f32 while every partial
    sum stays below 2**20. The longest segment compared exactly after
    squaring has 8193 elements. Identity values are multiples of 1/4, exact
    below 2**22, and the longest identity segment has 65537 elements.
    """
    values = _exact_values(0, 65537 + 67).double()
    squares = torch.cat([torch.zeros(1), values.square().cumsum(0)])
    magnitudes = torch.cat([torch.zeros(1), values.abs().cumsum(0)])
    starts = torch.arange(67)

    assert (squares[starts + 8193] - squares[starts]).max() < 2**20
    assert (magnitudes[starts + 65537] - magnitudes[starts]).max() < 2**22


def _pytorch_reference(values, offsets, kind):
    """Compute the PyTorch reference while preserving empty identities."""
    if len(offsets) < 2:
        return torch.empty(0, dtype=torch.float32)
    results = []
    for index in range(len(offsets) - 1):
        segment = values[offsets[index] : offsets[index + 1]]
        if kind == "sum":
            results.append(segment.sum())
        elif segment.numel():
            results.append(segment.max())
        else:
            results.append(torch.tensor(float("-inf"), dtype=torch.float32))
    return torch.stack(results)


def _transformed_values(values, transform):
    """Reference the element expression independently of the compiler."""
    if transform == "square":
        return values * values
    if transform == "maps":
        return (values + 1) * 2
    if transform == "exp2":
        return torch.exp2(values)
    if transform == "exp2_chain":
        for _ in range(8):
            values = torch.exp2(-0.5 * values)
    if transform == "rational8":
        for _ in range(8):
            values = (values + 0.125) / (1 + 0.25 * values * values)
    if transform in ("affine4", "affine32"):
        for _ in range(2 if transform == "affine4" else 16):
            values = -0.5 * values + 0.125
    return values


@pytest.mark.parametrize("kind", ["sum", "max"])
@pytest.mark.parametrize("transform", ["identity", "square", "maps"])
def test_composable_reduction_cpu_oracle(kind, transform):
    """The same native program supplies a sequential composition oracle."""
    values, offsets = _case([0, 1, 2, 3, 5, 6, 10, 11])
    printed = _execute(
        _runner_module(
            values,
            offsets,
            reduction_module(kind, transform),
            f"segmented_{kind}",
            len(offsets) - 1,
        )
    )
    expected = _pytorch_reference(
        _transformed_values(values, transform), offsets, kind
    )
    torch.testing.assert_close(
        torch.tensor(printed, dtype=torch.float32),
        expected,
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("kind", ["sum", "max"])
@pytest.mark.parametrize("transform", ["identity", "square", "maps"])
@pytest.mark.parametrize("small_chunks", [False, True])
def test_composable_reduction_gpu_schedules(kind, transform, small_chunks):
    """Every static schedule executes one program, including split tails."""
    lengths = (
        [0, 1, 2, 3, 5, 6, 10, 11]
        if small_chunks
        else [0, 1, 31, 32, 33, 4095, 4096, 4097, 8192, 8193]
    )
    host_values, host_offsets = _case(lengths)
    semantic = reduction_module(kind, transform)
    expected = _pytorch_reference(
        _transformed_values(host_values, transform), host_offsets, kind
    )
    values, offsets = host_values.cuda(), host_offsets.cuda()
    output = torch.full((len(lengths) + 1,), -123.0, device="cuda")
    limits = (
        {"warp_max_elements": 2, "cta_chunk_elements": 5}
        if (small_chunks)
        else {}
    )
    prepared = _prepare_planned_reduction(
        values,
        offsets,
        output,
        module_text=semantic,
        kernel_name=f"segmented_{kind}",
        **limits,
    )
    if small_chunks:
        printed = _execute(
            _runner_module(
                host_values,
                host_offsets,
                semantic,
                f"segmented_{kind}",
                len(lengths),
            )
        )
        torch.testing.assert_close(
            torch.tensor(printed, dtype=torch.float32),
            expected,
            rtol=0,
            atol=0,
        )
    for launch in prepared:
        for _ in range(2):
            output[:-1].fill_(float("nan"))
            launch()
            torch.testing.assert_close(
                output[:-1].cpu(),
                expected,
                rtol=0,
                atol=0,
            )
            assert output[-1].item() == -123.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("kind", ["sum", "max"])
@pytest.mark.parametrize(
    "transform",
    [
        "square", "maps", "exp2", "exp2_chain", "rational8",
        "affine4", "affine32",
    ],
)
def test_composable_reduction_nontrivial_f32(kind, transform):
    """Qualify rounding on finite non-dyadic data across static schedules."""
    lengths = [0, 7, 33, 4097, 8193]
    values = torch.sin(torch.arange(sum(lengths)) * 0.17) * 0.3 + 0.75
    offsets = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32
    )
    expected = _pytorch_reference(
        _transformed_values(values, transform), offsets, kind
    )
    output = torch.empty(len(lengths), device="cuda")
    prepared = _prepare_planned_reduction(
        values.cuda(),
        offsets.cuda(),
        output,
        module_text=reduction_module(kind, transform),
        kernel_name=f"segmented_{kind}",
    )
    for launch in prepared:
        launch()
        torch.testing.assert_close(
            output.cpu(),
            expected,
            rtol=1e-5,
            atol=1e-5,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("kind", ["sum", "max"])
@pytest.mark.parametrize(
    "transform", ["identity", "square", "maps", "affine4", "affine32", "exp2"]
)
def test_regular_split_batch_selects_cta_without_split_kernels(
    kind, transform, monkeypatch
):
    """A selected CTA reuses the program, output guards, and graph support."""
    from mlir_swage._mlir_libs._swageDialectsNanobind import swage as native

    count = torch.cuda.get_device_properties(0).multi_processor_count
    host_values, host_offsets = _case([8192] * count)
    expected = _pytorch_reference(
        _transformed_values(host_values, transform), host_offsets, kind
    ).cuda()
    output = torch.full((count + 1,), -123.0, device="cuda")

    def unexpected_split(*args, **kwargs):
        raise AssertionError("selected CTA must not compile split kernels")

    for name in ("partial", "merge"):
        monkeypatch.setattr(
            native, f"_compile_split_{name}_reduction_ptx", unexpected_split
        )
    prepared = _prepare_planned_reduction(
        host_values.cuda(), host_offsets.cuda(), output,
        module_text=reduction_module(kind, transform),
        kernel_name=f"segmented_{kind}",
    )
    assert prepared.mixed is prepared.cta
    prepared.mixed()
    tolerance = (
        {"rtol": 1e-5, "atol": 1e-5}
        if transform in ("exp2", "affine32") else {"rtol": 0, "atol": 0}
    )
    torch.testing.assert_close(output[:-1], expected, **tolerance)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        prepared.mixed()
    output[:-1].fill_(float("nan"))
    graph.replay()
    torch.testing.assert_close(output[:-1], expected, **tolerance)
    assert output[-1].item() == -123.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("kind", ["sum", "max"])
@pytest.mark.parametrize(
    "transform", ["exp2_chain", "rational8"]
)
def test_expensive_regular_batch_retains_split_execution(kind, transform):
    """A shape eligible for CTA still splits an expensive element program."""
    count = torch.cuda.get_device_properties(0).multi_processor_count
    values, offsets = _case([8192] * count)
    expected = _pytorch_reference(
        _transformed_values(values, transform), offsets, kind
    )
    output = torch.empty(count, device="cuda")
    prepared = _prepare_planned_reduction(
        values.cuda(), offsets.cuda(), output,
        module_text=reduction_module(kind, transform),
        kernel_name=f"segmented_{kind}",
    )
    assert prepared.mixed is not prepared.cta
    prepared.mixed()
    torch.testing.assert_close(output.cpu(), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize(
    "case", ["sparse", "long", "mixed", "empty", "custom", "disabled"]
)
def test_direct_cta_selection_retains_other_schedules(case):
    """Keep fixed planning for unsupported shapes and explicit opt-out."""
    count = torch.cuda.get_device_properties(0).multi_processor_count
    lengths = [8192] * count
    options = {}
    if case == "sparse":
        lengths.pop()
    elif case == "long":
        lengths[-1] = 8193
    elif case == "mixed":
        lengths[-1] = 4096
    elif case == "empty":
        lengths[-1] = 0
    elif case == "custom":
        options["cta_chunk_elements"] = 2048
    else:
        options["select_schedule"] = False
    values, offsets = _case(lengths)
    output = torch.empty(len(lengths), device="cuda")
    prepared = _prepare_planned_reduction(
        values.cuda(), offsets.cuda(), output,
        module_text=reduction_module("sum", "identity"),
        kernel_name="segmented_sum", **options,
    )
    assert prepared.mixed is not prepared.cta
    prepared.mixed()
    torch.testing.assert_close(
        output.cpu(), _pytorch_reference(values, offsets, "sum"),
        rtol=0, atol=0,
    )


def test_schedule_selection_requires_a_boolean():
    """Reject accidental string configuration before preparing any work."""
    with pytest.raises(TypeError, match="select_schedule must be a bool"):
        _prepare_planned_reduction(
            None, None, None, module_text="", kernel_name="",
            select_schedule="false",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_planned_max_preserves_special_values_across_chunks():
    """Keep NaNs, infinities, and signed zero through partials and merges."""
    length = 4097
    host_values = torch.cat(
        [
            torch.full((length,), 3.0),
            torch.full((length,), float("-inf")),
            torch.full((length,), float("inf")),
            torch.full((length,), -0.0),
            torch.full((length,), -0.0),
        ]
    )
    host_values[length - 1] = float("nan")
    host_values[-1] = 0.0
    offsets = torch.tensor(
        [0, 0, length, 2 * length, 3 * length, 4 * length, 5 * length],
        dtype=torch.int32,
        device="cuda",
    )
    output = torch.empty(6, device="cuda")
    prepared = _prepare_planned_reduction(
        host_values.cuda(),
        offsets,
        output,
        module_text=reduction_module("max", "identity"),
        kernel_name="segmented_max",
    )
    expected = torch.tensor(
        [
            float("-inf"),
            float("nan"),
            float("-inf"),
            float("inf"),
            -0.0,
            0.0,
        ]
    )
    for launch in prepared:
        launch()
        actual = output.cpu()
        torch.testing.assert_close(
            actual,
            expected,
            rtol=0,
            atol=0,
            equal_nan=True,
        )
        assert torch.equal(actual[-2:].signbit(), expected[-2:].signbit())


@pytest.mark.parametrize(
    ("offsets", "value_count", "output_count", "message"),
    [
        ([1, 1], 1, 1, "start at zero"),
        ([0, -1], 1, 1, "must not be negative"),
        ([0, 2, 1], 2, 2, "nondecreasing"),
        ([0, 2], 1, 1, "exceeds value count"),
        ([0, 1, 1], 1, 1, "output has 1 elements"),
    ],
)
def test_rejects_malformed_offsets(offsets, value_count, output_count, message):
    """Reject malformed metadata before a CUDA pointer is launched."""
    with pytest.raises(ValueError, match=message):
        _validate_offsets(offsets, value_count, output_count)


def test_cpu_oracle_rejects_empty_offsets_with_the_validator_message():
    """Fail closed on empty offsets instead of allocating a -1 tensor."""
    values = torch.randn(4, dtype=torch.float32)
    offsets = torch.tensor([], dtype=torch.int32)

    with pytest.raises(ValueError, match="at least the initial zero"):
        cpu_oracle(values, offsets, "sum")


def _sequential_f32_sum(values):
    """Accumulate left to right in float32, the order of the CPU oracle."""
    total = torch.zeros((), dtype=torch.float32)
    for value in values:
        total += value
    return total


def _bits(tensor):
    """Return the IEEE-754 bit patterns of a float32 tensor."""
    return tensor.contiguous().view(torch.int32).tolist()


def test_cpu_sum_oracle_is_bit_exact_for_seeded_randn():
    """The oracle transports result bits, not six-digit decimal text.

    A 65537-element zero-mean segment has a float32 sum that six significant
    digits cannot hold, so this fails on any decimal transport. The expected
    value is a scalar float32 loop in the oracle's own left-to-right order,
    which makes bit equality the correct assertion.
    """
    generator = torch.Generator().manual_seed(0)
    values = torch.randn(65537, generator=generator)
    offsets = torch.tensor([0, 65537], dtype=torch.int32)

    actual = cpu_oracle(values, offsets, "sum")

    expected = _sequential_f32_sum(values).reshape(1)
    assert _bits(actual) == _bits(expected)


def test_cpu_oracle_round_trips_every_float32_class():
    """Singleton maxima return their input bits through the transport."""
    values = torch.tensor(
        [
            1 / 3,
            -0.0,
            0.0,
            float("inf"),
            float("-inf"),
            torch.finfo(torch.float32).max,
            torch.finfo(torch.float32).tiny,
            1e-45,
            -16777215.0,
        ],
        dtype=torch.float32,
    )
    offsets = torch.arange(values.numel() + 1, dtype=torch.int32)

    actual = cpu_oracle(values, offsets, "max")

    assert _bits(actual) == _bits(values)


def test_cpu_softmax_oracle_is_bit_exact_for_equal_logits():
    """Three equal logits normalize to the float32 nearest one third.

    exp2 of zero is exactly one and the sum is exactly three, so the only
    rounding is the final division, which six decimal digits cannot carry.
    """
    values = torch.full((3,), 5.0)
    offsets = torch.tensor([0, 3], dtype=torch.int32)

    actual = cpu_softmax_oracle(values, offsets)

    expected = (torch.ones(3) / 3).to(torch.float32)
    assert _bits(actual) == _bits(expected)


@pytest.mark.parametrize("kind", ["sum", "max"])
def test_cpu_oracle_returns_nothing_for_zero_segments(kind):
    """A segment-free call yields an empty result, not an unwritten slot."""
    values = torch.empty(0, dtype=torch.float32)
    offsets = torch.zeros(1, dtype=torch.int32)

    actual = cpu_oracle(values, offsets, kind)

    assert actual.shape == (0,)
    assert actual.dtype == torch.float32


@pytest.mark.parametrize("count", [-1, 1 << 31])
def test_rejects_counts_outside_i32(count):
    """Keep explicit value and segment counts inside the CUDA ABI."""
    with pytest.raises(ValueError, match="nonnegative i32"):
        _validate_counts(count, 0)
    with pytest.raises(ValueError, match="nonnegative i32"):
        _validate_counts(0, count)


def test_rejects_wrong_offset_dtype_rank_and_undersized_output():
    """Validate tensor metadata before device or pointer access."""
    values = torch.empty(2)
    output = torch.empty(1)
    with pytest.raises(TypeError, match="offsets must have dtype torch.int32"):
        _validate_tensors(values, torch.tensor([0, 2]), output)
    with pytest.raises(TypeError, match="offsets must have rank one"):
        _validate_tensors(
            values,
            torch.tensor([[0, 2]], dtype=torch.int32),
            output,
        )
    with pytest.raises(ValueError, match="output has 1 elements"):
        _validate_tensors(
            values,
            torch.tensor([0, 1, 2], dtype=torch.int32),
            output,
        )


@pytest.mark.parametrize("kind", ["sum", "max"])
@pytest.mark.parametrize("lengths", CASES)
def test_cpu_reduction_matches_pytorch(lengths, kind):
    """Execute sequential reductions through upstream mlir-runner."""
    values, offsets = _case(lengths)

    actual = cpu_oracle(values, offsets, kind)
    expected = _pytorch_reference(values, offsets, kind)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("kind", ["sum", "max"])
@pytest.mark.parametrize("lengths", CASES)
def test_gpu_reduction_matches_pytorch_and_cpu_oracle(lengths, kind):
    """Qualify one-CTA reductions against both independent references."""
    host_values, host_offsets = _case(lengths)
    values = host_values.cuda()
    offsets = host_offsets.cuda()
    output = torch.empty(len(lengths), device="cuda")

    launch_gpu(values, offsets, output, kind)

    torch.testing.assert_close(
        output.cpu(),
        _pytorch_reference(host_values, host_offsets, kind),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        output.cpu(),
        cpu_oracle(host_values, host_offsets, kind),
        rtol=0,
        atol=0,
    )


RANDOM_LENGTHS = [0, 1, 31, 32, 33, 4095, 4096, 4097, 65537]
RANDOM_SUITES = ["randn", "cancellation", "magnitude"]
# Planning limits (warp_max_elements, cta_chunk_elements) of the two
# classified preparations. "split" lowers both so that every random segment
# longer than 16 elements takes the partial and merge kernels.
_PLANNING_LIMITS = {"mixed": (32, 4096), "split": (1, 16)}
_EPS32 = torch.finfo(torch.float32).eps


def _random_values(suite, count, generator):
    """Draw one segment of order-sensitive f32 values."""
    if suite == "randn":
        return torch.randn(count, generator=generator)
    if suite == "cancellation":
        # Exactly opposite pairs near 1e4 around terms near 1e-3, shuffled
        # so partners rarely share a lane. The true sum is the small terms.
        pairs = count // 3
        large = (1 + torch.rand(pairs, generator=generator)) * 1e4
        small = torch.randn(count - 2 * pairs, generator=generator) * 1e-3
        values = torch.cat([large, -large, small])
        return values[torch.randperm(count, generator=generator)]
    assert suite == "magnitude"
    # Log-uniform magnitudes from 1e-6 to 1e6 with random signs.
    exponents = torch.empty(count).uniform_(-6, 6, generator=generator)
    signs = torch.randint(0, 2, (count,), generator=generator) * 2 - 1
    return signs * torch.pow(10.0, exponents)


def _random_case(suite, seed):
    """Build seeded values and offsets for RANDOM_LENGTHS."""
    generator = torch.Generator().manual_seed(seed)
    values = torch.cat(
        [_random_values(suite, length, generator) for length in RANDOM_LENGTHS]
    )
    assert values.dtype == torch.float32
    return values, torch.tensor(_offsets(RANDOM_LENGTHS), dtype=torch.int32)


def _float64_reference(values, kind):
    """Reduce each random segment in float64, outside the compiler."""
    return torch.segment_reduce(
        values.double(), kind, lengths=torch.tensor(RANDOM_LENGTHS)
    )


def _summation_depth(policy, count):
    """Bound the rounding additions one element of a segment passes through.

    Every schedule is a summation tree. B lanes stride the segment, so a
    lane holds at most ceil(count / B) elements and chains one addition
    fewer than that after its first, because adding to the zero identity is
    exact. The lanes are then combined by a balanced tree.

    - sequential (CPU oracle): one chain, count - 1 additions.
    - warp: B = 32, then five XOR-shuffle levels.
    - cta: B = 128, then gpu.all_reduce, five levels inside each warp and
      two across the four warp leaders.
    - partial and merge kernels: B = 512, then gpu.all_reduce, five levels
      inside each warp and four across the sixteen warp leaders. A split
      segment passes through a partial over at most one chunk and then
      through the merge over its ceil(count / chunk) partials.
    - mixed and split: the path the planner classifies the length into
      under _PLANNING_LIMITS, which is warp, cta, or partial plus merge.

    An addition with an exact zero operand is exact and the subtrees merged
    along one path hold distinct elements, so no element sees more than
    count - 1 rounding additions whatever the tree.
    """

    def chain(elements, lanes):
        return max(-(-elements // lanes) - 1, 0)

    if policy == "sequential":
        depth = count - 1
    elif policy == "warp":
        depth = chain(count, 32) + 5
    elif policy == "cta":
        depth = chain(count, 128) + 7
    else:
        warp_max, chunk = _PLANNING_LIMITS[policy]
        if count <= warp_max:
            depth = chain(count, 32) + 5
        elif count <= chunk:
            depth = chain(count, 128) + 7
        else:
            partials = -(-count // chunk)
            depth = chain(chunk, 512) + 9 + chain(partials, 512) + 9
    return max(min(depth, count - 1), 0)


def _assert_matches_float64_reference(actual, values, kind, policy):
    """Compare one policy's f32 results with the float64 reference.

    Max must be exact. A sum must lie within k * eps32 * sum(|x|) of the
    reference per segment, where k is _summation_depth. The worst-case
    error of a summation tree is ((1 + u) ** k - 1) * sum(|x|) with unit
    roundoff u = eps32 / 2, which is below 2 * k * u while k * u <= 1 / 2.
    The factor of two in eps32 is that margin, and it also covers the
    rounding of the float64 reference.
    """
    reference = _float64_reference(values, kind)
    if kind == "max":
        torch.testing.assert_close(
            actual, reference.float(), rtol=0, atol=0, msg=policy
        )
        return
    magnitude = _float64_reference(values.abs(), "sum")
    depth = torch.tensor(
        [_summation_depth(policy, length) for length in RANDOM_LENGTHS],
        dtype=torch.float64,
    )
    error = (actual.double() - reference).abs()
    bound = depth * _EPS32 * magnitude
    assert (error <= bound).all(), (
        f"{policy}: error {error.tolist()} exceeds bound {bound.tolist()} "
        f"for lengths {RANDOM_LENGTHS}"
    )


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("suite", RANDOM_SUITES)
@pytest.mark.parametrize("kind", ["sum", "max"])
def test_cpu_oracle_matches_float64_reference_on_random_values(
    kind, suite, seed
):
    """Bound the sequential oracle against float64 on order-sensitive data.

    The oracle accumulates left to right, so k is count - 1 per segment in
    the bound k * eps32 * sum(|x|) of _assert_matches_float64_reference.
    """
    values, offsets = _random_case(suite, seed)

    actual = cpu_oracle(values, offsets, kind)

    _assert_matches_float64_reference(actual, values, kind, "sequential")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("suite", RANDOM_SUITES)
@pytest.mark.parametrize("kind", ["sum", "max"])
def test_gpu_policies_match_float64_reference_on_random_values(
    kind, suite, seed
):
    """Bound every static policy against float64 on order-sensitive data.

    Values are seeded normal draws, cancelling pairs around small terms,
    and magnitudes spread over twelve decades. Lengths cover the empty and
    singleton segments, both sides of the warp and chunk limits, and one
    segment of 65537 elements.

    Max is compared exactly. Each sum must lie within
    k * eps32 * sum(|x|) of torch.segment_reduce in float64, with eps32 =
    2**-23 and k the number of rounding additions on the longest path of
    that policy's reduction tree for that segment length, capped at
    count - 1 (see _summation_depth for the derivation):

    - warp: k = ceil(count / 32) - 1 + 5.
    - cta: k = ceil(count / 128) - 1 + 7.
    - split: k = (ceil(chunk / 512) - 1 + 9)
      + (ceil(ceil(count / chunk) / 512) - 1 + 9) with 16-element chunks,
      so 26 for 65537 elements.
    - mixed: the warp formula up to 32 elements, the cta formula up to
      4096, and the split formula with 4096-element chunks beyond, so 25
      for 65537 elements.

    The private API has no separate split closure. Splitting is the path
    the mixed closure takes for a segment longer than the chunk limit, so
    "split" is the mixed closure of a second preparation whose limits send
    every segment longer than 16 elements through partial and merge.

    Every policy also runs twice and must reproduce its own bits.
    """
    host_values, host_offsets = _random_case(suite, seed)
    values, offsets = host_values.cuda(), host_offsets.cuda()
    output = torch.empty(len(RANDOM_LENGTHS), device="cuda")
    launches = {}
    for policy, (warp_max, chunk) in _PLANNING_LIMITS.items():
        prepared = _prepare_planned_reduction(
            values,
            offsets,
            output,
            module_text=reduction_module(kind, "identity"),
            kernel_name=f"segmented_{kind}",
            warp_max_elements=warp_max,
            cta_chunk_elements=chunk,
            select_schedule=False,
        )
        if policy == "mixed":
            launches["warp"] = prepared.warp
            launches["cta"] = prepared.cta
        launches[policy] = prepared.mixed

    for policy, launch in launches.items():
        runs = []
        for _ in range(2):
            output.fill_(float("nan"))
            launch()
            runs.append(output.cpu())
        assert _bits(runs[0]) == _bits(runs[1]), policy
        _assert_matches_float64_reference(runs[0], host_values, kind, policy)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("block_size", [32, 128], ids=["warp", "cta"])
@pytest.mark.parametrize("lengths", TASK_CASES)
def test_gpu_task_sum_matches_exact_position_dependent_sums(
    lengths, block_size
):
    """Execute every segment through each private direct task policy."""
    host_values, host_offsets = _case(lengths)
    output = torch.full(
        (len(lengths),), float("nan"), device="cuda", dtype=torch.float32
    )
    task_ids = torch.arange(len(lengths), device="cuda", dtype=torch.int32)

    _launch_segmented_sum_tasks(
        host_values.cuda(),
        host_offsets.cuda(),
        output,
        task_ids,
        block_size=block_size,
    )

    torch.testing.assert_close(
        output.cpu(),
        _pytorch_reference(host_values, host_offsets, "sum"),
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("lengths", TASK_CASES)
def test_prepared_mixed_sum_matches_exact_position_dependent_sums(lengths):
    """Execute one fused launch with stable warp IDs before stable CTA IDs."""
    host_values, host_offsets = _case(lengths)
    output = torch.full(
        (len(lengths),), float("nan"), device="cuda", dtype=torch.float32
    )

    prepared = _prepare_planned_sum(
        host_values.cuda(), host_offsets.cuda(), output
    )
    prepared.mixed()

    torch.testing.assert_close(
        output.cpu(),
        _pytorch_reference(host_values, host_offsets, "sum"),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("resident_blocks", [0, -1, 1 << 32, True, "4"])
def test_persistent_sum_rejects_invalid_residency_before_tensor_work(
    resident_blocks,
):
    """Reject invalid physical grids before validation or native work."""
    with pytest.raises(ValueError, match="positive u32"):
        _prepare_persistent_sum(
            None, None, None, resident_blocks=resident_blocks
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize(
    "lengths",
    [
        pytest.param([0], id="empty"),
        pytest.param([32, 33], id="policy-boundary"),
        pytest.param([0, 2, 0, 0, 3], id="repeated-empty"),
        pytest.param([1, 257, 2, 3837], id="skewed"),
        pytest.param([4096], id="chunk-boundary"),
        pytest.param([4097], id="split-boundary"),
        pytest.param([0, 33, 4097, 1, 8193], id="mixed-split"),
    ],
)
def test_persistent_sum_matches_exact_position_dependent_sums(lengths):
    """Drain direct warp and CTA queues without dropped or duplicate work."""
    host_values, host_offsets = _case(lengths)
    output = torch.full(
        (len(lengths),), float("nan"), device="cuda", dtype=torch.float32
    )

    prepared = _prepare_persistent_sum(
        host_values.cuda(), host_offsets.cuda(), output, resident_blocks=4
    )
    prepared.launch()

    torch.testing.assert_close(
        output.cpu(),
        _pytorch_reference(host_values, host_offsets, "sum"),
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_persistent_sum_resets_queues_for_repeated_and_graph_launches():
    """Reset claims on every submission and preserve capture replay."""
    lengths = [1, 32, 33, 4096, 4097, 2, 8193]
    host_values, host_offsets = _case(lengths)
    output = torch.full((len(lengths),), float("nan"), device="cuda")
    prepared = _prepare_persistent_sum(
        host_values.cuda(), host_offsets.cuda(), output, resident_blocks=3
    )
    expected = _pytorch_reference(host_values, host_offsets, "sum")

    prepared.launch()
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    output.fill_(float("nan"))
    prepared.launch()
    torch.cuda.synchronize()
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        prepared.launch()
    output.fill_(float("nan"))
    graph.replay()
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_persistent_sum_uses_current_non_default_stream():
    """Submit queue reset and resident workers on the current stream."""
    lengths = [1, 33, 2, 4096]
    host_values, host_offsets = _case(lengths)
    output = torch.full((len(lengths),), float("nan"), device="cuda")
    prepared = _prepare_persistent_sum(
        host_values.cuda(), host_offsets.cuda(), output
    )
    stream = torch.cuda.Stream()

    with torch.cuda.stream(stream):
        prepared.launch()
    stream.synchronize()

    torch.testing.assert_close(
        output.cpu(),
        _pytorch_reference(host_values, host_offsets, "sum"),
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_persistent_split_sum_matches_nontrivial_oracles():
    """Publish partial completion before the unique dependent merge."""
    lengths = [0, 33, 4097, 1, 8193]
    host_values, host_offsets = _case(lengths)
    values = host_values.cuda()
    offsets = host_offsets.cuda()
    output = torch.full((len(lengths),), float("nan"), device="cuda")

    prepared = _prepare_persistent_sum(
        values, offsets, output, resident_blocks=5
    )
    prepared.launch()

    expected = _pytorch_reference(host_values, host_offsets, "sum")
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(
        output.cpu(),
        cpu_oracle(host_values, host_offsets, "sum"),
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_persistent_extreme_skew_completes_without_starvation():
    """Drain many claims and dependency groups repeatedly with seven CTAs."""
    lengths = [1] * 2048 + [65_537] * 8 + [32] * 2048
    host_values, host_offsets = _case(lengths)
    output = torch.full((len(lengths),), float("nan"), device="cuda")
    prepared = _prepare_persistent_sum(
        host_values.cuda(), host_offsets.cuda(), output, resident_blocks=7
    )
    expected = _pytorch_reference(host_values, host_offsets, "sum").cuda()

    assert prepared.resident_blocks == 7
    assert prepared.warp_tasks == 4096
    assert prepared.cta_tasks == 0
    assert prepared.partial_tasks == 136
    assert prepared.merge_tasks == 8
    for _ in range(10):
        output.fill_(float("nan"))
        prepared.launch()
        torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_empty_persistent_sum_does_not_compile_allocate_or_launch(monkeypatch):
    """Return a no-op after classifying an empty segment set."""
    from mlir_swage._mlir_libs._swageDialectsNanobind import (
        swage as native_swage,
    )
    from swage import _runtime

    values = torch.empty(0, device="cuda", dtype=torch.float32)
    offsets = torch.zeros(1, device="cuda", dtype=torch.int32)
    output = torch.empty(0, device="cuda", dtype=torch.float32)

    def fail(*_args, **_kwargs):
        pytest.fail("empty persistent work must not continue")

    monkeypatch.setattr(
        native_swage, "_compile_persistent_segmented_reduction_ptx", fail
    )
    monkeypatch.setattr(torch, "tensor", fail)
    monkeypatch.setattr(torch, "zeros", fail)
    monkeypatch.setattr(_runtime, "_get_driver", fail)

    prepared = _prepare_persistent_sum(values, offsets, output)

    assert prepared.resident_blocks == 0
    assert prepared.launch() is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_persistent_sum_rejects_mismatched_plan_before_work(monkeypatch):
    """Reject malformed dependency metadata before compilation or allocation."""
    from mlir_swage._mlir_libs._swageDialectsNanobind import (
        swage as native_swage,
    )
    from swage import _runtime

    values = torch.ones(4097, device="cuda")
    offsets = torch.tensor([0, 4097], device="cuda", dtype=torch.int32)
    output = torch.empty(1, device="cuda")

    def fail(*_args, **_kwargs):
        pytest.fail("malformed persistent work must not continue")

    monkeypatch.setattr(
        native_swage,
        "_materialize_segmented_plan",
        lambda *_args, **_kwargs: ([], [], [0, 4096], [0, 0, 1]),
    )
    monkeypatch.setattr(
        native_swage, "_compile_persistent_segmented_reduction_ptx", fail
    )
    monkeypatch.setattr(torch, "tensor", fail)
    monkeypatch.setattr(_runtime, "_get_driver", fail)

    with pytest.raises(RuntimeError, match="materialized plan does not match"):
        _prepare_persistent_sum(values, offsets, output)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_persistent_compile_failure_precedes_allocation_and_driver(monkeypatch):
    """Surface compilation failure before allocating private device state."""
    from mlir_swage._mlir_libs._swageDialectsNanobind import (
        swage as native_swage,
    )
    from swage import _runtime

    values = torch.ones(33, device="cuda")
    offsets = torch.tensor([0, 33], device="cuda", dtype=torch.int32)
    output = torch.empty(1, device="cuda")

    def compile_fail(*_args, **_kwargs):
        raise RuntimeError("persistent compile failed")

    def continue_fail(*_args, **_kwargs):
        pytest.fail("persistent compile failure must stop preparation")

    monkeypatch.setattr(
        native_swage,
        "_compile_persistent_segmented_reduction_ptx",
        compile_fail,
    )
    monkeypatch.setattr(torch, "tensor", continue_fail)
    monkeypatch.setattr(_runtime, "_get_driver", continue_fail)

    with pytest.raises(RuntimeError, match="persistent compile failed"):
        _prepare_persistent_sum(values, offsets, output)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_persistent_allocation_failure_precedes_launch(monkeypatch):
    """Do not submit resident work after private allocation fails."""
    from mlir_swage._mlir_libs._swageDialectsNanobind import (
        swage as native_swage,
    )
    from swage import _runtime

    class _Driver:
        def __init__(self):
            self.launches = []

        def load(self, _ptx, _kernel_name):
            return 1, 1

        def launch_persistent(self, *arguments):
            self.launches.append(arguments)

    values = torch.ones(4097, device="cuda")
    offsets = torch.tensor([0, 4097], device="cuda", dtype=torch.int32)
    output = torch.empty(1, device="cuda")
    monkeypatch.setattr(
        native_swage,
        "_compile_persistent_segmented_reduction_ptx",
        lambda *_args, **_kwargs: ("", "ptx"),
    )
    driver = _Driver()
    monkeypatch.setattr(_runtime, "_get_driver", lambda: driver)
    monkeypatch.setattr(
        torch,
        "tensor",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            MemoryError("persistent allocation failed")
        ),
    )

    with pytest.raises(MemoryError, match="persistent allocation failed"):
        _prepare_persistent_sum(values, offsets, output)
    assert not driver.launches


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_persistent_launch_failure_has_no_static_fallback(monkeypatch):
    """Propagate the resident launch error without submitting another policy."""
    from mlir_swage._mlir_libs._swageDialectsNanobind import (
        swage as native_swage,
    )
    from swage import _runtime

    class _Driver:
        def __init__(self):
            self.launches = 0

        def load(self, _ptx, _kernel_name):
            return 1, 1

        def launch_persistent(self, *_arguments):
            self.launches += 1
            raise RuntimeError("persistent launch failed")

    monkeypatch.setattr(
        native_swage,
        "_compile_persistent_segmented_reduction_ptx",
        lambda *_args, **_kwargs: ("", "ptx"),
    )
    driver = _Driver()
    monkeypatch.setattr(_runtime, "_get_driver", lambda: driver)
    values = torch.ones(33, device="cuda")
    offsets = torch.tensor([0, 33], device="cuda", dtype=torch.int32)
    output = torch.empty(1, device="cuda")
    prepared = _prepare_persistent_sum(values, offsets, output)

    with pytest.raises(RuntimeError, match="persistent launch failed"):
        prepared.launch()
    assert driver.launches == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_persistent_sum_rejects_a_different_current_device(monkeypatch):
    """Reject device drift before resetting counters or launching workers."""
    from mlir_swage._mlir_libs._swageDialectsNanobind import (
        swage as native_swage,
    )
    from swage import _runtime

    class _Driver:
        def __init__(self):
            self.launches = []

        def load(self, _ptx, _kernel_name):
            return 1, 1

        def launch_persistent(self, *arguments):
            self.launches.append(arguments)

    monkeypatch.setattr(
        native_swage,
        "_compile_persistent_segmented_reduction_ptx",
        lambda *_args, **_kwargs: ("", "ptx"),
    )
    driver = _Driver()
    monkeypatch.setattr(_runtime, "_get_driver", lambda: driver)
    values = torch.ones(33, device="cuda")
    offsets = torch.tensor([0, 33], device="cuda", dtype=torch.int32)
    output = torch.empty(1, device="cuda")
    prepared = _prepare_persistent_sum(values, offsets, output)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 999)

    with pytest.raises(ValueError, match="prepared device"):
        prepared.launch()
    assert not driver.launches


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_persistent_sum_retains_queue_and_dependency_storage(monkeypatch):
    """Keep every private device allocation alive across submissions."""
    from mlir_swage._mlir_libs._swageDialectsNanobind import (
        swage as native_swage,
    )
    from swage import _runtime

    class _Driver:
        def load(self, _ptx, _kernel_name):
            return 1, 1

        def launch_persistent(self, *_arguments):
            return None

    values = torch.ones(4098, device="cuda")
    offsets = torch.tensor([0, 1, 4098], device="cuda", dtype=torch.int32)
    output = torch.empty(2, device="cuda")
    monkeypatch.setattr(
        native_swage,
        "_compile_persistent_segmented_reduction_ptx",
        lambda *_args, **_kwargs: ("", "ptx"),
    )
    monkeypatch.setattr(_runtime, "_get_driver", _Driver)
    original_empty = torch.empty
    original_tensor = torch.tensor
    original_zeros = torch.zeros
    references = []

    def retain(result):
        references.append(weakref.ref(result))
        return result

    monkeypatch.setattr(
        torch,
        "tensor",
        lambda *args, **kwargs: retain(original_tensor(*args, **kwargs)),
    )
    monkeypatch.setattr(
        torch,
        "empty",
        lambda *args, **kwargs: retain(original_empty(*args, **kwargs)),
    )
    monkeypatch.setattr(
        torch,
        "zeros",
        lambda *args, **kwargs: retain(original_zeros(*args, **kwargs)),
    )

    prepared = _prepare_persistent_sum(values, offsets, output)
    gc.collect()
    assert references
    assert all(reference() is not None for reference in references)

    prepared.launch()
    gc.collect()
    assert all(reference() is not None for reference in references)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize(
    "lengths",
    [
        pytest.param([4097], id="split-only"),
        pytest.param([8192, 8193], id="multiple-split"),
        pytest.param([0, 33, 4097, 1, 8193], id="mixed"),
    ],
)
def test_prepared_split_sum_matches_nontrivial_oracles(lengths):
    """Match nontrivial f32 split sums with PyTorch and the CPU oracle."""
    host_values, host_offsets = _case(lengths)
    values = host_values.cuda()
    offsets = host_offsets.cuda()
    output = torch.full((len(lengths),), float("nan"), device="cuda")

    _prepare_planned_sum(values, offsets, output).mixed()

    expected = _pytorch_reference(host_values, host_offsets, "sum")
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(
        output.cpu(), cpu_oracle(host_values, host_offsets, "sum"),
        rtol=0, atol=0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_prepared_sum_rejects_output_aliases_before_driver_work(monkeypatch):
    """Reject either input alias before compilation or driver access."""
    from swage import _runtime

    monkeypatch.setattr(
        _runtime,
        "_get_driver",
        lambda: pytest.fail("driver must not be accessed"),
    )
    values = torch.ones(2, device="cuda", dtype=torch.float32)
    offsets = torch.tensor([0, 1, 2], device="cuda", dtype=torch.int32)

    with pytest.raises(ValueError, match="values buffer"):
        _prepare_planned_sum(values, offsets, values)

    offset_output = offsets[:2].view(torch.float32)
    with pytest.raises(ValueError, match="offsets buffer"):
        _prepare_planned_sum(values, offsets, offset_output)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("policy", ["warp", "cta", "mixed"])
def test_prepared_sum_uses_current_non_default_stream(policy):
    """Launch every prepared policy on PyTorch's current stream."""
    lengths = [0, 32, 33, 4097]
    host_values, host_offsets = _case(lengths)
    output = torch.full((len(lengths),), float("nan"), device="cuda")
    prepared = _prepare_planned_sum(
        host_values.cuda(), host_offsets.cuda(), output
    )
    stream = torch.cuda.Stream()

    with torch.cuda.stream(stream):
        getattr(prepared, policy)()
    stream.synchronize()

    torch.testing.assert_close(
        output.cpu(),
        _pytorch_reference(host_values, host_offsets, "sum"),
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("policy", ["warp", "cta", "mixed"])
def test_prepared_sum_supports_cuda_graph_replay(policy):
    """Capture prepared work after its immutable task IDs are ready."""
    lengths = [0, 32, 33, 4096]
    host_values, host_offsets = _case(lengths)
    output = torch.full((len(lengths),), float("nan"), device="cuda")
    prepared = _prepare_planned_sum(
        host_values.cuda(), host_offsets.cuda(), output
    )
    expected = _pytorch_reference(host_values, host_offsets, "sum")
    launch = getattr(prepared, policy)
    for _ in range(2):
        output.fill_(float("nan"))
        launch()
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    output.fill_(float("nan"))
    graph.replay()

    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("policy", ["warp", "cta", "mixed"])
def test_prepared_sum_waits_for_task_initialization_across_streams(
    monkeypatch, policy
):
    """Order asynchronous task initialization before another stream launches."""
    from swage import _runtime

    preparation_stream = torch.cuda.Stream()
    launch_stream = torch.cuda.Stream()
    launch_observed = torch.cuda.Event()
    launch_completed = threading.Event()

    class _Driver:
        def load(self, _ptx, _kernel_name):
            return 1, 1

        def launch_segmented_tasks(self, *_arguments):
            launch_observed.record(launch_stream)

        def launch_segmented_mixed(self, *_arguments):
            launch_observed.record(launch_stream)

    values = torch.ones(34, device="cuda", dtype=torch.float32)
    offsets = torch.tensor([0, 1, 34], device="cuda", dtype=torch.int32)
    output = torch.empty(2, device="cuda", dtype=torch.float32)
    all_task_storage = torch.empty(2, device="cuda", dtype=torch.int32)
    mixed_task_storage = torch.empty(2, device="cuda", dtype=torch.int32)
    torch.cuda.synchronize()
    monkeypatch.setattr(_runtime, "_get_driver", _Driver)

    def asynchronous_arange(*_args, **_kwargs):
        torch.cuda._sleep(2_000_000_000)
        return all_task_storage

    def asynchronous_tensor(_data, *, dtype, device):
        assert dtype == torch.int32
        assert device == offsets.device
        return mixed_task_storage

    monkeypatch.setattr(torch, "arange", asynchronous_arange)
    monkeypatch.setattr(torch, "tensor", asynchronous_tensor)
    with torch.cuda.stream(preparation_stream):
        prepared = _prepare_planned_sum(values, offsets, output)
    with torch.cuda.stream(launch_stream):
        getattr(prepared, policy)()

    def observe_launch():
        launch_observed.synchronize()
        launch_completed.set()

    observer = threading.Thread(target=observe_launch)
    observer.start()
    bypassed_initialization = launch_completed.wait(timeout=0.5)
    preparation_stream.synchronize()
    launch_stream.synchronize()
    observer.join()

    assert not bypassed_initialization


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize(
    ("lengths", "expected_ids", "expected_counts", "expected_grid"),
    [
        ([33, 1], [1, 0], (1, 1), (2,)),
        ([1, 2, 3, 4, 33], [0, 1, 2, 3, 4], (4, 1), (2,)),
        ([1, 2, 3, 4, 5, 33], [0, 1, 2, 3, 4, 5], (5, 1), (3,)),
        ([0, 32], [0, 1], (2, 0), (1,)),
        ([33, 34], [0, 1], (0, 2), (2,)),
    ],
)
def test_prepared_mixed_uses_one_ordered_fused_launch(
    monkeypatch, lengths, expected_ids, expected_counts, expected_grid
):
    """Submit stable warp IDs before CTA IDs through one fused kernel."""
    from swage import _runtime

    class _Driver:
        def __init__(self):
            self.loads = []
            self.launches = []

        def load(self, ptx, kernel_name):
            self.loads.append((ptx, kernel_name))
            function = len(self.loads)
            return function + 100, function

        def launch_segmented_tasks(
            self, function, grid, block, stream, arguments
        ):
            self.launches.append(
                ("pure", function, grid, block, stream, arguments)
            )

        def launch_segmented_mixed(
            self, function, grid, block, stream, arguments
        ):
            self.launches.append(
                ("mixed", function, grid, block, stream, arguments)
            )

    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    values = torch.ones(offsets[-1], device="cuda", dtype=torch.float32)
    device_offsets = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    output = torch.empty(len(lengths), device="cuda")
    driver = _Driver()
    monkeypatch.setattr(_runtime, "_get_driver", lambda: driver)
    original_tensor = torch.tensor
    task_tensors = []

    def capture_tensor(data, *args, **kwargs):
        tensor = original_tensor(data, *args, **kwargs)
        if kwargs.get("dtype") == torch.int32 and tensor.device.type == "cuda":
            task_tensors.append((list(data), tensor))
        return tensor

    monkeypatch.setattr(torch, "tensor", capture_tensor)

    prepared = _prepare_planned_sum(values, device_offsets, output)
    prepared.mixed()

    assert len(driver.launches) == 1
    assert len(task_tensors) == 1
    kind, _, grid, block, _, arguments = driver.launches[0]
    mixed_ids, mixed_tasks = task_tensors[-1]
    assert kind == "mixed"
    assert grid == expected_grid
    assert block == 128
    assert mixed_ids == expected_ids
    assert arguments[3] == mixed_tasks.data_ptr()
    assert arguments[5:] == expected_counts


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_prepared_mixed_orders_direct_partial_and_merge_phases(monkeypatch):
    """Submit direct work before every partial and every final merge."""
    from swage import _runtime

    class _Driver:
        def __init__(self):
            self.loads = []
            self.launches = []

        def load(self, ptx, kernel_name):
            self.loads.append((ptx, kernel_name))
            return 100 + len(self.loads), len(self.loads)

        def launch_segmented_mixed(
            self, function, grid, block, stream, arguments
        ):
            self.launches.append(
                ("direct", function, grid, block, stream, arguments)
            )

        def launch_segmented(
            self, function, grid, block, stream, arguments
        ):
            self.launches.append(
                ("split", function, grid, block, stream, arguments)
            )

    lengths = [1, 33, 4097, 8192, 0]
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    values = torch.ones(offsets[-1], device="cuda")
    device_offsets = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    output = torch.empty(len(lengths), device="cuda")
    driver = _Driver()
    monkeypatch.setattr(_runtime, "_get_driver", lambda: driver)
    original_tensor = torch.tensor
    descriptor_tensors = []

    def capture_tensor(data, *args, **kwargs):
        tensor = original_tensor(data, *args, **kwargs)
        if kwargs.get("dtype") == torch.int32 and tensor.device.type == "cuda":
            descriptor_tensors.append((list(data), tensor))
        return tensor

    monkeypatch.setattr(torch, "tensor", capture_tensor)

    prepared = _prepare_planned_sum(values, device_offsets, output)
    prepared.mixed()

    assert [launch[0] for launch in driver.launches] == [
        "direct",
        "split",
        "split",
    ]
    direct, partial, merge = driver.launches
    assert direct[2:4] == ((2,), 128)
    assert direct[5][5:] == (2, 1)
    assert partial[2:4] == ((4,), 512)
    assert partial[5][3:] == (offsets[-1], 4)
    assert merge[2:4] == ((2,), 512)
    assert merge[5][3:] == (4, 2)
    assert [data for data, _ in descriptor_tensors] == [
        [0, 4, 1],
        [34, 4130, 4130, 4131, 4131, 8227, 8227, 12323],
        [2, 0, 2, 3, 2, 4],
    ]
    assert direct[5][3] == descriptor_tensors[0][1].data_ptr()
    assert partial[5][1] == descriptor_tensors[1][1].data_ptr()
    assert merge[5][2] == descriptor_tensors[2][1].data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_split_only_mixed_skips_the_direct_phase(monkeypatch):
    """Do not compile, allocate, or launch an empty direct phase."""
    from mlir_swage._mlir_libs._swageDialectsNanobind import (
        swage as native_swage,
    )
    from swage import _runtime

    class _Driver:
        def __init__(self):
            self.launches = []

        def load(self, _ptx, _kernel_name):
            return 1, 1

        def launch_segmented(self, _function, grid, _block, _stream, _args):
            self.launches.append(grid)

    def reject_direct(*_args, **_kwargs):
        pytest.fail("split-only work must not compile a direct kernel")

    monkeypatch.setattr(
        native_swage, "_compile_fused_segmented_reduction_ptx", reject_direct
    )
    driver = _Driver()
    monkeypatch.setattr(_runtime, "_get_driver", lambda: driver)
    values = torch.ones(4097, device="cuda")
    offsets = torch.tensor([0, 4097], device="cuda", dtype=torch.int32)
    output = torch.empty(1, device="cuda")

    _prepare_planned_sum(values, offsets, output).mixed()

    assert driver.launches == [(2,), (1,)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize(
    "materialized",
    [
        ([0], [], [0, 4096, 4096, 4097], [0, 0, 2]),
        ([], [], [0, 4096, 4095, 4097], [0, 0, 2]),
        ([], [], [0, 4096, 4096, 4097], [1, 0, 2]),
        ([], [], [0, 4096], [0, 0, 1]),
    ],
)
def test_prepared_sum_rejects_mismatched_materialized_plan_before_work(
    monkeypatch, materialized
):
    """Reject duplicate, overlapping, misassigned, or omitted split work."""
    from mlir_swage._mlir_libs._swageDialectsNanobind import (
        swage as native_swage,
    )
    from swage import _runtime

    def fail(*_args, **_kwargs):
        pytest.fail("malformed split work must not continue")

    monkeypatch.setattr(
        native_swage,
        "_materialize_segmented_plan",
        lambda *_args, **_kwargs: materialized,
    )
    monkeypatch.setattr(
        native_swage, "_compile_segmented_reduction_ptx", fail
    )
    monkeypatch.setattr(torch, "arange", fail)
    monkeypatch.setattr(_runtime, "_get_driver", fail)
    values = torch.ones(4097, device="cuda")
    offsets = torch.tensor([0, 4097], device="cuda", dtype=torch.int32)
    output = torch.empty(1, device="cuda")

    with pytest.raises(
        RuntimeError, match="materialized plan does not match"
    ):
        _prepare_planned_sum(values, offsets, output)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_prepared_sum_rejects_invalid_limits_before_work(monkeypatch):
    """Classify invalid limits before compilation, allocation, or driver use."""
    from mlir_swage._mlir_libs._swageDialectsNanobind import (
        swage as native_swage,
    )
    from swage import _runtime

    def fail(*_args, **_kwargs):
        pytest.fail("invalid limits must not continue")

    monkeypatch.setattr(
        native_swage, "_compile_segmented_reduction_ptx", fail
    )
    monkeypatch.setattr(torch, "arange", fail)
    monkeypatch.setattr(_runtime, "_get_driver", fail)
    values = torch.ones(33, device="cuda")
    offsets = torch.tensor([0, 33], device="cuda", dtype=torch.int32)
    output = torch.empty(1, device="cuda")

    with pytest.raises(ValueError, match="planning limits must satisfy"):
        _prepare_planned_sum(
            values,
            offsets,
            output,
            warp_max_elements=32,
            cta_chunk_elements=31,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_split_compile_failure_precedes_allocation_and_driver(monkeypatch):
    """Surface split compilation failure without allocating runtime state."""
    from mlir_swage._mlir_libs._swageDialectsNanobind import (
        swage as native_swage,
    )
    from swage import _runtime

    def fail(*_args, **_kwargs):
        raise RuntimeError("partial compile failed")

    monkeypatch.setattr(
        native_swage,
        "_compile_segmented_reduction_ptx",
        lambda *_args, **_kwargs: ("", "ptx"),
    )
    monkeypatch.setattr(
        native_swage, "_compile_split_partial_reduction_ptx", fail
    )
    monkeypatch.setattr(
        torch,
        "arange",
        lambda *_args, **_kwargs: pytest.fail("must not allocate"),
    )
    monkeypatch.setattr(
        _runtime,
        "_get_driver",
        lambda: pytest.fail("must not access driver"),
    )
    values = torch.ones(4097, device="cuda")
    offsets = torch.tensor([0, 4097], device="cuda", dtype=torch.int32)
    output = torch.empty(1, device="cuda")

    with pytest.raises(RuntimeError, match="partial compile failed"):
        _prepare_planned_sum(values, offsets, output)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_split_allocation_failure_precedes_launch(monkeypatch):
    """Surface descriptor allocation failure without dispatching a kernel."""
    from mlir_swage._mlir_libs._swageDialectsNanobind import (
        swage as native_swage,
    )
    from swage import _runtime

    class _Driver:
        def __init__(self):
            self.launches = []

        def load(self, _ptx, _kernel_name):
            return 1, 1

        def launch_segmented(self, *arguments):
            self.launches.append(arguments)

    for name in (
        "_compile_segmented_reduction_ptx",
        "_compile_split_partial_reduction_ptx",
        "_compile_split_merge_reduction_ptx",
    ):
        monkeypatch.setattr(
            native_swage, name, lambda *_args, **_kwargs: ("", "ptx")
        )
    driver = _Driver()
    monkeypatch.setattr(_runtime, "_get_driver", lambda: driver)
    monkeypatch.setattr(
        torch,
        "arange",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            MemoryError("task allocation failed")
        ),
    )
    values = torch.ones(4097, device="cuda")
    offsets = torch.tensor([0, 4097], device="cuda", dtype=torch.int32)
    output = torch.empty(1, device="cuda")

    with pytest.raises(MemoryError, match="task allocation failed"):
        _prepare_planned_sum(values, offsets, output)
    assert not driver.launches


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize(
    ("failed_phase", "expected"),
    [
        ("direct", ["direct"]),
        ("partial", ["direct", "partial"]),
        ("merge", ["direct", "partial", "merge"]),
    ],
)
def test_split_launch_failures_stop_later_phases(
    monkeypatch, failed_phase, expected
):
    """Stop the ordered stream sequence at its first launch failure."""
    from mlir_swage._mlir_libs._swageDialectsNanobind import (
        swage as native_swage,
    )
    from swage import _runtime

    class _Driver:
        def __init__(self):
            self.phases = []
            self.split_launches = 0

        def load(self, _ptx, kernel_name):
            return 1, kernel_name

        def observe(self, phase):
            self.phases.append(phase)
            if phase == failed_phase:
                raise RuntimeError(f"{phase} launch failed")

        def launch_segmented_mixed(self, *_arguments):
            self.observe("direct")

        def launch_segmented(self, *_arguments):
            self.split_launches += 1
            self.observe("partial" if self.split_launches == 1 else "merge")

    for name in (
        "_compile_segmented_reduction_ptx",
        "_compile_fused_segmented_reduction_ptx",
        "_compile_split_partial_reduction_ptx",
        "_compile_split_merge_reduction_ptx",
    ):
        monkeypatch.setattr(
            native_swage, name, lambda *_args, **_kwargs: ("", "ptx")
        )
    driver = _Driver()
    monkeypatch.setattr(_runtime, "_get_driver", lambda: driver)
    values = torch.ones(4098, device="cuda")
    offsets = torch.tensor([0, 1, 4098], device="cuda", dtype=torch.int32)
    output = torch.empty(2, device="cuda")
    prepared = _prepare_planned_sum(values, offsets, output)

    with pytest.raises(RuntimeError, match=f"{failed_phase} launch failed"):
        prepared.mixed()
    assert driver.phases == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_prepared_sum_retains_private_split_storage(monkeypatch):
    """Keep task descriptors and scratch alive for every prepared launch."""
    from mlir_swage._mlir_libs._swageDialectsNanobind import (
        swage as native_swage,
    )
    from swage import _runtime

    class _Driver:
        def load(self, _ptx, _kernel_name):
            return 1, 1

        def launch_segmented_mixed(self, *_arguments):
            return None

        def launch_segmented(self, *_arguments):
            return None

    for name in (
        "_compile_segmented_reduction_ptx",
        "_compile_fused_segmented_reduction_ptx",
        "_compile_split_partial_reduction_ptx",
        "_compile_split_merge_reduction_ptx",
    ):
        monkeypatch.setattr(
            native_swage, name, lambda *_args, **_kwargs: ("", "ptx")
        )
    monkeypatch.setattr(_runtime, "_get_driver", _Driver)
    original_arange = torch.arange
    original_empty = torch.empty
    original_tensor = torch.tensor
    references = []

    def retain(result):
        references.append(weakref.ref(result))
        return result

    monkeypatch.setattr(
        torch,
        "arange",
        lambda *args, **kwargs: retain(original_arange(*args, **kwargs)),
    )
    monkeypatch.setattr(
        torch,
        "tensor",
        lambda *args, **kwargs: retain(original_tensor(*args, **kwargs)),
    )
    monkeypatch.setattr(
        torch,
        "empty",
        lambda *args, **kwargs: retain(original_empty(*args, **kwargs)),
    )
    values = torch.ones(4098, device="cuda")
    offsets = original_tensor([0, 1, 4098], device="cuda", dtype=torch.int32)
    output = original_empty(2, device="cuda")

    prepared = _prepare_planned_sum(values, offsets, output)
    gc.collect()
    assert references
    assert all(reference() is not None for reference in references)

    prepared.mixed()
    gc.collect()
    assert all(reference() is not None for reference in references)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_empty_prepared_sum_does_not_compile_allocate_or_launch(monkeypatch):
    """Return three no-ops after classification when there are no segments."""
    from mlir_swage._mlir_libs._swageDialectsNanobind import (
        swage as native_swage,
    )
    from swage import _runtime

    def fail(*_args, **_kwargs):
        pytest.fail("empty work must not compile, allocate, or access driver")

    monkeypatch.setattr(
        native_swage, "_compile_segmented_reduction_ptx", fail
    )
    monkeypatch.setattr(
        native_swage, "_compile_fused_segmented_reduction_ptx", fail
    )
    monkeypatch.setattr(torch, "arange", fail)
    monkeypatch.setattr(torch, "tensor", fail)
    monkeypatch.setattr(_runtime, "_get_driver", fail)
    values = torch.empty(0, device="cuda", dtype=torch.float32)
    offsets = torch.empty(1, device="cuda", dtype=torch.int32)
    offsets.zero_()
    output = torch.empty(0, device="cuda", dtype=torch.float32)

    prepared = _prepare_planned_sum(values, offsets, output)

    assert prepared.warp() is None
    assert prepared.cta() is None
    assert prepared.mixed() is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_prepared_sum_rejects_a_different_current_device(monkeypatch):
    """Do not dispatch a prepared policy after the current device changes."""
    from swage import _runtime

    class _Driver:
        def __init__(self):
            self.launches = []

        def load(self, _ptx, _kernel_name):
            return 1, 1

        def launch_segmented_tasks(self, *arguments):
            self.launches.append(arguments)

    values = torch.ones(34, device="cuda", dtype=torch.float32)
    offsets = torch.tensor([0, 1, 34], device="cuda", dtype=torch.int32)
    output = torch.empty(2, device="cuda")
    driver = _Driver()
    monkeypatch.setattr(_runtime, "_get_driver", lambda: driver)
    prepared = _prepare_planned_sum(values, offsets, output)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 999)

    with pytest.raises(ValueError, match="prepared device"):
        prepared.mixed()
    assert not driver.launches


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_prepared_mixed_sum_is_repeatable():
    """Run the same fused launch twice without stale warp or CTA state."""
    lengths = [0, 32, 33, 0, 1, 4097]
    host_values, host_offsets = _case(lengths)
    output = torch.empty(len(lengths), device="cuda", dtype=torch.float32)
    prepared = _prepare_planned_sum(
        host_values.cuda(), host_offsets.cuda(), output
    )
    expected = _pytorch_reference(host_values, host_offsets, "sum")

    for _ in range(2):
        output.fill_(float("nan"))
        prepared.mixed()
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("kind", ["sum", "max"])
def test_gpu_reduction_is_repeatable(kind):
    """Run the same loaded shape twice without stale CTA state."""
    host_values, host_offsets = _case([0, 2, 0, 129])
    values = host_values.cuda()
    offsets = host_offsets.cuda()
    output = torch.empty(4, device="cuda")

    launch_gpu(values, offsets, output, kind)
    first = output.clone()
    output.fill_(float("nan"))
    launch_gpu(values, offsets, output, kind)

    torch.testing.assert_close(output, first, rtol=0, atol=0)


def test_cpu_max_propagates_nan_and_uses_negative_infinity_identity():
    """Make max NaN and empty semantics explicit in the CPU oracle."""
    values = torch.tensor([1.0, float("nan"), 3.0, 4.0, 5.0])
    offsets = torch.tensor([0, 3, 3, 5], dtype=torch.int32)

    actual = cpu_oracle(values, offsets, "max")

    torch.testing.assert_close(
        actual,
        torch.tensor([float("nan"), float("-inf"), 5.0]),
        rtol=0,
        atol=0,
        equal_nan=True,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_gpu_max_propagates_nan_and_uses_negative_infinity_identity():
    """Match the specified max semantics through CTA reduction."""
    host_values = torch.tensor([1.0, float("nan"), 3.0, 4.0, 5.0])
    host_offsets = torch.tensor([0, 3, 3, 5], dtype=torch.int32)
    values = host_values.cuda()
    offsets = host_offsets.cuda()
    output = torch.empty(3, device="cuda")

    launch_gpu(values, offsets, output, "max")

    expected = torch.tensor([float("nan"), float("-inf"), 5.0])
    torch.testing.assert_close(
        output.cpu(), expected, rtol=0, atol=0, equal_nan=True
    )
    torch.testing.assert_close(
        output.cpu(),
        cpu_oracle(host_values, host_offsets, "max"),
        rtol=0,
        atol=0,
        equal_nan=True,
    )


# Softmax tolerances, measured on an RTX A6000 at sm_86.
#
# The dominant term is the f32 rounding of `x * 1.44269502`, whose relative
# effect on exp2 is about 6e-08 per unit of intra-segment spread. It is
# identical on both backends, so it does not cancel against PyTorch. Every
# distribution below except one-outlier caps its spread at 4, and _GPU_RTOL
# is sized for a cap of 8. Widening a segment past that needs a larger
# constant, so the fix for a failure just above rtol is the distribution.
_GPU_RTOL, _GPU_ATOL = 2e-6, 1e-7

# Anything compared against cpu_softmax_oracle carries a hard 5e-06 relative
# floor, because the oracle parses printMemrefF32's six-significant-digit
# text. That floor is the transport, not the arithmetic.
_ORACLE_RTOL, _ORACLE_ATOL = 1e-5, 1e-6


def _softmax_case(lengths, outlier=None):
    """Build a softmax case, optionally planting one dominant value.

    Softmax keeps its own logits, quarter steps from -2 to 2 with period
    17, instead of the reduction pattern. The tolerances below are sized
    by intra-segment spread, and every output element is compared, so the
    position of each value is already observable here.
    """
    offsets = _offsets(lengths)
    index = torch.arange(offsets[-1])
    values = ((index % 17) - 8).to(torch.float32) / 4
    if outlier is not None:
        values[offsets[1]] = outlier
    return values, torch.tensor(offsets, dtype=torch.int32)


SOFTMAX_CASES = [
    pytest.param([0] * 8, None, id="all-empty"),
    pytest.param([1] * 64, None, id="all-ones"),
    pytest.param([1, 2, 3, 4] * 24, None, id="many-tiny"),
    pytest.param([4096, 3, 2731], None, id="few-huge"),
    pytest.param([1, 127, 640], 100.0, id="one-outlier"),
    pytest.param([0, 5, 0, 7, 0, 3, 0, 1, 0], None, id="alternating-empty"),
]


def _pytorch_softmax_reference(values, offsets):
    """Compute segmented softmax in float32, with the shift written out.

    The maximum shift is the thing under test, so it is explicit rather than
    delegated to torch.softmax. Empty segments contribute nothing, so the
    result is a concatenation of length offsets[-1].
    """
    pieces = []
    for index in range(len(offsets) - 1):
        segment = values[offsets[index] : offsets[index + 1]]
        if not segment.numel():
            continue
        shifted = segment - segment.max()
        exponentials = shifted.exp()
        pieces.append(exponentials / exponentials.sum())
    if not pieces:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(pieces)


@pytest.mark.parametrize(("lengths", "outlier"), SOFTMAX_CASES)
def test_cpu_softmax_matches_pytorch(lengths, outlier):
    """Execute the sequential softmax through upstream mlir-runner."""
    values, offsets = _softmax_case(lengths, outlier)

    actual = cpu_softmax_oracle(values, offsets)
    expected = _pytorch_softmax_reference(values, offsets)

    torch.testing.assert_close(
        actual, expected, rtol=_ORACLE_RTOL, atol=_ORACLE_ATOL
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize(("lengths", "outlier"), SOFTMAX_CASES)
def test_gpu_softmax_matches_pytorch_and_cpu_oracle(lengths, outlier):
    """Qualify the three-phase kernel against both independent references."""
    host_values, host_offsets = _softmax_case(lengths, outlier)
    covered = int(host_offsets[-1])
    values = host_values.cuda()
    offsets = host_offsets.cuda()
    output = torch.empty(covered, device="cuda")

    launch_softmax_gpu(values, offsets, output)
    # A zero-length output enqueues no copy on .cpu(), so without this the
    # all-empty row would compare two empty tensors and observe nothing
    # about whether the launch even succeeded.
    torch.cuda.synchronize()

    expected = _pytorch_softmax_reference(host_values, host_offsets)
    torch.testing.assert_close(
        output.cpu(), expected, rtol=_GPU_RTOL, atol=_GPU_ATOL
    )
    torch.testing.assert_close(
        output.cpu(),
        cpu_softmax_oracle(host_values, host_offsets),
        rtol=_ORACLE_RTOL,
        atol=_ORACLE_ATOL,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_gpu_softmax_is_repeatable():
    """Run the same loaded shape twice without stale CTA state."""
    host_values, host_offsets = _softmax_case([0, 2, 0, 129])
    values = host_values.cuda()
    offsets = host_offsets.cuda()
    output = torch.empty(int(host_offsets[-1]), device="cuda")

    launch_softmax_gpu(values, offsets, output)
    first = output.clone()
    output.fill_(float("nan"))
    launch_softmax_gpu(values, offsets, output)

    torch.testing.assert_close(output, first, rtol=0, atol=0)


def test_cpu_softmax_of_singleton_is_exactly_one():
    """A one-element segment normalizes to 1.0 within the text transport.

    The assertion uses no tolerance, but the value reaches it through
    printMemrefF32's six significant digits, so its real strictness is the
    5e-06 floor documented above and not bit equality. That is still about
    twice as tight as _ORACLE_RTOL. The bit-exactness claim belongs to the
    GPU twin, which compares device memory directly.
    """
    values, offsets = _softmax_case([1, 1, 1])

    actual = cpu_softmax_oracle(values, offsets)

    torch.testing.assert_close(actual, torch.ones(3), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_gpu_softmax_of_singleton_is_exactly_one():
    """The recompute schedule makes the singleton quotient exact.

    map_store repeats the identical subtract, multiply, and exp2 sequence
    the reduce used, so the sum equals the exponential bit for bit and the
    quotient is exactly 1.0 regardless of ex2.approx's accuracy. This is the
    test that trips if the two region clones ever lower differently, which
    is the only way the recompute can stop agreeing with itself.
    """
    host_values, host_offsets = _softmax_case([1, 1, 1])
    output = torch.empty(3, device="cuda")

    launch_softmax_gpu(host_values.cuda(), host_offsets.cuda(), output)

    torch.testing.assert_close(output.cpu(), torch.ones(3), rtol=0, atol=0)


def _semantic_edge_case():
    """Three segments: a NaN, all negative infinity, and a finite maximum."""
    values = torch.tensor(
        [float("nan"), 1.0, float("-inf"), float("-inf"), float("-inf"), 0.0],
        dtype=torch.float32,
    )
    offsets = torch.tensor([0, 2, 4, 6], dtype=torch.int32)
    # A NaN propagates through maximumf and ex2. A segment that is entirely
    # negative infinity has a negative-infinity maximum, so every difference
    # is NaN. A segment where negative infinity sits under a finite maximum
    # is well defined, because exp2 of negative infinity is exactly zero.
    expected = torch.tensor(
        [float("nan"), float("nan"), float("nan"), float("nan"), 0.0, 1.0],
        dtype=torch.float32,
    )
    return values, offsets, expected


def test_cpu_softmax_propagates_nan_and_is_nan_for_all_negative_infinity():
    """Pin the three float edge cases on the sequential lowering."""
    values, offsets, expected = _semantic_edge_case()

    actual = cpu_softmax_oracle(values, offsets)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_gpu_softmax_propagates_nan_and_is_nan_for_all_negative_infinity():
    """Pin the same three edge cases on the one-CTA kernel."""
    host_values, host_offsets, expected = _semantic_edge_case()
    output = torch.empty(6, device="cuda")

    launch_softmax_gpu(host_values.cuda(), host_offsets.cuda(), output)

    torch.testing.assert_close(
        output.cpu(), expected, rtol=0, atol=0, equal_nan=True
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_gpu_softmax_leaves_output_beyond_final_offset_untouched():
    """An empty CTA that stored would write a neighbour's first slot."""
    host_values, host_offsets = _softmax_case([0, 5, 0, 7, 0])
    covered = int(host_offsets[-1])
    output = torch.full((covered + 8,), -1.0, device="cuda")

    launch_softmax_gpu(host_values.cuda(), host_offsets.cuda(), output)

    torch.testing.assert_close(
        output[covered:].cpu(), torch.full((8,), -1.0), rtol=0, atol=0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_rejects_softmax_output_shorter_than_final_offset():
    """The softmax ABI needs one output element per covered value."""
    host_values, host_offsets = _softmax_case([3, 4])
    output = torch.empty(6, device="cuda")

    with pytest.raises(ValueError, match="output has 6 elements for 7 values"):
        launch_softmax_gpu(host_values.cuda(), host_offsets.cuda(), output)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_rejects_softmax_output_aliasing_values():
    """map_store's no-alias obligation is enforced before any launch."""
    host_values, host_offsets = _softmax_case([4, 4])
    values = host_values.cuda()

    with pytest.raises(ValueError, match="must not overlap the values buffer"):
        launch_softmax_gpu(values, host_offsets.cuda(), values.narrow(0, 0, 8))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_rejects_softmax_output_aliasing_offsets():
    """The kernel re-reads offsets, so an aliased output voids validation."""
    host_values, host_offsets = _softmax_case([4, 4])
    count = host_offsets.numel()
    shared = torch.empty(16, dtype=torch.int32, device="cuda")
    shared[:count] = host_offsets.cuda()
    offsets = shared.narrow(0, 0, count)
    output = shared.view(torch.float32).narrow(0, 0, 8)

    with pytest.raises(ValueError, match="must not overlap the offsets buffer"):
        launch_softmax_gpu(host_values.cuda(), offsets, output)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_mismatched_blockdim_fails_at_launch_instead_of_wrong_sums():
    """Turn a launch-geometry mismatch into a driver error via reqntid."""
    from mlir_swage import ir
    from mlir_swage._mlir_libs._swageDialectsNanobind import (
        swage as native_swage,
    )
    from mlir_swage.dialects import swage as swage_dialect
    from swage import _runtime
    from swage._segmented_qualification import _semantic_module

    values = torch.ones(256, device="cuda")
    offsets = torch.tensor([0, 256], dtype=torch.int32, device="cuda")
    output = torch.zeros(1, device="cuda")
    major, minor = torch.cuda.get_device_capability()
    with ir.Context() as context:
        swage_dialect.register_dialects(context)
        module = ir.Module.parse(_semantic_module("sum"))
        _, ptx = native_swage._compile_segmented_reduction_ptx(
            module,
            kernel_name="segmented_sum",
            block_size=128,
            target=f"sm_{major}{minor}",
        )

    driver = _runtime._get_driver()
    _, function = driver.load(ptx, "segmented_sum")
    stream = torch.cuda.current_stream()
    arguments = (
        values.data_ptr(),
        offsets.data_ptr(),
        output.data_ptr(),
        256,
        1,
    )

    with pytest.raises(RuntimeError, match="cuLaunchKernel failed"):
        driver.launch_segmented(
            function, (1,), 64, stream.cuda_stream, arguments
        )

    driver.launch_segmented(
        function, (1,), 128, stream.cuda_stream, arguments
    )
    torch.cuda.synchronize()
    assert output.item() == 256.0
