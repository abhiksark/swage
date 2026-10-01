# python/tests/mlir/test_segmented_classification.py
"""Host validation and classification without per-element Python work.

Preparing a layout copies the offsets to the host once, validates them with
array operations, and classifies them in native code. Preparation does not
derive the plan a second time in Python. The Python derivation lives here as
a reference, and a seeded property test compares the native classifier with
it. The scalar validation loop is kept here in the same way, as the reference
for the exception types and messages of the array validation.
"""

import random
from itertools import accumulate, pairwise

import numpy
import pytest
import torch
from mlir_swage import ir
from mlir_swage._mlir_libs._swageDialectsNanobind import swage as native_swage
from mlir_swage.dialects import swage
from swage import _segmented_qualification as qualification

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"
)

_I32_MIN = -(1 << 31)
_I32_LIMIT = 1 << 31
_LIMITS = [(32, 4096), (1, 1), (7, 100), (64, 64)]
_SEEDS = range(20)


def _reference_plan(offsets, warp_max_elements=32, cta_chunk_elements=4096):
    """Classify one layout in plain Python, one segment at a time."""
    warp, cta, partial, merge = [], [], [], []
    for segment_id, (begin, end) in enumerate(pairwise(offsets)):
        length = end - begin
        if length <= warp_max_elements:
            warp.append(segment_id)
        elif length <= cta_chunk_elements:
            cta.append(segment_id)
        else:
            partial_begin = len(partial) // 2
            for chunk_begin in range(begin, end, cta_chunk_elements):
                chunk_end = min(end, chunk_begin + cta_chunk_elements)
                partial.extend([chunk_begin, chunk_end])
            merge.extend([segment_id, partial_begin, len(partial) // 2])
    return warp, cta, partial, merge


def _reference_partial_merges(merge, partial_count):
    """Map every partial task to its merge, rejecting gaps and overlaps."""
    partial_merges = [-1] * partial_count
    for merge_id, (begin, end) in enumerate(zip(merge[1::3], merge[2::3])):
        for partial_id in range(begin, end):
            assert partial_merges[partial_id] == -1, "partial has two merges"
            partial_merges[partial_id] = merge_id
    assert -1 not in partial_merges, "partial has no merge"
    return partial_merges


def _reference_validate(offsets, value_count):
    """Validate offsets one element at a time and return the segment count."""
    if not offsets:
        raise ValueError("offsets must contain at least the initial zero")
    if offsets[0] != 0:
        raise ValueError("offsets must start at zero")
    previous = 0
    for offset in offsets:
        if type(offset) is not int or not _I32_MIN <= offset < _I32_LIMIT:
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
    return len(offsets) - 1


def _random_lengths(seed, warp_max_elements, cta_chunk_elements):
    """Draw one layout that holds every classification boundary.

    Every layout has empty segments, the longest warp length and the length
    after it, the longest single-task length and the length after it, and one
    segment of several chunks, shuffled among random lengths of every class.
    """
    rng = random.Random(seed)
    chunk = cta_chunk_elements
    lengths = [
        0,
        0,
        warp_max_elements,
        warp_max_elements + 1,
        chunk,
        chunk + 1,
        rng.randint(3, 6) * chunk + rng.randint(0, chunk - 1),
    ]
    for _ in range(rng.randint(0, 200)):
        kind = rng.random()
        if kind < 0.15:
            lengths.append(0)
        elif kind < 0.6:
            lengths.append(rng.randint(1, warp_max_elements))
        elif kind < 0.9:
            lengths.append(rng.randint(warp_max_elements, chunk))
        else:
            lengths.append(rng.randint(chunk + 1, 4 * chunk + 1))
    rng.shuffle(lengths)
    return lengths


def _offsets(lengths):
    """Return the offsets of one list of segment lengths."""
    return [0, *accumulate(lengths)]


def _i32(numbers):
    """Return a host int32 array, the form preparation hands to native code."""
    return numpy.asarray(numbers, dtype=numpy.int32)


@pytest.fixture(scope="module")
def sum_module():
    """Parse the canonical sum once for every native classification here."""
    with ir.Context() as context:
        swage.register_dialects(context)
        yield ir.Module.parse(qualification._semantic_module("sum"))


def _native_plan(module, offsets, warp_max_elements, cta_chunk_elements):
    """Classify host offsets through the buffer form of the binding."""
    return native_swage._materialize_segmented_plan(
        module,
        offsets=_i32(offsets),
        value_count=offsets[-1],
        segment_count=len(offsets) - 1,
        warp_max_elements=warp_max_elements,
        cta_chunk_elements=cta_chunk_elements,
    )


def _matches_reference(plan, offsets, warp_max_elements, cta_chunk_elements):
    """Return whether four record arrays equal the reference plan."""
    expected = _reference_plan(offsets, warp_max_elements, cta_chunk_elements)
    return [records.tolist() for records in plan] == list(expected)


@pytest.mark.parametrize(("warp_max", "chunk"), _LIMITS)
@pytest.mark.parametrize("seed", _SEEDS)
def test_native_classification_matches_the_python_reference(
    sum_module, seed, warp_max, chunk
):
    """Compare native plans with the reference over seeded random layouts."""
    lengths = _random_lengths(seed, warp_max, chunk)
    offsets = _offsets(lengths)
    assert {0, warp_max, warp_max + 1, chunk, chunk + 1} <= set(lengths)
    assert max(lengths) >= 3 * chunk

    plan = _native_plan(sum_module, offsets, warp_max, chunk)

    assert _matches_reference(plan, offsets, warp_max, chunk)
    warp, cta, partial, merge = _reference_plan(offsets, warp_max, chunk)
    assert sorted([*warp, *cta, *merge[0::3]]) == list(range(len(lengths)))
    partial_merges = _reference_partial_merges(merge, len(partial) // 2)
    assert partial_merges == sorted(partial_merges)


@pytest.mark.parametrize(
    "malformed",
    [
        ([0], [], [0, 4096, 4096, 4097], [0, 0, 2]),
        ([], [], [0, 4096, 4095, 4097], [0, 0, 2]),
        ([], [], [0, 4096, 4096, 4097], [1, 0, 2]),
        ([], [], [0, 4096], [0, 0, 1]),
    ],
    ids=["duplicate", "overlapping", "misassigned", "omitted"],
)
def test_reference_comparison_rejects_malformed_split_work(malformed):
    """Keep the comparison able to see each kind of wrong split plan."""
    correct = ([], [], [0, 4096, 4096, 4097], [0, 0, 2])

    assert _matches_reference(map(_i32, correct), [0, 4097], 32, 4096)
    assert not _matches_reference(map(_i32, malformed), [0, 4097], 32, 4096)


def test_buffer_offsets_return_int32_record_arrays(sum_module):
    """Cross the native boundary as buffers in both directions."""
    plan = _native_plan(sum_module, [0, 32, 65, 4162, 12354], 32, 4096)

    assert all(isinstance(records, numpy.ndarray) for records in plan)
    assert all(records.dtype == numpy.int32 for records in plan)
    assert all(records.ndim == 1 for records in plan)
    assert [records.tolist() for records in plan] == [
        [0],
        [1],
        [65, 4161, 4161, 4162, 4162, 8258, 8258, 12354],
        [2, 0, 2, 3, 2, 4],
    ]


def test_record_arrays_outlive_the_call_that_returned_them(sum_module):
    """Own the native records, so a later call cannot overwrite them."""
    first = _native_plan(sum_module, _offsets([1] * 5000), 32, 4096)
    kept = first[0].copy()
    for _ in range(4):
        _native_plan(sum_module, _offsets([33] * 5000), 32, 4096)

    assert numpy.array_equal(first[0], kept)


def test_empty_layout_returns_four_empty_arrays(sum_module):
    """Classify a layout without segments before any early return."""
    plan = _native_plan(sum_module, [0], 32, 4096)

    assert [records.shape for records in plan] == [(0,)] * 4
    assert all(records.dtype == numpy.int32 for records in plan)


@pytest.mark.parametrize(
    "offsets",
    [
        numpy.asarray([0, 1], dtype=numpy.int64),
        numpy.asarray([0, 1], dtype=numpy.uint32),
        numpy.asarray([0.0, 1.0], dtype=numpy.float32),
        numpy.asarray([[0, 1]], dtype=numpy.int32),
        numpy.asarray([0, 9, 1, 9], dtype=numpy.int32)[::2],
        (0, 1),
    ],
    ids=["int64", "uint32", "float32", "rank-two", "strided", "tuple"],
)
def test_other_offset_buffers_are_rejected_without_conversion(
    sum_module, offsets
):
    """Refuse a buffer that would need a cast, a copy, or element access."""
    with pytest.raises(TypeError):
        native_swage._materialize_segmented_plan(
            sum_module, offsets=offsets, value_count=1, segment_count=1
        )


def test_buffer_offsets_keep_native_diagnostics(sum_module):
    """Report native classification failures as before, with their text."""
    with pytest.raises(ValueError, match="offsets must be nondecreasing"):
        native_swage._materialize_segmented_plan(
            sum_module, offsets=_i32([0, 2, 1]), value_count=2, segment_count=2
        )
    with pytest.raises(ValueError, match="planning limits must satisfy"):
        _native_plan(sum_module, [0, 1], 33, 32)


def _outcome(function, *arguments):
    """Return the result, or the type and message of the raised error."""
    try:
        return function(*arguments)
    except Exception as error:  # noqa: BLE001 (the comparison needs any type)
        return type(error), str(error)


def _random_offsets(seed):
    """Draw offsets that are valid or break one or more ordering rules."""
    rng = random.Random(seed)
    offsets = _offsets([rng.randint(0, 40) for _ in range(rng.randint(0, 30))])
    for _ in range(rng.choice([0, 0, 1, 1, 2, 3])):
        index = rng.randrange(len(offsets))
        changed = rng.choice(
            [
                -1,
                -rng.randint(1, 50),
                offsets[index] - rng.randint(1, 50),
                offsets[index] + rng.randint(1, 50),
                rng.randint(0, 50),
                _I32_MIN,
                _I32_LIMIT - 1,
            ]
        )
        offsets[index] = min(max(changed, _I32_MIN), _I32_LIMIT - 1)
    value_count = max(0, offsets[-1] + rng.choice([-3, -1, 0, 0, 1, 40]))
    return offsets, value_count


@pytest.mark.parametrize("seed", range(300))
def test_array_validation_matches_the_scalar_reference(seed):
    """Raise the same error for the same offsets, as an array or a list."""
    offsets, value_count = _random_offsets(seed)
    expected = _outcome(_reference_validate, offsets, value_count)

    assert (
        _outcome(
            qualification._validate_offset_sequence,
            _i32(offsets),
            value_count,
        )
        == expected
    )
    assert (
        _outcome(qualification._validate_offset_sequence, offsets, value_count)
        == expected
    )


@pytest.mark.parametrize(
    ("offsets", "value_count", "message"),
    [
        ([], 0, "offsets must contain at least the initial zero"),
        ([1, 1], 1, "offsets must start at zero"),
        ([0, -1], 1, "offsets must not be negative"),
        ([0, 5, -1, 3], 5, "offsets must not be negative"),
        ([0, 5, 3, -1], 5, "offsets must be nondecreasing"),
        ([0, _I32_LIMIT - 1, _I32_MIN], 5, "offsets must not be negative"),
        ([0, 2], 1, "final offset 2 exceeds value count 1"),
    ],
)
def test_array_validation_keeps_every_message(offsets, value_count, message):
    """Pin each message, and the first invalid offset deciding which one."""
    for form in (offsets, _i32(offsets)):
        with pytest.raises(ValueError) as error:
            qualification._validate_offset_sequence(form, value_count)
        assert str(error.value) == message


@pytest.mark.parametrize(
    ("offsets", "message"),
    [
        ([0, _I32_LIMIT], "offsets must contain signed i32 values"),
        ([0, _I32_MIN - 1], "offsets must contain signed i32 values"),
        ([0, 1.5], "offsets must contain signed i32 values"),
        ([0, True], "offsets must contain signed i32 values"),
        ([0, 5, 3, _I32_LIMIT], "offsets must be nondecreasing"),
        ([0, 5, _I32_LIMIT, 3], "offsets must contain signed i32 values"),
        ([_I32_LIMIT], "offsets must start at zero"),
    ],
)
def test_sequence_validation_rejects_values_outside_i32(offsets, message):
    """Keep the i32 check for a plain sequence, which no dtype guards."""
    with pytest.raises(ValueError) as error:
        qualification._validate_offset_sequence(offsets, 10)
    assert str(error.value) == message


def test_validation_returns_the_host_offsets_as_one_int32_array():
    """Hand preparation the host copy as a buffer, not as Python integers."""
    values = torch.ones(6)
    offsets = torch.tensor([0, 2, 6], dtype=torch.int32)
    output = torch.zeros(2)

    value_count, segment_count, host_offsets = qualification._validate_shapes(
        values,
        offsets,
        output,
        qualification._validate_offsets,
        require_cuda=False,
    )

    assert (value_count, segment_count) == (6, 2)
    assert isinstance(host_offsets, numpy.ndarray)
    assert host_offsets.dtype == numpy.int32
    assert host_offsets.tolist() == [0, 2, 6]


@pytest.mark.parametrize(
    ("offsets", "message"),
    [
        ([1, 6], "offsets must start at zero"),
        ([0, -1, 6], "offsets must not be negative"),
        ([0, 4, 2, 6], "offsets must be nondecreasing"),
        ([0, 2, 7], "final offset 7 exceeds value count 6"),
        ([0, 1, 2, 6], "output has 2 elements for 3 segments"),
    ],
)
def test_tensor_validation_keeps_every_message(offsets, message):
    """Raise the scalar messages for offsets that arrive as a tensor."""
    values = torch.ones(6)
    output = torch.zeros(2)

    with pytest.raises(ValueError) as error:
        qualification._validate_tensors(
            values,
            torch.tensor(offsets, dtype=torch.int32),
            output,
            require_cuda=False,
        )
    assert str(error.value) == message


def test_softmax_validation_names_the_covered_values_as_an_integer():
    """Format the final offset of an array as the integer it is."""
    values = torch.ones(7)
    offsets = torch.tensor([0, 3, 7], dtype=torch.int32)

    with pytest.raises(ValueError) as error:
        qualification._validate_softmax_tensors(
            values, offsets, torch.zeros(6), require_cuda=False
        )
    assert str(error.value) == "output has 6 elements for 7 values"


def _forbid_python_integers(monkeypatch):
    """Fail on the calls that turn a tensor into Python integers."""

    def fail(*_args, **_kwargs):
        pytest.fail("preparation must not build per-element Python objects")

    monkeypatch.setattr(torch.Tensor, "tolist", fail)
    monkeypatch.setattr(torch.Tensor, "__iter__", fail)


def _capture_uploads(monkeypatch):
    """Record the host data of every int32 tensor uploaded to the device."""
    original = torch.tensor
    uploads = []

    def capture(data, *args, **kwargs):
        tensor = original(data, *args, **kwargs)
        if kwargs.get("dtype") == torch.int32 and tensor.device.type == "cuda":
            uploads.append(data)
        return tensor

    monkeypatch.setattr(torch, "tensor", capture)
    return uploads


def _integer_case(lengths, seed=0):
    """Return small integer values whose f32 sums are exact in any order."""
    generator = torch.Generator().manual_seed(seed)
    offsets = torch.tensor(_offsets(lengths), dtype=torch.int32)
    values = torch.randint(
        0, 4, (int(offsets[-1]),), generator=generator
    ).float()
    expected = torch.segment_reduce(values, "sum", offsets=offsets)
    return values, offsets, expected


@requires_cuda
@pytest.mark.parametrize("seed", range(5))
def test_planned_preparation_uploads_the_reference_plan_as_arrays(
    monkeypatch, seed
):
    """Prepare from buffers alone and launch every policy exactly."""
    lengths = _random_lengths(seed, 32, 4096)
    values, offsets, expected = _integer_case(lengths, seed)
    warp, cta, partial, merge = _reference_plan(offsets.tolist())
    output = torch.empty(len(lengths), device="cuda")
    device_values, device_offsets = values.cuda(), offsets.cuda()
    uploads = _capture_uploads(monkeypatch)
    _forbid_python_integers(monkeypatch)

    prepared = qualification._prepare_planned_sum(
        device_values, device_offsets, output
    )

    assert all(isinstance(data, numpy.ndarray) for data in uploads)
    assert all(data.dtype == numpy.int32 for data in uploads)
    assert [data.tolist() for data in uploads] == [
        [*warp, *cta],
        partial,
        merge,
    ]
    monkeypatch.undo()
    for launch in prepared:
        output.fill_(float("nan"))
        launch()
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


@requires_cuda
@pytest.mark.parametrize("seed", range(5))
def test_persistent_preparation_uploads_the_reference_plan_as_arrays(
    monkeypatch, seed
):
    """Derive the merge of every partial task without a Python loop."""
    lengths = _random_lengths(seed, 32, 4096)
    values, offsets, expected = _integer_case(lengths, seed)
    warp, cta, partial, merge = _reference_plan(offsets.tolist())
    partial_merges = _reference_partial_merges(merge, len(partial) // 2)
    output = torch.empty(len(lengths), device="cuda")
    device_values, device_offsets = values.cuda(), offsets.cuda()
    uploads = _capture_uploads(monkeypatch)
    _forbid_python_integers(monkeypatch)

    prepared = qualification._prepare_persistent_sum(
        device_values, device_offsets, output
    )

    assert all(isinstance(data, numpy.ndarray) for data in uploads)
    assert all(data.dtype == numpy.int32 for data in uploads)
    assert [data.tolist() for data in uploads] == [
        warp,
        cta,
        partial,
        partial_merges,
        merge,
    ]
    assert (prepared.warp_tasks, prepared.cta_tasks) == (len(warp), len(cta))
    assert prepared.partial_tasks == len(partial) // 2
    assert prepared.merge_tasks == len(merge) // 3
    monkeypatch.undo()
    for _ in range(3):
        output.fill_(float("nan"))
        prepared.launch()
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


@requires_cuda
def test_direct_launches_validate_without_python_integers(monkeypatch):
    """Validate offsets and task ids of the one-shot launches as arrays."""
    lengths = [0, 1, 32, 33, 128, 0, 300]
    values, offsets, expected = _integer_case(lengths)
    device_values, device_offsets = values.cuda(), offsets.cuda()
    output = torch.empty(len(lengths), device="cuda")
    task_ids = torch.arange(len(lengths), dtype=torch.int32, device="cuda")
    softmax_output = torch.empty(values.numel(), device="cuda")
    _forbid_python_integers(monkeypatch)

    qualification.launch_gpu(device_values, device_offsets, output, "sum")
    first = output.clone()
    output.fill_(float("nan"))
    qualification._launch_segmented_sum_tasks(
        device_values, device_offsets, output, task_ids, block_size=128
    )
    qualification.launch_softmax_gpu(
        device_values, device_offsets, softmax_output
    )

    monkeypatch.undo()
    torch.testing.assert_close(first.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    assert bool(torch.isfinite(softmax_output).all())


@requires_cuda
@pytest.mark.parametrize("task_ids", [[0, 1, 2], [0, -1], [1, 0, -5]])
def test_task_launch_rejects_segment_ids_outside_the_layout(task_ids):
    """Reject an id below zero or past the last segment before launching."""
    values, offsets, _ = _integer_case([3, 4])
    output = torch.empty(2, device="cuda")

    with pytest.raises(ValueError, match="task_ids must contain valid"):
        qualification._launch_segmented_sum_tasks(
            values.cuda(),
            offsets.cuda(),
            output,
            torch.tensor(task_ids, dtype=torch.int32, device="cuda"),
            block_size=32,
        )


@requires_cuda
@pytest.mark.parametrize(
    ("merge", "message"),
    [
        ([0, 0, 2], "partial task has no merge dependency"),
        ([0, 1, 3], "partial task has no merge dependency"),
        ([0, 0, 2, 0, 1, 3], "partial task belongs to multiple merges"),
        ([0, 0, 4], "partial task belongs to multiple merges"),
    ],
    ids=["uncovered-tail", "uncovered-head", "overlap", "past-the-end"],
)
def test_persistent_preparation_rejects_merges_that_do_not_partition(
    monkeypatch, merge, message
):
    """Refuse merge ranges that leave a partial task without one merge.

    The resident kernel indexes the merge of every partial task unchecked,
    so the ranges are checked before any private state is allocated.
    """
    from swage import _runtime

    def fail(*_args, **_kwargs):
        pytest.fail("malformed merge records must not continue")

    values = torch.ones(8193, device="cuda")
    offsets = torch.tensor([0, 8193], device="cuda", dtype=torch.int32)
    output = torch.empty(1, device="cuda")
    partial = [0, 4096, 4096, 8192, 8192, 8193]
    monkeypatch.setattr(
        native_swage,
        "_materialize_segmented_plan",
        lambda *_args, **_kwargs: (
            _i32([]),
            _i32([]),
            _i32(partial),
            _i32(merge),
        ),
    )
    monkeypatch.setattr(torch, "tensor", fail)
    monkeypatch.setattr(_runtime, "_get_driver", fail)

    with pytest.raises(RuntimeError) as error:
        qualification._prepare_persistent_sum(values, offsets, output)
    assert str(error.value) == message


class _CountingModule:
    """Stand in for ir.Module and record every text that is parsed."""

    def __init__(self, module_class):
        self._module_class = module_class
        self.parsed = []

    def parse(self, text, *args, **kwargs):
        self.parsed.append(text)
        return self._module_class.parse(text, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._module_class, name)


@requires_cuda
def test_a_program_is_parsed_and_inspected_once_for_many_layouts(monkeypatch):
    """Reuse one parsed module across preparations with fresh offsets."""
    text = qualification._semantic_module("sum")
    layouts = [[5000] * 96, [4097] * 96, [6000, 7000] * 48]
    cases = [
        _integer_case(lengths, seed=index)
        for index, lengths in enumerate(layouts)
    ]
    output = torch.empty(96, device="cuda")

    def prepare(values, offsets):
        return qualification._prepare_planned_reduction(
            values.cuda(),
            offsets.cuda(),
            output,
            module_text=text,
            kernel_name="segmented_sum",
        )

    # Compile and load every kernel first, so only preparation parses below.
    prepare(*cases[0][:2])
    inspections = []
    inspect = qualification._has_small_element_program

    def counting_inspect(module):
        inspections.append(module)
        return inspect(module)

    counting_module = _CountingModule(ir.Module)
    monkeypatch.setattr(ir, "Module", counting_module)
    monkeypatch.setattr(
        qualification, "_has_small_element_program", counting_inspect
    )
    monkeypatch.setattr(qualification, "_module_memo", {})

    for values, offsets, expected in cases:
        prepared = prepare(values, offsets)
        prepared.mixed()
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)

    assert counting_module.parsed == [text]
    assert len(inspections) == 1
