# python/tests/mlir/test_segmented_classification.py
"""Host validation and classification without per-element Python work.

Preparing a layout copies the offsets to the host once, validates them with
array operations, and classifies them in native code. Preparation does not
derive the plan a second time in Python. The Python derivation lives here as
a reference, and a seeded property test compares the native classifier with
it. The scalar validation loop is kept here in the same way, as the reference
for the exception types and messages of the array validation.

A process that runs from an artifact classifies with the runtime library
instead, which holds a second implementation of the classifier. The same
seeded layouts and the same refusals hold it to the native one.
"""

import contextlib
import io
import random
from itertools import accumulate, pairwise

import numpy
import pytest
import torch
from mlir_swage import ir
from mlir_swage._mlir_libs._swageDialectsNanobind import swage as native_swage
from mlir_swage.dialects import swage
from swage import _artifact, compile
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


def _native_records(offsets, warp_max_elements, cta_chunk_elements):
    """Classify host offsets through the binding that takes no module."""
    return native_swage._classify_segments(
        _i32(offsets),
        value_count=offsets[-1],
        segment_count=len(offsets) - 1,
        warp_max_elements=warp_max_elements,
        cta_chunk_elements=cta_chunk_elements,
    )


def _split_records(records, warp_count, cta_count, partial_count, merge_count):
    """Cut one record buffer into its five lists.

    Returns:
        The warp ids, the CTA ids, the partial ranges, the merge records,
        and the merge of every partial task.
    """
    bounds = numpy.cumsum(
        [
            0,
            warp_count,
            cta_count,
            2 * partial_count,
            3 * merge_count,
            partial_count,
        ]
    )
    assert len(records) == bounds[-1]
    return [records[begin:end] for begin, end in pairwise(bounds)]


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
    *records, merge_of_partial = _split_records(
        *_native_records(offsets, warp_max, chunk)
    )

    assert _matches_reference(plan, offsets, warp_max, chunk)
    assert _matches_reference(records, offsets, warp_max, chunk)
    warp, cta, partial, merge = _reference_plan(offsets, warp_max, chunk)
    assert sorted([*warp, *cta, *merge[0::3]]) == list(range(len(lengths)))
    partial_merges = _reference_partial_merges(merge, len(partial) // 2)
    assert partial_merges == sorted(partial_merges)
    assert merge_of_partial.tolist() == partial_merges


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
        [0, 1],
    ],
    ids=[
        "int64", "uint32", "float32", "rank-two", "strided", "tuple", "list",
    ],
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


def test_classification_without_a_module_returns_one_buffer_and_counts():
    """Lay the four record lists out in one int32 array, in launch order."""
    records, *counts = _native_records([0, 32, 65, 4162, 12354], 32, 4096)

    assert isinstance(records, numpy.ndarray)
    assert records.dtype == numpy.int32
    assert counts == [1, 1, 4, 2]
    assert all(type(count) is int for count in counts)
    assert records.tolist() == [
        0,
        1,
        *[65, 4161, 4161, 4162, 4162, 8258, 8258, 12354],
        *[2, 0, 2, 3, 2, 4],
        *[0, 0, 1, 1],
    ]


def test_classification_without_a_module_handles_no_segments():
    """Classify a layout without segments into an empty buffer."""
    records, *counts = _native_records([0], 32, 4096)

    assert records.shape == (0,)
    assert records.dtype == numpy.int32
    assert counts == [0, 0, 0, 0]


@pytest.mark.parametrize(
    "offsets",
    [
        numpy.asarray([0, 1], dtype=numpy.int64),
        numpy.asarray([[0, 1]], dtype=numpy.int32),
        numpy.asarray([0, 9, 1, 9], dtype=numpy.int32)[::2],
        (0, 1),
        [0, 1],
    ],
    ids=["int64", "rank-two", "strided", "tuple", "list"],
)
def test_classification_without_a_module_never_converts_offsets(offsets):
    """Refuse what the plan binding refuses: only a host i32 buffer fits."""
    with pytest.raises(TypeError):
        native_swage._classify_segments(
            offsets, value_count=1, segment_count=1
        )


@pytest.mark.parametrize(
    ("offsets", "arguments", "message"),
    [
        ([0, 2, 1], {}, "offsets must be nondecreasing"),
        ([0, -1], {}, "offset must be a nonnegative i32 value"),
        ([1, 1], {}, "offsets must start at zero"),
        (
            [0, 2],
            {"value_count": 1},
            "final offset must not exceed value count",
        ),
        (
            [0, 2],
            {"segment_count": 2},
            "offset count must equal segment count plus one",
        ),
        (
            [0, 1],
            {"warp_max_elements": 33, "cta_chunk_elements": 32},
            "warp max elements must not exceed CTA chunk elements",
        ),
        (
            [0, 1],
            {"warp_max_elements": 0},
            "warp max elements must be positive",
        ),
    ],
)
def test_classification_without_a_module_reports_the_classifier_reason(
    sum_module, offsets, arguments, message
):
    """Raise the reason itself, which the plan binding wraps in a location."""
    arguments = {
        "value_count": max(offsets[-1], 0),
        "segment_count": len(offsets) - 1,
        **arguments,
    }

    with pytest.raises(ValueError) as error:
        native_swage._classify_segments(_i32(offsets), **arguments)
    assert str(error.value) == message
    if "warp_max_elements" not in arguments:
        with pytest.raises(ValueError, match=message):
            native_swage._materialize_segmented_plan(
                sum_module, offsets=_i32(offsets), **arguments
            )


@pytest.fixture(scope="module")
def runtime_classifier(tmp_path_factory):
    """Return the classifier of the runtime library an artifact ships."""
    directory = tmp_path_factory.mktemp("classifier") / "artifact"
    with contextlib.redirect_stdout(io.StringIO()):
        status = compile.main(
            ["--target", "sm_86", "--output", str(directory)]
            + ["--program", "softmax"]
        )
    assert status == 0
    return _artifact._Artifact(directory)._classify_segments


@pytest.mark.parametrize(("warp_max", "chunk"), _LIMITS)
@pytest.mark.parametrize("seed", _SEEDS)
def test_runtime_library_classification_matches_the_native_classifier(
    runtime_classifier, seed, warp_max, chunk
):
    """Hold the runtime library to the native records on seeded layouts."""
    lengths = _random_lengths(seed, warp_max, chunk)
    offsets = _offsets(lengths)
    arguments = {
        "value_count": offsets[-1],
        "segment_count": len(lengths),
        "warp_max_elements": warp_max,
        "cta_chunk_elements": chunk,
    }

    expected, *expected_counts = native_swage._classify_segments(
        _i32(offsets), **arguments
    )
    records, *counts = runtime_classifier(_i32(offsets), **arguments)

    assert isinstance(records, numpy.ndarray)
    assert records.dtype == numpy.int32
    assert records.tolist() == expected.tolist()
    assert counts == expected_counts
    assert all(type(count) is int for count in counts)
    *lists, merge_of_partial = _split_records(records, *counts)
    assert _matches_reference(lists, offsets, warp_max, chunk)
    assert merge_of_partial.tolist() == _reference_partial_merges(
        lists[3].tolist(), counts[2]
    )


def test_runtime_library_classification_uses_the_default_limits(
    runtime_classifier,
):
    """Classify with 32 and 4096 when a call names no limit."""
    records, *counts = runtime_classifier(
        _i32([0, 32, 65, 4162, 12354]), value_count=12354, segment_count=4
    )

    assert counts == [1, 1, 4, 2]
    assert records.tolist() == [
        0,
        1,
        *[65, 4161, 4161, 4162, 4162, 8258, 8258, 12354],
        *[2, 0, 2, 3, 2, 4],
        *[0, 0, 1, 1],
    ]


def test_runtime_library_classification_handles_no_segments(
    runtime_classifier,
):
    """Classify a layout without segments into an empty buffer."""
    records, *counts = runtime_classifier(
        _i32([0]), value_count=0, segment_count=0
    )

    assert records.shape == (0,)
    assert records.dtype == numpy.int32
    assert counts == [0, 0, 0, 0]


def _classified(classify, offsets, arguments):
    """Return the records and counts as lists, or the raised error."""
    try:
        records, *counts = classify(_i32(offsets), **arguments)
    except Exception as error:  # noqa: BLE001 (the comparison needs any type)
        return type(error), str(error)
    return records.tolist(), counts


@pytest.mark.parametrize("seed", range(300))
def test_runtime_library_classification_refuses_what_the_native_one_refuses(
    runtime_classifier, seed
):
    """Raise the same error for the same malformed offsets and counts."""
    offsets, value_count = _random_offsets(seed)
    rng = random.Random(seed)
    arguments = {
        "value_count": value_count,
        "segment_count": len(offsets) - 1 + rng.choice([0, 0, 0, 1, -1]),
        "warp_max_elements": 7,
        "cta_chunk_elements": 16,
    }

    assert _classified(runtime_classifier, offsets, arguments) == _classified(
        native_swage._classify_segments, offsets, arguments
    )


@pytest.mark.parametrize(
    ("offsets", "arguments", "message"),
    [
        ([0, 2, 1], {}, "offsets must be nondecreasing"),
        ([0, -1], {}, "offset must be a nonnegative i32 value"),
        ([1, 1], {}, "offsets must start at zero"),
        (
            [0, 2],
            {"value_count": 1},
            "final offset must not exceed value count",
        ),
        (
            [0, 2],
            {"segment_count": 2},
            "offset count must equal segment count plus one",
        ),
        (
            [0, 1],
            {"warp_max_elements": 33, "cta_chunk_elements": 32},
            "warp max elements must not exceed CTA chunk elements",
        ),
        (
            [0, 1],
            {"warp_max_elements": 0},
            "warp max elements must be positive",
        ),
        (
            [0, 1],
            {"cta_chunk_elements": 0},
            "CTA chunk elements must be positive",
        ),
        (
            [0, 1],
            {"value_count": -1},
            "value count must be a nonnegative i32 value",
        ),
        (
            [0, 1],
            {"cta_chunk_elements": _I32_LIMIT},
            "CTA chunk elements must be a nonnegative i32 value",
        ),
        (
            [],
            {"value_count": 0},
            "segment count must be a nonnegative i32 value",
        ),
        (
            [0, _I32_LIMIT - 1],
            {"warp_max_elements": 1, "cta_chunk_elements": 1},
            "descriptor count must fit in i32",
        ),
    ],
)
def test_runtime_library_classification_reports_the_classifier_reason(
    runtime_classifier, offsets, arguments, message
):
    """Raise each reason of the native classifier, word for word."""
    arguments = {
        "value_count": max(offsets[-1], 0) if offsets else 0,
        "segment_count": len(offsets) - 1,
        **arguments,
    }

    for classify in (runtime_classifier, native_swage._classify_segments):
        with pytest.raises(ValueError) as error:
            classify(_i32(offsets), **arguments)
        assert str(error.value) == message


@pytest.mark.parametrize(
    "offsets",
    [
        numpy.asarray([0, 1], dtype=numpy.int64),
        numpy.asarray([[0, 1]], dtype=numpy.int32),
        numpy.asarray([0, 9, 1, 9], dtype=numpy.int32)[::2],
        (0, 1),
        [0, 1],
    ],
    ids=["int64", "rank-two", "strided", "tuple", "list"],
)
def test_runtime_library_classification_never_converts_offsets(
    runtime_classifier, offsets
):
    """Refuse what the native binding refuses: only a host i32 buffer."""
    with pytest.raises(TypeError):
        runtime_classifier(offsets, value_count=1, segment_count=1)


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
def test_planned_preparation_uploads_the_reference_plan_in_one_array(
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
    partial_merges = _reference_partial_merges(merge, len(partial) // 2)
    assert [data.tolist() for data in uploads] == [
        [*warp, *cta, *partial, *merge, *partial_merges]
    ]
    monkeypatch.undo()
    for launch in prepared:
        output.fill_(float("nan"))
        launch()
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


@requires_cuda
@pytest.mark.parametrize("seed", range(5))
def test_persistent_preparation_uploads_the_reference_plan_in_one_array(
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
        [*warp, *cta, *partial, *merge, *partial_merges]
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
def test_persistent_launch_reads_each_list_at_its_place_in_the_upload(
    monkeypatch,
):
    """Pass one pointer per record list into the single uploaded buffer."""
    from swage import _runtime

    class _Driver:
        def __init__(self):
            self.arguments = None

        def load(self, _ptx, _kernel_name):
            return 1, 1

        def launch_persistent(self, _function, _grid, _block, _stream, args):
            self.arguments = args

    lengths = [1, 33, 4097, 8192, 0]
    values, offsets, _ = _integer_case(lengths)
    warp, cta, partial, merge = _reference_plan(offsets.tolist())
    driver = _Driver()
    monkeypatch.setattr(_runtime, "_get_driver", lambda: driver)
    original = torch.tensor
    uploads = []

    def capture(data, *args, **kwargs):
        tensor = original(data, *args, **kwargs)
        uploads.append(tensor)
        return tensor

    monkeypatch.setattr(torch, "tensor", capture)
    prepared = qualification._prepare_persistent_sum(
        values.cuda(), offsets.cuda(), torch.empty(5, device="cuda")
    )
    monkeypatch.undo()
    monkeypatch.setattr(_runtime, "_get_driver", lambda: driver)

    prepared.launch()

    assert len(uploads) == 1
    records = uploads[0].data_ptr()
    counts = (len(warp), len(cta), len(partial) // 2, len(merge) // 3)
    assert counts == (2, 1, 4, 2)
    assert driver.arguments[3:8] == (
        records,
        records + 4 * 2,
        records + 4 * (2 + 1),
        records + 4 * (2 + 1 + 8 + 6),
        records + 4 * (2 + 1 + 8),
    )
    assert driver.arguments[11:15] == counts
    assert uploads[0].tolist() == [
        *warp,
        *cta,
        *partial,
        *merge,
        *_reference_partial_merges(merge, 4),
    ]


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
    monkeypatch.setattr(qualification, "_admitted", {})

    for values, offsets, expected in cases:
        prepared = prepare(values, offsets)
        prepared.mixed()
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)

    assert counting_module.parsed == [text]
    assert len(inspections) == 1


class _CountingCalls:
    """Wrap one native entry and record the keyword arguments of each call."""

    def __init__(self, function):
        self._function = function
        self.calls = []

    def __call__(self, *arguments, **keywords):
        self.calls.append(keywords)
        return self._function(*arguments, **keywords)


@requires_cuda
def test_a_program_is_admitted_once_per_pair_of_limits(monkeypatch):
    """Run the planning pass once, then classify each layout without it."""
    layouts = [[1, 33, 5000], [40] * 7, [0, 9000, 2], [4097, 31]]
    cases = [
        _integer_case(lengths, seed=index)
        for index, lengths in enumerate(layouts)
    ]
    plan = _CountingCalls(native_swage._materialize_segmented_plan)
    classify = _CountingCalls(native_swage._classify_segments)
    monkeypatch.setattr(native_swage, "_materialize_segmented_plan", plan)
    monkeypatch.setattr(native_swage, "_classify_segments", classify)
    monkeypatch.setattr(qualification, "_admitted", {})

    def launch(case, **limits):
        values, offsets, expected = case
        output = torch.full((len(expected),), float("nan"), device="cuda")
        prepared = qualification._prepare_planned_sum(
            values.cuda(), offsets.cuda(), output, **limits
        )
        prepared.mixed()
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)

    for case in cases:
        launch(case)
    assert [call["segment_count"] for call in plan.calls] == [0]
    assert [call["segment_count"] for call in classify.calls] == [3, 7, 3, 2]

    launch(cases[0], warp_max_elements=16)
    launch(cases[1], warp_max_elements=16)
    qualification._prepare_persistent_sum(
        cases[2][0].cuda(), cases[2][1].cuda(), torch.empty(3, device="cuda")
    )
    assert [
        (call["warp_max_elements"], call["cta_chunk_elements"])
        for call in plan.calls
    ] == [(32, 4096), (16, 4096)]
    assert len(classify.calls) == 7


@requires_cuda
def test_a_refused_program_or_limit_is_refused_at_every_preparation(
    monkeypatch,
):
    """Keep no admission for what the planning pass rejects."""
    values, offsets, _ = _integer_case([3, 40])
    arguments = (values.cuda(), offsets.cuda(), torch.empty(2, device="cuda"))
    monkeypatch.setattr(qualification, "_admitted", {})

    def fail(*_args, **_kwargs):
        pytest.fail("a refused preparation must not compile")

    monkeypatch.setattr(qualification, "_compile_once", fail)
    for _ in range(2):
        with pytest.raises(ValueError, match="planning limits must satisfy"):
            qualification._prepare_planned_sum(
                *arguments, warp_max_elements=33, cta_chunk_elements=32
            )
        with pytest.raises(ValueError, match="capture-free maps"):
            qualification._prepare_planned_reduction(
                *arguments,
                module_text=qualification._SOFTMAX_MODULE,
                kernel_name="ragged_softmax",
            )
    assert qualification._admitted == {}


@requires_cuda
def test_later_preparations_repeat_no_device_or_program_lookup(monkeypatch):
    """Ask for the device target once, and never fill the counters."""
    cases = [_integer_case(lengths) for lengths in ([1, 33, 5000], [40] * 7)]
    capability = _CountingCalls(torch.cuda.get_device_capability)
    monkeypatch.setattr(torch.cuda, "get_device_capability", capability)
    monkeypatch.setattr(qualification, "_targets", {})

    def fail(*_args, **_kwargs):
        pytest.fail("the persistent counters are zeroed by every launch")

    monkeypatch.setattr(torch, "zeros", fail)
    for prepare, policy in (
        (qualification._prepare_planned_sum, "mixed"),
        (qualification._prepare_persistent_sum, "launch"),
    ):
        for values, offsets, expected in cases:
            output = torch.full((len(expected),), float("nan"), device="cuda")
            prepared = prepare(values.cuda(), offsets.cuda(), output)
            for _ in range(2):
                output.fill_(float("nan"))
                getattr(prepared, policy)()
                torch.testing.assert_close(
                    output.cpu(), expected, rtol=0, atol=0
                )

    assert len(capability.calls) == 1


@requires_cuda
def test_segment_ids_are_uploaded_once_and_shared_by_preparations(monkeypatch):
    """Read the task list of the pure policies from one tensor per device."""
    cases = [_integer_case(lengths) for lengths in ([1, 33, 5000], [40] * 7)]
    monkeypatch.setattr(qualification, "_identity_memo", {})
    uploads = _capture_uploads(monkeypatch)

    def fail(*_args, **_kwargs):
        pytest.fail("segment ids up to the limit are not filled per layout")

    monkeypatch.setattr(torch, "arange", fail)
    prepared = []
    for values, offsets, expected in cases:
        output = torch.full((len(expected),), float("nan"), device="cuda")
        prepared.append((
            qualification._prepare_planned_sum(
                values.cuda(), offsets.cuda(), output
            ),
            output,
            expected,
        ))
    monkeypatch.undo()

    limit = qualification._IDENTITY_LIMIT
    assert [len(data) for data in uploads] == [limit, 11, 7]
    assert numpy.array_equal(uploads[0], numpy.arange(limit))
    for policies, output, expected in prepared:
        for launch in policies:
            output.fill_(float("nan"))
            launch()
            torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


@requires_cuda
def test_segment_ids_past_the_shared_tensor_are_filled_per_preparation(
    monkeypatch,
):
    """Fall back to a device fill for more segments than the shared ids."""
    values, offsets, expected = _integer_case([1, 33, 5000, 2, 40])
    output = torch.full((5,), float("nan"), device="cuda")
    fills = _CountingCalls(torch.arange)
    monkeypatch.setattr(qualification, "_IDENTITY_LIMIT", 4)
    monkeypatch.setattr(torch, "arange", fills)

    prepared = qualification._prepare_planned_sum(
        values.cuda(), offsets.cuda(), output
    )

    monkeypatch.undo()
    assert len(fills.calls) == 1
    for launch in prepared:
        output.fill_(float("nan"))
        launch()
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


_INVALID_LAYOUTS = [
    ([1, 6], 2, "offsets must start at zero"),
    ([0, -1, 6], 2, "offsets must not be negative"),
    ([0, 4, 2, 6], 3, "offsets must be nondecreasing"),
    ([0, 2, 7], 2, "final offset 7 exceeds value count 6"),
    ([0, 1, 2, 6], 2, "output has 2 elements for 3 segments"),
]


@requires_cuda
@pytest.mark.parametrize(
    ("prepare", "policy"),
    [
        (qualification._prepare_planned_sum, "mixed"),
        (qualification._prepare_persistent_sum, "launch"),
    ],
    ids=["planned", "persistent"],
)
def test_preparation_validates_offsets_by_classifying_them(
    monkeypatch, prepare, policy
):
    """Walk valid offsets once, in the classifier, and keep every message."""
    values = torch.ones(6, device="cuda")
    output = torch.empty(2, device="cuda")

    for offsets, _, message in _INVALID_LAYOUTS:
        with pytest.raises(ValueError) as error:
            prepare(
                values,
                torch.tensor(offsets, dtype=torch.int32, device="cuda"),
                output,
            )
        assert str(error.value) == message

    def fail(*_args, **_kwargs):
        pytest.fail("valid offsets are validated by the classifier alone")

    monkeypatch.setattr(qualification, "_validate_offsets", fail)
    monkeypatch.setattr(qualification, "_validate_offset_sequence", fail)
    offsets = torch.tensor([0, 2, 6], dtype=torch.int32, device="cuda")
    getattr(prepare(values, offsets, output), policy)()
    assert output.tolist() == [2.0, 4.0]


@requires_cuda
def test_preparation_keeps_the_order_of_its_errors():
    """Name the offsets first, then an overlap, then the planning limits."""
    values = torch.ones(8, device="cuda")
    output = torch.empty(2, device="cuda")
    valid = torch.tensor([0, 2, 6], dtype=torch.int32, device="cuda")
    decreasing = torch.tensor([0, 6, 2], dtype=torch.int32, device="cuda")
    bad_limits = {"warp_max_elements": 33, "cta_chunk_elements": 32}

    def prepare(offsets, output=output, **limits):
        return qualification._prepare_planned_sum(
            values, offsets, output, **limits
        )

    with pytest.raises(ValueError, match="offsets must be nondecreasing"):
        prepare(decreasing, **bad_limits)
    with pytest.raises(ValueError, match="offsets must be nondecreasing"):
        prepare(decreasing, output=values.narrow(0, 0, 2))
    with pytest.raises(ValueError, match="must not overlap the values"):
        prepare(valid, output=values.narrow(0, 0, 2), **bad_limits)
    with pytest.raises(ValueError, match="planning limits must satisfy"):
        prepare(valid, **bad_limits)
    with pytest.raises(TypeError):
        prepare(valid, warp_max_elements=32.5)

