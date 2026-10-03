# python/tests/mlir/row_tile_model.py
"""A host model of the row-stripe tile of rank-two values.

The row-stripe tile reduces one column of the rows of a segment with the
threads of a block: thread `t` is stripe `(t / S) * (S / W) + (t mod S) / W`
of column `(t mod S) mod W` of a group of `W` columns, for the subgroup
width `S` and the column-group width `W`. Each stripe folds rows `s`,
`s + R`, `s + 2R`, and so on in row order, for `R = T / W` stripes in a
block of `T` threads. The stripes of one subgroup then combine by an XOR
butterfly over the bits of their stripe index, low to high, and the results
of the subgroups combine as a pairwise tree in subgroup order.

This module adds in exactly that order with numpy arithmetic of the element
type, so a test can require the bits of a kernel and not only a bound.
"""

import numpy

SUBGROUP_WIDTH = 32
CTA_BLOCK_THREADS = 128
SPLIT_BLOCK_THREADS = 512

_IDENTITY = {"sum": 0.0, "max": -numpy.inf, "min": numpy.inf}


def group_width(features, subgroup_width=SUBGROUP_WIDTH):
    """Return the column-group width of `features` columns.

    It is the smallest power of two that is at least `features`, capped at
    the subgroup width, and 1 for `features` of 1 or less.
    """
    width = 1
    while width < subgroup_width and width < features:
        width <<= 1
    return width


def _combine(kind, left, right):
    """Combine two arrays of partial results elementwise, in their dtype."""
    if kind == "sum":
        return left + right
    if kind == "max":
        return numpy.maximum(left, right)
    return numpy.minimum(left, right)


def reduce_rows(rows, kind, threads=CTA_BLOCK_THREADS,
                subgroup_width=SUBGROUP_WIDTH):
    """Reduce every column of `rows` as one block of the tile does.

    Args:
        rows: Array of shape `[n, D]` of float32 or float64.
        kind: `"sum"`, `"max"`, or `"min"`. A mean is the sum divided by
            the row count, which `mean_rows` gives.
        threads: The block width `T`, a multiple of the subgroup width
            whose subgroup count is a power of two.
        subgroup_width: The subgroup width `S`.

    Returns:
        An array of shape `[D]` in the dtype of `rows`.
    """
    rows = numpy.asarray(rows)
    dtype = rows.dtype.type
    count, features = rows.shape
    width = group_width(features, subgroup_width)
    stripes = threads // width
    identity = dtype(_IDENTITY[kind])
    # Stripe s folds rows s, s + R, ... in order: the rows of pass p are
    # p * R to p * R + R - 1, one per stripe, and a missing row is the
    # identity, which leaves an accumulator as it is.
    passes = -(-count // stripes) if count else 0
    padded = numpy.full((passes * stripes, features), identity, dtype=dtype)
    padded[:count] = rows
    accumulators = numpy.full((stripes, features), identity, dtype=dtype)
    for chunk in padded.reshape(passes, stripes, features):
        accumulators = _combine(kind, accumulators, chunk)
    # The butterfly within each subgroup, over the bits of the local stripe.
    local = subgroup_width // width
    subgroups = accumulators.reshape(threads // subgroup_width, local, features)
    step = 1
    while step < local:
        partner = numpy.arange(local) ^ step
        subgroups = _combine(kind, subgroups, subgroups[:, partner])
        step <<= 1
    partials = list(subgroups[:, 0])
    while len(partials) > 1:
        partials = [
            _combine(kind, partials[index], partials[index + 1])
            for index in range(0, len(partials), 2)
        ]
    return partials[0]


def mean_rows(rows, threads=CTA_BLOCK_THREADS, subgroup_width=SUBGROUP_WIDTH):
    """Return the sum of `reduce_rows` divided once by the row count."""
    rows = numpy.asarray(rows)
    dtype = rows.dtype.type
    total = reduce_rows(rows, "sum", threads, subgroup_width)
    with numpy.errstate(invalid="ignore", divide="ignore"):
        return total / dtype(rows.shape[0])


def reduce_segments(rows, offsets, kind, threads=CTA_BLOCK_THREADS):
    """Return the `[S, D]` results of the tile over the rows of each segment.

    Args:
        rows: Array of shape `[N, D]`.
        offsets: The `S + 1` row offsets, valid for `N`.
        kind: `"sum"`, `"max"`, `"min"`, or `"mean"`.
        threads: The block width of the tile.
    """
    rows = numpy.asarray(rows)
    offsets = [int(offset) for offset in offsets]
    results = numpy.empty((len(offsets) - 1, rows.shape[1]), dtype=rows.dtype)
    for index, (begin, end) in enumerate(zip(offsets, offsets[1:])):
        segment = rows[begin:end]
        results[index] = (
            mean_rows(segment, threads)
            if kind == "mean"
            else reduce_rows(segment, kind, threads)
        )
    return results
