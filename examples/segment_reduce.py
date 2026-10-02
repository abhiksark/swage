# examples/segment_reduce.py
"""Reduce and normalize ragged segments with the public segmented calls."""

from itertools import pairwise

import swage
import torch


def main():
    """Run sum, max, and softmax over four segments and check each result."""
    # Six values in four segments: [1, 2], [], [3, 4, 5], and [6]. The
    # repeated offset makes the second segment empty.
    values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], device="cuda")
    offsets = torch.tensor([0, 2, 2, 5, 6], dtype=torch.int32, device="cuda")

    totals = swage.segment_reduce(values, offsets, "sum")
    maxima = swage.segment_reduce(values, offsets, "max")
    weights = swage.segment_softmax(values, offsets)

    # Each call returns after it enqueued its kernels. Reading a result, as
    # the lines below do, waits for them.
    print("sum:", totals.tolist())
    print("max:", maxima.tolist())
    print("softmax:", [round(weight, 4) for weight in weights.tolist()])

    # torch.segment_reduce takes int64 offsets and gives the same values,
    # including 0.0 and negative infinity for the empty segment.
    long_offsets = offsets.long()
    torch.testing.assert_close(
        totals, torch.segment_reduce(values, "sum", offsets=long_offsets)
    )
    torch.testing.assert_close(
        maxima, torch.segment_reduce(values, "max", offsets=long_offsets)
    )
    for begin, end in pairwise(offsets.tolist()):
        torch.testing.assert_close(
            weights[begin:end], torch.softmax(values[begin:end], dim=0)
        )

    # A caller's tensor is written in place and returned.
    reused = torch.empty(4, device="cuda")
    assert swage.segment_reduce(values, offsets, "sum", out=reused) is reused
    torch.testing.assert_close(reused, totals)
    print("=== CUDA results match PyTorch ===")


if __name__ == "__main__":
    main()
