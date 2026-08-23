# benchmarks/distributions.py
"""Deterministic segment-length distributions for benchmarks and tests."""

import math
import random
import statistics
from collections.abc import Sequence

_I32_MAX = (1 << 31) - 1
_MAX_LENGTH = 4096
_MAX_COUNT = _I32_MAX // _MAX_LENGTH
_NAMES = {
    "uniform",
    "log-normal",
    "bimodal",
    "zipf-like",
    "many-tiny",
    "few-huge",
    "one-outlier",
    "alternating-empty",
}


def generate_lengths(name: str, count: int, seed: int) -> list[int]:
    """Generate one deterministic segment-length distribution.

    Args:
        name: Distribution name from ADR-0015.
        count: Positive segment count whose worst-case total fits in i32.
        seed: Integer seed for an isolated random-number generator.

    Returns:
        The generated integer segment lengths.

    Raises:
        TypeError: If the seed is not an integer.
        ValueError: If the name or count is invalid.
    """
    if name not in _NAMES:
        raise ValueError(f"unknown distribution {name!r}")
    if type(count) is not int or not 0 < count <= _MAX_COUNT:
        raise ValueError(
            f"count must be between 1 and {_MAX_COUNT} so the total fits i32"
        )
    if type(seed) is not int:
        raise TypeError("seed must be an integer")

    rng = random.Random(seed)
    if name == "uniform":
        return [rng.randint(0, _MAX_LENGTH) for _ in range(count)]
    if name == "log-normal":
        return [
            min(round(rng.lognormvariate(math.log(32), 1.5)), _MAX_LENGTH)
            for _ in range(count)
        ]
    if name == "bimodal":
        long_count = count // 10
        lengths = [rng.randint(1, 32) for _ in range(count - long_count)]
        lengths.extend(
            rng.randint(1024, _MAX_LENGTH) for _ in range(long_count)
        )
        rng.shuffle(lengths)
        return lengths
    if name == "zipf-like":
        return rng.choices(
            range(1, _MAX_LENGTH + 1),
            weights=[value**-1.2 for value in range(1, _MAX_LENGTH + 1)],
            k=count,
        )
    if name == "many-tiny":
        return [rng.randint(0, 32) for _ in range(count)]
    if name == "few-huge":
        long_count = count // 20
        lengths = [rng.randint(0, 4) for _ in range(count - long_count)]
        lengths.extend(
            rng.randint(1024, _MAX_LENGTH) for _ in range(long_count)
        )
        rng.shuffle(lengths)
        return lengths
    if name == "one-outlier":
        lengths = [rng.randint(1, 32) for _ in range(count - 1)] + [4096]
        rng.shuffle(lengths)
        return lengths
    return [
        0 if index % 2 == 0 else rng.randint(1, 32)
        for index in range(count)
    ]


def summarize_lengths(lengths: Sequence[int]) -> dict[str, int | float]:
    """Summarize generated lengths using the ADR-0015 statistics contract.

    Args:
        lengths: Nonempty sequence of generated segment lengths.

    Returns:
        Count, total, minimum, median, nearest-rank p95, and maximum.

    Raises:
        ValueError: If no lengths are provided.
    """
    if not lengths:
        raise ValueError("lengths must not be empty")
    ordered = sorted(lengths)
    p95_index = (95 * len(ordered) + 99) // 100 - 1
    return {
        "count": len(ordered),
        "total": sum(ordered),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "p95": ordered[p95_index],
        "max": ordered[-1],
    }
