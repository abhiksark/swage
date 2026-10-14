# benchmarks/distributions.py
"""Deterministic segment-length distributions for benchmarks and tests."""

import math
import random
import statistics
from collections.abc import Sequence

_I32_MAX = (1 << 31) - 1
_MAX_LENGTH = 4096
_SHORT_LENGTH = 32
_POWER_LAW_EXPONENT = 1.25
# An upper bound of zeta(1.25), the sum of rank ** -1.25 over every rank.
_POWER_LAW_TOTAL_FACTOR = 4.6
_NAMES = {
    "uniform",
    "log-normal",
    "bimodal",
    "zipf-like",
    "many-tiny",
    "few-huge",
    "one-outlier",
    "alternating-empty",
    "power-law",
}


def worst_case_total(name: str, count: int) -> int:
    """Return the largest total a distribution can reach at one count.

    A count is admitted when this bound fits a signed i32 offset. The bound
    follows the longest lengths each distribution can draw, so the
    distributions whose lengths are mostly short, and the ranked power law,
    admit far more segments than the ones that can draw 4096 everywhere.

    Args:
        name: Distribution name from ADR-0015, or ``power-law``.
        count: Segment count.

    Returns:
        An upper bound of the sum of the generated lengths for any seed.

    Raises:
        ValueError: If the name is unknown.
    """
    if name not in _NAMES:
        raise ValueError(f"unknown distribution {name!r}")
    if name == "bimodal":
        long_count = count // 10
        return (count - long_count) * _SHORT_LENGTH + long_count * _MAX_LENGTH
    if name == "few-huge":
        long_count = count // 20
        return (count - long_count) * 4 + long_count * _MAX_LENGTH
    if name == "one-outlier":
        return (count - 1) * _SHORT_LENGTH + _MAX_LENGTH
    if name == "many-tiny":
        return count * _SHORT_LENGTH
    if name == "alternating-empty":
        return (count // 2) * _SHORT_LENGTH
    if name == "power-law":
        return math.ceil(
            _POWER_LAW_TOTAL_FACTOR * count**_POWER_LAW_EXPONENT
        )
    return count * _MAX_LENGTH


def generate_lengths(name: str, count: int, seed: int) -> list[int]:
    """Generate one deterministic segment-length distribution.

    The eight ADR-0015 distributions cap every length at 4096 elements. The
    additional ``power-law`` distribution has no fixed cap: it draws one
    length per rank of a Pareto tail with shape 0.8, so its largest length is
    between ``(count / 2) ** 1.25`` and ``count ** 1.25`` for every seed. At
    32,768 segments that is at least 185,363 elements, far above the 4096
    elements one CTA task covers.

    Args:
        name: Distribution name from ADR-0015, or ``power-law``.
        count: Positive segment count whose ``worst_case_total`` fits i32.
        seed: Integer seed for an isolated random-number generator.

    Returns:
        The generated integer segment lengths.

    Raises:
        TypeError: If the seed is not an integer.
        ValueError: If the name or count is invalid.
    """
    if name not in _NAMES:
        raise ValueError(f"unknown distribution {name!r}")
    if type(count) is not int or count <= 0:
        raise ValueError("count must be a positive integer")
    if worst_case_total(name, count) > _I32_MAX:
        raise ValueError(
            f"count {count} is too large for {name}: its total can reach "
            f"{worst_case_total(name, count)} elements and must fit i32"
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
    if name == "power-law":
        # The jitter stays below one rank, which bounds the largest length
        # from below and the total by zeta(1.25) * count ** 1.25, the bound
        # that worst_case_total admits counts by.
        lengths = [
            int((count / (rank + rng.random())) ** _POWER_LAW_EXPONENT)
            for rank in range(1, count + 1)
        ]
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
