# tests/python/test_benchmark_distributions.py
"""Tests for deterministic benchmark segment distributions."""

import pytest

from benchmarks.distributions import generate_lengths, summarize_lengths

_NAMES = (
    "uniform",
    "log-normal",
    "bimodal",
    "zipf-like",
    "many-tiny",
    "few-huge",
    "one-outlier",
    "alternating-empty",
)


@pytest.mark.parametrize("name", _NAMES)
def test_distributions_are_seeded_and_have_the_requested_count(name):
    """Generate each distribution repeatably without global RNG state."""
    first = generate_lengths(name, 67, 11)
    second = generate_lengths(name, 67, 11)

    assert first == second
    assert len(first) == 67


@pytest.mark.parametrize(
    ("name", "member"),
    [
        ("uniform", lambda value: 0 <= value <= 4096),
        ("log-normal", lambda value: 0 <= value <= 4096),
        (
            "bimodal",
            lambda value: 1 <= value <= 32 or 1024 <= value <= 4096,
        ),
        ("zipf-like", lambda value: 1 <= value <= 4096),
        ("many-tiny", lambda value: 0 <= value <= 32),
        (
            "few-huge",
            lambda value: 0 <= value <= 4 or 1024 <= value <= 4096,
        ),
        ("one-outlier", lambda value: 1 <= value <= 32 or value == 4096),
        ("alternating-empty", lambda value: 0 <= value <= 32),
    ],
)
def test_distributions_stay_within_their_declared_support(name, member):
    """Keep every generated length inside its documented support."""
    assert all(member(value) for value in generate_lengths(name, 67, 13))


def test_fixed_class_distributions_have_exact_integer_quotas():
    """Assign incomplete ratio groups to the short class."""
    bimodal = generate_lengths("bimodal", 67, 17)
    few_huge = generate_lengths("few-huge", 67, 17)

    assert sum(value >= 1024 for value in bimodal) == 6
    assert sum(value >= 1024 for value in few_huge) == 3


def test_one_outlier_and_alternating_empty_have_exact_structure():
    """Preserve the deterministic structural distributions."""
    outlier = generate_lengths("one-outlier", 67, 19)
    alternating = generate_lengths("alternating-empty", 67, 19)

    assert outlier.count(4096) == 1
    assert all(value == 0 for value in alternating[::2])
    assert all(1 <= value <= 32 for value in alternating[1::2])


@pytest.mark.parametrize("count", [True, 0, -1, 524288])
def test_rejects_counts_that_cannot_guarantee_an_i32_total(count):
    """Reject invalid counts before allocating or sampling."""
    with pytest.raises(ValueError, match="count"):
        generate_lengths("uniform", count, 1)


@pytest.mark.parametrize("seed", [True, 1.5, "1"])
def test_rejects_non_integer_seeds(seed):
    """Keep reproducibility inputs explicit."""
    with pytest.raises(TypeError, match="seed"):
        generate_lengths("uniform", 1, seed)


def test_rejects_unknown_distribution():
    """Reject misspelled policies instead of silently substituting one."""
    with pytest.raises(ValueError, match="unknown distribution"):
        generate_lengths("normal", 1, 1)


def test_summarizes_with_nearest_rank_p95():
    """Report the committed benchmark statistics contract."""
    assert summarize_lengths([1, 2, 3, 4]) == {
        "count": 4,
        "total": 10,
        "min": 1,
        "median": 2.5,
        "p95": 4,
        "max": 4,
    }
