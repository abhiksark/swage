# tests/python/test_benchmark_distributions.py
"""Tests for deterministic benchmark segment distributions."""

import hashlib

import pytest

from benchmarks.distributions import (
    generate_lengths,
    summarize_lengths,
    worst_case_total,
)

_NAMES = (
    "uniform",
    "log-normal",
    "bimodal",
    "zipf-like",
    "many-tiny",
    "few-huge",
    "one-outlier",
    "alternating-empty",
    "power-law",
)
# SHA-256 of the comma-joined lengths at 32,768 segments and seed 7, taken
# from the generators as they were before the power-law distribution existed.
_FROZEN_DIGESTS = {
    "uniform": (
        "a7d52af0ccf278bca9c0b17f9434490f68b7164c2bb1c592765bf99d98b18afc"
    ),
    "log-normal": (
        "402fe89dbf7ed9847713a289c01c9c196cb5e95c427e45230f5f69beeb37313f"
    ),
    "bimodal": (
        "ed7085b9523a63f199bc43a95f1936b3842ba7d2e53bfc7968bedcce9502a085"
    ),
    "zipf-like": (
        "6868b9898de0ff4ff4832913bbc82a16e6b513f144f1823ec380d22caa54a800"
    ),
    "many-tiny": (
        "b70457f2d3f931f9df32d92a6fdd43bbde256be9195999e4986336235cb0c736"
    ),
    "few-huge": (
        "91f605fb3ab17531443157415670a050f13bfed568a423ef3976bc659bf435d2"
    ),
    "one-outlier": (
        "e30f920aa7ff95baf63db93ea143767ca00053b2e9705ff31e5b2d3f08e15eda"
    ),
    "alternating-empty": (
        "9205adf3593f0f52d9069e3988a4923971e44d0c87ba87f00cdc84a5b84ca9e5"
    ),
}
_DEFAULT_COUNT = 32_768
_MAX_COUNT = 524_287
_MILLION = 1_000_000
_I32_MAX = (1 << 31) - 1
_CTA_CHUNK_ELEMENTS = 4096


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
        ("power-law", lambda value: 0 <= value <= 67**1.25),
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


@pytest.mark.parametrize("name", sorted(_FROZEN_DIGESTS))
def test_capped_distributions_match_their_frozen_output(name):
    """Keep every earlier distribution byte-identical for its seed."""
    lengths = generate_lengths(name, _DEFAULT_COUNT, 7)
    digest = hashlib.sha256(",".join(map(str, lengths)).encode()).hexdigest()

    assert digest == _FROZEN_DIGESTS[name]
    assert max(lengths) <= _CTA_CHUNK_ELEMENTS


def test_power_law_is_pinned_for_its_default_seed():
    """Change the heavy-tailed lengths only through a deliberate edit."""
    lengths = generate_lengths("power-law", _DEFAULT_COUNT, 7)

    assert lengths[:8] == [1, 3, 2, 8, 1, 7, 3, 2]
    assert summarize_lengths(lengths) == {
        "count": 32_768,
        "total": 1_684_109,
        "min": 0,
        "median": 2.0,
        "p95": 42,
        "max": 310_470,
    }


@pytest.mark.parametrize("seed", range(8))
def test_power_law_reaches_split_lengths_for_every_seed(seed):
    """Guarantee the stated maximum by construction, not by a lucky seed."""
    lengths = generate_lengths("power-law", _DEFAULT_COUNT, seed)

    assert (_DEFAULT_COUNT / 2) ** 1.25 <= max(lengths)
    assert max(lengths) >= 65_536
    assert max(lengths) <= _DEFAULT_COUNT**1.25
    assert sum(length > _CTA_CHUNK_ELEMENTS for length in lengths) >= 32
    assert sum(length <= 32 for length in lengths) > _DEFAULT_COUNT // 2


def test_power_law_lengths_depend_on_the_seed():
    """Draw different lengths, not only a different order, per seed."""
    first = generate_lengths("power-law", _DEFAULT_COUNT, 7)
    second = generate_lengths("power-law", _DEFAULT_COUNT, 8)

    assert sorted(first) != sorted(second)


def test_power_law_total_fits_i32_at_the_largest_count():
    """Keep the heavy tail inside the signed-i32 offset contract."""
    lengths = generate_lengths("power-law", _MAX_COUNT, 7)

    assert len(lengths) == _MAX_COUNT
    assert sum(lengths) < 4.6 * _MAX_COUNT**1.25 < (1 << 31) - 1


@pytest.mark.parametrize("count", [True, 0, -1, 524288])
def test_rejects_counts_that_cannot_guarantee_an_i32_total(count):
    """Reject invalid counts before allocating or sampling."""
    with pytest.raises(ValueError, match="count"):
        generate_lengths("uniform", count, 1)


@pytest.mark.parametrize(
    ("name", "largest"),
    [
        ("uniform", 524_287),
        ("log-normal", 524_287),
        ("zipf-like", 524_287),
        ("bimodal", 4_898_459),
        ("few-huge", 10_294_759),
        ("one-outlier", 67_108_736),
        ("many-tiny", 67_108_863),
        ("alternating-empty", 134_217_727),
        ("power-law", 8_616_633),
    ],
)
def test_count_is_admitted_exactly_while_the_worst_case_total_fits_i32(
    name, largest
):
    """Bound each distribution by its own longest lengths, not by 4096."""
    assert worst_case_total(name, largest) <= _I32_MAX
    assert worst_case_total(name, largest + 1) > _I32_MAX
    with pytest.raises(ValueError, match=f"count.*{name}.*fit i32"):
        generate_lengths(name, largest + 1, 1)


@pytest.mark.parametrize(
    "name",
    [
        "bimodal",
        "few-huge",
        "one-outlier",
        "many-tiny",
        "alternating-empty",
        "power-law",
    ],
)
def test_a_million_segments_fit_where_the_lengths_are_short_or_ranked(name):
    """Generate 10^6 segments for every distribution that guarantees i32."""
    lengths = generate_lengths(name, _MILLION, 7)

    assert len(lengths) == _MILLION
    assert sum(lengths) <= worst_case_total(name, _MILLION) <= _I32_MAX


@pytest.mark.parametrize("name", ["uniform", "log-normal", "zipf-like"])
def test_a_million_segments_are_refused_where_every_length_can_be_4096(name):
    """Refuse 10^6 segments when the total is not guaranteed to fit i32."""
    assert worst_case_total(name, _MILLION) == 4096 * _MILLION
    with pytest.raises(ValueError, match="count"):
        generate_lengths(name, _MILLION, 7)


def test_worst_case_total_rejects_an_unknown_distribution():
    """Name the misspelled distribution instead of returning a bound."""
    with pytest.raises(ValueError, match="unknown distribution"):
        worst_case_total("normal", 1)


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
