# tests/python/test_campaign_tables.py
"""Tests for the generated page of the segmented-sum campaign record.

The page beside the record and the fragments that the documentation
includes are generated from the committed summaries. These tests keep them
in step with the records, and they pin the rules the generator uses for a
ratio and for a "best" configuration.
"""

import importlib
import pathlib

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_FRESH_RUNS = [
    f"fresh-{size}-{kind}"
    for size in (2048, 8192, 32768)
    for kind in ("nopad", "all")
]


@pytest.fixture
def tables(monkeypatch):
    """Import the generator as its script entry point does."""
    monkeypatch.syspath_prepend(str(_ROOT / "benchmarks"))
    return importlib.import_module("campaign_tables")


def _candidate(*process_values):
    """Return a summary candidate with the given per-process medians."""
    ordered = sorted(process_values)
    return {
        "median_us": {
            "median": ordered[len(ordered) // 2],
            "process_values": list(process_values),
        }
    }


def test_committed_page_fragments_and_digests_are_current(tables):
    """The page, every fragment, and every process record match."""
    assert tables.check() == []


def test_every_fragment_that_a_page_includes_is_generated(tables):
    """A documentation page includes only fragments the generator writes."""
    generated = {path.name for path in tables.outputs()}
    included = set()
    for page in (_ROOT / "docs").rglob("*.md"):
        for line in page.read_text().splitlines():
            if line.startswith('--8<-- "') and tables.NAME in line:
                included.add(pathlib.Path(line.split('"')[1]).name)

    assert included
    assert included <= generated


def test_ratio_is_formed_inside_each_process(tables):
    """A ratio pairs the medians of one process, not the two medians."""
    numerator = _candidate(10.0, 30.0, 20.0)
    denominator = _candidate(5.0, 10.0, 20.0)

    assert tables._ratio(numerator, denominator) == (2.0, 1.0, 3.0)


@pytest.mark.parametrize("run", [*_FRESH_RUNS, "comparison"])
def test_torch_ratios_equal_the_ones_the_summaries_hold(tables, run):
    """The generator's ratio rule reproduces the driver's ratio to torch."""
    summary = tables._load_summary(run)
    checked = 0
    for methods in summary["rows"].values():
        for candidates in methods.values():
            for name, candidate in candidates.items():
                recorded = candidate["ratio_to_torch"]
                median, low, high = tables._ratio(
                    candidate, candidates["torch"]
                )
                assert median == pytest.approx(recorded["median"]), name
                assert low == pytest.approx(recorded["min"]), name
                assert high == pytest.approx(recorded["max"]), name
                checked += 1

    assert checked


def test_family_does_not_mix_planned_and_looping_planned(tables):
    """A family is selected by its block or warp suffix."""
    candidates = {
        "triton_b128_w1": _candidate(4.0),
        "triton_looped_b128_w1": _candidate(3.0),
        "triton_planned_w2": _candidate(2.0),
        "triton_planned_looped_b256_w4": _candidate(1.0),
        "torch": _candidate(5.0),
    }

    assert tables._family(candidates, "triton") == ["triton_b128_w1"]
    assert tables._family(candidates, "triton_planned") == ["triton_planned_w2"]
    assert tables._family(candidates, "triton_planned_looped") == [
        "triton_planned_looped_b256_w4"
    ]
    assert tables._config("triton_planned_looped_b256_w4") == "b256_w4"
    assert tables._config("triton_planned_w2") == "w2"


def test_best_is_the_lowest_median_and_none_without_candidates(tables):
    """The best configuration has the lowest median across processes."""
    candidates = {
        "triton_looped_b128_w1": _candidate(9.0, 9.0, 1.0),
        "triton_looped_b256_w1": _candidate(8.0, 8.0, 8.0),
    }

    assert (
        tables._best(candidates, sorted(candidates)) == "triton_looped_b256_w1"
    )
    assert tables._best(candidates, []) is None


def test_classes_use_the_parity_fraction(tables):
    """Ratios within the parity fraction are equal, not faster or slower."""
    faster, equal, slower = tables._classes(
        {"a": 0.97, "b": 0.985, "c": 1.0, "d": 1.02, "e": 1.03}
    )

    assert sorted(faster) == ["a"]
    assert sorted(equal) == ["b", "c", "d"]
    assert sorted(slower) == ["e"]


def test_record_text_is_escaped_for_markdown(tables):
    """Record text with markup characters is shown as written."""
    assert tables._quote("order_seed '<seed>' a|b") == (
        "order\\_seed '\\<seed\\>' a\\|b"
    )


def test_a_false_template_sentence_stops_generation(tables, monkeypatch):
    """A fixed sentence that the records contradict fails the generator."""
    real = tables._fresh_ratios

    def faster_once(size):
        ratios = real(size)
        ratios["mixed_low"] = [0.9, *ratios["mixed_low"][1:]]
        return ratios

    monkeypatch.setattr(tables, "_fresh_ratios", faster_once)

    with pytest.raises(ValueError, match="swage_mixed is slower than torch"):
        tables.check_claims()


def test_page_states_the_process_that_saw_another_compute_process(tables):
    """The conditions come from the records, including the exception."""
    text = tables._conditions_text()

    assert "34 of 35 processes saw none" in text
    assert "`fresh-2048-nopad/process-1.json` listed" in text
    assert "`powersave`" in text
