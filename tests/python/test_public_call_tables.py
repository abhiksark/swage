# tests/python/test_public_call_tables.py
"""Tests for the generated pages of the records of the public calls.

The page beside each record and the fragments that the documentation
includes are generated from the committed summaries. These tests keep them
in step with the records, and they pin the rules the generator uses for a
row, a ratio, and a best looped Triton configuration.
"""

import importlib
import json
import pathlib

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_RECORD = "segment-reduce-a6000-sm86-c6099ec"
_RUNS = [
    "public-r1-8192",
    "public-r2-2048",
    "public-r2-8192",
    "public-r2-32768",
    "public-r2-768-2048",
]


@pytest.fixture
def tables(monkeypatch):
    """Import the generator as its script entry point does."""
    monkeypatch.syspath_prepend(str(_ROOT / "benchmarks"))
    return importlib.import_module("public_call_tables")


def _candidate(*process_values):
    """Return a summary candidate with the given per-process medians."""
    ordered = sorted(process_values)
    return {
        "median_us": {
            "median": ordered[len(ordered) // 2],
            "process_values": list(process_values),
        }
    }


def test_committed_page_fragments_digests_and_ptx_are_current(tables):
    """The page, every fragment, every process record, and the PTX match."""
    assert _RECORD in tables.RECORDS
    assert tables.check(_RECORD) == []


def test_every_fragment_that_a_page_includes_is_generated(tables):
    """A documentation page includes only fragments the generator writes."""
    generated = {
        path.name for name in tables.RECORDS for path in tables.outputs(name)
    }
    included = set()
    for page in (_ROOT / "docs").rglob("*.md"):
        for line in page.read_text().splitlines():
            if line.startswith('--8<-- "') and any(
                name in line for name in tables.RECORDS
            ):
                included.add(pathlib.Path(line.split('"')[1]).name)

    assert included
    assert included <= generated


def test_runs_are_ordered_by_rank_width_and_size(tables):
    """The page lists the rank-one run first, then the widest last."""
    assert list(tables._runs(_RECORD)) == _RUNS


def test_row_labels_follow_the_process_driver(tables):
    """A row is labelled as `benchmark_processes.py` labels it."""
    assert tables._label({"distribution": "bimodal", "features": None}) == (
        "bimodal"
    )
    assert (
        tables._label(
            {
                "distribution": "power-law",
                "features": 64,
                "kind": "max",
                "dtype": "float64",
            }
        )
        == "power-law D=64 max float64"
    )


def test_ratio_is_formed_inside_each_process(tables):
    """A ratio pairs the medians of one process, not the two medians."""
    numerator = _candidate(10.0, 30.0, 20.0)
    denominator = _candidate(5.0, 10.0, 20.0)

    assert tables._ratio(numerator, denominator) == (2.0, 1.0, 3.0)


@pytest.mark.parametrize("run", _RUNS)
def test_torch_ratios_equal_the_ones_the_summaries_hold(tables, run):
    """The generator's ratio rule reproduces the driver's ratio to torch."""
    checked = 0
    for candidates in tables._rows(_RECORD, run).values():
        for name, candidate in candidates.items():
            recorded = candidate["ratio_to_torch"]
            median, low, high = tables._ratio(candidate, candidates["torch"])
            assert median == pytest.approx(recorded["median"]), name
            assert low == pytest.approx(recorded["min"]), name
            assert high == pytest.approx(recorded["max"]), name
            checked += 1

    assert checked


def test_best_looped_spans_both_ranks_and_skips_planned(tables):
    """The best looped configuration is looped Triton of either rank."""
    candidates = {
        "triton_looped_b128_w1": _candidate(4.0),
        "triton_rows_looped_r16_w4": _candidate(3.0),
        "triton_planned_looped_b256_w4": _candidate(1.0),
        "triton_planned_w2": _candidate(0.5),
        "torch": _candidate(5.0),
    }

    assert tables._best_looped(candidates) == "triton_rows_looped_r16_w4"
    assert tables._best_looped({"torch": _candidate(5.0)}) is None
    assert tables._config("triton_rows_looped_r16_w4") == "r16_w4"
    assert tables._config("triton_looped_b128_w1") == "b128_w1"


def test_configurations_map_to_the_described_families(tables):
    """A timed configuration names the family its description uses."""
    assert tables._family("triton_looped_b1024_w8") == "triton_looped"
    assert tables._family("triton_rows_looped_r4_w1") == "triton_rows_looped"
    assert tables._family("triton_planned_w4") == "triton_planned"
    assert (
        tables._family("triton_planned_looped_b128_w2")
        == "triton_planned_looped"
    )
    assert tables._family("swage_public_call_int64") == (
        "swage_public_call_int64"
    )


def test_record_text_is_escaped_for_markdown(tables):
    """Record text with markup characters is shown as written."""
    assert tables._quote("order_seed '<seed>' a|b") == (
        "order\\_seed '\\<seed\\>' a\\|b"
    )


def test_a_ptx_file_that_no_process_loaded_is_reported(tables, monkeypatch):
    """The committed PTX must be a module that the summaries list."""
    monkeypatch.setattr(tables, "_loaded_ptx", lambda name: {})

    errors = tables.check_records(_RECORD)

    assert errors == [
        "no process loaded this PTX: "
        f"{tables.RESULTS / _RECORD / 'segmented_sum_r2.ptx'}"
    ]


def test_conditions_come_from_the_records(tables):
    """The conditions and load averages come from the committed files."""
    text = tables._conditions_text(_RECORD)

    assert "25 of 25 processes saw none" in text
    assert "`powersave`" in text
    assert tables._load_averages(_RECORD) == ["1.18", "4.72"]


def test_public_statement_names_the_extreme_rows(tables):
    """Each run names the rows of its lowest and highest ratio to torch."""
    text = tables._public_statement(_RECORD)

    assert "lowest on `uniform`" in text
    assert "highest on `power-law D=768`" in text
    assert text.count("\n- ") == len(_RUNS)


def test_rank_one_kernels_equal_those_of_the_453c56e_record(tables):
    """The benchmarks page states that the rank-one PTX is unchanged."""
    earlier = _ROOT / "benchmarks/results/segmented-sum-a6000-sm86-453c56e"
    summary = json.loads(
        (earlier / "fresh-8192-nopad/summary.json").read_text()
    )
    then = {module["sha256"] for module in summary["code"]["loaded_ptx"]}
    code = tables._load_summary(_RECORD, "public-r1-8192")["code"]
    now = {module["sha256"] for module in code["loaded_ptx"]}

    assert now and now == then
