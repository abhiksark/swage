# tests/python/test_public_call_tables.py
"""Tests for the generated pages of the records of the public calls.

The page beside each record and the fragments that the documentation
includes are generated from the committed summaries. These tests keep them
in step with the records, pin the rules the generator uses for a row, a
ratio, a best looped Triton configuration, and a change between records,
and check the sentences that the documentation states about the records
in prose.
"""

import copy
import importlib
import json
import pathlib

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_BASELINE = "segment-reduce-a6000-sm86-c6099ec"
_LATEST = "segment-reduce-a6000-sm86-2cf88ae"
_RECORDS = [_BASELINE, _LATEST]
_RUNS = [
    "public-r1-8192",
    "public-r2-2048",
    "public-r2-8192",
    "public-r2-32768",
    "public-r2-768-2048",
]
# The longest segment, in rows, of the layouts that the documentation calls
# short.
_SHORT = 32


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


@pytest.mark.parametrize("name", _RECORDS)
def test_committed_page_fragments_digests_and_ptx_are_current(tables, name):
    """The page, every fragment, every process record, and the PTX match.

    For the first record this also shows that rendering the second one left
    its committed page as it was.
    """
    assert name in tables.RECORDS
    assert tables.check(name) == []


def test_the_latest_record_is_compared_with_the_baseline(tables):
    """The second record names the first as its baseline."""
    assert tables.BASELINES == {_LATEST: _BASELINE}


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

    assert {name for name in included if _LATEST in name}
    assert included <= generated


@pytest.mark.parametrize("name", _RECORDS)
def test_runs_are_ordered_by_rank_width_and_size(tables, name):
    """The page lists the rank-one run first, then the widest last."""
    assert list(tables._runs(name)) == _RUNS


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


@pytest.mark.parametrize("name", _RECORDS)
@pytest.mark.parametrize("run", _RUNS)
def test_torch_ratios_equal_the_ones_the_summaries_hold(tables, name, run):
    """The generator's ratio rule reproduces the driver's ratio to torch."""
    checked = 0
    for candidates in tables._rows(name, run).values():
        for candidate_name, candidate in candidates.items():
            recorded = candidate["ratio_to_torch"]
            median, low, high = tables._ratio(candidate, candidates["torch"])
            assert median == pytest.approx(recorded["median"]), candidate_name
            assert low == pytest.approx(recorded["min"]), candidate_name
            assert high == pytest.approx(recorded["max"]), candidate_name
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


@pytest.mark.parametrize(
    ("name", "files"),
    [
        (_BASELINE, ["segmented_sum_r2.ptx"]),
        (
            _LATEST,
            [
                "segmented_sum_r2.cta.ptx",
                "segmented_sum_r2.merge.ptx",
                "segmented_sum_r2.partial.ptx",
            ],
        ),
    ],
)
def test_a_ptx_file_that_no_process_loaded_is_reported(
    tables, monkeypatch, name, files
):
    """Every committed PTX file must be a module that the summaries list."""
    monkeypatch.setattr(tables, "_loaded_ptx", lambda record: {})

    errors = tables.check_records(name)

    assert errors == [
        f"no process loaded this PTX: {tables.RESULTS / name / file}"
        for file in files
    ]


@pytest.mark.parametrize(
    ("name", "loads"),
    [(_BASELINE, ["1.18", "4.72"]), (_LATEST, ["1.89", "7.25"])],
)
def test_conditions_come_from_the_records(tables, name, loads):
    """The conditions and load averages come from the committed files."""
    text = tables._conditions_text(name)

    assert "25 of 25 processes saw none" in text
    assert "`powersave`" in text
    assert tables._load_averages(name) == loads


def test_public_statement_names_the_extreme_rows(tables):
    """Each run names the rows of its lowest and highest ratio to torch."""
    text = tables._public_statement(_BASELINE)

    assert "lowest on `uniform`" in text
    assert "highest on `power-law D=768`" in text
    assert "faster in every process" not in text
    assert text.count("\n- ") == len(_RUNS)


def test_losses_name_the_rows_only_when_some_rows_lose(tables):
    """A loss of every row names none; otherwise the fewer side is named."""
    items = [{"row": row} for row in ("a", "b", "c")]

    assert tables._named(["a", "b", "c"], items) == ""
    assert tables._named(["a"], items) == " (`a`)"
    assert tables._named(["a", "b"], items) == ", all but `c`"


@pytest.mark.parametrize(
    ("late", "change"),
    [
        ((8.0, 9.0, 9.5), "faster"),
        ((10.0, 11.0, 12.0), "within the spread"),
        ((12.5, 13.0, 14.0), "slower"),
    ],
)
def test_a_change_needs_process_medians_that_do_not_overlap(
    tables, late, change
):
    """Two records differ only when their process medians do not overlap."""
    before = _candidate(10.0, 11.0, 12.0)

    ratio, found = tables._change(before, _candidate(*late))

    assert found == change
    assert ratio == pytest.approx(sorted(late)[1] / 11.0)


def test_a_comparison_refuses_runs_with_other_commands(tables, monkeypatch):
    """A run is compared only with the baseline run of the same command."""
    real = tables._load_summary

    def other_command(name, run):
        summary = real(name, run)
        if name != _BASELINE:
            return summary
        summary = copy.deepcopy(summary)
        summary["command"] = [*summary["command"], "--seed", "8"]
        return summary

    monkeypatch.setattr(tables, "_load_summary", other_command)

    with pytest.raises(ValueError, match="differs from the run"):
        tables._changes.__wrapped__(_LATEST)


def test_rank_one_is_unchanged_within_the_spread_of_the_baseline(tables):
    """Rank one ran the same kernels, and no row left the process spread."""
    items = tables._changes(_LATEST)["public-r1-8192"]

    assert {item["change"] for item in items} == {"within the spread"}
    assert tables._modules(_LATEST, "public-r1-8192") == tables._modules(
        _BASELINE, "public-r1-8192"
    )
    assert "Both runs loaded the same" in tables._comparison_statement(
        _LATEST
    )


def _longest_segments(tables, name, run):
    """Return the longest segment of the layouts of every row of a run."""
    return {
        tables._label(result): max(
            iteration["layout_statistics"]["max"]
            for iteration in result["iterations"]
        )
        for result in tables._load_process(name, run)["results"]
    }


def test_documented_claims_about_the_latest_record_hold(tables):
    """The README, the reference, and the guide state these in prose.

    The call is slower in every process on every rank-one row and on every
    `[N, D]` row whose segments have at most 32 rows, and its median is
    below torch on most `[N, D]` rows with longer segments.
    """
    by_run = tables._by_run(_LATEST)
    short, longer = [], []
    for run, items in by_run.items():
        if run == "public-r1-8192":
            assert all(item["public"][1] > 1 for item in items)
            continue
        longest = _longest_segments(tables, _LATEST, run)
        for item in items:
            group = short if longest[item["row"]] <= _SHORT else longer
            group.append(item["public"])

    assert short and all(low > 1 for _, low, _ in short)
    assert sum(1 for median, _, _ in longer if median < 1) * 2 > len(longer)
    for page in ("README.md", "docs/reference/swage.md"):
        text = " ".join((_ROOT / page).read_text().split())
        assert "segments have at most 32 rows, and faster on most" in text
    guide = (_ROOT / "docs/user-guide/segmented-calls.md").read_text()
    assert "whose segments have at most 32\nrows" in guide


def test_the_load_average_rose_during_the_latest_campaign(tables):
    """The benchmarks page says the load average rose."""
    start, end = (float(value) for value in tables._load_averages(_LATEST))

    assert end > start


def test_rank_one_kernels_equal_those_of_the_453c56e_record(tables):
    """The benchmarks page states that the rank-one PTX is unchanged."""
    earlier = _ROOT / "benchmarks/results/segmented-sum-a6000-sm86-453c56e"
    summary = json.loads(
        (earlier / "fresh-8192-nopad/summary.json").read_text()
    )
    then = {module["sha256"] for module in summary["code"]["loaded_ptx"]}

    assert then == tables._modules(_BASELINE, "public-r1-8192")
