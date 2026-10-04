# benchmarks/campaign_tables.py
"""Generate the summary page of the segmented-sum campaign at 453c56e.

The campaign record is `benchmarks/results/segmented-sum-a6000-sm86-453c56e`:
seven runs of five independent processes each. This script reads the
committed `summary.json` of every run, and the first compressed process
record of every run for the run facts that a summary does not repeat. It
fills the tokens of `page.md.in` in the record directory to write the
Markdown page beside the record, and it writes the fragments that the
documentation includes, so no measured number in either is typed by hand.

Run from the repository root:

    python benchmarks/campaign_tables.py           # write page and fragments
    python benchmarks/campaign_tables.py --check   # verify them and the
                                                   # process record digests

It needs only the standard library.
"""

import argparse
import functools
import hashlib
import json
import lzma
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]
NAME = "segmented-sum-a6000-sm86-453c56e"
RECORD = REPO_ROOT / "benchmarks/results" / NAME
PAGE = REPO_ROOT / "benchmarks/results" / f"{NAME}.md"
TEMPLATE = RECORD / "page.md.in"
FRAGMENTS = REPO_ROOT / "docs/internals/_generated"
SIZES = (2048, 8192, 32768)
ROW_ORDER = (
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
# Two medians that differ by no more than this fraction are called equal.
PARITY = 0.02
# A candidate whose per-process medians spread further than this is listed
# under the irregular processes.
IRREGULAR = 1.25


@functools.cache
def _load_summary(run):
    """Return the committed summary of one run. Callers must not change it."""
    return json.loads((RECORD / run / "summary.json").read_text())


@functools.cache
def _load_process(run, index=1):
    """Return one decompressed per-process record of one run."""
    packed = RECORD / run / f"process-{index}.json.xz"
    return json.loads(lzma.decompress(packed.read_bytes()))


def _row_name(key):
    """Strip the seed that the comparison appends to a row name."""
    return key.split(" ")[0]


def _rows(summary, method):
    """Return the candidates of every row for one timing method.

    Returns:
        A mapping from distribution name to its candidates, in the order
        the harnesses list the distributions.
    """
    found = {
        _row_name(key): methods[method]
        for key, methods in summary["rows"].items()
    }
    return {name: found[name] for name in ROW_ORDER if name in found}


def _median(candidate):
    """Return the median of the per-process medians, in microseconds."""
    return candidate["median_us"]["median"]


def _ratio(numerator, denominator):
    """Form the ratio of two candidates inside each process.

    This is the rule the process driver states for its own ratios: every
    process contributes the ratio of its two medians.

    Returns:
        The median, minimum, and maximum of the per-process ratios.
    """
    ratios = [
        top / bottom
        for top, bottom in zip(
            numerator["median_us"]["process_values"],
            denominator["median_us"]["process_values"],
            strict=True,
        )
    ]
    return statistics.median(ratios), min(ratios), max(ratios)


def _family(candidates, family):
    """Return the configurations of one Triton family in a row.

    The family name is followed by `_b<block>` or `_w<warps>`, so
    `triton_planned` does not select `triton_planned_looped`.
    """
    return sorted(
        name
        for name in candidates
        if name.startswith((f"{family}_b", f"{family}_w"))
    )


def _best(candidates, names):
    """Return the name with the lowest median, or None for no names."""
    if not names:
        return None
    return min(names, key=lambda name: _median(candidates[name]))


def _config(name):
    """Return the block and warp part of a Triton candidate name."""
    for family in (
        "triton_planned_looped_",
        "triton_planned_",
        "triton_looped_",
        "triton_",
    ):
        if name.startswith(family):
            return name[len(family):]
    return name


def _us(value):
    """Format microseconds with one decimal."""
    return f"{value:,.1f}"


def _times(value):
    """Format a ratio with two decimals."""
    return f"{value:,.2f}"


def _spread(ratio):
    """Format a ratio as its median and its range across processes."""
    middle, low, high = ratio
    return f"{_times(middle)} [{_times(low)} to {_times(high)}]"


def _span(values):
    """Format the lowest and highest of several ratios, or the one value."""
    low, high = _times(min(values)), _times(max(values))
    return low if len(values) == 1 else f"{low} to {high}"


def _quote(text):
    """Escape record text so that Markdown shows it as written."""
    for character in "\\`*_<>|":
        text = text.replace(character, "\\" + character)
    return text


def _table(headers, rows, numeric=True):
    """Return a Markdown table.

    Args:
        headers: The column titles.
        rows: The cells of every row, as text.
        numeric: Whether the columns after the first hold numbers, which
            are aligned to the right.
    """
    rule = "---:|" if numeric else "---|"
    lines = [
        "| " + " | ".join(headers) + " |",
        "|---|" + rule * (len(headers) - 1),
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def _listed(names):
    """Join row names as code spans in a sentence."""
    spans = [f"`{name}`" for name in names]
    if len(spans) <= 2:
        return " and ".join(spans)
    return ", ".join(spans[:-1]) + ", and " + spans[-1]


def _count(count, total):
    """Say in how many rows something holds."""
    if count == total:
        return f"every one of the {total} rows"
    if count == 0:
        return f"none of the {total} rows"
    return f"{count} of the {total} rows"


# Fresh offsets.


def _fresh_rows(size, padded=False):
    """Return the end-to-end candidates of one fresh-offsets run."""
    run = f"fresh-{size}-{'all' if padded else 'nopad'}"
    return _rows(_load_summary(run), "end_to_end")


def _prepare_medians(size):
    """Return the median preparation time of `swage_mixed` per row.

    The summaries do not hold the part of a sample spent in preparation, so
    this reads the five process records: every process contributes the
    median of its samples, and the row reports the median across processes.
    """
    per_row = {}
    for index in range(1, 6):
        record = _load_process(f"fresh-{size}-nopad", index)
        for row in record["results"]:
            per_row.setdefault(row["distribution"], []).append(
                statistics.median(row["swage_mixed_prepare_samples_us"])
            )
    return {name: statistics.median(values) for name, values in per_row.items()}


def _planned_names(candidates):
    """Return every planned Triton configuration of a row, both kinds."""
    return _family(candidates, "triton_planned") + _family(
        candidates, "triton_planned_looped"
    )


def _fresh_swage_table(size):
    """Tabulate the two Swage candidates against torch at one size."""
    prepare = _prepare_medians(size)
    rows = []
    for name, candidates in _fresh_rows(size).items():
        torch = candidates["torch"]
        mixed = candidates["swage_mixed"]
        call = candidates["swage_cta_call"]
        rows.append(
            [
                f"`{name}`",
                _us(_median(torch)),
                _us(_median(mixed)),
                _us(prepare[name]),
                _spread(_ratio(mixed, torch)),
                _us(_median(call)),
                _spread(_ratio(call, torch)),
            ]
        )
    return _table(
        [
            "Distribution",
            "torch us",
            "`swage_mixed` us",
            "of which preparation us",
            "`swage_mixed` / torch",
            "`swage_cta_call` us",
            "`swage_cta_call` / torch",
        ],
        rows,
    )


def _fresh_triton_table(size):
    """Tabulate the Triton candidates of the fresh regime at one size."""
    rows = []
    for name, candidates in _fresh_rows(size).items():
        torch = candidates["torch"]
        mixed = candidates["swage_mixed"]
        looped_names = _family(candidates, "triton_looped")
        looped = _best(candidates, looped_names)
        below = sum(
            1
            for config in looped_names
            if _median(candidates[config]) <= _median(torch)
        )
        planned = _best(candidates, _planned_names(candidates))
        rows.append(
            [
                f"`{name}`",
                f"`{_config(looped)}`",
                _us(_median(candidates[looped])),
                _spread(_ratio(candidates[looped], torch)),
                f"{below} of {len(looped_names)}",
                f"`{planned[len('triton_'):]}`",
                _us(_median(candidates[planned])),
                _spread(_ratio(candidates[planned], torch)),
                _spread(_ratio(mixed, candidates[planned])),
            ]
        )
    return _table(
        [
            "Distribution",
            "Best looped",
            "us",
            "Best looped / torch",
            "Looped at or below torch",
            "Best planned",
            "us",
            "Best planned / torch",
            "`swage_mixed` / best planned",
        ],
        rows,
    )


def _fresh_ratios(size):
    """Collect the per-row median ratios that the statements quote.

    Returns:
        A mapping from a short key to the per-row values at this size.
    """
    out = {
        "mixed": [],
        "mixed_low": [],
        "call": [],
        "call_low": [],
        "looped": [],
        "looped_high": [],
        "looped_below": {},
        "planned": [],
        "mixed_planned": {},
        "mixed_looped": [],
    }
    for name, candidates in _fresh_rows(size).items():
        torch = candidates["torch"]
        mixed = _ratio(candidates["swage_mixed"], torch)
        call = _ratio(candidates["swage_cta_call"], torch)
        looped_names = _family(candidates, "triton_looped")
        looped = _best(candidates, looped_names)
        planned = _best(candidates, _planned_names(candidates))
        looped_ratio = _ratio(candidates[looped], torch)
        out["looped_below"][name] = sum(
            1
            for config in looped_names
            if _median(candidates[config]) <= _median(torch)
        )
        out["mixed"].append(mixed[0])
        out["mixed_low"].append(mixed[1])
        out["call"].append(call[0])
        out["call_low"].append(call[1])
        out["looped"].append(looped_ratio[0])
        out["looped_high"].append(looped_ratio[2])
        out["planned"].append(_ratio(candidates[planned], torch)[0])
        out["mixed_planned"][name] = _ratio(
            candidates["swage_mixed"], candidates[planned]
        )[0]
        out["mixed_looped"].append(
            _ratio(candidates["swage_mixed"], candidates[looped])[0]
        )
    return out


def _fresh_ranges_table():
    """Tabulate the range of every fresh-offsets ratio at each size."""
    rows = []
    for size in SIZES:
        ratios = _fresh_ratios(size)
        rows.append(
            [
                f"{size:,}",
                _span(ratios["mixed"]),
                _span(ratios["call"]),
                _span(ratios["looped"]),
                _span(ratios["planned"]),
                _span(list(ratios["mixed_planned"].values())),
            ]
        )
    return _table(
        [
            "Segments",
            "`swage_mixed` / torch",
            "`swage_cta_call` / torch",
            "Best looped Triton / torch",
            "Best planned Triton / torch",
            "`swage_mixed` / best planned Triton",
        ],
        rows,
    )


def _fresh_swage_statement():
    """State how the private planned sum compares with torch per size."""
    parts = []
    for size in SIZES:
        ratios = _fresh_ratios(size)
        slower = sum(1 for low in ratios["mixed_low"] if low > 1)
        parts.append(
            f"{_span(ratios['mixed'])} times as long at {size:,} segments "
            f"(slower in every process in {_count(slower, len(ROW_ORDER))})"
        )
    return (
        "With a new offsets layout on every call, the private planned sum "
        "(`swage_mixed`: preparation with schedule selection disabled, "
        "then the mixed launch) took longer than `torch.segment_reduce`: "
        + "; ".join(parts)
        + ". Each figure is the median across five processes of the ratio "
        "of the two per-process medians, and the range is over the nine "
        "distributions.\n"
    )


def _fresh_call_statement():
    """State how the single-policy Swage call compares with torch."""
    parts = []
    for size in SIZES:
        ratios = _fresh_ratios(size)
        slower = sum(1 for low in ratios["call_low"] if low > 1)
        parts.append(
            f"{_span(ratios['call'])} at {size:,} segments (slower in every "
            f"process in {_count(slower, len(ROW_ORDER))})"
        )
    return (
        "The single-policy call (`swage_cta_call`), which validates the "
        "offsets and launches the pure CTA kernel without classifying, also "
        "took longer than `torch.segment_reduce`: "
        + "; ".join(parts)
        + ". The loss is therefore not the planner alone.\n"
    )


def _fresh_looped_statement():
    """State how the best looped Triton configuration compares."""
    parts = []
    fewest = []
    for size in SIZES:
        ratios = _fresh_ratios(size)
        at_or_below = sum(1 for value in ratios["looped"] if value <= 1)
        parts.append(
            f"{_span(ratios['looped'])} at {size:,} segments (median at or "
            f"below one in {_count(at_or_below, len(ROW_ORDER))}; the "
            f"largest ratio of any process is "
            f"{_times(max(ratios['looped_high']))})"
        )
        row = min(ratios["looped_below"], key=ratios["looped_below"].get)
        fewest.append(
            f"{ratios['looped_below'][row]} of 15 on `{row}` at {size:,} "
            "segments"
        )
    return (
        "A looped Triton kernel with no planner, one program per segment "
        "that walks the segment in fixed blocks, is at or below "
        "`torch.segment_reduce` in the same regime. Taking the best of its "
        "15 block and warp configurations per row, chosen after the run, "
        "its ratio to torch is "
        + "; ".join(parts)
        + ". Not every configuration is: the fewest at or below torch are "
        + ", ".join(fewest)
        + ". The tables give the count for each row.\n"
    )


def _fresh_planned_statement():
    """State how planned Triton with a timed partition compares."""
    parts = []
    for size in SIZES:
        ratios = _fresh_ratios(size)
        versus = ratios["mixed_planned"]
        faster = sum(1 for value in versus.values() if value < 1)
        parts.append(
            f"at {size:,} segments it took {_span(ratios['planned'])} times "
            f"as long as torch, and `swage_mixed` took "
            f"{_span(list(versus.values()))} times as long as it (Swage "
            f"faster in {_count(faster, len(versus))})"
        )
    return (
        "Planned Triton with its partition inside the timed region (two "
        "`torch.nonzero` calls over the lengths, then a packed launch) is "
        "slower than `torch.segment_reduce` at every size. Taking the best "
        "of its configurations per row, chosen after the run: "
        + "; ".join(parts)
        + ".\n"
    )


def _pad_table():
    """Tabulate the pad-to-max baseline at every size."""
    rows = []
    per_size = {size: _fresh_rows(size, padded=True) for size in SIZES}
    for name in ROW_ORDER:
        row = [f"`{name}`"]
        for size in SIZES:
            candidates = per_size[size][name]
            if "torch_pad_to_max" in candidates:
                padded = candidates["torch_pad_to_max"]
                row.append(_us(_median(padded)))
                row.append(_times(_ratio(padded, candidates["torch"])[0]))
            else:
                row.extend(["not timed", "not timed"])
        rows.append(row)
    headers = ["Distribution"]
    for size in SIZES:
        headers.extend([f"{size:,}: us", f"{size:,}: / torch"])
    return _table(headers, rows)


def _pad_statement():
    """State how the pad-to-max baseline compares with torch."""
    ratios = []
    skipped = []
    for size in SIZES:
        for name, candidates in _fresh_rows(size, padded=True).items():
            if "torch_pad_to_max" in candidates:
                ratios.append(
                    _ratio(candidates["torch_pad_to_max"], candidates["torch"])[
                        0
                    ]
                )
            else:
                skipped.append(f"`{name}` at {size:,} segments")
    text = (
        "Pad-to-max in pure PyTorch (pad every segment with zeros to the "
        "longest one, then sum the masked rows) took "
        f"{_span(ratios)} times as long as `torch.segment_reduce` on the "
        "rows where it ran. The low end is the rows whose longest segment "
        "is 32 elements; one long segment sets the cost of every row."
    )
    if skipped:
        text += (
            " It was not timed on "
            + " and ".join(skipped)
            + ", where the padded matrix does not fit the free device memory."
        )
    return text + "\n"


def _pad_agreement_statement():
    """State how far the two runs of each size agree on `swage_mixed`."""
    worst = 0.0
    for size in SIZES:
        plain = _fresh_rows(size)
        padded = _fresh_rows(size, padded=True)
        for name in ROW_ORDER:
            first = _ratio(plain[name]["swage_mixed"], plain[name]["torch"])[0]
            second = _ratio(
                padded[name]["swage_mixed"], padded[name]["torch"]
            )[0]
            worst = max(worst, abs(second / first - 1))
    return (
        "Each size was run twice, once without the pad-to-max candidate and "
        "once with it. The fresh-offsets tables use the run without it, "
        "because that candidate allocates a large buffer on every call. The "
        "`swage_mixed` to torch ratios of the two runs differ by at most "
        f"{worst * 100:.1f} percent in any row.\n"
    )


# Frozen comparison.


def _frozen_rows(method):
    """Return the candidates of every comparison row for one method."""
    return _rows(_load_summary("comparison"), method)


def _frozen_swage_table(method):
    """Tabulate the three Swage policies against the PyTorch baselines."""
    rows = []
    for name, candidates in _frozen_rows(method).items():
        mixed = candidates["swage_mixed"]
        padded = candidates.get("torch_padded")
        rows.append(
            [
                f"`{name}`",
                _us(_median(mixed)),
                _us(_median(candidates["swage_cta"])),
                _us(_median(candidates["swage_warp"])),
                _us(_median(candidates["torch"])),
                _spread(_ratio(mixed, candidates["torch"])),
                _us(_median(padded)) if padded else "not timed",
            ]
        )
    return _table(
        [
            "Distribution",
            "`swage_mixed` us",
            "`swage_cta` us",
            "`swage_warp` us",
            "torch us",
            "`swage_mixed` / torch",
            "padded torch us",
        ],
        rows,
    )


_TRITON_FAMILIES = {
    "triton_looped": "looped",
    "triton_planned": "matched planned",
    "triton_planned_looped": "looping planned",
}


def _frozen_overview_table(method):
    """Tabulate `swage_mixed` against torch and each Triton family.

    Every comparator cell gives the median time of the comparator and, in
    parentheses, the median ratio of `swage_mixed` to it.
    """
    rows = []
    for name, candidates in _frozen_rows(method).items():
        mixed = candidates["swage_mixed"]
        torch = candidates["torch"]
        row = [
            f"`{name}`",
            _us(_median(mixed)),
            f"{_us(_median(torch))} ({_times(_ratio(mixed, torch)[0])})",
        ]
        for family in _TRITON_FAMILIES:
            best = _best(candidates, _family(candidates, family))
            if best is None:
                row.append("not timed")
                continue
            row.append(
                f"{_us(_median(candidates[best]))} "
                f"({_times(_ratio(mixed, candidates[best])[0])})"
            )
        rows.append(row)
    headers = ["Distribution", "`swage_mixed` us", "torch us"]
    headers.extend(
        f"Best {label} Triton us" for label in _TRITON_FAMILIES.values()
    )
    return _table(headers, rows)


def _frozen_family_table(method, family):
    """Tabulate `swage_mixed` against the best of one Triton family."""
    rows = []
    for name, candidates in _frozen_rows(method).items():
        mixed = candidates["swage_mixed"]
        names = _family(candidates, family)
        best = _best(candidates, names)
        if best is None:
            rows.append([f"`{name}`", _us(_median(mixed))] + ["not timed"] * 4)
            continue
        faster = sum(
            1
            for config in names
            if _median(candidates[config]) < _median(mixed)
        )
        rows.append(
            [
                f"`{name}`",
                _us(_median(mixed)),
                f"`{_config(best)}`",
                _us(_median(candidates[best])),
                _spread(_ratio(mixed, candidates[best])),
                f"{faster} of {len(names)}",
            ]
        )
    return _table(
        [
            "Distribution",
            "`swage_mixed` us",
            "Best configuration",
            "us",
            "`swage_mixed` / best",
            "Configurations faster than `swage_mixed`",
        ],
        rows,
    )


def _frozen_fixed_table(method):
    """Tabulate `swage_mixed` against the one-block-per-segment kernel."""
    rows = []
    for name, candidates in _frozen_rows(method).items():
        names = _family(candidates, "triton")
        best = _best(candidates, names)
        if best is None:
            rows.append([f"`{name}`", "not timed", "not timed", "not timed"])
            continue
        rows.append(
            [
                f"`{name}`",
                f"`{_config(best)}`",
                _us(_median(candidates[best])),
                _spread(_ratio(candidates["swage_mixed"], candidates[best])),
            ]
        )
    return _table(
        [
            "Distribution",
            "Best fixed Triton",
            "us",
            "`swage_mixed` / best fixed",
        ],
        rows,
    )


def _classes(ratios):
    """Sort rows into faster, equal, and slower by a median ratio.

    Args:
        ratios: The median ratio of `swage_mixed` to a comparator per row.

    Returns:
        Three mappings from row name to ratio: below one by more than the
        parity fraction, within it, and above one by more than it.
    """
    faster = {n: r for n, r in ratios.items() if r < 1 - PARITY}
    slower = {n: r for n, r in ratios.items() if r > 1 + PARITY}
    equal = {
        n: r for n, r in ratios.items() if n not in faster and n not in slower
    }
    return faster, equal, slower


def _class_sentence(label, ratios, not_timed):
    """Describe where `swage_mixed` is faster, equal, and slower."""
    faster, equal, slower = _classes(ratios)
    parts = []
    if faster:
        parts.append(
            f"faster on {_listed(faster)} "
            f"({_span(list(faster.values()))} times as long)"
        )
    if equal:
        parts.append(
            f"within {PARITY * 100:.0f} percent on {_listed(equal)} "
            f"({_span(list(equal.values()))})"
        )
    if slower:
        parts.append(
            f"slower on {_listed(slower)} "
            f"({_span(list(slower.values()))} times as long)"
        )
    text = f"- Against {label}, `swage_mixed` is " + "; ".join(parts) + "."
    if not_timed:
        text += f" The comparator was not timed on {_listed(not_timed)}."
    return text + "\n"


def _frozen_statements(method="graph"):
    """State the frozen comparison against each comparator."""
    rows = _frozen_rows(method)
    torch = {
        name: _ratio(c["swage_mixed"], c["torch"])[0]
        for name, c in rows.items()
    }
    lines = [_class_sentence("`torch.segment_reduce`", torch, [])]
    labels = {
        "triton_planned": (
            "the matched planned Triton scheduler (best of 4 warp settings, "
            "chosen after the run)"
        ),
        "triton_looped": (
            "the looped Triton kernel (best of 15 configurations, chosen "
            "after the run)"
        ),
        "triton_planned_looped": (
            "the looping planned Triton scheduler (best of 15 "
            "configurations, chosen after the run)"
        ),
    }
    for family, label in labels.items():
        ratios = {}
        missing = []
        for name, candidates in rows.items():
            best = _best(candidates, _family(candidates, family))
            if best is None:
                missing.append(name)
            else:
                ratios[name] = _ratio(
                    candidates["swage_mixed"], candidates[best]
                )[0]
        lines.append(_class_sentence(label, ratios, missing))
    return "".join(lines)


def _frozen_family_ratios(family, method="graph"):
    """Return the median ratio of `swage_mixed` to the best of a family."""
    ratios = {}
    for name, candidates in _frozen_rows(method).items():
        best = _best(candidates, _family(candidates, family))
        if best is not None:
            ratios[name] = _ratio(candidates["swage_mixed"], candidates[best])[
                0
            ]
    return ratios


def _losses():
    """List where the private Swage sum is the slower candidate."""
    fresh = {size: _fresh_ratios(size) for size in SIZES}
    lines = [
        "- A new offsets layout per call, against `torch.segment_reduce`: "
        "`swage_mixed` took "
        + ", ".join(
            f"{_span(fresh[size]['mixed'])} times as long at {size:,} "
            "segments"
            for size in SIZES
        )
        + ", in every row.",
        "- A new offsets layout per call, against a looped Triton kernel "
        "(best of 15 configurations, chosen after the run): `swage_mixed` "
        "took "
        + ", ".join(
            f"{_span(fresh[size]['mixed_looped'])} times as long at "
            f"{size:,} segments"
            for size in SIZES
        )
        + ", in every row.",
    ]
    largest = fresh[SIZES[-1]]["mixed_planned"]
    slower = {name: ratio for name, ratio in largest.items() if ratio > 1}
    lines.append(
        "- A new offsets layout per call, against planned Triton with its "
        "partition timed (best configuration, chosen after the run): at "
        f"{SIZES[-1]:,} segments `swage_mixed` took "
        f"{_span(list(slower.values()))} times as long in "
        f"{_count(len(slower), len(largest))}. At the two smaller sizes it "
        "was faster or equal in most rows."
    )
    labels = {
        "triton_looped": "a looped Triton kernel",
        "triton_planned": "the matched planned Triton scheduler",
        "triton_planned_looped": "the looping planned Triton scheduler",
    }
    for family, label in labels.items():
        _, _, slower = _classes(_frozen_family_ratios(family))
        lines.append(
            f"- A frozen layout under graph replay, against {label} (best "
            "configuration, chosen after the run): `swage_mixed` took "
            f"{_span(list(slower.values()))} times as long on "
            f"{_listed(slower)}."
        )
    return "\n".join(lines) + "\n"


# Run facts and conditions.


def _runs():
    """Return every run directory of the record, fresh runs first."""
    names = [
        f"fresh-{size}-{kind}" for size in SIZES for kind in ("nopad", "all")
    ]
    return [*names, "comparison"]


def _process_records():
    """Return `(run, record)` for every process of every run."""
    return [
        (run, record)
        for run in _runs()
        for record in _load_summary(run)["process_records"]
    ]


def _percent(text):
    """Read a utilization such as `7 %` as an integer."""
    return int(text.split()[0])


def _conditions_text():
    """Describe the machine state that the process records hold."""
    records = _process_records()
    before = [record["gpu_state_before"]["gpu"] for _, record in records]
    utilization = sorted(_percent(gpu["utilization.gpu"]) for gpu in before)
    temperature = sorted(int(gpu["temperature.gpu"]) for gpu in before)
    busy = [
        f"`{run}/{record['record']}`"
        for run, record in records
        if _percent(record["gpu_state_before"]["gpu"]["utilization.gpu"])
        == 100
    ]
    idle = [value for value in utilization if value < 100]
    governors = set()
    preferences = set()
    drivers = set()
    for _, record in records:
        for key in ("cpu_frequency_before", "cpu_frequency_after"):
            frequency = record[key]
            governors.update(frequency["governors"])
            preferences.add(frequency["energy_performance_preference"])
            drivers.add(frequency["scaling_driver"])
    seen = []
    for run, record in records:
        for moment in ("before", "after"):
            for other in record[f"gpu_state_{moment}"][
                "other_compute_processes"
            ]:
                seen.append(
                    f"`{run}/{record['record']}` listed `"
                    f"{other['process_name']}` with {other['used_memory']} "
                    f"{moment} it ran"
                )
    clean = sum(1 for _, record in records if record["worktree_clean"])
    unseen = sum(
        1 for _, record in records if not record["other_compute_process_seen"]
    )
    started = min(
        record["gpu_state_before"]["sampled_at"] for _, record in records
    )
    ended = max(
        record["gpu_state_after"]["sampled_at"] for _, record in records
    )
    lines = [
        f"- All {len(records)} processes ran between `{started}` and "
        f"`{ended}` (UTC), one after another, and {clean} of them recorded a "
        "clean worktree.",
        f"- CPU frequency governor: `{'`, `'.join(sorted(governors))}` on "
        "every CPU, before and after every process, with scaling driver `"
        f"{'`, `'.join(sorted(drivers))}` and energy performance preference "
        f"`{'`, `'.join(sorted(preferences))}`.",
        f"- GPU utilization sampled before each process: {idle[0]} to "
        f"{idle[-1]} percent in {len(idle)} processes and 100 percent in "
        f"{len(busy)} ({' and '.join(busy)}). GPU temperature before each "
        f"process: {temperature[0]} to {temperature[-1]} C.",
        f"- Other compute processes on the GPU: {unseen} of "
        f"{len(records)} processes saw none in either sample. "
        + ("; ".join(seen) + "." if seen else "None was listed."),
    ]
    return "\n".join(lines) + "\n"


def _facts():
    """Collect the run facts that the page states, from the records."""
    fresh = _load_process("fresh-32768-nopad")
    comparison = _load_process("comparison")
    code = _load_summary("comparison")["code"]
    for run in _runs():
        other = _load_summary(run)["code"]
        for key in ("revision", "gpu", "cpu_model", "pytorch", "triton"):
            if other[key] != code[key]:
                raise ValueError(f"{run} differs from comparison in {key}")
    gpu = _load_summary("comparison")["process_records"][0]["gpu_state_before"]
    return {
        "revision": code["revision"],
        "gpu": code["gpu"],
        "cpu": code["cpu_model"],
        "pytorch": code["pytorch"],
        "triton": code["triton"],
        "nvidia_driver": gpu["gpu"]["driver_version"],
        "cuda_driver": comparison["environment"]["cuda_driver"],
        "target": comparison["environment"]["compute_capability"],
        "python": comparison["environment"]["python"].split(" ")[0],
        "platform": comparison["environment"]["platform"],
        "llvm": comparison["provenance"]["llvm_linked"],
        "swage": comparison["provenance"]["swage"],
        "fresh": fresh["configuration"],
        "comparison": comparison["methodology"],
    }


def _command(run):
    """Return the driver command that produced one run."""
    summary = _load_summary(run)
    harness = " ".join(summary["command"])
    return (
        "python benchmarks/benchmark_processes.py "
        f"--processes {summary['processes']} "
        f'--output-dir "$OUT/{run}" \\\n  -- {harness}'
    )


def _irregular_table():
    """List every candidate whose process medians spread widely."""
    rows = []
    for run in _runs():
        summary = _load_summary(run)
        for key, methods in summary["rows"].items():
            for method, candidates in methods.items():
                for name, candidate in sorted(candidates.items()):
                    values = candidate["median_us"]["process_values"]
                    if max(values) / min(values) > IRREGULAR:
                        rows.append(
                            [
                                f"`{run}`",
                                f"`{_row_name(key)}`",
                                f"`{method}`",
                                f"`{name}`",
                                ", ".join(_us(value) for value in values),
                            ]
                        )
    return (
        _table(
            ["Run", "Distribution", "Timing", "Candidate", "Process us"],
            rows,
            numeric=False,
        ),
        len(rows),
    )


# Page and fragments.


def _fact_table(rows):
    """Return a two-column table of facts with text cells."""
    return _table(["Fact", "Value"], rows, numeric=False)


def _blocks():
    """Return every generated block of the page by its template token."""
    facts = _facts()
    fresh = facts["fresh"]
    comparison = facts["comparison"]
    irregular, irregular_count = _irregular_table()
    distributions = ", ".join(f"`{name}`" for name in ROW_ORDER)
    blocks = {
        "gpu": facts["gpu"],
        "target": facts["target"],
        "revision": facts["revision"],
        "seed": str(fresh["seed"]),
        "parity_percent": f"{PARITY * 100:.0f}",
        "runs_table": _table(
            ["Run", "Harness command"],
            [
                [f"`{run}`", f"`{' '.join(_load_summary(run)['command'])}`"]
                for run in _runs()
            ],
            numeric=False,
        ),
        "fresh_facts_table": _fact_table(
            [
                ["Segments per layout", "2,048, 8,192, and 32,768"],
                ["Distributions", distributions],
                ["Seed", str(fresh["seed"])],
                ["Layout seeds", _quote(fresh["layout_seeds"])],
                ["Layout reuse", _quote(fresh["layout_reuse"])],
                ["Warmup layouts per row", str(fresh["warmups"])],
                ["Timed layouts per row", str(fresh["samples"])],
                ["Warm step", _quote(fresh["warm_step"]["step"])],
                ["Values", _quote(fresh["values"])],
                ["Candidate order", _quote(fresh["candidate_order"])],
                ["Clock", _quote(fresh["clock"])],
                ["Correctness", _quote(fresh["correctness"])],
            ]
        ),
        "fresh_candidates_table": _table(
            ["Candidate", "Timed region"],
            [
                [f"`{name}`", _quote(text)]
                for name, text in fresh["timed_region"].items()
            ],
            numeric=False,
        ),
        "fresh_policy_note": "> " + _quote(fresh["swage_policy"]) + "\n",
        "comparison_facts_table": _fact_table(
            [
                ["Segments per row", f"{comparison['segment_count']:,}"],
                ["Distributions", distributions],
                ["Seeds", ", ".join(str(s) for s in comparison["seeds"])],
                ["Values", f"`{comparison['values']}`: every value is one"],
                ["Warmups per candidate", str(comparison["warmups"])],
                ["Samples per candidate", str(comparison["samples"])],
                [
                    "Launches per event or graph sample",
                    str(comparison["batched_launches"]),
                ],
                ["Batching", _quote(comparison["batching"])],
                ["Candidate order", _quote(comparison["candidate_order"])],
                ["Correctness", _quote(comparison["correctness"])],
            ]
        ),
        "triton_families_table": _table(
            ["Candidate", "Description"],
            [
                [f"`{name}`", _quote(comparison[key])]
                for name, key in (
                    ("triton (fixed)", "triton_fixed"),
                    ("triton_looped", "triton_looped"),
                    ("triton_planned", "triton_planned"),
                    ("triton_planned_looped", "triton_planned_looped"),
                    ("torch_padded", "torch_padded"),
                )
            ],
            numeric=False,
        ),
        "machine_table": _fact_table(
            [
                ["Revision", f"`{facts['revision']}`"],
                ["GPU", f"{facts['gpu']} (`{facts['target']}`)"],
                ["NVIDIA driver", facts["nvidia_driver"]],
                ["CUDA driver API", facts["cuda_driver"]],
                ["CPU", facts["cpu"]],
                ["Platform", facts["platform"]],
                ["Python", facts["python"]],
                ["PyTorch", facts["pytorch"]],
                ["Triton", facts["triton"]],
                ["Swage package version", facts["swage"]],
                ["LLVM linked by the bindings", facts["llvm"]],
            ]
        ),
        "conditions_list": _conditions_text(),
        "conditions_file": (RECORD / "conditions.txt").read_text(),
        "fresh_swage_statement": _fresh_swage_statement(),
        "fresh_call_statement": _fresh_call_statement(),
        "fresh_looped_statement": _fresh_looped_statement(),
        "fresh_planned_statement": _fresh_planned_statement(),
        "fresh_ranges_table": _fresh_ranges_table(),
        "pad_statement": _pad_statement(),
        "pad_agreement_statement": _pad_agreement_statement(),
        "pad_table": _pad_table(),
        "frozen_statements": _frozen_statements("graph"),
        "frozen_fixed_table_graph": _frozen_fixed_table("graph"),
        "irregular_count": str(irregular_count),
        "irregular_ratio": str(IRREGULAR),
        "irregular_table": irregular,
        "reproduce_commands": "\n".join(_command(run) for run in _runs())
        + "\n",
    }
    for size in SIZES:
        blocks[f"fresh_swage_table_{size}"] = _fresh_swage_table(size)
        blocks[f"fresh_triton_table_{size}"] = _fresh_triton_table(size)
    for method in ("graph", "call"):
        blocks[f"frozen_swage_table_{method}"] = _frozen_swage_table(method)
        blocks[f"frozen_overview_table_{method}"] = _frozen_overview_table(
            method
        )
    for family in _TRITON_FAMILIES:
        blocks[f"frozen_{family}_table_graph"] = _frozen_family_table(
            "graph", family
        )
    return blocks


def _require(condition, claim):
    """Fail generation when a fixed sentence of the template is false."""
    if not condition:
        raise ValueError(f"the records do not support: {claim}")


def check_claims():
    """Check the qualitative sentences that the template states as text.

    The numbers on the page are generated. Its sentences about which
    candidate is faster are written in the template, so each one is tested
    here against the summaries before a page is produced.
    """
    for size in SIZES:
        ratios = _fresh_ratios(size)
        _require(
            min(ratios["mixed_low"]) > 1,
            f"swage_mixed is slower than torch in every process at {size}",
        )
        _require(
            min(ratios["call_low"]) > 1,
            f"swage_cta_call is slower than torch in every process at {size}",
        )
        _require(
            max(ratios["looped"]) <= 1,
            f"the best looped Triton is at or below torch at {size}",
        )
        _require(
            min(ratios["planned"]) > 1,
            f"the best planned Triton is slower than torch at {size}",
        )
        _require(
            min(ratios["mixed_looped"]) > 1,
            f"swage_mixed is slower than the best looped Triton at {size}",
        )
        if size != SIZES[-1]:
            versus = ratios["mixed_planned"].values()
            _require(
                sum(1 for value in versus if value <= 1 + PARITY) * 2
                > len(ROW_ORDER),
                f"swage_mixed is faster than or equal to planned at {size}",
            )
    rows = _frozen_rows("graph")
    torch = {
        name: _ratio(c["swage_mixed"], c["torch"])[0]
        for name, c in rows.items()
    }
    faster, equal, slower = _classes(torch)
    _require(
        set(equal) == {"uniform"} and not slower and len(faster) == 8,
        "frozen swage_mixed is faster than torch except on uniform",
    )
    looped = {}
    for name, candidates in rows.items():
        best = _best(candidates, _family(candidates, "triton_looped"))
        looped[name] = _ratio(candidates["swage_mixed"], candidates[best])[0]
    faster, equal, slower = _classes(looped)
    _require(
        set(faster)
        == {"many-tiny", "one-outlier", "alternating-empty", "power-law"},
        "frozen swage_mixed beats looped Triton on the short and split rows",
    )
    _require(
        set(slower) == {"log-normal", "bimodal", "zipf-like", "few-huge"},
        "frozen looped Triton beats swage_mixed on the mid-length rows",
    )
    for family in ("triton_planned", "triton_planned_looped"):
        faster, _, _ = _classes(_frozen_family_ratios(family))
        _require(
            len(faster) * 2 < len(_frozen_family_ratios(family)),
            f"{family} matches or beats swage_mixed on most rows",
        )
    _, equal, slower = _classes(_frozen_family_ratios("triton_planned"))
    _require(
        set(equal) == {"uniform", "bimodal", "zipf-like", "few-huge"}
        and set(slower) == {"many-tiny", "one-outlier", "alternating-empty"},
        "matched planned Triton ties four rows and wins the packed rows",
    )
    faster, _, slower = _classes(
        _frozen_family_ratios("triton_planned_looped")
    )
    _require(
        "log-normal" in slower and "power-law" in faster,
        "looping planned Triton wins log-normal and loses power-law",
    )
    faster, _, _ = _classes(_frozen_family_ratios("triton_planned"))
    _require(
        set(faster) == {"log-normal"},
        "swage_mixed is ahead of matched planned Triton on log-normal only",
    )


def page():
    """Return the whole summary page: the template with its blocks."""
    check_claims()
    text = TEMPLATE.read_text()
    for token, block in _blocks().items():
        text = text.replace("{{" + token + "}}", block.rstrip("\n"))
    if "{{" in text:
        unknown = text[text.index("{{"): text.index("{{") + 40]
        raise ValueError(f"the template has an unknown token near {unknown!r}")
    return text.replace(
        f"<!-- benchmarks/results/{NAME}/page.md.in -->",
        f"<!-- benchmarks/results/{NAME}.md -->\n"
        "<!-- Generated by benchmarks/campaign_tables.py from "
        f"{NAME}/page.md.in. Do not edit; rerun the script. -->",
        1,
    )


def fragments():
    """Return the documentation fragments by file name."""
    header = (
        "<!-- docs/internals/_generated/{name} -->\n"
        "<!-- Generated by benchmarks/campaign_tables.py from "
        f"benchmarks/results/{NAME}. Do not edit. -->\n\n"
    )
    bodies = {
        "fresh-ranges.inc": _fresh_ranges_table(),
        "fresh-32768.inc": _fresh_swage_table(32768),
        "fresh-triton-32768.inc": _fresh_triton_table(32768),
        "fresh-statement.inc": _fresh_swage_statement(),
        "fresh-call-statement.inc": _fresh_call_statement(),
        "fresh-looped-statement.inc": _fresh_looped_statement(),
        "fresh-planned-statement.inc": _fresh_planned_statement(),
        "pad-statement.inc": _pad_statement(),
        "frozen-swage.inc": _frozen_swage_table("graph"),
        "frozen-overview.inc": _frozen_overview_table("graph"),
        "frozen-looped.inc": _frozen_family_table("graph", "triton_looped"),
        "frozen-statements.inc": _frozen_statements("graph"),
        "losses.inc": _losses(),
        "reproduce.inc": (
            "```sh\n"
            'export PYTHONPATH="$PWD/python:$PWD/build/python_packages"\n'
            + "\n".join(_command(run) for run in _runs())
            + "\n```\n"
        ),
    }
    return {
        f"{NAME}-{name}": header.format(name=f"{NAME}-{name}") + body
        for name, body in bodies.items()
    }


def check_digests():
    """Return one message per process record that fails its digest."""
    errors = []
    for run, record in _process_records():
        packed = RECORD / run / f"{record['record']}.xz"
        try:
            digest = hashlib.sha256(
                lzma.decompress(packed.read_bytes())
            ).hexdigest()
        except (OSError, lzma.LZMAError) as error:
            errors.append(f"cannot read {packed}: {error}")
            continue
        if digest != record["sha256"]:
            errors.append(f"digest mismatch: {packed}")
    return errors


def outputs():
    """Return every generated file and its content."""
    files = {PAGE: page()}
    for name, body in fragments().items():
        files[FRAGMENTS / name] = body
    return files


def check():
    """Return one message per stale, missing, or orphaned output."""
    errors = check_digests()
    files = outputs()
    for path, content in files.items():
        if not path.is_file():
            errors.append(f"missing generated file: {path}")
        elif path.read_text() != content:
            errors.append(f"stale generated file: {path}")
    for path in sorted(FRAGMENTS.glob(f"{NAME}-*")):
        if path not in files:
            errors.append(f"orphaned generated file: {path}")
    return errors


def main(argv=None):
    """Write or check the page and the fragments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the committed files and digests instead of writing",
    )
    arguments = parser.parse_args(argv)
    if arguments.check:
        errors = check()
        for error in errors:
            print(error)
        return 1 if errors else 0
    FRAGMENTS.mkdir(parents=True, exist_ok=True)
    for path, content in outputs().items():
        path.write_text(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
