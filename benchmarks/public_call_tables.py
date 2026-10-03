# benchmarks/public_call_tables.py
"""Generate the summary pages of the records of the public segmented calls.

Such a record is a directory under `benchmarks/results/` that holds runs of
`benchmarks/benchmark_fresh_offsets.py`, each repeated in independent
processes by `benchmarks/benchmark_processes.py`, which time the public
`swage.segment_reduce` call with a new offsets layout on every call. This
script reads the committed `summary.json` of every run, and the first
compressed process record of every run for the facts that a summary does not
repeat. It fills the tokens of `page.md.in` in the record directory to write
the Markdown page beside the record, and it writes the fragments that the
documentation includes, so no measured number in either is typed by hand.
The ratio rule and the cell formats are those of `campaign_tables.py`.

`RECORDS` names every record it renders. Run from the repository root:

    python benchmarks/public_call_tables.py           # write every record
    python benchmarks/public_call_tables.py --check   # verify the outputs,
                                                     # digests, and PTX
    python benchmarks/public_call_tables.py --record NAME   # one record

It needs only the standard library.
"""

import argparse
import functools
import hashlib
import json
import lzma
import re
from pathlib import Path

from campaign_tables import (
    _count,
    _fact_table,
    _median,
    _percent,
    _quote,
    _ratio,
    _span,
    _spread,
    _table,
    _us,
)

REPO_ROOT = Path(__file__).parents[1]
RESULTS = REPO_ROOT / "benchmarks/results"
FRAGMENTS = REPO_ROOT / "docs/internals/_generated"
# Every record of the public calls that this script renders.
RECORDS = ("segment-reduce-a6000-sm86-c6099ec",)
PUBLIC = "swage_public_call"
PUBLIC_INT64 = "swage_public_call_int64"
# The looped Triton families: rank-one values, then `[N, D]` values.
LOOPED = ("triton_looped_", "triton_rows_looped_")
# The configuration facts that every run of a record must share.
SHARED = (
    "seed", "layout_seeds", "layout_reuse", "warmups", "samples",
    "warm_step", "kinds", "dtype", "values", "pipeline", "candidate_order",
    "clock", "correctness", "rank_two", "timed_region",
    "candidate_descriptions",
)


@functools.cache
def _load_summary(name, run):
    """Return the committed summary of one run. Callers must not change it."""
    return json.loads((RESULTS / name / run / "summary.json").read_text())


@functools.cache
def _load_process(name, run, index=1):
    """Return one decompressed per-process record of one run."""
    packed = RESULTS / name / run / f"process-{index}.json.xz"
    return json.loads(lzma.decompress(packed.read_bytes()))


def _configuration(name, run):
    """Return the harness configuration of one run."""
    return _load_process(name, run)["configuration"]


@functools.cache
def _runs(name):
    """Return every run of a record: by rank, widest D, then segments."""

    def key(run):
        configuration = _configuration(name, run)
        return (
            configuration["rank"],
            max(configuration["features"] or [0]),
            configuration["segment_count"],
        )

    summaries = (RESULTS / name).glob("*/summary.json")
    return tuple(sorted((path.parent.name for path in summaries), key=key))


def _label(result):
    """Label a result row as `benchmark_processes.py` labels it.

    The label is the distribution, then `D=<features>` for `[N, D]`
    values, the kind when it is not a sum, and the type when it is not
    float32.
    """
    parts = [result["distribution"]]
    if result.get("features") is not None:
        parts.append(f"D={result['features']}")
    if result.get("kind", "sum") != "sum":
        parts.append(result["kind"])
    if result.get("dtype", "float32") != "float32":
        parts.append(result["dtype"])
    return " ".join(parts)


def _rows(name, run):
    """Return the end-to-end candidates of every row of one run.

    Returns:
        A mapping from row label to its candidates, in the order the first
        process ran the rows.

    Raises:
        ValueError: The process record and the summary hold other rows.
    """
    summary = _load_summary(name, run)["rows"]
    labels = [_label(result) for result in _load_process(name, run)["results"]]
    if sorted(labels) != sorted(summary):
        raise ValueError(f"{name}/{run}: process rows differ from the summary")
    return {label: summary[label]["end_to_end"] for label in labels}


def _best_looped(candidates):
    """Return the looped Triton configuration with the lowest median.

    Returns:
        The candidate name, or None when the row timed no looped Triton.
    """
    names = sorted(name for name in candidates if name.startswith(LOOPED))
    if not names:
        return None
    return min(names, key=lambda name: _median(candidates[name]))


def _config(name):
    """Return the block or row and warp part of a looped Triton name."""
    for family in LOOPED:
        if name.startswith(family):
            return name[len(family):]
    return name


def _family(name):
    """Return the candidate family that a timed configuration belongs to."""
    return re.sub(r"(_[br]\d+)?_w\d+$", "", name)


def _title(name, run):
    """Describe one run: its rank, its segment count, and its widths."""
    configuration = _configuration(name, run)
    rank = {1: "Rank one", 2: "Rank two"}[configuration["rank"]]
    text = f"{rank}, {configuration['segment_count']:,} segments"
    widths = [str(width) for width in configuration["features"] or []]
    if len(widths) > 1:
        widths = [", ".join(widths[:-1]), widths[-1]]
    if widths:
        text += ", D of " + " and ".join(widths)
    return text


# Ratios, tables, and statements.


@functools.cache
def _by_run(name):
    """Return the median ratios that the page uses, per run and row.

    Returns:
        A mapping from run to one mapping per row, which holds the row
        label and the per-process ratios of the public call and of the
        best looped Triton configuration.
    """
    out = {}
    for run in _runs(name):
        out[run] = []
        for label, candidates in _rows(name, run).items():
            torch = candidates["torch"]
            public = candidates[PUBLIC]
            looped = candidates[_best_looped(candidates)]
            int64 = candidates[PUBLIC_INT64]
            out[run].append(
                {
                    "row": label,
                    "public": _ratio(public, torch),
                    "int64": _ratio(int64, torch),
                    "int64_public": _ratio(int64, public),
                    "looped": _ratio(looped, torch),
                    "public_looped": _ratio(public, looped),
                }
            )
    return out


def _medians(items, key):
    """Return the median ratio of one kind for every row."""
    return [item[key][0] for item in items]


def _run_table(name, run, candidates_of):
    """Tabulate Swage candidates of every row of a run against torch.

    Args:
        name: The record.
        run: The run.
        candidates_of: Returns the Swage candidates to show for a row.

    Returns:
        The table, with the best looped Triton configuration when the
        candidates are the public ones, or None for no candidates.
    """
    rows = []
    names = []
    for label, candidates in _rows(name, run).items():
        names = candidates_of(candidates)
        if not names:
            return None
        torch = candidates["torch"]
        row = [f"`{label}`", _us(_median(torch))]
        for candidate in names:
            row.append(_us(_median(candidates[candidate])))
            row.append(_spread(_ratio(candidates[candidate], torch)))
        if PUBLIC in names:
            looped = _best_looped(candidates)
            row.append(f"`{_config(looped)}`")
            row.append(_us(_median(candidates[looped])))
            row.append(_spread(_ratio(candidates[looped], torch)))
        rows.append(row)
    headers = ["Row", "torch us"]
    for candidate in names:
        headers.extend([f"`{candidate}` us", f"`{candidate}` / torch"])
    if PUBLIC in names:
        headers.extend(["Best looped Triton", "us", "Best looped / torch"])
    return _table(headers, rows)


def _private(name, run):
    """Return a function that lists the other Swage candidates of a row."""
    order = _configuration(name, run)["candidates"]
    return lambda candidates: [
        candidate
        for candidate in order
        if candidate.startswith("swage_")
        and candidate not in (PUBLIC, PUBLIC_INT64)
        and candidate in candidates
    ]


def _run_tables(name):
    """Return a section with the tables of every run."""
    parts = []
    for run in _runs(name):
        parts.append(f"### {_title(name, run)}\n")
        parts.append(f"Run `{run}`. The public call and looped Triton:\n")
        parts.append(_run_table(name, run, lambda _: [PUBLIC, PUBLIC_INT64]))
        private = _run_table(name, run, _private(name, run))
        if private is not None:
            parts.append("The other Swage candidates of the run:\n")
            parts.append(private)
    return "\n".join(parts)


def _ranges_table(name):
    """Tabulate the range of every median ratio over the rows of each run."""
    keys = ("public", "int64", "looped", "public_looped")
    return _table(
        [
            "Run",
            "Rows",
            f"`{PUBLIC}` / torch",
            f"`{PUBLIC_INT64}` / torch",
            "Best looped Triton / torch",
            f"`{PUBLIC}` / best looped Triton",
        ],
        [
            [_title(name, run), str(len(items))]
            + [_span(_medians(items, key)) for key in keys]
            for run, items in _by_run(name).items()
        ],
    )


def _public_statement(name):
    """State how the public call compares with torch in every run."""
    lines = [
        f"`swage.segment_reduce` (`{PUBLIC}`) against "
        "`torch.segment_reduce`, with a new offsets layout on every call. "
        "Each figure is the median across processes of the ratio of the "
        "two per-process medians, and the range is over the rows of the "
        "run:",
        "",
    ]
    everything = []
    for run, items in _by_run(name).items():
        everything.extend(items)
        slower = sum(1 for item in items if item["public"][1] > 1)
        low = min(items, key=lambda item: item["public"][0])["row"]
        high = max(items, key=lambda item: item["public"][0])["row"]
        lines.append(
            f"- {_title(name, run)}: {_span(_medians(items, 'public'))} "
            f"times as long, lowest on `{low}` and highest on `{high}`; "
            f"slower in every process in {_count(slower, len(items))}."
        )
    lines += [
        "",
        f"On int64 offsets (`{PUBLIC_INT64}`) the call took "
        f"{_span(_medians(everything, 'int64_public'))} times as long as on "
        "the int32 offsets of the same layouts, over all "
        f"{len(everything)} rows.",
    ]
    return "\n".join(lines) + "\n"


def _looped_statement(name):
    """State how the best looped Triton configuration compares."""
    lines = [
        "The best looped Triton configuration of each row, chosen after the "
        "run, against `torch.segment_reduce`. A looped kernel has no "
        "planner: one program per segment, and per block of columns for "
        "`[N, D]` values, walks its segment in fixed blocks:",
        "",
    ]
    everything = []
    for run, items in _by_run(name).items():
        everything.extend(items)
        at_or_below = sum(1 for item in items if item["looped"][0] <= 1)
        lines.append(
            f"- {_title(name, run)}: {_span(_medians(items, 'looped'))} "
            "times as long; the median is at or below one in "
            f"{_count(at_or_below, len(items))}."
        )
    slower = sum(1 for item in everything if item["public_looped"][1] > 1)
    lines += [
        "",
        f"`{PUBLIC}` took {_span(_medians(everything, 'public_looped'))} "
        "times as long as the best looped configuration of the row, and "
        f"longer in every process in {_count(slower, len(everything))}.",
    ]
    return "\n".join(lines) + "\n"


def _losses(name):
    """List, per run, the rows where the public call is the slower one."""
    lines = []
    for run, items in _by_run(name).items():
        clauses = []
        for key, label in (
            ("public", "`torch.segment_reduce`"),
            ("public_looped", "the best looped Triton configuration"),
        ):
            slower = [value for value in _medians(items, key) if value > 1]
            if slower:
                clauses.append(
                    f"{_span(slower)} times as long as {label} in "
                    f"{_count(len(slower), len(items))}"
                )
        if clauses:
            lines.append(
                f"- {_title(name, run)}: `{PUBLIC}` took "
                + ", and ".join(clauses)
                + "."
            )
    return "\n".join(lines) + "\n"


# Run facts, conditions, and kernels.


def _process_records(name):
    """Return `(run, record)` for every process of every run."""
    return [
        (run, record)
        for run in _runs(name)
        for record in _load_summary(name, run)["process_records"]
    ]


def _load_averages(name):
    """Return the one-minute load averages that `conditions.txt` holds."""
    text = (RESULTS / name / "conditions.txt").read_text()
    return re.findall(r"load average: ([0-9.]+),", text)


def _conditions_text(name):
    """Describe the machine state that the process records hold.

    The names of other compute processes are not published, because a
    desktop program can be among them.
    """
    records = [record for _, record in _process_records(name)]
    before = [record["gpu_state_before"] for record in records]
    gpus = [state["gpu"] for state in before]
    utilization = sorted(_percent(gpu["utilization.gpu"]) for gpu in gpus)
    temperature = sorted(int(gpu["temperature.gpu"]) for gpu in gpus)
    governors = sorted(
        {
            governor
            for record in records
            for key in ("cpu_frequency_before", "cpu_frequency_after")
            for governor in record[key]["governors"]
        }
    )
    seen = [
        f"`{run}/{record['record']}`"
        for run, record in _process_records(name)
        if record["other_compute_process_seen"] is not False
    ]
    clean = sum(1 for record in records if record["worktree_clean"])
    lines = [
        f"- All {len(records)} processes ran between "
        f"`{min(state['sampled_at'] for state in before)}` and "
        f"`{max(r['gpu_state_after']['sampled_at'] for r in records)}` "
        f"(UTC), one after another, and {clean} of them recorded a clean "
        "worktree.",
        f"- CPU frequency governor: `{'`, `'.join(governors)}` on every "
        "CPU, before and after every process.",
        f"- GPU utilization sampled before each process: {utilization[0]} to "
        f"{utilization[-1]} percent. GPU temperature before each process: "
        f"{temperature[0]} to {temperature[-1]} C.",
        f"- Other compute processes on the GPU: {len(records) - len(seen)} "
        f"of {len(records)} processes saw none in either sample"
        + (f"; {', '.join(seen)} saw one or could not read a sample."
           if seen else "."),
    ]
    return "\n".join(lines) + "\n"


def _shared(name):
    """Return the code and configuration that every run of a record shares.

    Raises:
        ValueError: Two runs differ in code, machine, or configuration.
    """
    first, *others = _runs(name)
    code = _load_summary(name, first)["code"]
    configuration = _configuration(name, first)
    for run in others:
        for key in ("revision", "gpu", "cpu_model", "pytorch", "triton"):
            if _load_summary(name, run)["code"][key] != code[key]:
                raise ValueError(f"{run} differs from {first} in {key}")
        for key in SHARED:
            if _configuration(name, run)[key] != configuration[key]:
                raise ValueError(f"{run} differs from {first} in {key}")
    return code, configuration, _load_process(name, first)


def _families_table(name, headers, texts, cells):
    """Tabulate the record text of every candidate family a run timed.

    Args:
        name: The record.
        headers: The column titles after the candidate.
        texts: A mapping from candidate family to its record text.
        cells: Returns the cells after the candidate from a record text.
    """
    timed = {
        _family(candidate)
        for run in _runs(name)
        for candidates in _rows(name, run).values()
        for candidate in candidates
    }
    return _table(
        ["Candidate", *headers],
        [
            [f"`{family}`", *cells(text)]
            for family, text in texts.items()
            if family in timed
        ],
        numeric=False,
    )


@functools.cache
def _loaded_ptx(name):
    """Return every PTX module the processes loaded, by SHA-256.

    Returns:
        A mapping from digest to the kernel name, the size in bytes, and
        the runs whose summaries list it.
    """
    found = {}
    for run in _runs(name):
        for module in _load_summary(name, run)["code"]["loaded_ptx"]:
            entry = found.setdefault(
                module["sha256"], {**module, "runs": []}
            )
            entry["runs"].append(run)
    return found


def _ptx_table(name):
    """Tabulate every committed PTX file and the runs that loaded it."""
    rows = []
    for path in sorted((RESULTS / name).glob("*.ptx")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        module = _loaded_ptx(name)[digest]
        rows.append(
            [
                f"`{path.name}`",
                f"`{module['kernel']}`",
                f"{module['bytes']:,}",
                f"`{digest}`",
                ", ".join(f"`{run}`" for run in module["runs"]),
            ]
        )
    return _table(
        ["File", "Kernel", "Bytes", "SHA-256", "Loaded in"],
        rows,
        numeric=False,
    )


def _harness(name, run):
    """Return the harness command that the driver ran for one run."""
    return " ".join(_load_summary(name, run)["command"])


def _command(name, run):
    """Return the driver command that produced one run."""
    return (
        "python benchmarks/benchmark_processes.py "
        f"--processes {_load_summary(name, run)['processes']} "
        f'--output-dir "$OUT/{run}" \\\n'
        f"  -- {_harness(name, run)}"
    )


# Page and fragments.


def _blocks(name):
    """Return every generated block of a page by its template token."""
    code, configuration, process = _shared(name)
    environment = process["environment"]
    provenance = process["provenance"]
    target = environment["compute_capability"]
    loads = _load_averages(name)
    return {
        "gpu": code["gpu"],
        "target": target,
        "revision": code["revision"],
        "seed": str(configuration["seed"]),
        "load_start": loads[0],
        "load_end": loads[-1],
        "runs_table": _table(
            ["Run", "Harness command"],
            [
                [f"`{run}`", f"`{_harness(name, run)}`"]
                for run in _runs(name)
            ],
            numeric=False,
        ),
        "facts_table": _fact_table(
            [
                ["Seed", str(configuration["seed"])],
                ["Layout seeds", _quote(configuration["layout_seeds"])],
                ["Layout reuse", _quote(configuration["layout_reuse"])],
                ["Warmup layouts per row", str(configuration["warmups"])],
                ["Timed layouts per row", str(configuration["samples"])],
                ["Warm step", _quote(configuration["warm_step"]["step"])],
                ["Kinds", ", ".join(configuration["kinds"])],
                ["Type", configuration["dtype"]],
                ["Values", _quote(configuration["values"])],
                ["Pipeline", _quote(configuration["pipeline"]["timer"])],
                ["Candidate order", _quote(configuration["candidate_order"])],
                ["Clock", _quote(configuration["clock"])],
                ["Correctness", _quote(configuration["correctness"])],
                ["`[N, D]` rows", _quote(configuration["rank_two"])],
            ]
        ),
        "candidates_table": _families_table(
            name,
            ["Surface", "Entry point", "Covers"],
            configuration["candidate_descriptions"],
            lambda text: [
                text["surface"],
                _quote(text["entry"]),
                _quote(text["times"]),
            ],
        ),
        "timed_region_table": _families_table(
            name,
            ["Timed region"],
            configuration["timed_region"],
            lambda text: [_quote(text)],
        ),
        "machine_table": _fact_table(
            [
                ["Revision", f"`{code['revision']}`"],
                ["GPU", f"{code['gpu']} (`{target}`)"],
                [
                    "NVIDIA driver",
                    provenance["gpu_state_before"]["gpu"]["driver_version"],
                ],
                ["CUDA driver API", environment["cuda_driver"]],
                ["CPU", code["cpu_model"]],
                ["Platform", environment["platform"]],
                ["Python", environment["python"].split(" ")[0]],
                ["PyTorch", code["pytorch"]],
                ["Triton", code["triton"]],
                ["Swage package version", provenance["swage"]],
                ["LLVM linked by the bindings", provenance["llvm_linked"]],
            ]
        ),
        "conditions_list": _conditions_text(name),
        "conditions_file": (RESULTS / name / "conditions.txt").read_text(),
        "public_statement": _public_statement(name),
        "looped_statement": _looped_statement(name),
        "ranges_table": _ranges_table(name),
        "run_tables": _run_tables(name),
        "ptx_table": _ptx_table(name),
        "reproduce_commands": "\n".join(
            _command(name, run) for run in _runs(name)
        ),
    }


def page(name):
    """Return the summary page of a record: its template with its blocks."""
    text = (RESULTS / name / "page.md.in").read_text()
    for token, block in _blocks(name).items():
        text = text.replace("{{" + token + "}}", block.rstrip("\n"))
    if "{{" in text:
        unknown = text[text.index("{{"): text.index("{{") + 40]
        raise ValueError(f"the template has an unknown token near {unknown!r}")
    return text.replace(
        f"<!-- benchmarks/results/{name}/page.md.in -->",
        f"<!-- benchmarks/results/{name}.md -->\n"
        "<!-- Generated by benchmarks/public_call_tables.py from "
        f"{name}/page.md.in. Do not edit; rerun the script. -->",
        1,
    )


def fragments(name):
    """Return the documentation fragments of a record by file name."""
    bodies = {
        "public-statement.inc": _public_statement(name),
        "looped-statement.inc": _looped_statement(name),
        "ranges.inc": _ranges_table(name),
        "losses.inc": _losses(name),
    }
    return {
        f"{name}-{part}": (
            f"<!-- docs/internals/_generated/{name}-{part} -->\n"
            "<!-- Generated by benchmarks/public_call_tables.py from "
            f"benchmarks/results/{name}. Do not edit. -->\n\n{body}"
        )
        for part, body in bodies.items()
    }


def check_records(name):
    """Return one message per process record or PTX file that fails.

    A process record fails when its decompressed SHA-256 is not the one
    its summary lists. A committed PTX file fails when no summary lists a
    loaded module of its SHA-256 and size.
    """
    errors = []
    for run, record in _process_records(name):
        packed = RESULTS / name / run / f"{record['record']}.xz"
        try:
            content = lzma.decompress(packed.read_bytes())
        except (OSError, lzma.LZMAError) as error:
            errors.append(f"cannot read {packed}: {error}")
            continue
        if hashlib.sha256(content).hexdigest() != record["sha256"]:
            errors.append(f"digest mismatch: {packed}")
    for path in sorted((RESULTS / name).glob("*.ptx")):
        content = path.read_bytes()
        module = _loaded_ptx(name).get(hashlib.sha256(content).hexdigest())
        if module is None or module["bytes"] != len(content):
            errors.append(f"no process loaded this PTX: {path}")
    return errors


def outputs(name):
    """Return every generated file of a record and its content."""
    files = {RESULTS / f"{name}.md": page(name)}
    for part, body in fragments(name).items():
        files[FRAGMENTS / part] = body
    return files


def check(name):
    """Return one message per failed record or stale or missing output."""
    errors = check_records(name)
    if errors:
        return errors
    files = outputs(name)
    for path, content in files.items():
        if not path.is_file():
            errors.append(f"missing generated file: {path}")
        elif path.read_text() != content:
            errors.append(f"stale generated file: {path}")
    for path in sorted(FRAGMENTS.glob(f"{name}-*")):
        if path not in files:
            errors.append(f"orphaned generated file: {path}")
    return errors


def main(argv=None):
    """Write or check the pages and the fragments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the committed files, digests, and PTX instead of writing",
    )
    parser.add_argument(
        "--record",
        action="append",
        choices=RECORDS,
        help="a record to render or check; the default is every record",
    )
    arguments = parser.parse_args(argv)
    errors = []
    for name in arguments.record or RECORDS:
        if arguments.check:
            errors += check(name)
            continue
        errors += check_records(name)
        if not errors:
            FRAGMENTS.mkdir(parents=True, exist_ok=True)
            for path, content in outputs(name).items():
                path.write_text(content)
    for error in errors:
        print(error)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
