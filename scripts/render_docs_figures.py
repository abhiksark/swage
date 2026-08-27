# scripts/render_docs_figures.py
"""Render the TikZ figure atlas used by the documentation.

Render mode compiles each `figures/*.tex` source with the local
`tectonic` binary, converts the PDF to SVG with PyMuPDF, injects
accessibility metadata, and stamps the output with a digest of its
sources. It renders only missing or stale outputs and removes orphans.

Check mode needs only the standard library: it recomputes each source
digest and compares it with the committed stamp, so continuous
integration verifies freshness without a TeX toolchain.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).parents[1]
SOURCE_DIR = REPO_ROOT / "figures"
OUTPUT_DIR = REPO_ROOT / "docs/assets/figures"
PREAMBLE_NAME = "common-preamble.tex"
SNAPSHOT_PATH = REPO_ROOT / "benchmarks/results/perf-5090-sm120.json"
FALLBACK_TECTONIC = Path.home() / ".local/bin/tectonic"
STAMP_PATTERN = re.compile(r"<!-- source-sha256: ([0-9a-f]{64}) -->")


class FigureSpec(NamedTuple):
    """One rendered figure and the repo files that define it."""

    name: str
    title: str
    description: str
    data: tuple[str, ...] = ()


FIGURES = (
    FigureSpec(
        name="fixed-block-thread-map",
        title="Fixed-block launch geometry",
        description=(
            "A 22-element input covered by three 8-thread blocks, with "
            "the third block expanded to show global indices and the "
            "two masked tail lanes."
        ),
    ),
    FigureSpec(
        name="warp-vs-cta-tiles",
        title="Warp, CTA, and split tile shapes",
        description=(
            "A 32-thread warp tile reduces one short segment through "
            "an xor shuffle butterfly, a 128-thread CTA tile strides "
            "one longer segment before a shared block reduction, and "
            "512-thread split tiles cover the 4096-element chunks of "
            "an oversized segment."
        ),
    ),
    FigureSpec(
        name="oracle-topology",
        title="One module, three executions",
        description=(
            "One verified semantic module runs on the GPU path, the "
            "sequential CPU oracle, and the PyTorch reference, and "
            "every correctness claim is a differential comparison "
            "between them."
        ),
    ),
    FigureSpec(
        name="ownership-map",
        title="Ownership map",
        description=(
            "Three lanes separate what Swage owns from what upstream "
            "MLIR and LLVM own and what PyTorch owns, with one launch "
            "traced across the domains."
        ),
    ),
    FigureSpec(
        name="m5-softmax-phases",
        title="Stable softmax in one CTA",
        description=(
            "One CTA sweeps its segment three times, for the maximum, "
            "the shifted exponential sum, and the normalizing stores, "
            "with uniform all-reduce operations acting as both "
            "broadcast and phase barrier."
        ),
    ),
    FigureSpec(
        name="plan-classification",
        title="SwagePlan classification",
        description=(
            "Observed segment lengths flow through the empty, warp, "
            "CTA, and split buckets into four task lists, under the "
            "validated planning-limit invariant."
        ),
    ),
    FigureSpec(
        name="specialization-key-cache",
        title="Specialization key and cache verification",
        description=(
            "The specialization key fields hash into one digest that "
            "selects a cache entry, which is verified before module "
            "load; a rejected entry raises while a plain miss "
            "compiles in process."
        ),
    ),
    FigureSpec(
        name="dispatch-path",
        title="Launch dispatch lanes",
        description=(
            "A validated launch enqueues through the compiled nanobind "
            "launcher when the bindings import, or through the ctypes "
            "fallback otherwise; both submit the same driver call on "
            "the current PyTorch stream."
        ),
    ),
    FigureSpec(
        name="timing-methods",
        title="Three timing methods",
        description=(
            "Timelines contrast synchronized per-call wall clock, "
            "batched CUDA-event timing with the launcher still "
            "visible, and CUDA-graph replay with the host removed."
        ),
    ),
    FigureSpec(
        name="segsum-graph-comparison",
        title="Segmented sum under graph timing",
        description=(
            "Grouped bars compare graph-replay medians for the best "
            "Swage policy, the tuned Triton baseline, and torch "
            "segment_reduce across seven segment distributions on an "
            "RTX 5090."
        ),
        data=("benchmarks/results/perf-5090-sm120.json",),
    ),
    FigureSpec(
        name="dispatch-ladder",
        title="Warm dispatch cost ladder",
        description=(
            "Log-scale bars follow warm per-launch dispatch cost from "
            "the baseline host path through the cached and compiled "
            "launchers, beside the Triton and torch dispatch costs."
        ),
        data=("benchmarks/results/perf-5090-sm120.json",),
    ),
    FigureSpec(
        name="fused-mixed-schedule",
        title="Fused mixed-policy schedule",
        description=(
            "One 128-thread launch covers six warp tasks at four "
            "independent slots per block, then three CTA tasks at one "
            "segment per block, with task_ids mapping blocks to "
            "segments."
        ),
    ),
)


def tectonic_binary() -> str | None:
    """Locate the tectonic binary on PATH or at the pinned fallback."""
    found = shutil.which("tectonic")
    if found:
        return found
    if FALLBACK_TECTONIC.is_file():
        return str(FALLBACK_TECTONIC)
    return None


def _load_snapshot() -> dict:
    """Load the committed performance snapshot."""
    return json.loads(SNAPSHOT_PATH.read_text())


def _series_line(rows: list, impl: str, style: str) -> str:
    """Build one pgfplots series from snapshot rows for one impl."""
    coordinates = " ".join(
        f"({row['distribution']},{row[impl]['median']:.1f})" for row in rows
    )
    return f"\\addplot[{style}] coordinates {{{coordinates}}};\n"


def _segsum_include() -> str:
    """Generate the segmented-sum chart series from the snapshot."""
    rows = _load_snapshot()["segsum_graph_us"]
    styles = (
        ("swage", "ybar, fill=swbluefill, draw=swblue"),
        ("triton", "ybar, fill=sworangefill, draw=sworange"),
        ("torch", "ybar, fill=swpurplefill, draw=swpurple"),
    )
    lines = [_series_line(rows, impl, style) for impl, style in styles]
    lines.append("\\legend{swage, triton (tuned), torch (CUB)}\n")
    return "".join(lines)


def _dispatch_include() -> str:
    """Generate the dispatch-ladder series from the snapshot."""
    snapshot = _load_snapshot()
    stages = snapshot["dispatch_call_us"]
    styles = {
        "swage": "xbar, bar shift=0pt, fill=swbluefill, draw=swblue",
        "triton": "xbar, bar shift=0pt, fill=sworangefill, draw=sworange",
        "torch": "xbar, bar shift=0pt, fill=swpurplefill, draw=swpurple",
    }
    total = len(stages)
    lines = ["\\newcommand{\\dispatchplots}{%\n"]
    for impl in ("swage", "triton", "torch"):
        coordinates = " ".join(
            f"({stage['median']:.1f},{total - 1 - index})"
            for index, stage in enumerate(stages)
            if stage["impl"] == impl
        )
        lines.append(
            f"\\addplot[{styles[impl]}] coordinates {{{coordinates}}};\n"
        )
    lines.append("}\n")
    cold = snapshot["cold_start_ms"]
    lines.append(f"\\def\\swagecoldms{{{cold['swage']}}}\n")
    lines.append(f"\\def\\tritoncoldms{{{cold['triton']}}}\n")
    return "".join(lines)


def chart_include(spec: FigureSpec) -> str | None:
    """Return the generated data include for one chart figure."""
    if spec.name == "segsum-graph-comparison":
        return _segsum_include()
    if spec.name == "dispatch-ladder":
        return _dispatch_include()
    return None


def figure_digest(spec: FigureSpec) -> str:
    """Hash the source closure that defines one rendered figure."""
    hasher = hashlib.sha256()
    paths = [SOURCE_DIR / PREAMBLE_NAME, SOURCE_DIR / f"{spec.name}.tex"]
    paths += [REPO_ROOT / entry for entry in spec.data]
    for path in paths:
        payload = path.read_bytes()
        relative = path.relative_to(REPO_ROOT).as_posix()
        hasher.update(f"{relative}:{len(payload)}:".encode())
        hasher.update(payload)
    include = chart_include(spec)
    if include is not None:
        hasher.update(b"figure-data.tex:")
        hasher.update(include.encode())
    return hasher.hexdigest()


def _escape(value: str) -> str:
    """Escape text destined for SVG title and description elements."""
    return (
        value.replace("&", "&amp;").replace("<", "&lt;").replace('"', "&quot;")
    )


def _pdf_to_svg(pdf_path: Path) -> str:
    """Convert a one-page PDF to SVG markup with PyMuPDF."""
    import pymupdf

    with pymupdf.open(pdf_path) as document:
        return document[0].get_svg_image()


def _finalize_svg(spec: FigureSpec, markup: str) -> bytes:
    """Inject header comments and accessible metadata into raw SVG."""
    start = markup.index("<svg")
    opening_end = markup.index(">", start)
    prefix = spec.name
    injected = (
        f' role="img" aria-labelledby="{prefix}-title '
        f'{prefix}-description"'
    )
    metadata = (
        f'\n<title id="{prefix}-title">{_escape(spec.title)}</title>'
        f'\n<desc id="{prefix}-description">'
        f"{_escape(spec.description)}</desc>"
    )
    body = (
        markup[:opening_end]
        + injected
        + ">"
        + metadata
        + markup[opening_end + 1:]
    )
    header = (
        f"<!-- docs/assets/figures/{spec.name}.svg -->\n"
        f"<!-- source-sha256: {figure_digest(spec)} -->\n"
    )
    return (header + body.rstrip("\n") + "\n").encode()


def _render_figure(spec: FigureSpec, tectonic: str) -> bytes:
    """Compile one figure source to finalized SVG bytes."""
    with tempfile.TemporaryDirectory() as scratch:
        workdir = Path(scratch)
        shutil.copy(SOURCE_DIR / PREAMBLE_NAME, workdir)
        shutil.copy(SOURCE_DIR / f"{spec.name}.tex", workdir)
        include = chart_include(spec)
        if include is not None:
            (workdir / "figure-data.tex").write_text(include)
        environment = dict(os.environ, SOURCE_DATE_EPOCH="0")
        completed = subprocess.run(
            [
                tectonic,
                "--chatter",
                "minimal",
                "--outdir",
                str(workdir),
                str(workdir / f"{spec.name}.tex"),
            ],
            capture_output=True,
            text=True,
            env=environment,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"tectonic failed for {spec.name}: {completed.stderr[-2000:]}"
            )
        markup = _pdf_to_svg(workdir / f"{spec.name}.pdf")
    return _finalize_svg(spec, markup)


def _committed_digest(path: Path) -> str | None:
    """Read the source digest stamped into one committed SVG."""
    try:
        text = path.read_text()
    except OSError:
        return None
    match = STAMP_PATTERN.search(text)
    return match.group(1) if match else None


def render_figures(output_dir: Path, *, check: bool = False) -> list[str]:
    """Render or verify the figure atlas and report drift diagnostics."""
    errors = []
    expected = {f"{spec.name}.svg" for spec in FIGURES}
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(output_dir.glob("*.svg")):
        if path.name in expected:
            continue
        if check:
            errors.append(f"orphaned generated figure: {path}")
        else:
            path.unlink()
    tectonic = tectonic_binary()
    for spec in FIGURES:
        path = output_dir / f"{spec.name}.svg"
        digest = figure_digest(spec)
        stamped = _committed_digest(path)
        if stamped == digest:
            continue
        if check:
            if stamped is None:
                errors.append(f"missing generated figure: {path}")
            else:
                errors.append(f"stale generated figure: {path}")
            continue
        if tectonic is None:
            errors.append(
                f"cannot render {spec.name}: tectonic binary not found"
            )
            continue
        path.write_bytes(_render_figure(spec, tectonic))
    return sorted(errors)


def main() -> int:
    """Run the renderer from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify committed figures instead of rendering",
    )
    arguments = parser.parse_args()
    errors = render_figures(OUTPUT_DIR, check=arguments.check)
    for line in errors:
        print(line)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
