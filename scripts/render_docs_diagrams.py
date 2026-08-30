# scripts/render_docs_diagrams.py
"""Render the deterministic SVG atlas used by the documentation."""

import argparse
from pathlib import Path

PALETTE = {
    "canvas": "#FFFDF8",
    "panel": "#F5F3ED",
    "ink": "#17212B",
    "muted": "#4B5563",
    "line": "#59636E",
    "blue": "#00689D",
    "blue_fill": "#DCEFFD",
    "purple": "#8E4775",
    "purple_fill": "#F7E2EF",
    "orange": "#A94500",
    "orange_fill": "#FCE5DC",
    "green": "#00664B",
    "green_fill": "#DDF4EB",
    "gray_fill": "#EEF0F2",
}
CONTRAST_PAIRS = (
    ("ink", "canvas", 4.5),
    ("muted", "canvas", 4.5),
    ("line", "canvas", 3.0),
    ("blue", "blue_fill", 4.5),
    ("purple", "purple_fill", 4.5),
    ("orange", "orange_fill", 4.5),
    ("green", "green_fill", 4.5),
    ("line", "gray_fill", 3.0),
)
OUTPUT_NAMES = (
    "capability-boundary.svg",
    "frontend-boundary.svg",
    "ragged-storage.svg",
    "segments-tasks-tiles.svg",
    "compiler-pipeline.svg",
    "split-lifecycle.svg",
    "runtime-lifecycle.svg",
)
DEFAULT_OUTPUT_DIR = Path(__file__).parents[1] / "docs/assets/diagrams"


def _escape(value: object) -> str:
    """Escape text and attribute values without changing visible arrows."""
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace('"', "&quot;")
    )


class Svg:
    """Small deterministic SVG writer for the fixed documentation atlas."""

    def __init__(
        self,
        name: str,
        title: str,
        description: str,
        *,
        width: int = 1200,
        height: int = 700,
    ) -> None:
        """Initialize one SVG document with its accessible metadata."""
        self.name = name
        self.width = width
        self.height = height
        self.prefix = name.removesuffix(".svg")
        self.lines = [
            f"<!-- docs/assets/diagrams/{name} -->",
            '<svg xmlns="http://www.w3.org/2000/svg"',
            f'  viewBox="0 0 {width} {height}" role="img"',
            (
                f'  aria-labelledby="{self.prefix}-title '
                f'{self.prefix}-description"'
            ),
            (
                '  font-family="system-ui, -apple-system, BlinkMacSystemFont, '
                "'Segoe UI', sans-serif\">"
            ),
            f'  <title id="{self.prefix}-title">{_escape(title)}</title>',
            (
                f'  <desc id="{self.prefix}-description">'
                f"{_escape(description)}</desc>"
            ),
            "  <defs>",
            (
                f'    <marker id="{self.prefix}-arrow" viewBox="0 0 10 10" '
                'refX="9" refY="5" markerWidth="8" markerHeight="8" '
                'orient="auto">'
            ),
            (
                f'      <path d="M 0 0 L 10 5 L 0 10 z" '
                f'fill="{PALETTE["line"]}"/>'
            ),
            "    </marker>",
            "  </defs>",
            (
                f'  <rect data-role="canvas" x="0" y="0" width="{width}" '
                f'height="{height}" fill="{PALETTE["canvas"]}"/>'
            ),
        ]

    def box(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        *,
        fill: str = "panel",
        stroke: str = "line",
        dashed: bool = False,
        radius: int = 14,
    ) -> None:
        """Draw one rounded structural box."""
        dash = ' stroke-dasharray="9 7"' if dashed else ""
        self.lines.append(
            f'  <rect x="{x}" y="{y}" width="{width}" height="{height}" '
            f'rx="{radius}" fill="{PALETTE[fill]}" '
            f'stroke="{PALETTE[stroke]}" stroke-width="2"{dash}/>'
        )

    def text(
        self,
        x: int,
        y: int,
        value: str,
        *,
        size: int = 18,
        weight: int = 500,
        color: str = "ink",
        anchor: str = "start",
    ) -> None:
        """Draw visible text with an explicit accessible font size."""
        self.lines.append(
            f'  <text x="{x}" y="{y}" font-size="{size}" '
            f'font-weight="{weight}" fill="{PALETTE[color]}" '
            f'text-anchor="{anchor}">{_escape(value)}</text>'
        )

    def multiline(
        self,
        x: int,
        y: int,
        values: tuple[str, ...],
        *,
        size: int = 18,
        weight: int = 500,
        color: str = "ink",
        gap: int = 25,
        anchor: str = "start",
    ) -> None:
        """Draw a short fixed list of text lines."""
        for index, value in enumerate(values):
            self.text(
                x,
                y + index * gap,
                value,
                size=size,
                weight=weight,
                color=color,
                anchor=anchor,
            )

    def arrow(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        *,
        dashed: bool = False,
        directional: bool = True,
    ) -> None:
        """Draw one directional spine segment."""
        dash = ' stroke-dasharray="8 7"' if dashed else ""
        marker = (
            f' marker-end="url(#{self.prefix}-arrow)"' if directional else ""
        )
        self.lines.append(
            f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{PALETTE["line"]}" stroke-width="3"{dash}{marker}/>'
        )

    def status_tag(
        self,
        x: int,
        y: int,
        width: int,
        label: str,
        *,
        color: str,
        fill: str,
    ) -> None:
        """Draw a visible status label whose meaning is also textual."""
        self.box(x, y, width, 38, fill=fill, stroke=color, radius=19)
        self.text(
            x + width // 2,
            y + 26,
            label,
            size=16,
            weight=750,
            color=color,
            anchor="middle",
        )

    def finish(self) -> bytes:
        """Close and encode the SVG with one final newline."""
        return ("\n".join((*self.lines, "</svg>")) + "\n").encode()


def capability_boundary() -> bytes:
    """Render the public, private, and planned capability lanes."""
    svg = Svg(
        "capability-boundary.svg",
        "Swage capability boundary",
        "Three separate lanes distinguish the public fixed vector-add "
        "surface, private segmented qualification, and planned segmented "
        "APIs.",
        height=650,
    )
    svg.text(48, 54, "Capability boundary", size=32, weight=750)
    svg.text(
        48,
        84,
        "The lanes describe status; none is a fallback from another.",
        color="muted",
    )
    lanes = (
        (
            120,
            "PUBLIC TODAY",
            "blue",
            "blue_fill",
            ("Python capture", "emit_mlir()", "launch()"),
            "canonical fixed vector add only",
            False,
        ),
        (
            300,
            "PRIVATE QUALIFICATION",
            "purple",
            "purple_fill",
            ("segment semantics", "planning + split", "GPU evidence"),
            "not a public API",
            False,
        ),
        (
            480,
            "PLANNED",
            "line",
            "gray_fill",
            ("public segmented API", "packed work", "split max / softmax"),
            "persistent scheduling",
            True,
        ),
    )
    for y, label, color, fill, boxes, note, dashed in lanes:
        svg.box(40, y, 1120, 138, fill=fill, stroke=color, dashed=dashed)
        svg.status_tag(62, y + 18, 250, label, color=color, fill=fill)
        for index, box_label in enumerate(boxes):
            x = 350 + index * 245
            svg.box(x, y + 24, 205, 64, fill="canvas", stroke=color)
            svg.text(
                x + 102,
                y + 63,
                box_label,
                size=17,
                weight=650,
                anchor="middle",
            )
            if index:
                svg.arrow(x - 34, y + 56, x - 8, y + 56, dashed=dashed)
        svg.text(350, y + 116, note, size=16, weight=650, color=color)
    return svg.finish()


def frontend_boundary() -> bytes:
    """Render frontend verification and the compile versus launch branch."""
    svg = Svg(
        "frontend-boundary.svg",
        "Frontend verification and execution boundary",
        "Python source passes restricted AST validation and verified semantic "
        "MLIR before emit_mlir stops or canonical launch continues to CUDA.",
        height=680,
    )
    svg.text(48, 54, "Frontend boundary", size=32, weight=750)
    common = (
        (60, "Python source", ("captured, not executed",)),
        (350, "restricted AST", ("fail closed", "source-located errors")),
        (640, "verified semantic MLIR", ("one production IR",)),
    )
    for index, (x, heading, body) in enumerate(common):
        svg.box(x, 130, 230, 126, fill="panel")
        svg.text(x + 115, 170, heading, size=17, weight=700, anchor="middle")
        svg.multiline(
            x + 115,
            204,
            body,
            size=16,
            color="muted",
            anchor="middle",
        )
        if index:
            svg.arrow(x - 56, 193, x - 10, 193)
    svg.arrow(870, 193, 930, 193)
    svg.box(940, 130, 210, 126, fill="green_fill", stroke="green")
    svg.text(1045, 170, "frontend gate", weight=700, anchor="middle")
    svg.text(
        1045, 204, "module verified", size=16, color="green", anchor="middle"
    )

    svg.arrow(1045, 256, 1045, 320)
    svg.arrow(1045, 320, 820, 340)
    svg.arrow(1045, 320, 1085, 340)
    svg.box(650, 350, 340, 154, fill="blue_fill", stroke="blue")
    svg.status_tag(
        674, 370, 174, "NO GPU NEEDED", color="blue", fill="blue_fill"
    )
    svg.text(820, 438, "emit_mlir()", size=22, weight=750, anchor="middle")
    svg.text(
        820, 470, "compile-only stop", size=17, color="blue", anchor="middle"
    )

    svg.box(1010, 350, 150, 154, fill="purple_fill", stroke="purple")
    svg.status_tag(
        1024, 370, 122, "PUBLIC", color="purple", fill="purple_fill"
    )
    svg.text(1085, 438, "launch()", size=22, weight=750, anchor="middle")
    svg.text(
        1085, 470, "canonical only", size=16, color="purple", anchor="middle"
    )
    svg.arrow(1045, 504, 1045, 558)
    svg.box(865, 570, 295, 72, fill="orange_fill", stroke="orange")
    svg.text(1012, 600, "native compile -> CUDA", weight=700, anchor="middle")
    svg.text(
        1012,
        626,
        "nonzero target gate: sm_80+",
        size=16,
        color="orange",
        anchor="middle",
    )
    return svg.finish()


def ragged_storage() -> bytes:
    """Render dense values and offsets with an empty segment."""
    svg = Svg(
        "ragged-storage.svg",
        "Dense ragged storage with repeated offsets",
        "Values stay dense while offsets define four half-open ranges, "
        "including one empty range created by repeated offsets.",
        height=640,
    )
    svg.text(48, 54, "Ragged storage", size=32, weight=750)
    svg.text(
        48,
        84,
        "One values strip plus offsets reconstructs every logical segment.",
        color="muted",
    )
    svg.text(74, 146, "values", size=20, weight=750)
    values = ("4.0", "1.0", "3.0", "8.0", "2.0", "9.0")
    for index, value in enumerate(values):
        x = 210 + index * 135
        fill = (
            "orange_fill"
            if index < 2
            else "blue_fill"
            if index < 5
            else "green_fill"
        )
        svg.box(x, 112, 120, 74, fill=fill)
        svg.text(x + 60, 157, value, weight=700, anchor="middle")
        svg.text(x, 210, str(index), size=16, color="muted", anchor="middle")
    svg.text(1020, 210, "6", size=16, color="muted", anchor="middle")

    svg.text(74, 284, "offsets", size=20, weight=750)
    offsets = (0, 2, 2, 5, 6)
    for index, value in enumerate(offsets):
        x = 250 + index * 175
        svg.box(x, 246, 92, 64, fill="gray_fill")
        svg.text(x + 46, 286, str(value), weight=700, anchor="middle")
    svg.text(250, 340, "offsets = [0, 2, 2, 5, 6]", size=18, weight=650)

    ranges = (
        (60, 390, "S0", "[0, 2)", "2 values", "orange_fill", "orange"),
        (345, 390, "S1", "[2, 2)", "empty segment", "gray_fill", "line"),
        (630, 390, "S2", "[2, 5)", "3 values", "blue_fill", "blue"),
        (915, 390, "S3", "[5, 6)", "1 value", "green_fill", "green"),
    )
    for x, y, segment, span, note, fill, color in ranges:
        svg.box(x, y, 225, 116, fill=fill, stroke=color)
        svg.text(x + 112, y + 36, segment, weight=750, anchor="middle")
        svg.text(x + 112, y + 68, span, size=20, weight=700, anchor="middle")
        svg.text(x + 112, y + 96, note, size=16, color=color, anchor="middle")
    svg.text(
        600,
        564,
        "segment i = values[offsets[i] : offsets[i + 1]]  (half-open)",
        size=20,
        weight=650,
        anchor="middle",
    )
    svg.text(
        600,
        602,
        "Repeated offsets preserve dense storage and describe zero work.",
        size=17,
        color="muted",
        anchor="middle",
    )
    return svg.finish()


def segments_tasks_tiles() -> bytes:
    """Render the segment, task, and fixed hardware-step distinction."""
    svg = Svg(
        "segments-tasks-tiles.svg",
        "One segment becomes policy-bearing tasks and fixed GPU steps",
        "A runtime-length segment keeps its meaning while qualification "
        "derives one or more ordered tasks executed by fixed warp or CTA "
        "steps.",
        height=650,
    )
    svg.text(48, 54, "Segment -> Task -> Tile", size=32, weight=750)
    svg.text(
        48,
        84,
        "Meaning, scheduling policy, and physical execution remain separate.",
        color="muted",
    )
    columns = ((45, "1  SEGMENT"), (430, "2  TASK"), (815, "3  TILE"))
    for x, heading in columns:
        svg.box(x, 120, 340, 460, fill="panel")
        svg.text(x + 24, 160, heading, size=18, weight=750, color="muted")

    svg.box(75, 210, 280, 150, fill="orange_fill", stroke="orange")
    svg.multiline(
        215,
        252,
        (
            "one runtime-length",
            "segment",
            "logical dense slice",
            "identity is an SSA value",
        ),
        weight=650,
        gap=27,
        anchor="middle",
    )
    svg.text(
        215, 410, "semantic meaning", size=16, color="orange", anchor="middle"
    )
    svg.text(215, 440, "no GPU IDs", size=16, color="muted", anchor="middle")
    svg.text(
        215,
        470,
        "no runtime-sized register array",
        size=16,
        color="muted",
        anchor="middle",
    )

    svg.box(460, 190, 280, 104, fill="blue_fill", stroke="blue")
    svg.multiline(
        600,
        232,
        ("one direct task", "short or medium", "admitted work"),
        size=16,
        weight=650,
        gap=25,
        anchor="middle",
    )
    svg.text(600, 330, "or", size=18, weight=750, anchor="middle")
    svg.box(460, 360, 280, 130, fill="purple_fill", stroke="purple")
    svg.multiline(
        600,
        400,
        (
            "ordered partial tasks",
            "+ one merge task",
            "for qualified split work",
        ),
        size=16,
        weight=650,
        gap=30,
        anchor="middle",
    )
    svg.text(
        600,
        540,
        "task = policy-bearing work unit",
        size=16,
        color="purple",
        anchor="middle",
    )

    svg.box(845, 205, 280, 112, fill="blue_fill", stroke="blue")
    svg.text(
        985, 250, "32-thread warp step", size=20, weight=750, anchor="middle"
    )
    svg.text(
        985, 282, "fixed physical shape", size=16, color="blue", anchor="middle"
    )
    svg.box(845, 365, 280, 112, fill="purple_fill", stroke="purple")
    svg.text(
        985, 410, "128-thread CTA step", size=20, weight=750, anchor="middle"
    )
    svg.text(
        985,
        442,
        "fixed physical shape",
        size=16,
        color="purple",
        anchor="middle",
    )
    svg.text(
        985,
        525,
        "a tile is an execution step",
        size=16,
        color="muted",
        anchor="middle",
    )

    svg.arrow(385, 286, 420, 286)
    svg.arrow(740, 242, 805, 242)
    svg.arrow(740, 424, 805, 424)
    svg.text(
        600,
        620,
        "One segment may require one task or several ordered tasks.",
        size=18,
        weight=650,
        anchor="middle",
    )
    return svg.finish()


def compiler_pipeline() -> bytes:
    """Render the canonical compiler spine and three admitted branches."""
    svg = Svg(
        "compiler-pipeline.svg",
        "Swage compiler pipeline",
        "Verified semantic MLIR enters three admitted branches. The public "
        "fixed-block branch, the private direct segmented "
        "GPU: one CTA / segment path, and "
        "GPU work rejoin upstream MLIR and LLVM before NVPTX, PTX, and CUDA. "
        "The direct segmented sequential CPU oracle exits separately through "
        "SCF and "
        "memref.",
        height=850,
    )
    svg.text(
        48,
        54,
        "One compiler spine, three admitted branches",
        size=32,
        weight=750,
    )
    svg.box(390, 92, 420, 76, fill="orange_fill", stroke="orange")
    svg.text(
        600, 139, "verified semantic MLIR", size=22, weight=750, anchor="middle"
    )
    svg.arrow(600, 168, 600, 215, directional=False)

    branches = (
        (
            55,
            "PUBLIC TODAY",
            "fixed-block vector add",
            ("canonical vector add", "fixed-block GPU conversion"),
            "blue",
            "blue_fill",
        ),
        (
            420,
            "PRIVATE QUALIFICATION",
            "segmented direct",
            ("sum, max, stable softmax", "two qualified paths"),
            "purple",
            "purple_fill",
        ),
        (
            785,
            "PRIVATE QUALIFICATION",
            "SwagePlan direct + split",
            ("classification companion", "direct or split identity sum"),
            "green",
            "green_fill",
        ),
    )
    for x, status, heading, body, color, fill in branches:
        svg.box(x, 245, 330, 180, fill=fill, stroke=color)
        svg.status_tag(x + 22, 265, 250, status, color=color, fill=fill)
        svg.text(x + 165, 344, heading, size=19, weight=750, anchor="middle")
        svg.multiline(
            x + 165,
            384,
            body,
            size=16,
            color=color,
            gap=30,
            anchor="middle",
        )
    svg.arrow(600, 215, 220, 235)
    svg.arrow(600, 215, 585, 235)
    svg.arrow(600, 215, 950, 235)

    svg.arrow(585, 425, 475, 435)
    svg.arrow(585, 425, 695, 435)
    svg.box(380, 445, 190, 105, fill="canvas", stroke="purple")
    svg.multiline(
        475,
        480,
        ("sequential CPU", "oracle", "SCF / memref stop"),
        size=16,
        weight=650,
        color="purple",
        gap=27,
        anchor="middle",
    )
    svg.box(600, 445, 190, 105, fill="canvas", stroke="purple")
    svg.multiline(
        695,
        480,
        ("GPU: one CTA", "per segment", "continues below"),
        size=16,
        weight=650,
        color="purple",
        gap=27,
        anchor="middle",
    )

    svg.arrow(220, 425, 350, 565)
    svg.arrow(695, 550, 600, 565)
    svg.arrow(950, 425, 850, 565)
    svg.box(330, 580, 540, 78, fill="gray_fill")
    svg.text(
        600,
        613,
        "GPU / SCF / NVVM / LLVM",
        size=22,
        weight=750,
        anchor="middle",
    )
    svg.text(
        600,
        640,
        "upstream MLIR and LLVM paths",
        size=16,
        color="muted",
        anchor="middle",
    )
    svg.arrow(600, 658, 310, 690)
    backend = (
        (210, "LLVM NVPTX"),
        (500, "PTX"),
        (790, "CUDA Driver API"),
    )
    for x, label in backend:
        svg.box(x, 700, 200, 68, fill="canvas")
        svg.text(x + 100, 742, label, size=17, weight=700, anchor="middle")
    svg.arrow(410, 734, 490, 734)
    svg.arrow(700, 734, 780, 734)
    svg.text(
        600,
        818,
        "No second production IR and no silent backend fallback.",
        size=16,
        color="muted",
        anchor="middle",
    )
    return svg.finish()


def split_lifecycle() -> bytes:
    """Render the private split partial and merge ownership lifecycle."""
    svg = Svg(
        "split-lifecycle.svg",
        "Private split CTA identity-sum lifecycle",
        "An oversized segment is divided into ordered absolute input ranges, "
        "each partial writes one scratch slot, and one merge writes output "
        "once.",
        height=760,
    )
    svg.text(48, 54, "Private split lifecycle", size=32, weight=750)
    svg.status_tag(
        900, 28, 245, "identity sum only", color="purple", fill="purple_fill"
    )
    svg.text(
        48,
        88,
        "One oversized segment; split max and softmax remain planned.",
        color="muted",
    )

    svg.box(50, 122, 1100, 92, fill="orange_fill", stroke="orange")
    svg.text(80, 158, "segment 7", size=20, weight=750)
    svg.text(
        80, 190, "absolute input range [100, 9500)", size=18, color="orange"
    )
    svg.text(
        1040,
        175,
        "9,400 values",
        size=16,
        weight=650,
        color="orange",
        anchor="middle",
    )

    partials = (
        (55, "partial 0", "[100, 4196)", "scratch[0]"),
        (405, "partial 1", "[4196, 8292)", "scratch[1]"),
        (755, "partial 2", "[8292, 9500)", "scratch[2]"),
    )
    for x, label, span, scratch in partials:
        svg.box(x, 275, 310, 132, fill="blue_fill", stroke="blue")
        svg.text(x + 155, 312, label, size=18, weight=750, anchor="middle")
        svg.text(x + 155, 345, span, size=18, anchor="middle")
        svg.text(
            x + 155,
            382,
            f"unique scratch writer -> {scratch}",
            size=16,
            color="blue",
            anchor="middle",
        )
        svg.arrow(x + 155, 214, x + 155, 265)

    svg.arrow(210, 407, 490, 475)
    svg.arrow(560, 407, 600, 475)
    svg.arrow(910, 407, 710, 475)
    svg.box(355, 490, 490, 112, fill="purple_fill", stroke="purple")
    svg.text(600, 530, "merge record", size=20, weight=750, anchor="middle")
    svg.text(
        600,
        562,
        "segment 7, compact scratch range [0, 3)",
        size=18,
        anchor="middle",
    )
    svg.text(
        600, 588, "one merge CTA", size=16, color="purple", anchor="middle"
    )
    svg.arrow(600, 602, 600, 635)
    svg.box(430, 650, 340, 70, fill="green_fill", stroke="green")
    svg.text(600, 680, "output[7]", size=20, weight=750, anchor="middle")
    svg.text(
        600, 705, "one final writer", size=16, color="green", anchor="middle"
    )

    svg.text(
        48,
        742,
        "launch order: direct fused > partial CTAs > merge CTAs; "
        "empty phases skipped",
        size=16,
        weight=650,
    )
    return svg.finish()


def runtime_lifecycle() -> bytes:
    """Render the fail-closed launch and stream-retention lifecycle."""
    svg = Svg(
        "runtime-lifecycle.svg",
        "Runtime validation, launch, and tensor retention lifecycle",
        "Validation precedes specialization, compilation, and allocation; "
        "work is enqueued on the current stream and tensors are retained.",
        height=680,
    )
    svg.text(48, 54, "Runtime lifecycle", size=32, weight=750)
    svg.text(
        48,
        84,
        "Asynchronous launch with no synchronization or fallback.",
        color="muted",
    )

    svg.status_tag(55, 120, 145, "VALIDATE", color="orange", fill="orange_fill")
    svg.box(55, 172, 220, 138, fill="orange_fill", stroke="orange")
    svg.multiline(
        165,
        205,
        ("tensors / ABI", "grid / device", "before compile", "or allocation"),
        size=16,
        weight=650,
        gap=27,
        anchor="middle",
    )
    svg.status_tag(
        82, 326, 165, "FAIL CLOSED", color="orange", fill="orange_fill"
    )
    svg.text(
        165,
        392,
        "invalid input stops",
        size=16,
        color="orange",
        anchor="middle",
    )

    stages = (
        (
            330,
            "specialize / cache",
            ("exact sm_*", "verified reuse"),
            "blue",
            "blue_fill",
        ),
        (
            595,
            "compile / load",
            ("LLVM NVPTX", "CUDA Driver"),
            "purple",
            "purple_fill",
        ),
        (
            860,
            "enqueue",
            ("current PyTorch stream", "asynchronous"),
            "green",
            "green_fill",
        ),
    )
    for index, (x, heading, body, color, fill) in enumerate(stages):
        svg.box(x, 172, 220, 138, fill=fill, stroke=color)
        svg.text(x + 110, 214, heading, size=18, weight=750, anchor="middle")
        svg.multiline(
            x + 110,
            252,
            body,
            size=16,
            color=color,
            gap=28,
            anchor="middle",
        )
        svg.arrow(x - 45, 241, x - 10, 241)

    svg.box(55, 430, 220, 78, fill="gray_fill")
    svg.text(165, 463, "n == 0", size=18, weight=750, anchor="middle")
    svg.text(
        165, 491, "zero-work no-op", size=16, color="muted", anchor="middle"
    )
    svg.arrow(275, 469, 320, 469, dashed=True)
    svg.text(330, 475, "returns before specialization", size=16, color="muted")

    svg.arrow(970, 310, 920, 380)
    svg.box(690, 400, 460, 122, fill="green_fill", stroke="green")
    svg.text(
        920,
        440,
        "retain submitted tensors",
        size=20,
        weight=750,
        anchor="middle",
    )
    svg.text(
        920,
        475,
        "record_stream() on the launch stream",
        size=18,
        color="green",
        anchor="middle",
    )
    svg.text(
        920,
        505,
        "storage stays owned by PyTorch",
        size=16,
        color="muted",
        anchor="middle",
    )
    svg.text(
        600,
        600,
        "No copy, cast, device change, context creation, synchronization or "
        "fallback.",
        size=17,
        weight=650,
        anchor="middle",
    )
    svg.text(
        600,
        638,
        "Cache or driver integrity failures stop instead of selecting another "
        "backend.",
        size=16,
        color="muted",
        anchor="middle",
    )
    return svg.finish()


def render_all() -> dict[str, bytes]:
    """Render all seven diagrams in fixed output order."""
    renderers = (
        capability_boundary,
        frontend_boundary,
        ragged_storage,
        segments_tasks_tiles,
        compiler_pipeline,
        split_lifecycle,
        runtime_lifecycle,
    )
    return {name: renderer() for name, renderer in zip(OUTPUT_NAMES, renderers)}


def render_diagrams(output_dir: Path, *, check: bool = False) -> list[str]:
    """Write or byte-check the generated diagram set."""
    rendered = render_all()
    errors = []
    existing = {
        path.name: path for path in output_dir.glob("*.svg") if path.is_file()
    }
    if check:
        for name in OUTPUT_NAMES:
            path = output_dir / name
            if not path.is_file():
                errors.append(f"missing generated diagram: {path}")
            elif path.read_bytes() != rendered[name]:
                errors.append(f"stale generated diagram: {path}")
        for name in sorted(existing.keys() - rendered.keys()):
            errors.append(f"orphaned generated diagram: {existing[name]}")
        return errors

    output_dir.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_NAMES:
        (output_dir / name).write_bytes(rendered[name])
    for name in existing.keys() - rendered.keys():
        existing[name].unlink()
    return []


def main() -> int:
    """Render or check the committed documentation diagrams."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    errors = render_diagrams(DEFAULT_OUTPUT_DIR, check=args.check)
    for error in errors:
        print(error)
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
