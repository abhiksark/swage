# python/swage/bench.py
"""Measure the frozen installed-wheel fixed vector-add CUDA benchmark."""

import argparse
import pathlib


def main(argv=None) -> int:
    """Parse the fixed-only command and return its benchmark exit status."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    vector_add = commands.add_parser(
        "vector-add",
        help="measure the frozen fixed vector-add benchmark",
        description=__doc__,
        allow_abbrev=False,
    )
    vector_add.add_argument(
        "--output", type=pathlib.Path, help="write raw JSON evidence to PATH"
    )
    vector_add.add_argument(
        "--enforce",
        action="store_true",
        help="enforce the frozen NVIDIA RTX A6000 / sm_86 gates",
    )
    vector_add.add_argument(
        "--cold-child", action="store_true", help=argparse.SUPPRESS
    )
    args = parser.parse_args(argv)
    if not args.cold_child and args.output is None:
        vector_add.error("--output is required")

    from ._benchmark import run_fixed_vector_add

    return run_fixed_vector_add(
        output=args.output, enforce=args.enforce, cold_child=args.cold_child
    )


if __name__ == "__main__":
    raise SystemExit(main())
