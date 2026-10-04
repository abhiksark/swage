#!/usr/bin/env bash
# scripts/qualify_installed_segments.sh
# Qualify the public segmented calls of an installed swage-compiler wheel on
# a CUDA GPU.
#
#   scripts/qualify_installed_segments.sh PYTHON ORACLE_BUILD WORK_DIR
#
# PYTHON is the interpreter of an environment with the wheel installed and
# with pytest, numpy, and CUDA PyTorch. ORACLE_BUILD is a Swage build
# directory that holds bin/swage-opt and CMakeCache.txt: the private CPU
# oracle of the tests runs that swage-opt and the LLVM runner of the LLVM
# install the build names, and the wheel holds neither. WORK_DIR must not
# exist; it receives a copy of the tests and of the repository files they
# read, and the evidence: segmented-calls.xml (JUnit) and environment.json.
#
# The tests run from the copy with PYTHONPATH unset, so only the installed
# packages can be imported. They cover `swage.segment_reduce` (sum, max,
# min, and mean over float32 and float64) and `swage.segment_softmax`
# (float32), over rank-one values and [N, D] values on the row-stripe tile,
# int32 and int64 offsets, empty batches and segments, and `out=`, against
# PyTorch, float64 references, and the CPU oracle; the first and second
# derivatives of both calls; and the artifact path: `python -m swage.compile`
# writes the kernels, and a fresh process with no compiler runs both calls
# and their derivatives from SWAGE_ARTIFACT_DIR. A skipped test fails the
# qualification, because it shows nothing about the wheel.
set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "usage: $0 PYTHON ORACLE_BUILD WORK_DIR" >&2
    exit 2
fi
PYTHON="$1"
ORACLE_BUILD="$(cd "$2" && pwd)"
WORK="$3"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

for name in CMakeCache.txt bin/swage-opt; do
    if [ ! -e "$ORACLE_BUILD/$name" ]; then
        echo "error: $ORACLE_BUILD lacks $name, which the CPU oracle runs" >&2
        exit 2
    fi
done
if [ -e "$WORK" ]; then
    echo "error: $WORK exists; name a new directory" >&2
    exit 2
fi

mkdir -p "$WORK/python/tests" "$WORK/cmake" "$WORK/docs/internals" \
    "$WORK/benchmarks"
cp -r "$REPO_ROOT/python/tests/mlir" "$WORK/python/tests/mlir"
cp -r "$REPO_ROOT/examples" "$REPO_ROOT/test" "$WORK/"
cp "$REPO_ROOT/cmake/llvm-version.txt" "$WORK/cmake/"
cp "$REPO_ROOT/docs/internals/segmented-reductions.md" "$WORK/docs/internals/"
cp "$REPO_ROOT/benchmarks/distributions.py" "$WORK/benchmarks/"
find "$WORK" -name __pycache__ -prune -exec rm -rf {} +

cd "$WORK"
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export SWAGE_ORACLE_BUILD_DIR="$ORACLE_BUILD"

# The installed package and bindings, with CUDA, or nothing is qualified.
"$PYTHON" -m swage.env --json --check cuda > environment.json
"$PYTHON" - "$REPO_ROOT" <<'PY'
import pathlib
import sys

import mlir_swage
import swage

checkout = pathlib.Path(sys.argv[1]).resolve()
for module in (swage, mlir_swage):
    for location in getattr(module, "__path__", [module.__file__]):
        path = pathlib.Path(location).resolve()
        if path.is_relative_to(checkout):
            sys.exit(f"{module.__name__} is imported from the checkout: {path}")
PY

"$PYTHON" -m pytest -q -p no:cacheprovider -rs \
    --junitxml="$WORK/segmented-calls.xml" \
    python/tests/mlir/test_public_segments.py \
    python/tests/mlir/test_segment_columns.py \
    python/tests/mlir/test_segment_gradients.py \
    python/tests/mlir/test_artifact.py

"$PYTHON" - "$WORK/segmented-calls.xml" <<'PY'
import sys
import xml.etree.ElementTree as ElementTree

suite = ElementTree.parse(sys.argv[1]).getroot()
suite = suite if suite.tag == "testsuite" else suite.find("testsuite")
counts = {name: int(suite.get(name, 0)) for name in ("tests", "skipped")}
print(f"segmented calls qualified: {counts['tests']} tests")
if counts["skipped"]:
    sys.exit(f"{counts['skipped']} tests skipped; nothing qualifies them")
PY
