#!/usr/bin/env bash
# scripts/build_native_wheel.sh
# Build a wheel of the native `mlir_swage` package against an existing
# install of the pinned LLVM/MLIR release.
#
#   scripts/build_native_wheel.sh [OUTPUT_DIR]
#
# The wheel, `swage_compiler_native-<version>-<tag>.whl`, lands in
# OUTPUT_DIR (default ./dist). It holds the MLIR Python bindings, the Swage
# dialect bindings, the compiler library they share, the license, and the
# third-party notices. It records the swage version and the source revision
# it was built from, and it requires the `swage-compiler` wheel of the same
# version, which `python -m build` produces from the same checkout.
#
# The wheel is built for the `python` found on PATH (see build_swage.sh)
# and carries the plain `linux` platform tag: it runs on systems that are
# at least as new as the build host and provide zlib and Zstandard, and it
# is not checked against a `manylinux` policy.
#
# Environment overrides:
#   SWAGE_LLVM_HOME        LLVM root used by scripts/build_llvm.sh
#   MLIR_DIR/LLVM_DIR      explicit CMake package dirs of an install of the
#                          pinned release
#   SWAGE_WHEEL_BUILD_DIR  build tree (default ./build-native-wheel)
#   CMAKE_BUILD_PARALLEL_LEVEL  number of parallel build jobs
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG="$(cat "$REPO_ROOT/cmake/llvm-version.txt")"
LLVM_HOME="${SWAGE_LLVM_HOME:-$HOME/.swage/llvm}"
INSTALL_DIR="$LLVM_HOME/install-$TAG"
BUILD_DIR="${SWAGE_WHEEL_BUILD_DIR:-$REPO_ROOT/build-native-wheel}"
OUTPUT_DIR="${1:-$REPO_ROOT/dist}"
STAGE_DIR="$BUILD_DIR/wheel-stage"

MLIR_DIR="${MLIR_DIR:-$INSTALL_DIR/lib/cmake/mlir}"
LLVM_DIR="${LLVM_DIR:-$INSTALL_DIR/lib/cmake/llvm}"

if [ ! -d "$MLIR_DIR" ]; then
    echo "error: MLIR not found at $MLIR_DIR (run scripts/build_llvm.sh or set MLIR_DIR)" >&2
    exit 1
fi

PYTHON=""
for candidate in python python3; do
    if command -v "$candidate" >/dev/null &&
        "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 10))' \
            >/dev/null 2>&1; then
        PYTHON="$(command -v "$candidate")"
        break
    fi
done
if [ -z "$PYTHON" ]; then
    echo "error: found no Python 3.10 or newer on PATH as python or python3;" \
        "the wheel is built for that interpreter" >&2
    exit 1
fi
echo "Python interpreter: $PYTHON"

cmake -G Ninja -S "$REPO_ROOT" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DMLIR_DIR="$MLIR_DIR" \
    -DLLVM_DIR="$LLVM_DIR" \
    -DPython3_EXECUTABLE="$PYTHON" \
    -DSWAGE_PYTHON_BINDINGS=ON \
    -DLLVM_EXTERNAL_LIT="$(command -v lit || true)"

cmake --build "$BUILD_DIR" --target SwagePythonModules

# The install step copies real files instead of the build tree's links to
# the LLVM install, and leaves each library searching only next to itself.
rm -rf "$STAGE_DIR"
cmake --install "$BUILD_DIR" --component SwagePythonModules \
    --prefix "$STAGE_DIR" --strip >/dev/null

# The staged package is loaded once, away from the build tree and from any
# installed swage, and reports the identity that was compiled into it.
IDENTITY="$(cd "$STAGE_DIR" && "$PYTHON" -I -c '
import sys

sys.path.insert(0, "python_packages")
from mlir_swage._mlir_libs._swageDialectsNanobind import swage as native

print(native.__version__, native.__source_revision__, native.__llvm_version__)
')"
read -r VERSION REVISION LLVM_VERSION <<<"$IDENTITY"
case "$REVISION" in
*-dirty | unknown)
    echo "notice: the wheel records source revision $REVISION, which does" \
        "not name one commit" >&2
    ;;
esac

"$PYTHON" "$REPO_ROOT/scripts/assemble_native_wheel.py" \
    --package "$STAGE_DIR/python_packages/mlir_swage" \
    --version "$VERSION" \
    --revision "$REVISION" \
    --llvm-version "$LLVM_VERSION" \
    --license "$REPO_ROOT/LICENSE" \
    --notices "$REPO_ROOT/THIRD_PARTY_NOTICES.md" \
    --output "$OUTPUT_DIR"
