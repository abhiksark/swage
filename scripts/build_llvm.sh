#!/usr/bin/env bash
# scripts/build_llvm.sh
# Configure, build, and install the pinned LLVM/MLIR release.
#
# Environment overrides:
#   SWAGE_LLVM_HOME            source/build/install root (default ~/.swage/llvm)
#   SWAGE_LLVM_BUILD_TYPE      CMake build type (default RelWithDebInfo)
#   SWAGE_LLVM_PYTHON_BINDINGS ON/OFF for MLIR Python bindings (default ON)
#   CC / CXX                   host compiler
#
# The MLIR Python bindings are built for the `python` found on PATH, the
# interpreter the documented commands run, or for `python3` where `python`
# is absent or older than Python 3.10.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG="$(cat "$REPO_ROOT/cmake/llvm-version.txt")"
LLVM_HOME="${SWAGE_LLVM_HOME:-$HOME/.swage/llvm}"
SRC_DIR="$LLVM_HOME/src-$TAG"
BUILD_DIR="$LLVM_HOME/build-$TAG"
INSTALL_DIR="$LLVM_HOME/install-$TAG"
BUILD_TYPE="${SWAGE_LLVM_BUILD_TYPE:-RelWithDebInfo}"
ENABLE_PYTHON="${SWAGE_LLVM_PYTHON_BINDINGS:-ON}"

if [ ! -d "$SRC_DIR" ]; then
    echo "error: LLVM source not found at $SRC_DIR (run scripts/fetch_llvm.sh)" >&2
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
        "the LLVM build and the MLIR Python bindings need one" >&2
    exit 1
fi
echo "Python interpreter: $PYTHON"

EXTRA_ARGS=()
if command -v ccache >/dev/null; then
    EXTRA_ARGS+=(
        -DCMAKE_C_COMPILER_LAUNCHER=ccache
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
    )
fi

cmake -G Ninja -S "$SRC_DIR/llvm" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_TARGETS_TO_BUILD="Native;NVPTX" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_INSTALL_GTEST=ON \
    -DLLVM_INSTALL_UTILS=ON \
    -DMLIR_ENABLE_BINDINGS_PYTHON="$ENABLE_PYTHON" \
    -DPython3_EXECUTABLE="$PYTHON" \
    "${EXTRA_ARGS[@]}"

ninja -C "$BUILD_DIR" install
echo "LLVM/MLIR installed: $INSTALL_DIR"
