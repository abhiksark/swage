#!/usr/bin/env bash
# scripts/build_llvm.sh
# Configure, build, and install the pinned LLVM/MLIR release.
#
# Environment overrides:
#   SWAGE_LLVM_HOME            source/build/install root (default ~/.swage/llvm)
#   SWAGE_LLVM_BUILD_TYPE      CMake build type (default RelWithDebInfo)
#   SWAGE_LLVM_PYTHON_BINDINGS ON/OFF for MLIR Python bindings (default ON)
#   SWAGE_PYTHON_EXECUTABLE    active Python interpreter (default python3)
#   SWAGE_LLVM_TARGETS        LLVM targets (default Native;NVPTX)
#   SWAGE_LLVM_SANITIZERS     LLVM sanitizer list (default empty)
#   CMAKE_BUILD_PARALLEL_LEVEL maximum concurrent build jobs (CMake default)
#   CC / CXX                   host compiler
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG="$(cat "$REPO_ROOT/cmake/llvm-version.txt")"
LLVM_HOME="${SWAGE_LLVM_HOME:-$HOME/.swage/llvm}"
SRC_DIR="$LLVM_HOME/src-$TAG"
BUILD_DIR="$LLVM_HOME/build-$TAG"
INSTALL_DIR="$LLVM_HOME/install-$TAG"
BUILD_TYPE="${SWAGE_LLVM_BUILD_TYPE:-RelWithDebInfo}"
ENABLE_PYTHON="${SWAGE_LLVM_PYTHON_BINDINGS:-ON}"
PYTHON_EXECUTABLE="$(command -v "${SWAGE_PYTHON_EXECUTABLE:-python3}")"

if [ ! -d "$SRC_DIR" ]; then
    echo "error: LLVM source not found at $SRC_DIR (run scripts/fetch_llvm.sh)" >&2
    exit 1
fi

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
    -DLLVM_TARGETS_TO_BUILD="${SWAGE_LLVM_TARGETS:-Native;NVPTX}" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_INSTALL_GTEST=ON \
    -DLLVM_INSTALL_UTILS=ON \
    -DLLVM_USE_SANITIZER="${SWAGE_LLVM_SANITIZERS:-}" \
    -DMLIR_ENABLE_BINDINGS_PYTHON="$ENABLE_PYTHON" \
    -DPython3_EXECUTABLE="$PYTHON_EXECUTABLE" \
    "${EXTRA_ARGS[@]}"

cmake --build "$BUILD_DIR" --target install
echo "LLVM/MLIR installed: $INSTALL_DIR"
