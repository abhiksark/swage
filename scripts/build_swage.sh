#!/usr/bin/env bash
# scripts/build_swage.sh
# Configure and build the Swage MLIR components against the pinned
# LLVM/MLIR install, then run the lit test suite (check-swage).
#
# Environment overrides:
#   SWAGE_LLVM_HOME   LLVM root used by scripts/build_llvm.sh
#   MLIR_DIR/LLVM_DIR explicit CMake package dirs (any MLIR install works)
#   SWAGE_BUILD_DIR   build tree (default ./build)
#   SWAGE_BUILD_TYPE  CMake build type (default RelWithDebInfo)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG="$(cat "$REPO_ROOT/cmake/llvm-version.txt")"
LLVM_HOME="${SWAGE_LLVM_HOME:-$HOME/.swage/llvm}"
INSTALL_DIR="$LLVM_HOME/install-$TAG"
BUILD_DIR="${SWAGE_BUILD_DIR:-$REPO_ROOT/build}"

MLIR_DIR="${MLIR_DIR:-$INSTALL_DIR/lib/cmake/mlir}"
LLVM_DIR="${LLVM_DIR:-$INSTALL_DIR/lib/cmake/llvm}"

if [ ! -d "$MLIR_DIR" ]; then
    echo "error: MLIR not found at $MLIR_DIR (run scripts/build_llvm.sh or set MLIR_DIR)" >&2
    exit 1
fi

cmake -G Ninja -S "$REPO_ROOT" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE="${SWAGE_BUILD_TYPE:-RelWithDebInfo}" \
    -DMLIR_DIR="$MLIR_DIR" \
    -DLLVM_DIR="$LLVM_DIR" \
    -DLLVM_EXTERNAL_LIT="$(command -v lit || true)"

ninja -C "$BUILD_DIR"
ninja -C "$BUILD_DIR" check-swage
