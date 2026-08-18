#!/usr/bin/env bash
# scripts/fetch_llvm.sh
# Download and extract the pinned LLVM/MLIR source release.
#
# The pin lives in cmake/llvm-version.txt. Sources land in
# $SWAGE_LLVM_HOME (default: ~/.swage/llvm), outside the repository.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG="$(cat "$REPO_ROOT/cmake/llvm-version.txt")"
VERSION="${TAG#llvmorg-}"
LLVM_HOME="${SWAGE_LLVM_HOME:-$HOME/.swage/llvm}"
SRC_DIR="$LLVM_HOME/src-$TAG"
TARBALL="llvm-project-$VERSION.src.tar.xz"
URL="https://github.com/llvm/llvm-project/releases/download/$TAG/$TARBALL"

if [ -d "$SRC_DIR" ]; then
    echo "LLVM source already present: $SRC_DIR"
    exit 0
fi

mkdir -p "$LLVM_HOME"
cd "$LLVM_HOME"
if [ ! -f "$TARBALL" ]; then
    echo "Downloading $URL"
    curl -fL --retry 3 -o "$TARBALL" "$URL"
fi
echo "Extracting $TARBALL"
tar -xf "$TARBALL"
mv "llvm-project-$VERSION.src" "$SRC_DIR"
echo "LLVM source ready: $SRC_DIR"
