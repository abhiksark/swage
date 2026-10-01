#!/usr/bin/env bash
# scripts/fetch_llvm.sh
# Download, verify, and extract the pinned LLVM/MLIR source release.
#
# The pin lives in cmake/llvm-version.txt and the SHA-256 of its source
# tarball in cmake/llvm-source-sha256.txt. Sources land in
# $SWAGE_LLVM_HOME (default: ~/.swage/llvm), outside the repository. The
# tarball is verified before extraction, whether it was just downloaded or
# was already present.
#
# Environment overrides:
#   SWAGE_LLVM_HOME  download and source root (default ~/.swage/llvm)
#   SWAGE_LLVM_URL   full URL of the pinned tarball, for example on a
#                    mirror (default: the llvm-project GitHub release
#                    asset). The digest check applies to every source.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG="$(cat "$REPO_ROOT/cmake/llvm-version.txt")"
VERSION="${TAG#llvmorg-}"
LLVM_HOME="${SWAGE_LLVM_HOME:-$HOME/.swage/llvm}"
SRC_DIR="$LLVM_HOME/src-$TAG"
TARBALL="llvm-project-$VERSION.src.tar.xz"
DIGEST_FILE="$REPO_ROOT/cmake/llvm-source-sha256.txt"
RELEASES="https://github.com/llvm/llvm-project/releases/download"
URL="${SWAGE_LLVM_URL:-$RELEASES/$TAG/$TARBALL}"

die() {
    echo "error: $*" >&2
    exit 1
}

if [ -d "$SRC_DIR" ]; then
    echo "LLVM source already present: $SRC_DIR"
    exit 0
fi

# Resolve the recorded digest and a hashing tool before any download, so an
# unverifiable fetch stops early.
if [ ! -f "$DIGEST_FILE" ]; then
    die "$DIGEST_FILE is missing; it must record the SHA-256 of $TARBALL"
fi
EXPECTED="$(awk -v name="$TARBALL" \
    '$1 !~ /^#/ && $2 == name { print $1; exit }' "$DIGEST_FILE")"
if [ -z "$EXPECTED" ]; then
    die "$DIGEST_FILE has no SHA-256 for $TARBALL, the tarball of the pin" \
        "$TAG; the recorded digest and cmake/llvm-version.txt must change" \
        "together"
fi
if [[ ! "$EXPECTED" =~ ^[0-9a-f]{64}$ ]]; then
    die "$DIGEST_FILE records '$EXPECTED' for $TARBALL, which is not a" \
        "SHA-256 digest (64 lowercase hexadecimal characters)"
fi

if command -v sha256sum >/dev/null; then
    SHA256=(sha256sum)
elif command -v shasum >/dev/null; then
    SHA256=(shasum -a 256)
else
    die "found neither sha256sum nor shasum; one is required to verify" \
        "$TARBALL"
fi

mkdir -p "$LLVM_HOME"
cd "$LLVM_HOME"
if [ ! -f "$TARBALL" ]; then
    echo "Downloading $URL"
    # An interrupted download never leaves a file of the expected name.
    curl -fL --retry 3 -o "$TARBALL.partial" "$URL"
    mv "$TARBALL.partial" "$TARBALL"
fi

ACTUAL="$("${SHA256[@]}" < "$TARBALL" | awk '{ print $1 }')"
if [ "$ACTUAL" != "$EXPECTED" ]; then
    cat >&2 <<EOF
error: SHA-256 mismatch for $LLVM_HOME/$TARBALL
  expected: $EXPECTED (cmake/llvm-source-sha256.txt)
  actual:   $ACTUAL
Nothing was extracted. Remove the file to download it again.
EOF
    exit 1
fi
echo "Verified SHA-256 of $TARBALL: $ACTUAL"

echo "Extracting $TARBALL"
tar -xf "$TARBALL"
mv "llvm-project-$VERSION.src" "$SRC_DIR"
echo "LLVM source ready: $SRC_DIR"
