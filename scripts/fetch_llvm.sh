#!/usr/bin/env bash
# scripts/fetch_llvm.sh
# Download, verify, and extract the pinned LLVM/MLIR source release.
#
# The pin lives in cmake/llvm-version.txt and the SHA-256 of its source
# tarball in cmake/llvm-source-sha256.txt. Sources land in
# $SWAGE_LLVM_HOME (default: ~/.swage/llvm), outside the repository.
#
# The tarball is verified before extraction, whether it was just downloaded
# or was already present. It is unpacked into a fresh temporary directory
# and renamed into place, so the source tree holds only what the verified
# tarball held, plus a .swage-source-sha256 marker that records the digest.
# A source tree that already exists is accepted when its marker matches the
# recorded digest, used with a notice when it has no marker, and rejected
# when its marker differs or the directory is empty.
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
UNPACKED="llvm-project-$VERSION.src"
TARBALL="$UNPACKED.tar.xz"
MARKER=".swage-source-sha256"
DIGEST_FILE="$REPO_ROOT/cmake/llvm-source-sha256.txt"
RELEASES="https://github.com/llvm/llvm-project/releases/download"
URL="${SWAGE_LLVM_URL:-$RELEASES/$TAG/$TARBALL}"

die() {
    echo "error: $*" >&2
    exit 1
}

# Resolve the recorded digest first. An existing source tree is compared
# with it, and an unverifiable fetch stops before any download.
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

if [ -d "$SRC_DIR" ]; then
    if [ -f "$SRC_DIR/$MARKER" ]; then
        VERIFIED="$(awk 'NR == 1 { print $1 }' "$SRC_DIR/$MARKER")"
        if [ "$VERIFIED" != "$EXPECTED" ]; then
            cat >&2 <<EOF
error: $SRC_DIR was not extracted from the recorded tarball
  expected: $EXPECTED (cmake/llvm-source-sha256.txt)
  marker:   ${VERIFIED:-<none>} ($MARKER)
Remove the directory and rerun scripts/fetch_llvm.sh.
EOF
            exit 1
        fi
    elif [ -z "$(ls -A "$SRC_DIR")" ]; then
        die "$SRC_DIR is empty, so it is not an LLVM source tree; remove" \
            "the directory and rerun scripts/fetch_llvm.sh"
    else
        echo "notice: $SRC_DIR has no $MARKER marker, so it was not verified" \
            "by this script and is used as it is; to replace it with a" \
            "verified tree, remove the directory and rerun" \
            "scripts/fetch_llvm.sh" >&2
    fi
    echo "LLVM source already present: $SRC_DIR"
    exit 0
fi
if [ -e "$SRC_DIR" ] || [ -L "$SRC_DIR" ]; then
    die "$SRC_DIR exists but is not a directory; remove it and rerun" \
        "scripts/fetch_llvm.sh"
fi

# Earlier versions of this script unpacked here before renaming. Whatever
# is in such a directory did not come from this run's verified tarball.
if [ -e "$LLVM_HOME/$UNPACKED" ] || [ -L "$LLVM_HOME/$UNPACKED" ]; then
    die "$LLVM_HOME/$UNPACKED already exists and is not verified; it is" \
        "left over from an interrupted earlier run or was placed there." \
        "Remove it and rerun scripts/fetch_llvm.sh"
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
if [ ! -f "$LLVM_HOME/$TARBALL" ]; then
    echo "Downloading $URL"
    # An interrupted download never leaves a file of the expected name.
    curl -fL --retry 3 -o "$LLVM_HOME/$TARBALL.partial" "$URL"
    mv "$LLVM_HOME/$TARBALL.partial" "$LLVM_HOME/$TARBALL"
fi

ACTUAL="$("${SHA256[@]}" < "$LLVM_HOME/$TARBALL" | awk '{ print $1 }')"
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

# Unpack into a fresh directory on the same filesystem and rename the
# result into place. A failed or interrupted extraction is removed on exit
# and never has the name of the source tree.
UNPACK_DIR="$(mktemp -d "$LLVM_HOME/.unpack-$TAG.XXXXXX")"
trap 'rm -rf "$UNPACK_DIR"' EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

echo "Extracting $TARBALL"
tar -xf "$LLVM_HOME/$TARBALL" -C "$UNPACK_DIR"
if [ "$(ls -A "$UNPACK_DIR")" != "$UNPACKED" ] ||
    [ ! -d "$UNPACK_DIR/$UNPACKED" ] || [ -L "$UNPACK_DIR/$UNPACKED" ]; then
    die "$TARBALL must unpack to the single directory $UNPACKED, but it" \
        "unpacked to: $(ls -A "$UNPACK_DIR" | tr '\n' ' ')"
fi
printf '%s  %s\n' "$EXPECTED" "$TARBALL" > "$UNPACK_DIR/$UNPACKED/$MARKER"
if [ -e "$SRC_DIR" ] || [ -L "$SRC_DIR" ]; then
    die "$SRC_DIR appeared during extraction; it was left untouched"
fi
mv "$UNPACK_DIR/$UNPACKED" "$SRC_DIR"
echo "LLVM source ready: $SRC_DIR"
