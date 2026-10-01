# tests/python/test_fetch_llvm.py
"""Tests for verified retrieval of the pinned LLVM source release.

Every test runs offline: a curl wrapper placed first on PATH refuses any
URL that is not a local file, so a script that ignored its mirror override
would fail instead of reaching the network. The real script is exercised
against the committed pin and digest with a wrong tarball and with source
directories that already exist. The accepting paths run a copy of the
script inside a temporary repository root whose digest file records a small
synthetic tarball, because the script resolves its inputs relative to its
own location.
"""

import hashlib
import io
import os
import re
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "fetch_llvm.sh"
_PIN_FILE = _REPO_ROOT / "cmake" / "llvm-version.txt"
_DIGEST_FILE = _REPO_ROOT / "cmake" / "llvm-source-sha256.txt"
_TAG = _PIN_FILE.read_text().strip()
_VERSION = _TAG.removeprefix("llvmorg-")
_TARBALL = f"llvm-project-{_VERSION}.src.tar.xz"
_UNPACKED = f"llvm-project-{_VERSION}.src"
_SOURCE = f"src-{_TAG}"
_MARKER = ".swage-source-sha256"

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("tar") is None,
    reason="bash and tar are required to run scripts/fetch_llvm.sh",
)
_needs_curl = pytest.mark.skipif(
    shutil.which("curl") is None, reason="curl unavailable"
)


def _source_tarball(path, members=(f"{_UNPACKED}/README",)):
    """Write a small tarball shaped like an LLVM source release.

    Args:
        path: Destination of the xz-compressed tarball.
        members: Names of the files the tarball holds.

    Returns:
        The SHA-256 hex digest of the written file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"synthetic release used by tests/python/test_fetch_llvm.py\n"
    with tarfile.open(path, "w:xz") as archive:
        for name in members:
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repository(root, digest_text):
    """Create a repository root holding a copy of the script and the pin.

    Args:
        root: Directory that becomes the temporary repository root.
        digest_text: Contents of its digest file, or None to omit the file.

    Returns:
        The path of the copied script.
    """
    (root / "scripts").mkdir(parents=True)
    (root / "cmake").mkdir()
    script = root / "scripts" / "fetch_llvm.sh"
    shutil.copy(_SCRIPT, script)
    shutil.copy(_PIN_FILE, root / "cmake" / "llvm-version.txt")
    if digest_text is not None:
        (root / "cmake" / "llvm-source-sha256.txt").write_text(digest_text)
    return script


_OFFLINE_CURL = """\
#!/bin/sh
# Record the call, refuse network URLs, and pass local files through.
touch "{marker}"
for argument in "$@"; do
    case "$argument" in
        *://*)
            case "$argument" in
                file://*) ;;
                *)
                    echo "offline test: refusing $argument" >&2
                    exit 97
                    ;;
            esac
            ;;
    esac
done
exec "{curl}" "$@"
"""
# Stands in for tar: unpack half a tree into the -C directory, then stop.
_INTERRUPTED_TAR = """\
#!/bin/sh
destination=.
while [ "$#" -gt 0 ]; do
    if [ "$1" = "-C" ]; then
        destination="$2"
    fi
    shift
done
mkdir -p "$destination/{unpacked}/llvm"
echo "half a file" > "$destination/{unpacked}/llvm/PARTIAL"
{ending}
"""


def _fetch(script, tmp_path, llvm_home, url=None, tar_ending=None):
    """Run a fetch script with an isolated home and no reachable network.

    Args:
        script: Script to run.
        tmp_path: Scratch directory of the test.
        llvm_home: Value of SWAGE_LLVM_HOME.
        url: Value of SWAGE_LLVM_URL. Defaults to a missing local file.
        tar_ending: Shell command that ends a tar stand-in after it has
            unpacked half a tree. The real tar runs when this is None.

    Returns:
        The completed process and a marker path that exists when the
        script invoked curl.
    """
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    marker = tmp_path / "curl-was-called"
    stubs = tmp_path / "stubs"
    stubs.mkdir(exist_ok=True)
    stub = stubs / "curl"
    stub.write_text(
        _OFFLINE_CURL.format(marker=marker, curl=shutil.which("curl") or "")
    )
    stub.chmod(0o755)
    if tar_ending is not None:
        stub = stubs / "tar"
        stub.write_text(
            _INTERRUPTED_TAR.format(unpacked=_UNPACKED, ending=tar_ending)
        )
        stub.chmod(0o755)
    environment = dict(os.environ)
    environment.update(
        HOME=str(home),
        PATH=f"{stubs}{os.pathsep}{environment['PATH']}",
        SWAGE_LLVM_HOME=str(llvm_home),
        SWAGE_LLVM_URL=url or (tmp_path / "no-such-mirror").as_uri(),
    )
    result = subprocess.run(
        ["bash", str(script)],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    return result, marker


def _recorded_digests():
    """Return the (digest, tarball name) entries of the committed file."""
    return [
        tuple(line.split())
        for line in _DIGEST_FILE.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _entries(directory):
    """Return the sorted names directly inside a directory."""
    return sorted(path.name for path in directory.iterdir())


def _assert_nothing_extracted(llvm_home):
    """Assert that no source tree and no unpacking leftover exists."""
    assert not (llvm_home / _SOURCE).exists()
    assert not (llvm_home / _UNPACKED).exists()
    assert not list(llvm_home.glob(".unpack-*"))


def _existing_source(llvm_home, marker=None):
    """Create a source directory that looks like an earlier extraction.

    Args:
        llvm_home: Value of SWAGE_LLVM_HOME.
        marker: Contents of its verification marker, or None for a tree
            extracted before the marker existed.

    Returns:
        The path of the source directory.
    """
    source = llvm_home / _SOURCE
    (source / "llvm").mkdir(parents=True)
    (source / "llvm" / "CMakeLists.txt").write_text("# earlier extraction\n")
    if marker is not None:
        (source / _MARKER).write_text(marker)
    return source


def test_committed_digest_belongs_to_the_pinned_tarball():
    """The pin and the recorded digest name the same release tarball."""
    entries = _recorded_digests()

    assert len(entries) == 1
    digest, name = entries[0]
    assert re.fullmatch(r"[0-9a-f]{64}", digest)
    assert name == _TARBALL


def test_wrong_existing_tarball_is_rejected_before_extraction(tmp_path):
    """A tarball of the expected name is not trusted without its digest."""
    llvm_home = tmp_path / "llvm"
    wrong_digest = _source_tarball(llvm_home / _TARBALL)
    original = (llvm_home / _TARBALL).read_bytes()
    ((expected_digest, _),) = _recorded_digests()

    result, curl_marker = _fetch(_SCRIPT, tmp_path, llvm_home)

    assert result.returncode != 0
    assert wrong_digest != expected_digest
    assert f"expected: {expected_digest}" in result.stderr
    assert f"actual:   {wrong_digest}" in result.stderr
    _assert_nothing_extracted(llvm_home)
    assert (llvm_home / _TARBALL).read_bytes() == original
    assert not curl_marker.exists(), "an existing tarball was downloaded again"


@_needs_curl
def test_wrong_download_is_rejected_before_extraction(tmp_path):
    """A freshly downloaded tarball is verified like an existing one."""
    llvm_home = tmp_path / "llvm"
    mirror = tmp_path / "mirror" / "release.tar.xz"
    wrong_digest = _source_tarball(mirror)

    result, curl_marker = _fetch(
        _SCRIPT, tmp_path, llvm_home, url=mirror.as_uri()
    )

    assert result.returncode != 0
    assert curl_marker.exists()
    assert f"actual:   {wrong_digest}" in result.stderr
    _assert_nothing_extracted(llvm_home)


def test_matching_existing_tarball_is_extracted(tmp_path):
    """A tarball whose digest matches the recorded one is unpacked."""
    llvm_home = tmp_path / "llvm"
    digest = _source_tarball(llvm_home / _TARBALL)
    script = _repository(tmp_path / "repo", f"{digest}  {_TARBALL}\n")

    result, curl_marker = _fetch(script, tmp_path, llvm_home)

    assert result.returncode == 0, result.stderr
    assert _entries(llvm_home / _SOURCE) == [_MARKER, "README"]
    assert (llvm_home / _SOURCE / _MARKER).read_text().split() == [
        digest,
        _TARBALL,
    ]
    assert _entries(llvm_home) == [_TARBALL, _SOURCE]
    assert not curl_marker.exists()


@_needs_curl
def test_mirror_url_is_downloaded_verified_and_extracted(tmp_path):
    """SWAGE_LLVM_URL selects the download source of the pinned tarball."""
    llvm_home = tmp_path / "llvm"
    mirror = tmp_path / "mirror" / "release.tar.xz"
    digest = _source_tarball(mirror)
    script = _repository(
        tmp_path / "repo", f"# a comment line\n{digest}  {_TARBALL}\n"
    )

    result, curl_marker = _fetch(
        script, tmp_path, llvm_home, url=mirror.as_uri()
    )

    assert result.returncode == 0, result.stderr
    assert curl_marker.exists()
    assert mirror.as_uri() in result.stdout
    assert (llvm_home / _SOURCE / "README").is_file()
    assert (llvm_home / _TARBALL).read_bytes() == mirror.read_bytes()


def test_leftover_unpack_directory_is_refused(tmp_path):
    """Files outside the verified tarball never join the source tree."""
    llvm_home = tmp_path / "llvm"
    digest = _source_tarball(llvm_home / _TARBALL)
    script = _repository(tmp_path / "repo", f"{digest}  {_TARBALL}\n")
    leftover = llvm_home / _UNPACKED
    leftover.mkdir()
    (leftover / "PLANTED.cmake").write_text("not in the tarball\n")

    result, curl_marker = _fetch(script, tmp_path, llvm_home)

    assert result.returncode != 0
    assert str(leftover) in result.stderr
    assert "not verified" in result.stderr
    assert not (llvm_home / _SOURCE).exists()
    assert not list(llvm_home.glob(".unpack-*"))
    assert _entries(leftover) == ["PLANTED.cmake"]
    assert not curl_marker.exists()


@pytest.mark.parametrize(
    "members",
    [
        ("another-project/README",),
        (f"{_UNPACKED}/README", "EXTRA.cmake"),
        (f"{_UNPACKED}/README", "second-tree/README"),
    ],
    ids=["renamed-tree", "extra-file", "extra-tree"],
)
def test_tarball_of_another_shape_is_rejected(tmp_path, members):
    """Only the single release directory is moved into place."""
    llvm_home = tmp_path / "llvm"
    digest = _source_tarball(llvm_home / _TARBALL, members)
    script = _repository(tmp_path / "repo", f"{digest}  {_TARBALL}\n")

    result, _ = _fetch(script, tmp_path, llvm_home)

    assert result.returncode != 0
    assert _UNPACKED in result.stderr
    _assert_nothing_extracted(llvm_home)
    assert _entries(llvm_home) == [_TARBALL]


@pytest.mark.parametrize(
    "tar_ending",
    ["exit 2", 'kill -TERM "$PPID"'],
    ids=["tar-fails", "script-terminated"],
)
def test_interrupted_extraction_leaves_no_source_tree(tmp_path, tar_ending):
    """A half unpacked tree never gets the name of the source tree."""
    llvm_home = tmp_path / "llvm"
    digest = _source_tarball(llvm_home / _TARBALL)
    script = _repository(tmp_path / "repo", f"{digest}  {_TARBALL}\n")

    result, _ = _fetch(script, tmp_path, llvm_home, tar_ending=tar_ending)

    assert result.returncode != 0
    _assert_nothing_extracted(llvm_home)
    assert _entries(llvm_home) == [_TARBALL]


def test_verified_source_tree_is_reused_quietly(tmp_path):
    """A tree this script extracted is accepted by its marker alone."""
    llvm_home = tmp_path / "llvm"
    digest = _source_tarball(llvm_home / _TARBALL)
    script = _repository(tmp_path / "repo", f"{digest}  {_TARBALL}\n")
    first, _ = _fetch(script, tmp_path, llvm_home)
    assert first.returncode == 0, first.stderr
    (llvm_home / _TARBALL).unlink()

    second, curl_marker = _fetch(script, tmp_path, llvm_home)

    assert second.returncode == 0, second.stderr
    assert second.stderr == ""
    assert str(llvm_home / _SOURCE) in second.stdout
    assert _entries(llvm_home) == [_SOURCE]
    assert not curl_marker.exists()


def test_source_tree_marked_with_the_committed_digest_is_reused(tmp_path):
    """The real script accepts a tree marked with the recorded digest."""
    llvm_home = tmp_path / "llvm"
    ((expected_digest, _),) = _recorded_digests()
    _existing_source(llvm_home, f"{expected_digest}  {_TARBALL}\n")

    result, curl_marker = _fetch(_SCRIPT, tmp_path, llvm_home)

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert not curl_marker.exists()


def test_unmarked_source_tree_is_used_with_one_notice(tmp_path):
    """A tree extracted before the marker existed keeps working."""
    llvm_home = tmp_path / "llvm"
    source = _existing_source(llvm_home)

    result, curl_marker = _fetch(_SCRIPT, tmp_path, llvm_home)

    assert result.returncode == 0, result.stderr
    (notice,) = result.stderr.splitlines()
    assert notice.startswith("notice: ")
    assert str(source) in notice
    assert "not verified by this script" in notice
    assert "remove the directory and rerun scripts/fetch_llvm.sh" in notice
    assert _entries(source) == ["llvm"]
    assert _entries(llvm_home) == [_SOURCE]
    assert not curl_marker.exists()


@pytest.mark.parametrize(
    ("marker", "reported"),
    [
        (f"{'0' * 64}  {_TARBALL}\n", "0" * 64),
        ("", "<none>"),
    ],
    ids=["another-digest", "empty-marker"],
)
def test_source_tree_marked_with_another_digest_is_rejected(
    tmp_path, marker, reported
):
    """A marker that disagrees with the recorded digest is an error."""
    llvm_home = tmp_path / "llvm"
    source = _existing_source(llvm_home, marker)
    ((expected_digest, _),) = _recorded_digests()

    result, curl_marker = _fetch(_SCRIPT, tmp_path, llvm_home)

    assert result.returncode != 0
    assert f"expected: {expected_digest}" in result.stderr
    assert f"marker:   {reported}" in result.stderr
    assert str(source) in result.stderr
    assert _entries(source) == [_MARKER, "llvm"]
    assert (source / _MARKER).read_text() == marker
    assert not curl_marker.exists()


def test_empty_source_directory_is_rejected(tmp_path):
    """A directory of the right name is not a source tree by itself."""
    llvm_home = tmp_path / "llvm"
    (llvm_home / _SOURCE).mkdir(parents=True)

    result, curl_marker = _fetch(_SCRIPT, tmp_path, llvm_home)

    assert result.returncode != 0
    assert str(llvm_home / _SOURCE) in result.stderr
    assert "is empty" in result.stderr
    assert _entries(llvm_home) == [_SOURCE]
    assert _entries(llvm_home / _SOURCE) == []
    assert not curl_marker.exists()


def test_existing_source_tree_still_needs_the_digest_record(tmp_path):
    """An existing tree is not accepted before the record is read."""
    llvm_home = tmp_path / "llvm"
    _existing_source(llvm_home)
    script = _repository(tmp_path / "repo", None)

    result, curl_marker = _fetch(script, tmp_path, llvm_home)

    assert result.returncode != 0
    assert "llvm-source-sha256.txt is missing" in result.stderr
    assert not curl_marker.exists()


@pytest.mark.parametrize(
    ("digest_text", "message"),
    [
        (None, "missing"),
        ("", "no SHA-256 for"),
        ("# only a comment\n", "no SHA-256 for"),
        (f"{'0' * 64}  llvm-project-0.0.0.src.tar.xz\n", "no SHA-256 for"),
        (f"not-a-digest  {_TARBALL}\n", "not a SHA-256 digest"),
        (f"{'A' * 64}  {_TARBALL}\n", "not a SHA-256 digest"),
    ],
    ids=[
        "missing-file",
        "empty-file",
        "comment-only",
        "other-release",
        "malformed-digest",
        "uppercase-digest",
    ],
)
def test_unusable_digest_record_fails_closed(tmp_path, digest_text, message):
    """Without a usable recorded digest nothing is fetched or extracted."""
    llvm_home = tmp_path / "llvm"
    _source_tarball(llvm_home / _TARBALL)
    script = _repository(tmp_path / "repo", digest_text)

    result, curl_marker = _fetch(script, tmp_path, llvm_home)

    assert result.returncode != 0
    assert message in result.stderr
    assert "llvm-source-sha256.txt" in result.stderr
    _assert_nothing_extracted(llvm_home)
    assert not curl_marker.exists()
