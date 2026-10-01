# tests/python/test_fetch_llvm.py
"""Tests for verified retrieval of the pinned LLVM source release.

Every test runs offline: a curl wrapper placed first on PATH refuses any
URL that is not a local file, so a script that ignored its mirror override
would fail instead of reaching the network. The real script is exercised
against the committed pin and digest with a wrong tarball. The accepting
paths run a copy of the script inside a temporary repository root whose
digest file records a small synthetic tarball, because the script resolves
its inputs relative to its own location.
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

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("tar") is None,
    reason="bash and tar are required to run scripts/fetch_llvm.sh",
)
_needs_curl = pytest.mark.skipif(
    shutil.which("curl") is None, reason="curl unavailable"
)


def _source_tarball(path):
    """Write a small tarball shaped like an LLVM source release.

    Args:
        path: Destination of the xz-compressed tarball.

    Returns:
        The SHA-256 hex digest of the written file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"synthetic release used by tests/python/test_fetch_llvm.py\n"
    member = tarfile.TarInfo(f"{_UNPACKED}/README")
    member.size = len(payload)
    with tarfile.open(path, "w:xz") as archive:
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


def _fetch(script, tmp_path, llvm_home, url=None):
    """Run a fetch script with an isolated home and no reachable network.

    Args:
        script: Script to run.
        tmp_path: Scratch directory of the test.
        llvm_home: Value of SWAGE_LLVM_HOME.
        url: Value of SWAGE_LLVM_URL. Defaults to a missing local file.

    Returns:
        The completed process and a marker path that exists when the
        script invoked curl.
    """
    home = tmp_path / "home"
    home.mkdir()
    marker = tmp_path / "curl-was-called"
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    stub = stubs / "curl"
    stub.write_text(
        _OFFLINE_CURL.format(marker=marker, curl=shutil.which("curl") or "")
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


def _assert_nothing_extracted(llvm_home):
    """Assert that neither the unpacked nor the final source tree exists."""
    assert not (llvm_home / _SOURCE).exists()
    assert not (llvm_home / _UNPACKED).exists()


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
    assert (llvm_home / _SOURCE / "README").is_file()
    assert not (llvm_home / _UNPACKED).exists()
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


def test_existing_source_tree_needs_no_tarball(tmp_path):
    """An extracted source tree is reused without a download."""
    llvm_home = tmp_path / "llvm"
    (llvm_home / _SOURCE).mkdir(parents=True)

    result, curl_marker = _fetch(_SCRIPT, tmp_path, llvm_home)

    assert result.returncode == 0, result.stderr
    assert not curl_marker.exists()
    assert not (llvm_home / _TARBALL).exists()


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
