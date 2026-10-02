# tests/python/test_native_identity.py
"""Tests for how `swage` identifies and accepts its native bindings.

Two things are covered. The disk cache key names each native library by its
contents, so equal bytes give one key whatever the file times are, and other
bytes give another key whatever the file size and times are. And `swage`
refuses bindings that were not built for its version.

No test here loads a native library: the libraries are small fake files and
the bindings are fake modules.
"""

import hashlib
import importlib.machinery
import os
import pathlib
import shutil
import struct
import subprocess
import sys
import tarfile
import types
import warnings

import pytest
import swage as sw
import swage.language as sl
from swage import _runtime

_EXTENSION = "_swageDialectsNanobind.cpython-313-x86_64-linux-gnu.so"
_LIBRARY = "libSwagePythonCAPI.so.22.1"
_NATIVE = "mlir_swage._mlir_libs._swageDialectsNanobind"
# A fixed time, as an image build that normalizes file times would set it.
_NORMALIZED_NS = 1_000_000_000 * 10**9


@sw.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n, BLOCK: sl.constexpr):  # noqa: D103
    pid = sl.program_id(0)
    offsets = pid * BLOCK + sl.arange(0, BLOCK)
    mask = offsets < n
    x = sl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = sl.load(y_ptr + offsets, mask=mask, other=0.0)
    sl.store(output_ptr + offsets, x + y, mask=mask)


def _install(directory, extension=b"extension", library=b"compiler library"):
    """Write a fake `mlir_swage/_mlir_libs` holding the two libraries.

    Args:
        directory: Root of the fake install.
        extension: Contents of the nanobind extension.
        library: Contents of the C API library.

    Returns:
        The directory that holds the libraries.
    """
    libraries = directory / "mlir_swage" / "_mlir_libs"
    libraries.mkdir(parents=True)
    (libraries / _EXTENSION).write_bytes(extension)
    (libraries / _LIBRARY).write_bytes(library)
    (libraries / "libSwagePythonCAPI.so").symlink_to(_LIBRARY)
    (libraries / "__init__.py").write_text("")
    return libraries


def _use(monkeypatch, libraries):
    """Make `libraries` the directory `mlir_swage._mlir_libs` is found in."""
    spec = types.SimpleNamespace(submodule_search_locations=[str(libraries)])
    monkeypatch.setattr(
        _runtime.importlib.util, "find_spec", lambda _name: spec
    )


def _identity_and_key(monkeypatch, libraries):
    """Return the native identity and the launch cache key of an install."""
    _use(monkeypatch, libraries)
    monkeypatch.setattr(_runtime, "_identity_cache", None)
    data = _runtime._specialization_data(
        add_kernel,
        descriptors=("ptr<f32>", "ptr<f32>", "ptr<f32>", "i32"),
        constexprs={"BLOCK": 128},
        target="sm_86",
    )
    return data["native"], _runtime._cache_key(data)


def _normalize_times(libraries):
    """Give every file one fixed modification time."""
    for path in libraries.iterdir():
        if not path.is_symlink():
            os.utime(path, ns=(_NORMALIZED_NS, _NORMALIZED_NS))


def _elf(build_id, *, alignment=4, tail=b""):
    """Build a 64-bit little-endian ELF file that carries a GNU build id.

    Args:
        build_id: The id bytes, or None for a file with an unrelated note
            only.
        alignment: Alignment of the note segment, 4 or 8.
        tail: Bytes after the notes, standing in for code and symbols.
    """

    def padded(field):
        return field + b"\0" * (-len(field) % alignment)

    def note(name, descriptor, kind):
        return (
            struct.pack("<III", len(name), len(descriptor), kind)
            + padded(name)
            + padded(descriptor)
        )

    notes = note(b"GNU\0", b"\0" * 16, 5)
    if build_id is not None:
        notes += note(b"GNU\0", build_id, 3)
    header_size, entry_size = 64, 56
    header = struct.pack(
        "<16sHHIQQQIHHHHHH",
        b"\x7fELF\x02\x01\x01" + b"\0" * 9,
        3,
        62,
        1,
        0,
        header_size,
        0,
        0,
        header_size,
        entry_size,
        1,
        0,
        0,
        0,
    )
    segment = struct.pack(
        "<IIQQQQQQ",
        4,
        4,
        header_size + entry_size,
        0,
        0,
        len(notes),
        len(notes),
        alignment,
    )
    return header + segment + notes + tail


def _sha256(contents):
    return hashlib.sha256(contents).hexdigest()


def test_equal_bytes_give_one_key_whatever_the_file_times_are(
    tmp_path, monkeypatch
):
    """Keep the key across a copy, an archive, and normalized file times."""
    original = _install(tmp_path / "original")
    identity, key = _identity_and_key(monkeypatch, original)

    copied = tmp_path / "copied" / "mlir_swage" / "_mlir_libs"
    shutil.copytree(original, copied, symlinks=True)
    archive = tmp_path / "install.tar"
    with tarfile.open(archive, "w") as packed:
        packed.add(original, arcname="_mlir_libs")
    with tarfile.open(archive) as packed:
        packed.extractall(tmp_path / "extracted", filter="data")
    extracted = tmp_path / "extracted" / "_mlir_libs"
    normalized = _install(tmp_path / "normalized")
    _normalize_times(normalized)

    times = {
        (libraries / _LIBRARY).stat().st_mtime_ns
        for libraries in (original, copied, extracted, normalized)
    }
    assert len(times) > 1
    assert identity == [
        [_EXTENSION, f"sha256:{_sha256(b'extension')}"],
        [_LIBRARY, f"sha256:{_sha256(b'compiler library')}"],
    ]
    for libraries in (copied, extracted, normalized):
        assert _identity_and_key(monkeypatch, libraries) == (identity, key)


def test_other_bytes_give_another_key_whatever_the_metadata_is(
    tmp_path, monkeypatch
):
    """Tell two installs apart that agree in name, size, and file times."""
    first = _install(tmp_path / "first", library=b"compiler library")
    second = _install(tmp_path / "second", library=b"COMPILER LIBRARY")
    for libraries in (first, second):
        _normalize_times(libraries)
    assert [
        (path.name, path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(first.iterdir())
    ] == [
        (path.name, path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(second.iterdir())
    ]

    first_identity, first_key = _identity_and_key(monkeypatch, first)
    second_identity, second_key = _identity_and_key(monkeypatch, second)

    assert first_identity[0] == second_identity[0]
    assert first_identity[1] != second_identity[1]
    assert first_key != second_key


def test_library_rewritten_in_place_is_read_again(tmp_path, monkeypatch):
    """See new bytes behind an unchanged name, size, and modification time."""
    libraries = _install(tmp_path)
    _use(monkeypatch, libraries)
    before = _runtime._native_identity()

    (libraries / _LIBRARY).write_bytes(b"COMPILER LIBRARY")
    os.utime(libraries / _LIBRARY, ns=(_NORMALIZED_NS, _NORMALIZED_NS))
    (libraries / _EXTENSION).write_bytes(b"extension")

    after = _runtime._native_identity()
    assert after[0] == before[0]
    assert after[1] == [_LIBRARY, f"sha256:{_sha256(b'COMPILER LIBRARY')}"]


def test_build_id_identifies_a_library_across_stripping(
    tmp_path, monkeypatch
):
    """Name an ELF library by its build id, not by the rest of its bytes."""
    build_id = bytes(range(20))
    linked = _install(
        tmp_path / "linked",
        extension=_elf(b"\x11" * 20),
        library=_elf(build_id, tail=b"code" + b"symbols" * 100),
    )
    stripped = _install(
        tmp_path / "stripped",
        extension=_elf(b"\x11" * 20),
        library=_elf(build_id, tail=b"code"),
    )
    relinked = _install(
        tmp_path / "relinked",
        extension=_elf(b"\x11" * 20),
        library=_elf(bytes(range(1, 21)), tail=b"code"),
    )

    identity, key = _identity_and_key(monkeypatch, linked)

    assert identity == [
        [_EXTENSION, "build-id:" + "11" * 20],
        [_LIBRARY, f"build-id:{build_id.hex()}"],
    ]
    assert _identity_and_key(monkeypatch, stripped) == (identity, key)
    relinked_identity, relinked_key = _identity_and_key(monkeypatch, relinked)
    assert relinked_identity[1] == [
        _LIBRARY,
        f"build-id:{bytes(range(1, 21)).hex()}",
    ]
    assert relinked_key != key


@pytest.mark.parametrize("alignment", [4, 8])
def test_build_id_is_read_from_a_note_segment(tmp_path, alignment):
    """Find the build id after another note, at either note alignment."""
    library = tmp_path / "library.so"
    library.write_bytes(_elf(b"\xab\xcd\xef", alignment=alignment))

    assert _runtime._elf_build_id(library) == "abcdef"


@pytest.mark.parametrize(
    "contents",
    [
        b"",
        b"not an ELF file",
        _elf(None),
        _elf(b"\x01" * 20)[:70],
        _elf(b"\x01" * 20)[:-4],
        b"\x7fELF\x01\x01" + _elf(b"\x01" * 20)[6:],
        b"\x7fELF\x02\x02" + _elf(b"\x01" * 20)[6:],
    ],
    ids=[
        "empty",
        "text",
        "no-build-id-note",
        "cut-in-the-program-header",
        "cut-in-the-note",
        "32-bit",
        "big-endian",
    ],
)
def test_file_without_a_readable_build_id_is_hashed(tmp_path, contents):
    """Fall back to a digest of the whole file, and never raise."""
    library = tmp_path / "library.so"
    library.write_bytes(contents)

    assert _runtime._elf_build_id(library) is None
    assert _runtime._content_identity(library) == f"sha256:{_sha256(contents)}"


def test_library_without_a_build_id_is_hashed_once_per_process(
    tmp_path, monkeypatch
):
    """Hash a still library once, and again only after it changes."""
    libraries = _install(tmp_path)
    _use(monkeypatch, libraries)
    hashed = []
    digest = _runtime._file_digest
    monkeypatch.setattr(
        _runtime,
        "_file_digest",
        lambda path: hashed.append(path.name) or digest(path),
    )
    # The files were written by this process, so they count as still only
    # for a process that started after them.
    newest = max(_runtime._changed_ns(path) for path in libraries.iterdir())
    monkeypatch.setattr(_runtime, "_PROCESS_START_NS", newest + 1)

    first = _runtime._native_identity()
    second = _runtime._native_identity()
    (libraries / _LIBRARY).write_bytes(b"a longer compiler library")
    third = _runtime._native_identity()

    assert first == second
    assert third != first
    assert sorted(hashed) == [_EXTENSION, _LIBRARY, _LIBRARY]


def test_cache_key_is_the_same_in_a_checkout_and_in_an_install(monkeypatch):
    """Leave the LLVM pin of a checkout out of the key."""
    identity = {
        "revision": "0" * 40,
        "clean": True,
        "llvm": "llvmorg-test",
        "frontend": "f" * 64,
        "native": [[_EXTENSION, "build-id:00"]],
    }

    def key(**overrides):
        monkeypatch.setattr(
            _runtime, "_cached_identity", lambda: {**identity, **overrides}
        )
        data = _runtime._specialization_data(
            add_kernel,
            descriptors=("ptr<f32>", "ptr<f32>", "ptr<f32>", "i32"),
            constexprs={"BLOCK": 128},
            target="sm_86",
        )
        return _runtime._cache_key(data)

    assert key() == key(revision=None, clean=False, llvm=None)
    assert key() != key(native=[[_EXTENSION, "build-id:01"]])


def _bindings(**attributes):
    """Return a fake native `swage` module with a build identity."""
    identity = {
        "__version__": sw.__version__,
        "__source_revision__": "unknown",
        "__llvm_version__": "22.1.8",
    }
    identity.update(attributes)
    return types.SimpleNamespace(
        **{name: value for name, value in identity.items() if value is not None}
    )


def _importable(monkeypatch, native):
    """Make `native` the module that importing the bindings returns."""
    extension = types.ModuleType(_NATIVE)
    extension.swage = native
    libraries = types.ModuleType("mlir_swage._mlir_libs")
    libraries._swageDialectsNanobind = extension
    package = types.ModuleType("mlir_swage")
    package._mlir_libs = libraries
    for module in (package, libraries, extension):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(_runtime, "_verified_bindings", None)


def test_bindings_built_for_this_swage_are_accepted_once(monkeypatch):
    """Check matching bindings at the first use and not again."""
    native = _bindings()
    _importable(monkeypatch, native)
    checks = []
    monkeypatch.setattr(
        _runtime,
        "_stale_native_sources",
        lambda revision: checks.append(revision),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _runtime._native_bindings() is native
        assert _runtime._native_bindings() is native

    assert checks == ["unknown"]


@pytest.mark.parametrize(
    ("attributes", "reasons"),
    [
        (
            {"__version__": "0.0.1", "__source_revision__": "a" * 40},
            ("were built for swage 0.0.1", "source revision " + "a" * 40),
        ),
        ({"__version__": None}, ("record no swage version",)),
    ],
    ids=["another-version", "no-identity"],
)
def test_bindings_built_for_another_swage_are_refused(
    monkeypatch, attributes, reasons
):
    """Refuse bindings of another version, naming both sides."""
    native = _bindings(**attributes)
    _importable(monkeypatch, native)

    for _ in range(2):
        with pytest.raises(_runtime._BindingsMismatch) as refused:
            _runtime._native_bindings()

    message = str(refused.value)
    for reason in reasons:
        assert reason in message
    assert f"swage {sw.__version__}" in message
    assert str(_runtime._package_dir()) in message
    assert "ebuild" in message


def test_refusal_raised_while_the_extension_loads_is_not_hidden(monkeypatch):
    """Report the mismatch itself when the import of the bindings fails."""

    class _Finder:
        """Fail the import the way the extension does when it is refused."""

        def find_spec(self, name, path=None, target=None):
            if name != _NATIVE:
                return None
            return importlib.machinery.ModuleSpec(name, self)

        def create_module(self, spec):
            try:
                raise _runtime._BindingsMismatch("built for swage 0.0.1")
            except _runtime._BindingsMismatch as refused:
                raise ImportError("error initializing") from refused

        def exec_module(self, module):
            raise AssertionError("the module is never created")

    libraries = types.ModuleType("mlir_swage._mlir_libs")
    libraries.__path__ = []
    package = types.ModuleType("mlir_swage")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "mlir_swage", package)
    monkeypatch.setitem(sys.modules, "mlir_swage._mlir_libs", libraries)
    monkeypatch.delitem(sys.modules, _NATIVE, raising=False)
    monkeypatch.setattr(sys, "meta_path", [_Finder(), *sys.meta_path])

    with pytest.raises(_runtime._BindingsMismatch, match="built for swage"):
        _runtime._native_bindings()


def test_emit_and_compile_report_the_mismatch_itself(monkeypatch):
    """Do not describe refused bindings as missing bindings."""
    _importable(monkeypatch, _bindings(__version__="0.0.1"))

    with pytest.raises(_runtime._BindingsMismatch, match="0.0.1"):
        add_kernel.emit_mlir(
            signature={
                "x_ptr": sl.pointer(sl.float32),
                "y_ptr": sl.pointer(sl.float32),
                "output_ptr": sl.pointer(sl.float32),
                "n": sl.int32,
            },
            constexprs={"BLOCK": 128},
        )
    with pytest.raises(_runtime._BindingsMismatch, match="0.0.1"):
        _runtime._compile_native(object(), "add_kernel", 128, "sm_86")


def _git(root, *arguments):
    """Run git in `root` with an identity that needs no user configuration."""
    return subprocess.run(
        [
            "git",
            "-c",
            "user.name=Swage Tests",
            "-c",
            "user.email=tests@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "init.defaultBranch=main",
            *arguments,
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _checkout(root, monkeypatch):
    """Commit a small Swage checkout and run `swage` from its package.

    Returns:
        The package directory, the native source file, and the HEAD.
    """
    package = root / "python" / "swage"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("VERSION = 1\n")
    (root / "cmake").mkdir()
    (root / "cmake" / "llvm-version.txt").write_text("llvmorg-test\n")
    (root / "lib").mkdir()
    native_source = root / "lib" / "Codegen.cpp"
    native_source.write_text("// lowering 1\n")
    _git(root, "init", "--quiet")
    _git(root, "add", "--all")
    _git(root, "commit", "--quiet", "--message", "initial")
    monkeypatch.setattr(_runtime, "_package_dir", lambda: package)
    monkeypatch.setattr(_runtime, "_native_identity", lambda: None)
    monkeypatch.setattr(_runtime, "_identity_cache", None)
    return package, native_source, _git(root, "rev-parse", "HEAD")


def _problem(monkeypatch, built_from):
    """Compare bindings built from `built_from` with the checkout as it is."""
    monkeypatch.setattr(_runtime, "_identity_cache", None)
    return _runtime._stale_native_sources(built_from)


@pytest.mark.skipif(shutil.which("git") is None, reason="git unavailable")
def test_another_revision_matters_only_when_native_sources_differ(
    tmp_path, monkeypatch
):
    """Accept a frontend-only change, and name a native one."""
    package, native_source, built_from = _checkout(tmp_path, monkeypatch)

    assert _problem(monkeypatch, built_from) is None
    (package / "__init__.py").write_text("VERSION = 2\n")
    assert _problem(monkeypatch, built_from) is None
    _git(tmp_path, "commit", "--quiet", "--all", "--message", "frontend")
    assert _git(tmp_path, "rev-parse", "HEAD") != built_from
    assert _problem(monkeypatch, built_from) is None

    native_source.write_text("// lowering 2\n")
    uncommitted = _problem(monkeypatch, built_from)
    _git(tmp_path, "commit", "--quiet", "--all", "--message", "native")
    committed = _problem(monkeypatch, built_from)

    for problem in (uncommitted, committed):
        assert f"built from revision {built_from}" in problem
        assert "native sources" in problem
        assert str(tmp_path) in problem
        assert "rebuild the bindings" in problem


@pytest.mark.skipif(shutil.which("git") is None, reason="git unavailable")
def test_revision_that_cannot_be_compared_is_reported_or_skipped(
    tmp_path, monkeypatch
):
    """Name a revision the checkout lacks; skip one that names no sources."""
    _, native_source, built_from = _checkout(tmp_path, monkeypatch)
    native_source.write_text("// lowering 2\n")

    missing = _problem(monkeypatch, "f" * 40)

    assert "does not have" in missing
    assert "f" * 40 in missing
    assert _problem(monkeypatch, "unknown") is None
    assert _problem(monkeypatch, f"{built_from}-dirty") is None


def test_installed_swage_compares_no_revision(tmp_path, monkeypatch):
    """Run no comparison for a package that is not in a Swage checkout."""
    package = tmp_path / "site-packages" / "swage"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("VERSION = 1\n")
    monkeypatch.setattr(_runtime, "_package_dir", lambda: package)
    monkeypatch.setattr(_runtime, "_native_identity", lambda: None)
    monkeypatch.setattr(_runtime, "_identity_cache", None)
    monkeypatch.setattr(
        _runtime.subprocess,
        "run",
        lambda *_args, **_options: pytest.fail("git must not run"),
    )

    assert _runtime._stale_native_sources("a" * 40) is None


def test_stale_native_sources_warn_once_and_do_not_refuse(monkeypatch):
    """Warn about bindings behind the checkout, then use them."""
    native = _bindings(__source_revision__="a" * 40)
    _importable(monkeypatch, native)
    monkeypatch.setattr(
        _runtime,
        "_stale_native_sources",
        lambda revision: f"built from revision {revision}; rebuild",
    )

    with pytest.warns(RuntimeWarning, match="built from revision a+; rebuild"):
        assert _runtime._native_bindings() is native
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _runtime._native_bindings() is native


def test_import_of_swage_does_not_load_or_check_the_bindings():
    """Keep `import swage` free of the bindings and of their warnings."""
    completed = subprocess.run(
        [
            sys.executable,
            "-W",
            "error",
            "-c",
            "import sys, swage\n"
            "from swage import _runtime\n"
            "assert 'mlir_swage' not in sys.modules\n"
            "assert _runtime._verified_bindings is None",
        ],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        capture_output=True,
        text=True,
        check=False,
        cwd=pathlib.Path(__file__).parent,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
