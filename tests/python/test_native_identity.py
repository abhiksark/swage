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
        _runtime, "_pair_problem", lambda bindings: checks.append(bindings)
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _runtime._native_bindings() is native
        assert _runtime._native_bindings() is native

    assert checks == [native]


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


def _problem(monkeypatch, built_from, frontend_digest=None):
    """Compare bindings built from `built_from` with the checkout as it is."""
    monkeypatch.setattr(_runtime, "_identity_cache", None)
    return _runtime._pair_problem(
        _bindings(
            __source_revision__=built_from,
            __frontend_digest__=frontend_digest,
        )
    )


def _refused(problem):
    """Return the description of a problem that refuses the bindings."""
    assert problem is not None and problem[0] is True, problem
    return problem[1]


def _warned(problem):
    """Return the description of a problem that only warns."""
    assert problem is not None and problem[0] is False, problem
    return problem[1]


@pytest.mark.skipif(shutil.which("git") is None, reason="git unavailable")
def test_a_checkout_warns_for_a_frontend_move_and_refuses_a_native_one(
    tmp_path, monkeypatch
):
    """Apply the rule of a development checkout to another revision.

    A change outside the frontend and the native sources is no problem. A
    frontend that moved while the native sources did not is used with a
    warning, committed or not. Native sources that moved refuse the
    bindings, committed or not.
    """
    package, native_source, built_from = _checkout(tmp_path, monkeypatch)

    assert _problem(monkeypatch, built_from) is None
    (tmp_path / "NOTES.md").write_text("notes\n")
    _git(tmp_path, "add", "NOTES.md")
    _git(tmp_path, "commit", "--quiet", "--message", "notes")
    assert _problem(monkeypatch, built_from) is None

    (package / "__init__.py").write_text("VERSION = 2\n")
    uncommitted = _warned(_problem(monkeypatch, built_from))
    _git(tmp_path, "commit", "--quiet", "--all", "--message", "frontend")
    committed = _warned(_problem(monkeypatch, built_from))
    for problem in (uncommitted, committed):
        assert f"built from revision {built_from}" in problem
        assert "frontend" in problem
        assert "native sources" in problem
        assert "rebuild the bindings" in problem

    native_source.write_text("// lowering 2\n")
    uncommitted = _refused(_problem(monkeypatch, built_from))
    _git(tmp_path, "commit", "--quiet", "--all", "--message", "native")
    committed = _refused(_problem(monkeypatch, built_from))
    for problem in (uncommitted, committed):
        assert f"built from revision {built_from}" in problem
        assert "native sources" in problem
        assert str(tmp_path) in problem
        assert "rebuild the bindings" in problem


@pytest.mark.skipif(shutil.which("git") is None, reason="git unavailable")
def test_a_revision_the_checkout_lacks_is_refused(tmp_path, monkeypatch):
    """Refuse bindings whose revision cannot be compared with the checkout."""
    _checkout(tmp_path, monkeypatch)

    problem = _refused(_problem(monkeypatch, "f" * 40))

    assert "does not have" in problem
    assert "f" * 40 in problem


@pytest.mark.skipif(shutil.which("git") is None, reason="git unavailable")
def test_a_build_of_a_modified_tree_compares_the_frontend_digest(
    tmp_path, monkeypatch
):
    """Compare only the frontend when the bindings name no exact revision.

    A build of a modified tree, or one that records no revision, cannot be
    diffed with the checkout. The frontend digest it recorded can be
    compared: a moved frontend warns. Without the digest nothing is
    compared.
    """
    package, _, built_from = _checkout(tmp_path, monkeypatch)
    digest = _runtime._frontend_build_digest(package)

    for revision in (f"{built_from}-dirty", "unknown"):
        assert _problem(monkeypatch, revision) is None
        assert _problem(monkeypatch, revision, digest) is None
        (package / "__init__.py").write_text("VERSION = 3\n")
        problem = _warned(_problem(monkeypatch, revision, digest))
        assert revision in problem
        assert "frontend" in problem
        (package / "__init__.py").write_text("VERSION = 1\n")


def test_installed_swage_compares_the_frontend_digest(tmp_path, monkeypatch):
    """Compare an installed frontend with the one the bindings were built by.

    Outside a checkout no git runs and no revision is compared: the frontend
    digest the bindings recorded is the record both sides share. Another
    digest refuses the bindings; bindings that record none are accepted.
    """
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
    digest = _runtime._frontend_build_digest(package)

    assert _problem(monkeypatch, "a" * 40) is None
    assert _problem(monkeypatch, "a" * 40, digest) is None
    problem = _refused(_problem(monkeypatch, "a" * 40, "0" * 64))
    assert str(package) in problem
    assert "a" * 40 in problem
    assert "built together" in problem


def test_a_refused_pair_raises_and_a_warned_one_is_used(monkeypatch):
    """Raise for a refused pair at every use, and warn once otherwise."""
    native = _bindings(__source_revision__="a" * 40)
    _importable(monkeypatch, native)
    monkeypatch.setattr(
        _runtime,
        "_pair_problem",
        lambda bindings: (True, "native sources differ; rebuild"),
    )
    for _ in range(2):
        with pytest.raises(_runtime._BindingsMismatch, match="differ"):
            _runtime._native_bindings()

    monkeypatch.setattr(
        _runtime,
        "_pair_problem",
        lambda bindings: (False, f"built from {bindings.__source_revision__}"),
    )
    with pytest.warns(RuntimeWarning, match="built from a+"):
        assert _runtime._native_bindings() is native
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _runtime._native_bindings() is native


@pytest.mark.skipif(shutil.which("cmake") is None, reason="cmake unavailable")
def test_the_build_records_the_frontend_digest_swage_computes(tmp_path):
    """Compute the frontend digest in the build and in `swage` alike.

    The build script hashes `python/swage` of the checkout it builds from,
    and `swage` hashes the package it runs from. Both skip names that start
    with a dot and files that are not Python sources, and both see a
    changed source.
    """
    package = tmp_path / "python" / "swage"
    (package / "nested").mkdir(parents=True)
    (package / ".hidden").mkdir()
    (package / "__init__.py").write_text('__version__ = "9.9.9"\n')
    (package / "nested" / "module.py").write_text("VALUE = 1\n")
    (package / ".hidden" / "skipped.py").write_text("SKIPPED = 1\n")
    (package / "notes.txt").write_text("not a source\n")
    header = tmp_path / "BuildIdentity.h"
    script = pathlib.Path(__file__).parents[2] / "cmake"
    script = script / "SwageBuildIdentity.cmake"

    def recorded():
        subprocess.run(
            [
                "cmake",
                f"-DSWAGE_SOURCE_DIR={tmp_path}",
                f"-DSWAGE_IDENTITY_HEADER={header}",
                "-P",
                str(script),
            ],
            check=True,
            capture_output=True,
        )
        (line,) = [
            line
            for line in header.read_text().splitlines()
            if line.startswith("#define SWAGE_BUILD_FRONTEND ")
        ]
        return line.split('"')[1]

    first = recorded()
    assert first == _runtime._frontend_build_digest(package)
    (package / ".hidden" / "skipped.py").write_text("SKIPPED = 2\n")
    assert recorded() == first
    (package / "nested" / "module.py").write_text("VALUE = 2\n")
    assert recorded() != first
    assert recorded() == _runtime._frontend_build_digest(package)


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


@pytest.mark.parametrize(
    "recorded", [None, "0" * 64], ids=["no-digest", "another-frontend"]
)
def test_the_compile_command_refuses_bindings_of_another_frontend(
    tmp_path, monkeypatch, recorded
):
    """Write no artifact whose manifest names a revision that is not its own.

    The manifest records the revision of the bindings, and the program texts
    come from the frontend. Bindings built beside another frontend, or
    bindings that cannot say which frontend they were built beside, are
    refused before anything is compiled or written.
    """
    from swage import compile

    monkeypatch.delenv("SWAGE_ARTIFACT_DIR", raising=False)
    output = tmp_path / "artifact"
    native = _bindings(
        __source_revision__="a" * 40, __frontend_digest__=recorded
    )
    monkeypatch.setattr(_runtime, "_native_bindings", lambda: native)

    with pytest.raises(RuntimeError, match="did not produce the kernels"):
        compile._write_artifact(output, "sm_86", ["sum"], None)
    assert list(tmp_path.iterdir()) == []

    # The frontend the bindings were built beside passes the check and
    # reaches the next requirement, the runtime library.
    native.__frontend_digest__ = _runtime._frontend_build_digest(
        _runtime._package_dir()
    )

    def runtime_library():
        raise RuntimeError("the runtime library is reached")

    monkeypatch.setattr(compile, "_packaged_runtime", runtime_library)
    with pytest.raises(RuntimeError, match="runtime library is reached"):
        compile._write_artifact(output, "sm_86", ["sum"], None)
