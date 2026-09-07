# tests/python/test_native_wheel.py
"""Artifact mutations exercise release gates with real ELF bytes, not LLVM."""

import csv
import hashlib
import importlib.util
import io
import json
import struct
import sys
import tarfile
import zipfile
from base64 import urlsafe_b64encode
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_REVISION = "a" * 40
_PYTHON = f"cp{sys.version_info.major}{sys.version_info.minor}"
_PLATFORM = "manylinux_2_28_x86_64"
_TAG = f"{_PYTHON}-{_PYTHON}-{_PLATFORM}"
_DIST = "swage_compiler-0.5.2.dist-info"
_LIBDIR = "mlir_swage/_mlir_libs"
_EXTENSION = f"{_LIBDIR}/_mlir.cpython-{_PYTHON[2:]}-x86_64-linux-gnu.so"
_SWAGE_EXTENSION = (
    f"{_LIBDIR}/_swageDialectsNanobind."
    f"cpython-{_PYTHON[2:]}-x86_64-linux-gnu.so"
)
_CAPI = f"{_LIBDIR}/libSwagePythonCAPI.so.22.1"
_SUPPORT = f"{_LIBDIR}/libMLIRPythonSupport-mlir_swage.so"
_NANOBIND = f"{_LIBDIR}/libnanobind-mlir_swage.so"
_INFO = {
    "schema_version": 1,
    "package_version": "0.5.2",
    "source_revision": _REVISION,
    "source_clean": True,
    "llvm_version": "llvmorg-22.1.8",
    "build_type": "Release",
}


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name,
        _ROOT / "scripts" / f"{name}.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _elf(*, needed=(), paths=((29, "$ORIGIN"),), machine=62, trailer=b""):
    # Minimal ELF64 ET_DYN with a LOAD segment, dynamic segment, dynamic symbol
    # table and matching sections. No host compiler or architecture is needed.
    strings = bytearray(b"\0")

    def string(value):
        offset = len(strings)
        strings.extend(value.encode() + b"\0")
        return offset

    symbol = string("PyInit__mlir")
    dynamic = [(1, string(value)) for value in needed]
    dynamic += [(kind, string(value)) for kind, value in paths]
    str_offset = 64 + 2 * 56
    dynamic_offset = (str_offset + len(strings) + 7) & ~7
    dynamic += [(5, str_offset), (10, len(strings)), (0, 0)]
    dynamic_data = b"".join(struct.pack("<qQ", *entry) for entry in dynamic)
    symbol_offset = dynamic_offset + len(dynamic_data)
    symbol_data = bytes(24) + struct.pack("<IBBHQQ", symbol, 18, 0, 1, 0, 0)
    section_strings = b"\0.dynstr\0.dynamic\0.dynsym\0.shstrtab\0"
    names_offset = symbol_offset + len(symbol_data)
    section_offset = (names_offset + len(section_strings) + 7) & ~7
    size = section_offset + 5 * 64
    ident = b"\x7fELF\x02\x01\x01" + bytes(9)
    header = ident + struct.pack(
        "<HHIQQQIHHHHHH",
        3,
        machine,
        1,
        0,
        64,
        section_offset,
        0,
        64,
        56,
        2,
        64,
        5,
        4,
    )
    segments = struct.pack("<IIQQQQQQ", 1, 4, 0, 0, 0, size, size, 4096)
    segments += struct.pack(
        "<IIQQQQQQ",
        2,
        4,
        dynamic_offset,
        dynamic_offset,
        dynamic_offset,
        len(dynamic_data),
        len(dynamic_data),
        8,
    )
    data = bytearray(header + segments)
    data.extend(strings)
    data.extend(bytes(dynamic_offset - len(data)))
    data.extend(dynamic_data + symbol_data + section_strings)
    data.extend(bytes(section_offset - len(data)))
    sections = [
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        (1, 3, 2, str_offset, str_offset, len(strings), 0, 0, 1, 0),
        (
            9,
            6,
            2,
            dynamic_offset,
            dynamic_offset,
            len(dynamic_data),
            1,
            0,
            8,
            16,
        ),
        (
            18,
            11,
            2,
            symbol_offset,
            symbol_offset,
            len(symbol_data),
            1,
            1,
            8,
            24,
        ),
        (26, 3, 0, 0, names_offset, len(section_strings), 0, 0, 1, 0),
    ]
    for section in sections:
        data.extend(struct.pack("<IIQQQQIIQQ", *section))
    return bytes(data) + trailer


def _wheel(
    tmp_path,
    *,
    edits=None,
    remove=(),
    tag=_TAG,
    info=None,
    corrupt_record=False,
):
    metadata = (
        "Metadata-Version: 2.4\nName: swage-compiler\nVersion: 0.5.2\n"
        "Requires-Python: >=3.10,<3.14\n"
        "License-Expression: MIT AND Apache-2.0 WITH LLVM-exception\n"
        "License-File: LICENSE\nLicense-File: LICENSES/LLVM.txt\n\n"
    )
    members = {
        "swage/__init__.py": b'__version__ = "0.5.2"\n',
        "swage/language.py": b"",
        "swage/py.typed": b"",
        "swage/__init__.pyi": b"def jit(f): ...\n",
        "swage/language.pyi": b"constexpr: object\n",
        "mlir_swage/ir.py": b"",
        "mlir_swage/dialects/swage.py": b"",
        f"{_LIBDIR}/__init__.py": b"",
        "mlir_swage/_build_info.json": json.dumps(
            _INFO if info is None else info,
        ).encode(),
        _EXTENSION: _elf(needed=(Path(_CAPI).name,)),
        _SWAGE_EXTENSION: _elf(needed=(Path(_SUPPORT).name,)),
        _CAPI: _elf(),
        _SUPPORT: _elf(),
        _NANOBIND: _elf(),
        f"{_DIST}/METADATA": metadata.encode(),
        f"{_DIST}/WHEEL": (
            f"Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: {_TAG}\n"
        ).encode(),
        f"{_DIST}/licenses/LICENSE": (_ROOT / "LICENSE").read_bytes(),
        f"{_DIST}/licenses/LICENSES/LLVM.txt": (
            _ROOT / "LICENSES/LLVM.txt"
        ).read_bytes(),
    }
    members.update(edits or {})
    for name in remove:
        del members[name]
    record = io.StringIO(newline="")
    writer = csv.writer(record, lineterminator="\n")
    for name, data in sorted(members.items()):
        digest = urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
        writer.writerow([name, "sha256=" + digest.decode(), len(data)])
    writer.writerow([f"{_DIST}/RECORD", "", ""])
    members[f"{_DIST}/RECORD"] = record.getvalue().encode()
    if corrupt_record:
        members["swage/language.py"] += b"# modified after RECORD generation\n"
    output = tmp_path / f"swage_compiler-0.5.2-{tag}.whl"
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for name, data in sorted(members.items()):
            archive.writestr(zipfile.ZipInfo(name, (2024, 1, 1, 0, 0, 0)), data)
    return output


@pytest.fixture
def checker(monkeypatch):
    """Keep ELF/ZIP checks real while isolating external policy reporting."""
    module = _load("check_native_wheel")
    monkeypatch.setattr(module, "_auditwheel_show", lambda path: _PLATFORM)
    return module


def test_checked_artifact_identity(checker, tmp_path):
    """Release evidence identifies the exact checked bytes and native ABI."""
    wheel = _wheel(tmp_path)
    result = checker.check_wheel(wheel, expected_revision=_REVISION)
    assert result["sha256"] == hashlib.sha256(wheel.read_bytes()).hexdigest()
    assert result["size"] == wheel.stat().st_size
    assert result["python"] == _PYTHON
    assert result["build_info"]["source_revision"] == _REVISION


def test_real_auditwheel_analysis_accepts_self_contained_elf(tmp_path):
    """Run the real policy analyzer on compiler-free, genuine ELF fixtures."""
    module = _load("check_native_wheel")
    result = module.check_wheel(_wheel(tmp_path), expected_revision=_REVISION)
    assert result["python"] == _PYTHON


@pytest.mark.parametrize(
    "tag,diagnostic",
    [
        (f"{_PYTHON}-{_PYTHON}-linux_x86_64", "platform must be exactly"),
        (
            f"{_PYTHON}-{_PYTHON}-manylinux_2_34_x86_64",
            "platform must be exactly",
        ),
        (f"{_PYTHON}-{_PYTHON}-{_PLATFORM}.manylinux2014_x86_64", "one exact"),
        (f"{_PYTHON}-abi3-{_PLATFORM}", "regular CPython"),
        (f"cp313-cp313t-{_PLATFORM}", "regular CPython"),
        (f"pp310-pypy310_pp73-{_PLATFORM}", "regular CPython"),
        (f"cp314-cp314-{_PLATFORM}", "regular CPython"),
    ],
)
def test_filename_rejects_unsupported_contract(
    checker, tmp_path, tag, diagnostic
):
    """Renaming native bytes cannot broaden the supported platform or ABI."""
    with pytest.raises(ValueError, match=diagnostic):
        checker.check_wheel(
            _wheel(tmp_path, tag=tag), expected_revision=_REVISION
        )


def test_internal_tag_and_purelib_must_agree(checker, tmp_path):
    """A supported filename cannot hide contradictory installer metadata."""
    for header, diagnostic in (
        ("Tag: py3-none-any", "internal WHEEL tags"),
        (f"Tag: {_TAG}\nRoot-Is-Purelib: true", "exactly one Root-Is-Purelib"),
    ):
        wheel = _wheel(
            tmp_path,
            edits={
                f"{_DIST}/WHEEL": (
                    f"Wheel-Version: 1.0\nRoot-Is-Purelib: false\n{header}\n"
                ).encode()
            },
        )
        with pytest.raises(ValueError, match=diagnostic):
            checker.check_wheel(wheel, expected_revision=_REVISION)


@pytest.mark.parametrize(
    "payload",
    [
        b"{not-json}",
        b"null",
        b'{"schema_version": 1}',
        json.dumps({**_INFO, "unexpected": "value"}).encode(),
        (json.dumps(_INFO)[:-1] + ', "source_clean": true}').encode(),
    ],
)
def test_malformed_build_identity_fails(checker, tmp_path, payload):
    """Malformed or ambiguous provenance can never qualify a native artifact."""
    wheel = _wheel(tmp_path, edits={"mlir_swage/_build_info.json": payload})
    with pytest.raises(ValueError, match="build info"):
        checker.check_wheel(wheel, expected_revision=_REVISION)


@pytest.mark.parametrize(
    "header,value,diagnostic",
    [
        ("Name", "another-project", "project Name"),
        ("Version", "0.5.1", "Version"),
        ("Requires-Python", ">=3.10", "Requires-Python"),
    ],
)
def test_distribution_metadata_matches_release(
    checker, tmp_path, header, value, diagnostic
):
    """Installer metadata must match the release, not just the ZIP filename."""
    wheel = _wheel(tmp_path)
    with zipfile.ZipFile(wheel) as archive:
        metadata = archive.read(f"{_DIST}/METADATA").decode()
    lines = metadata.splitlines()
    metadata = "\n".join(
        f"{header}: {value}" if line.startswith(f"{header}:") else line
        for line in lines
    )
    wheel = _wheel(tmp_path, edits={f"{_DIST}/METADATA": metadata.encode()})
    with pytest.raises(ValueError, match=diagnostic):
        checker.check_wheel(wheel, expected_revision=_REVISION)


@pytest.mark.parametrize(
    "field,value,diagnostic",
    [
        ("source_revision", "b" * 40, "expected revision"),
        ("source_revision", "A" * 40, "lowercase hex"),
        ("source_clean", 1, "must be a boolean"),
        ("schema_version", True, "integer 1"),
        ("llvm_version", "llvmorg-22.1.7", "llvm_version"),
        ("build_type", "Debug", "build_type"),
        ("package_version", "0.5.1", "package_version"),
    ],
)
def test_provenance_contract(checker, tmp_path, field, value, diagnostic):
    """Reject contradictory provenance, never fabricate build identity."""
    wheel = _wheel(tmp_path, info={**_INFO, field: value})
    with pytest.raises(ValueError, match=diagnostic):
        checker.check_wheel(wheel, expected_revision=_REVISION)


def test_dirty_artifacts_require_explicit_local_opt_in(checker, tmp_path):
    """Reject dirty official artifacts unless local checking is requested."""
    wheel = _wheel(tmp_path, info={**_INFO, "source_clean": False})
    with pytest.raises(ValueError, match="source_clean=true"):
        checker.check_wheel(wheel, expected_revision=_REVISION)
    result = checker.check_wheel(
        wheel,
        expected_revision=_REVISION,
        allow_dirty=True,
    )
    assert result["build_info"]["source_clean"] is False


@pytest.mark.parametrize(
    "member",
    [
        "swage/__init__.py",
        "mlir_swage/ir.py",
        _EXTENSION,
        _SWAGE_EXTENSION,
        _CAPI,
        _SUPPORT,
        _NANOBIND,
        "swage/py.typed",
        "swage/language.pyi",
        f"{_DIST}/licenses/LICENSE",
        f"{_DIST}/licenses/LICENSES/LLVM.txt",
    ],
)
def test_missing_release_capability_fails(checker, tmp_path, member):
    """Removing a binding, runtime, public typing or license fails the gate."""
    wheel = _wheel(tmp_path, remove=[member])
    with pytest.raises(ValueError, match="missing"):
        checker.check_wheel(wheel, expected_revision=_REVISION)


@pytest.mark.parametrize(
    "member,diagnostic",
    [
        ("swage/_segmented_plan.py", "segmented Python"),
        ("swage/__pycache__/language.pyc", "bytecode"),
    ],
)
def test_unshipped_python_payload_fails(checker, tmp_path, member, diagnostic):
    """Research entry points and bytecode never enter release wheels."""
    wheel = _wheel(tmp_path, edits={member: b"private"})
    with pytest.raises(ValueError, match=diagnostic):
        checker.check_wheel(wheel, expected_revision=_REVISION)


@pytest.mark.parametrize(
    "kind,path",
    [
        (15, "/tmp/llvm/lib"),
        (29, "relative/lib"),
        (29, ""),
        (29, "$ORIGIN:$LIB"),
        (15, "${ORIGIN}/$PLATFORM"),
        (29, "$ORIGIN/../../../host"),
    ],
)
def test_loader_paths_are_relocatable(checker, tmp_path, kind, path):
    """Every dynamic section rejects absolute, ambient or escaping paths."""
    wheel = _wheel(tmp_path, edits={_NANOBIND: _elf(paths=((kind, path),))})
    with pytest.raises(ValueError, match="RPATH/RUNPATH"):
        checker.check_wheel(wheel, expected_revision=_REVISION)


def test_origin_relative_dependency_resolution(checker, tmp_path):
    """Sibling bundled libraries resolve through the ELF's own origin paths."""
    wheel = _wheel(
        tmp_path,
        edits={
            _NANOBIND: _elf(
                needed=("libextra.so.1",), paths=((29, "${ORIGIN}/../extra"),)
            ),
            "mlir_swage/extra/libextra.so.1": _elf(),
        },
    )
    assert (
        checker.check_wheel(wheel, expected_revision=_REVISION)["python"]
        == _PYTHON
    )


def test_system_loader_does_not_require_bundling(checker, tmp_path):
    """The glibc x86_64 loader is a system dependency, not a wheel graft."""
    wheel = _wheel(
        tmp_path,
        edits={_EXTENSION: _elf(needed=("ld-linux-x86-64.so.2",))},
    )
    assert checker.main([str(wheel), "--expected-revision", _REVISION]) == 0


@pytest.mark.parametrize(
    "library,diagnostic",
    [
        ("libcuda.so.1", "forbidden libcuda DT_NEEDED"),
        ("libcuda-deadbeef.so.1", "forbidden libcuda DT_NEEDED"),
        ("libmissing.so.1", "unresolved non-system dependency"),
        ("/usr/lib/libextra.so.1", "non-relocatable DT_NEEDED"),
    ],
)
def test_dynamic_dependency_policy(checker, tmp_path, library, diagnostic):
    """Driver grafts and unresolved non-system libraries cannot be published."""
    wheel = _wheel(tmp_path, edits={_NANOBIND: _elf(needed=(library,))})
    with pytest.raises(ValueError, match=diagnostic):
        checker.check_wheel(wheel, expected_revision=_REVISION)


@pytest.mark.parametrize(
    "prefix", [b"/project/swage", b"/tmp/swage-wheel-build"]
)
def test_both_original_build_roots_are_absent(checker, tmp_path, prefix):
    """Reject leaked roots anywhere in the binary, not just dynamic strings."""
    wheel = _wheel(
        tmp_path, edits={_CAPI: _elf(trailer=prefix + b"/lib/file.cpp")}
    )
    with pytest.raises(ValueError, match="forbidden build-root prefix"):
        checker.check_wheel(
            wheel,
            expected_revision=_REVISION,
            forbidden_prefixes=("/project/swage", "/tmp/swage-wheel-build"),
        )


def test_all_elf_files_need_x86_64_headers(checker, tmp_path):
    """An incorrectly tagged ELF cannot hide in a non-library-named member."""
    wheel = _wheel(tmp_path, edits={"mlir_swage/extra.bin": _elf(machine=183)})
    with pytest.raises(ValueError, match="ELF x86_64"):
        checker.check_wheel(wheel, expected_revision=_REVISION)


def test_payload_integrity_and_size_boundary(checker, tmp_path):
    """An edited payload or wheel reaching the publish ceiling is rejected."""
    wheel = _wheel(tmp_path, corrupt_record=True)
    with pytest.raises(ValueError, match="RECORD digest"):
        checker.check_wheel(wheel, expected_revision=_REVISION)
    wheel = _wheel(tmp_path)
    with wheel.open("ab") as output:
        output.truncate(checker.MAX_WHEEL_SIZE)
    with pytest.raises(ValueError, match="must be below"):
        checker.check_wheel(wheel, expected_revision=_REVISION)


@pytest.mark.parametrize(
    "report,returncode,diagnostic",
    [
        (
            "is consistent with the following platform tag: "
            '"manylinux_2_35_x86_64".',
            0,
            "policy is not manylinux",
        ),
        (
            'is consistent with the following platform tag: "linux_x86_64".',
            0,
            "policy is not manylinux",
        ),
        (
            "input filename has manylinux_2_28_x86_64",
            0,
            "policy is not manylinux",
        ),
        ("dependency analysis failed", 1, "auditwheel show failed"),
    ],
)
def test_auditwheel_policy_fails_closed(
    monkeypatch, tmp_path, report, returncode, diagnostic
):
    """Successful execution does not prove compatible native symbols."""
    module = _load("check_native_wheel")
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            returncode=returncode,
            stdout=report,
            stderr="",
        ),
    )
    with pytest.raises(ValueError, match=diagnostic):
        module.check_wheel(_wheel(tmp_path), expected_revision=_REVISION)


def test_checker_cli_json_and_failure_status(checker, tmp_path, capsys):
    """The CLI emits machine-readable evidence or a nonzero diagnostic."""
    wheel = _wheel(tmp_path)
    assert checker.main([str(wheel), "--expected-revision", _REVISION]) == 0
    assert json.loads(capsys.readouterr().out)["filename"] == wheel.name
    assert checker.main([str(wheel), "--expected-revision", "b" * 40]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "expected revision" in captured.err


def _sdist(tmp_path, *, extra=None):
    path = tmp_path / "swage_compiler-0.5.2.tar.gz"
    members = {
        "swage_compiler-0.5.2/pyproject.toml": (
            '[build-system]\nrequires = ["scikit-build-core==1.0.3"]\n'
            'build-backend = "scikit_build_core.build"\n'
        ).encode()
    }
    members.update(extra or {})
    with tarfile.open(path, "w:gz") as archive:
        for name, data in members.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    return path


@pytest.fixture
def repair(checker, monkeypatch):
    """Share the real checker with a separately loaded repair CLI."""
    monkeypatch.setitem(sys.modules, "check_native_wheel", checker)
    return _load("repair_native_wheel")


def test_failed_repair_never_promotes_an_artifact(
    repair, monkeypatch, tmp_path
):
    """Auditwheel command failures propagate with no unchecked wheel output."""

    def fail(command, **kwargs):
        raise ValueError("auditwheel: too-recent versioned symbols")

    monkeypatch.setattr(repair, "_run", fail)
    with pytest.raises(ValueError, match="too-recent versioned symbols"):
        repair.repair_wheel(
            _wheel(tmp_path),
            wheel_dir=tmp_path / "release",
            expected_revision=_REVISION,
        )
    assert not (tmp_path / "release").exists()


def _simulate_builds(repair, monkeypatch, *, leak=None, mismatch=False):
    def run(command, *, cwd=None, env=None):
        if command[2:4] == ["auditwheel", "repair"]:
            original = Path(command[4])
            output = Path(command[command.index("--wheel-dir") + 1])
            (output / original.name).write_bytes(original.read_bytes())
            return
        assert command[2:4] == ["build", "--wheel"]
        settings = dict(
            value[2:].split("=", 1)
            for value in command
            if value.startswith("-C")
        )
        build_dir = Path(settings["build-dir"])
        build_dir.mkdir()
        (build_dir / "CMakeCache.txt").write_text(
            f"CMAKE_HOME_DIRECTORY:INTERNAL={cwd}\n"
            f"CMAKE_CACHEFILE_DIR:INTERNAL={build_dir}\n",
            encoding="utf-8",
        )
        output = Path(command[command.index("--outdir") + 1])
        edits = {}
        if leak:
            root = str(cwd if leak == "source" else build_dir).encode()
            edits[_CAPI] = _elf(trailer=root + b"/compiler.cpp")
        if mismatch:
            edits["swage/language.py"] = b"# different build output\n"
        _wheel(output, edits=edits)

    monkeypatch.setattr(repair, "_run", run)
    monkeypatch.setattr(repair, "version", lambda name: "1.0.3")
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1704067200")


def test_reproducibility_checks_and_promotes_identical_bytes(
    repair, monkeypatch, tmp_path
):
    """A requested second build must be checked and match the published hash."""
    _simulate_builds(repair, monkeypatch)
    result = repair.repair_wheel(
        _wheel(tmp_path),
        wheel_dir=tmp_path / "release",
        expected_revision=_REVISION,
        rebuild_sdist=_sdist(tmp_path),
    )
    published = tmp_path / "release" / result["filename"]
    digest = hashlib.sha256(published.read_bytes()).hexdigest()
    assert result["sha256"] == result["reproducibility"]["sha256"] == digest


@pytest.mark.parametrize("leak", ["source", "build"])
def test_rebuild_rejects_each_actual_fresh_root(
    repair, monkeypatch, tmp_path, leak
):
    """Fresh second-build roots join first-build prefixes in the binary scan."""
    _simulate_builds(repair, monkeypatch, leak=leak)
    with pytest.raises(ValueError, match="forbidden build-root prefix"):
        repair.repair_wheel(
            _wheel(tmp_path),
            wheel_dir=tmp_path / "release",
            expected_revision=_REVISION,
            rebuild_sdist=_sdist(tmp_path),
        )
    assert not (tmp_path / "release").exists()


def test_reproducibility_mismatch_never_promotes_first_wheel(
    repair, monkeypatch, tmp_path
):
    """A valid but byte-different second artifact blocks release."""
    _simulate_builds(repair, monkeypatch, mismatch=True)
    with pytest.raises(ValueError, match="not byte-for-byte reproducible"):
        repair.repair_wheel(
            _wheel(tmp_path),
            wheel_dir=tmp_path / "release",
            expected_revision=_REVISION,
            rebuild_sdist=_sdist(tmp_path),
        )
    assert not (tmp_path / "release").exists()


def test_missing_epoch_never_skips_requested_rebuild(
    repair, monkeypatch, tmp_path
):
    """A missing reproducibility prerequisite fails, not skips the rebuild."""
    _simulate_builds(repair, monkeypatch)
    monkeypatch.delenv("SOURCE_DATE_EPOCH")
    with pytest.raises(ValueError, match="SOURCE_DATE_EPOCH"):
        repair.repair_wheel(
            _wheel(tmp_path),
            wheel_dir=tmp_path / "release",
            expected_revision=_REVISION,
            rebuild_sdist=_sdist(tmp_path),
        )
    assert not (tmp_path / "release").exists()


@pytest.mark.parametrize(
    "member", ["../escape", "/absolute", "source/../../escape"]
)
def test_sdist_traversal_never_extracts(repair, tmp_path, member):
    """Validate the entire archive before writing any untrusted member."""
    sdist = _sdist(tmp_path, extra={member: b"escape"})
    with pytest.raises(ValueError, match="unsafe sdist member"):
        repair._extract_sdist(sdist, tmp_path / "source")
    assert list((tmp_path / "source").iterdir()) == []


@pytest.mark.parametrize(
    "kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.CHRTYPE]
)
def test_sdist_links_and_devices_never_extract(repair, tmp_path, kind):
    """Fresh rebuild directories cannot be redirected through archive links."""
    sdist = tmp_path / "unsafe.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        member = tarfile.TarInfo("source/link")
        member.type = kind
        member.linkname = "../../escape"
        archive.addfile(member)
    with pytest.raises(ValueError, match="unsafe sdist member"):
        repair._extract_sdist(sdist, tmp_path / "source")
    assert list((tmp_path / "source").iterdir()) == []
