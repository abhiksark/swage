# tests/python/test_assemble_native_wheel.py
"""Tests for packing a staged `mlir_swage` package into a wheel.

The staged package is a small fake. Its libraries are a few bytes behind
the ELF magic number, and a stand-in `readelf` placed first on PATH answers
with the dynamic section recorded for each file name.
"""

import base64
import email
import hashlib
import importlib.util
import os
import shutil
import stat
import zipfile
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "assemble_native_wheel.py"
_SUFFIX = ".cpython-313-x86_64-linux-gnu.so"
_REVISION = "0123456789abcdef0123456789abcdef01234567"
_INFORMATION = "swage_compiler_native-1.2.3.dist-info"
_WHEEL = "swage_compiler_native-1.2.3-cp313-cp313-linux_x86_64.whl"
_SYSTEM = ("libc.so.6", "libstdc++.so.6", "libz.so.1", "libzstd.so.1")

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None, reason="bash runs the stand-in readelf"
)


def _load_assembler():
    spec = importlib.util.spec_from_file_location("assemble", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dynamic_section(needed, run_path):
    """Return what `readelf --dynamic --wide` prints for a library."""
    lines = ["Dynamic section at offset 0x1000 contains 9 entries:"]
    lines += [
        f" 0x0000000000000001 (NEEDED)             Shared library: [{name}]"
        for name in needed
    ]
    if run_path is not None:
        lines.append(
            " 0x000000000000001d (RUNPATH)            Library runpath: "
            f"[{run_path}]"
        )
    return "\n".join(lines) + "\n"


class _Stage:
    """A fake staged install and the stand-in tool that describes it."""

    def __init__(self, root, monkeypatch):
        self.root = root
        self.package = root / "python_packages" / "mlir_swage"
        self.libraries = self.package / "_mlir_libs"
        self.sections = root / "sections"
        self.sections.mkdir()
        tools = root / "tools"
        tools.mkdir()
        readelf = tools / "readelf"
        readelf.write_text(
            f"#!{shutil.which('bash')}\n"
            f'exec {shutil.which("cat")} "{self.sections}/${{3##*/}}"\n'
        )
        readelf.chmod(0o755)
        monkeypatch.setenv("PATH", str(tools))
        (root / "LICENSE").write_text("MIT License\n")
        (root / "NOTICES.md").write_text("# Third-party notices\n")

        (self.package / "dialects").mkdir(parents=True)
        self.libraries.mkdir()
        (self.package / "ir.py").write_text("IR = 1\n")
        (self.package / "dialects" / "swage.py").write_text("OPS = 1\n")
        (self.libraries / "__init__.py").write_text("")
        cache = self.package / "__pycache__"
        cache.mkdir()
        (cache / "ir.cpython-313.pyc").write_bytes(b"cached")
        (self.package / "stray.pyc").write_bytes(b"cached")
        compiler = "libSwagePythonCAPI.so.22.1"
        self.library(compiler, needed=_SYSTEM)
        (self.libraries / "libSwagePythonCAPI.so").symlink_to(compiler)
        for extension in ("_mlir", "_swageDialectsNanobind"):
            self.library(extension + _SUFFIX, needed=(compiler, "libc.so.6"))

    def library(self, name, *, needed=(), run_path="$ORIGIN"):
        """Write a fake shared library and its dynamic section."""
        path = self.libraries / name
        path.write_bytes(b"\x7fELF" + name.encode())
        path.chmod(0o755)
        (self.sections / name).write_text(_dynamic_section(needed, run_path))
        return path

    def assemble(self, assembler, output=None):
        """Pack the stage and return the result of `assemble`."""
        return assembler.assemble(
            self.package,
            version="1.2.3",
            revision=_REVISION,
            llvm_version="22.1.8",
            license_file=self.root / "LICENSE",
            notices=self.root / "NOTICES.md",
            output=output or self.root / "dist",
        )


@pytest.fixture
def stage(tmp_path, monkeypatch):
    """Return a fake staged install that packs without complaint."""
    return _Stage(tmp_path, monkeypatch)


def test_wheel_holds_the_package_the_licenses_and_its_metadata(stage):
    """Pack real files only, and describe what the bindings were built from."""
    wheel, tag, external = stage.assemble(_load_assembler())

    assert wheel.name == _WHEEL
    assert tag == "cp313-cp313-linux_x86_64"
    assert external == sorted(_SYSTEM)
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        metadata = email.message_from_bytes(
            archive.read(f"{_INFORMATION}/METADATA")
        )
        wheel_file = archive.read(f"{_INFORMATION}/WHEEL").decode()
        notices = archive.read(f"{_INFORMATION}/licenses/NOTICES.md")
        library = archive.getinfo(
            "mlir_swage/_mlir_libs/libSwagePythonCAPI.so.22.1"
        )
    assert sorted(names) == sorted(
        [
            "mlir_swage/_mlir_libs/__init__.py",
            f"mlir_swage/_mlir_libs/_mlir{_SUFFIX}",
            f"mlir_swage/_mlir_libs/_swageDialectsNanobind{_SUFFIX}",
            "mlir_swage/_mlir_libs/libSwagePythonCAPI.so.22.1",
            "mlir_swage/dialects/swage.py",
            "mlir_swage/ir.py",
            f"{_INFORMATION}/METADATA",
            f"{_INFORMATION}/WHEEL",
            f"{_INFORMATION}/licenses/LICENSE",
            f"{_INFORMATION}/licenses/NOTICES.md",
            f"{_INFORMATION}/RECORD",
        ]
    )
    assert names[-1] == f"{_INFORMATION}/RECORD"
    assert notices == b"# Third-party notices\n"
    assert stat.S_IMODE(library.external_attr >> 16) == 0o755
    assert metadata["Name"] == "swage-compiler-native"
    assert metadata["Version"] == "1.2.3"
    assert metadata["Requires-Dist"] == "swage-compiler==1.2.3"
    assert metadata["Requires-Python"] == "==3.13.*"
    assert metadata.get_all("License-File") == ["LICENSE", "NOTICES.md"]
    description = metadata.get_payload()
    assert f"Source revision: `{_REVISION}`" in description
    assert "Linked LLVM: 22.1.8" in description
    assert "`libzstd.so.1`" in description
    assert "manylinux" in description
    assert "Root-Is-Purelib: false\n" in wheel_file
    assert "Tag: cp313-cp313-linux_x86_64\n" in wheel_file


def test_record_states_the_digest_and_size_of_every_member(stage):
    """Let an installer verify each file against the record."""
    wheel, _, _ = stage.assemble(_load_assembler())

    with zipfile.ZipFile(wheel) as archive:
        record = archive.read(f"{_INFORMATION}/RECORD").decode().splitlines()
        recorded = dict(line.split(",", 1) for line in record)
        for name in archive.namelist():
            contents = archive.read(name)
            if name.endswith("/RECORD"):
                assert recorded[name] == ","
                continue
            digest = base64.urlsafe_b64encode(
                hashlib.sha256(contents).digest()
            ).rstrip(b"=")
            assert recorded[name] == (
                f"sha256={digest.decode()},{len(contents)}"
            )
    assert len(recorded) == len(record) == 11


def test_wheel_bytes_depend_only_on_the_staged_files(stage):
    """Produce the same archive when the same stage is packed again."""
    assembler = _load_assembler()
    first, _, _ = stage.assemble(assembler, stage.root / "first")
    for path in stage.package.rglob("*"):
        if not path.is_symlink():
            os.utime(path, ns=(10**18, 10**18))
    second, _, _ = stage.assemble(assembler, stage.root / "second")

    assert first.read_bytes() == second.read_bytes()


def _absolute_link(stage):
    (stage.package / "rewrite.py").symlink_to(stage.root / "LICENSE")


def _link_out_of_the_package(stage):
    (stage.package / "rewrite.py").symlink_to("../../LICENSE")


def _build_tree_run_path(stage):
    stage.library(
        "libnanobind-mlir_swage.so", run_path="$ORIGIN:/home/build/lib"
    )


def _unexpected_dependency(stage):
    stage.library("libnanobind-mlir_swage.so", needed=("libxml2.so.2",))


def _missing_extension(stage):
    (stage.libraries / f"_mlir{_SUFFIX}").unlink()


def _extensions_of_two_interpreters(stage):
    (stage.libraries / f"_mlir{_SUFFIX}").rename(
        stage.libraries / "_mlir.cpython-310-x86_64-linux-gnu.so"
    )


def _unreadable_library(stage):
    (stage.sections / "libSwagePythonCAPI.so.22.1").unlink()


@pytest.mark.parametrize(
    ("damage", "reason"),
    [
        (_absolute_link, "absolute or outside the package"),
        (_link_out_of_the_package, "absolute or outside the package"),
        (_build_tree_run_path, "searches /home/build/lib"),
        (_unexpected_dependency, "needs libxml2.so.2"),
        (_missing_extension, "expected one _mlir.cpython-"),
        (_extensions_of_two_interpreters, "different interpreters"),
        (_unreadable_library, "readelf could not read"),
    ],
)
def test_package_that_would_not_relocate_is_refused(stage, damage, reason):
    """Write no wheel for a package tied to the machine that built it."""
    assembler = _load_assembler()
    damage(stage)

    with pytest.raises(assembler.PackagingError, match=reason):
        stage.assemble(assembler)

    assert not (stage.root / "dist").exists()


def test_run_path_below_the_library_is_accepted(stage):
    """Allow a library to search a directory next to itself."""
    stage.library("libnanobind-mlir_swage.so", run_path="$ORIGIN/support")

    wheel, _, _ = stage.assemble(_load_assembler())

    with zipfile.ZipFile(wheel) as archive:
        assert "mlir_swage/_mlir_libs/libnanobind-mlir_swage.so" in (
            archive.namelist()
        )


def test_command_reports_the_wheel_or_the_refusal(stage, capsys):
    """Exit with a status, and name the wheel, its size, and its digest."""
    assembler = _load_assembler()
    arguments = [
        "--package",
        str(stage.package),
        "--version",
        "1.2.3",
        "--revision",
        _REVISION,
        "--llvm-version",
        "22.1.8",
        "--license",
        str(stage.root / "LICENSE"),
        "--notices",
        str(stage.root / "NOTICES.md"),
        "--output",
        str(stage.root / "dist"),
    ]

    assert assembler.main(arguments) == 0
    wheel = stage.root / "dist" / _WHEEL
    report = capsys.readouterr().out
    assert f"wheel: {wheel}" in report
    assert f"size: {wheel.stat().st_size} bytes" in report
    assert hashlib.sha256(wheel.read_bytes()).hexdigest() in report
    assert "needs from the system: libc.so.6, libstdc++.so.6" in report

    arguments[1] = str(stage.package / "dialects")
    assert assembler.main(arguments) == 1
    assert "error: " in capsys.readouterr().err
