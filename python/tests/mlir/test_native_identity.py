# python/tests/mlir/test_native_identity.py
"""Tests for the identity the built bindings carry and how it is checked.

The extension records the `swage` version and source revision it was built
from, and asks an imported `swage` to check them while it loads. The checks
that need a first import of the extension run in a fresh interpreter.
"""

import importlib.util
import os
import pathlib
import re
import shutil
import subprocess
import sys

import swage
from mlir_swage._mlir_libs._swageDialectsNanobind import swage as native_swage
from swage import _runtime

_REVISION = re.compile(r"unknown|[0-9a-f]{40}(-dirty)?")
_CONTENT = re.compile(r"build-id:([0-9a-f]{2})+|sha256:[0-9a-f]{64}")
_NATIVE = "mlir_swage._mlir_libs._swageDialectsNanobind"


def _fresh_interpreter(program, *, first_on_path=None, warnings="default"):
    """Run `program` where neither `swage` nor the bindings are loaded yet.

    Args:
        program: Source the interpreter runs.
        first_on_path: A directory to search before the `swage` and
            `mlir_swage` this process imported, or None.
        warnings: The warning filter of the interpreter.
    """
    bindings = importlib.util.find_spec("mlir_swage._mlir_libs")
    roots = [
        pathlib.Path(swage.__file__).parents[1],
        pathlib.Path(list(bindings.submodule_search_locations)[0]).parents[1],
    ]
    if first_on_path is not None:
        roots.insert(0, first_on_path)
    return subprocess.run(
        [sys.executable, "-W", warnings, "-c", program],
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(str(root) for root in roots),
        },
        cwd=pathlib.Path(sys.executable).parent,
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )


def test_extension_records_what_it_was_built_from():
    """Carry the swage version, a source revision, and the linked LLVM."""
    assert native_swage.__version__ == swage.__version__
    assert _REVISION.fullmatch(native_swage.__source_revision__)
    assert re.fullmatch(r"\d+\.\d+\.\d+", native_swage.__llvm_version__)


def test_matching_bindings_are_accepted():
    """Hand out the loaded bindings once they passed the check."""
    assert _runtime._native_bindings() is native_swage
    assert _runtime._verified_bindings is native_swage


def test_frontend_was_asked_to_check_while_the_extension_loaded():
    """Check the pair at the import, before any accessor of swage runs."""
    completed = _fresh_interpreter(
        "import swage\n"
        "from swage import _runtime\n"
        "assert _runtime._verified_bindings is None\n"
        f"import {_NATIVE} as extension\n"
        "assert _runtime._verified_bindings is extension.swage\n",
        warnings="error",
    )

    assert completed.returncode == 0, completed.stderr


def test_bindings_refuse_to_load_into_another_swage_version():
    """Fail the import, and every later one, with the reason."""
    completed = _fresh_interpreter(
        "import swage\n"
        "from swage import _runtime\n"
        "swage.__version__ = '0.0.1'\n"
        "for attempt in range(2):\n"
        "    try:\n"
        f"        import {_NATIVE}\n"
        "    except ImportError as error:\n"
        "        refusal = error.__cause__\n"
        "        assert isinstance(refusal, _runtime._BindingsMismatch)\n"
        "    else:\n"
        "        raise AssertionError('mismatched bindings were imported')\n"
        "try:\n"
        "    _runtime._native_bindings()\n"
        "except _runtime._BindingsMismatch as error:\n"
        "    print(error)\n"
    )

    assert completed.returncode == 0, completed.stderr
    assert f"were built for swage {swage.__version__}" in completed.stdout
    assert "but swage 0.0.1 is loaded from" in completed.stdout
    assert native_swage.__source_revision__ in completed.stdout


def test_bindings_loaded_before_swage_are_checked_at_their_first_use():
    """Refuse a mismatch that the import of the extension could not see."""
    completed = _fresh_interpreter(
        f"import {_NATIVE}\n"
        "import swage\n"
        "from swage import _runtime\n"
        "assert _runtime._verified_bindings is None\n"
        "swage.__version__ = '0.0.1'\n"
        "try:\n"
        "    _runtime._native_bindings()\n"
        "except _runtime._BindingsMismatch as error:\n"
        "    print(error)\n"
    )

    assert completed.returncode == 0, completed.stderr
    assert "but swage 0.0.1 is loaded from" in completed.stdout


def test_frontend_from_before_the_check_is_reported(tmp_path):
    """Warn when a frontend that cannot check the pair uses the bindings."""
    old = tmp_path / "swage"
    old.mkdir()
    (old / "__init__.py").write_text(f'__version__ = "{swage.__version__}"\n')
    program = f"import swage\nimport {_NATIVE}\n"

    reported = _fresh_interpreter(program, first_on_path=tmp_path)
    refused = _fresh_interpreter(
        program, first_on_path=tmp_path, warnings="error"
    )

    assert reported.returncode == 0, reported.stderr
    assert "RuntimeWarning: the swage package at " in reported.stderr
    assert str(old / "__init__.py") in reported.stderr
    assert "predates the check that pairs a frontend" in reported.stderr
    assert f"built for swage {swage.__version__}" in reported.stderr
    assert native_swage.__source_revision__ in reported.stderr
    assert refused.returncode != 0


def test_bindings_load_without_swage():
    """Stay usable on their own: nothing is checked without a frontend."""
    completed = _fresh_interpreter(
        "import sys\n"
        f"import {_NATIVE} as extension\n"
        "assert 'swage' not in sys.modules\n"
        "print(extension.swage.__version__)\n",
        warnings="error",
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == swage.__version__


def test_native_libraries_keep_their_identity_when_copied(tmp_path):
    """Identify the built libraries by content, not by file times."""
    libraries = sorted(
        {path.resolve() for path in _runtime._native_libraries()}
    )
    identity = _runtime._native_identity()

    assert [name for name, _ in identity] == [path.name for path in libraries]
    for library, (_, content) in zip(libraries, identity, strict=True):
        assert _CONTENT.fullmatch(content)
        copy = tmp_path / library.name
        shutil.copyfile(library, copy)
        os.utime(copy, ns=(10**18, 10**18))
        assert copy.stat().st_mtime_ns != library.stat().st_mtime_ns
        assert _runtime._content_identity(copy) == content
        assert content == (
            f"build-id:{_runtime._elf_build_id(library)}"
            if content.startswith("build-id:")
            else f"sha256:{_runtime._file_digest(library)}"
        )
