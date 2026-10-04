# python/tests/mlir/test_cache_process_reuse.py
"""Native tests that a second process reuses the first one's cache entry.

Each test runs the fixed vector-add compile in fresh interpreters with the
real native bindings. The real compiler identity (frontend digest, native
library metadata, and the process-start check) is therefore persisted by
one process and read back by another, which the host tests cannot show
because they stub the identity.

This file is also the program those interpreters run: executed as a script
it compiles or launches the kernel once per block size and prints a JSON
report. Every subprocess is given its own cache directory explicitly.
"""

import hashlib
import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
import warnings

import pytest
import swage as sw
import swage.language as sl

_ENTRY_FILES = ["kernel.ptx", "lowered.mlir", "metadata.json"]
_SIZE = 129


@sw.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n, BLOCK: sl.constexpr):  # noqa: D103
    pid = sl.program_id(0)
    offsets = pid * BLOCK + sl.arange(0, BLOCK)
    mask = offsets < n
    x = sl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = sl.load(y_ptr + offsets, mask=mask, other=0.0)
    sl.store(output_ptr + offsets, x + y, mask=mask)


def _compile_only(block):
    """Run the compile that `launch()` runs, without needing a device."""
    from swage import _runtime

    constexprs = {"BLOCK": block}
    signature = {
        "x_ptr": sl.pointer(sl.float32),
        "y_ptr": sl.pointer(sl.float32),
        "output_ptr": sl.pointer(sl.float32),
        "n": sl.int32,
    }
    specialization = _runtime._specialization_data(
        add_kernel,
        descriptors=("ptr<f32>", "ptr<f32>", "ptr<f32>", "i32"),
        constexprs=constexprs,
        target="sm_80",
    )
    return _runtime._compile_cached(
        specialization,
        add_kernel.__name__,
        block,
        lambda: add_kernel.emit_mlir(
            signature=signature, constexprs=constexprs
        ),
        key=_runtime._cache_key(specialization),
    )


def _launch(block):
    """Launch through the public boundary and check the result."""
    import torch
    from swage import _runtime

    x = torch.arange(_SIZE, device="cuda", dtype=torch.float32)
    y = 2 * x
    output = torch.full_like(x, -1.0)
    add_kernel.launch(
        arguments={"x_ptr": x, "y_ptr": y, "output_ptr": output, "n": _SIZE},
        constexprs={"BLOCK": block},
        grid=((_SIZE + block - 1) // block,),
    )
    torch.cuda.synchronize()
    assert torch.equal(output, x + y)
    return list(_runtime._ptx_cache.values())[-1]


def _main(mode, touch, blocks):
    """Compile or launch once per block size and print a JSON report."""
    from swage import _runtime

    compiles = []
    compile_native = _runtime._compile_native

    def counting_compile(*arguments):
        compiles.append(arguments[1])
        return compile_native(*arguments)

    _runtime._compile_native = counting_compile
    package = pathlib.Path(sw.__file__).parent
    if touch == "touch":
        # Same bytes, so the same key, but newer than this process.
        os.utime(package / "_frontend.py")
    run = _launch if mode == "launch" else _compile_only
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        artifacts = [run(int(block)) for block in blocks]
    identity = _runtime._cached_identity()
    cache = pathlib.Path(os.environ["SWAGE_CACHE_DIR"])
    report = {
        "package": str(package),
        "keys": [artifact.key for artifact in artifacts],
        "ptx": [
            hashlib.sha256(artifact.ptx.encode()).hexdigest()
            for artifact in artifacts
        ],
        "compiles": len(compiles),
        "entries": sorted(path.name for path in cache.glob("*")),
        "warnings": [
            str(warning.message)
            for warning in caught
            if "persistent cache" in str(warning.message)
        ],
        "frontend": identity["frontend"],
        "native": identity["native"],
    }
    print(json.dumps(report))


def _run(cache, *, mode="compile", touch=False, blocks=(128,), site=None):
    """Run this file as a program in a fresh interpreter.

    Args:
        cache: Cache directory the interpreter must use.
        mode: `compile` for the device-free path or `launch` for CUDA.
        touch: Whether to touch a frontend file after the process started.
        blocks: Block sizes to specialize, one compile or launch each.
        site: Directory holding the `swage` package to import, or None for
            the package this test process imported.

    Returns:
        The JSON report the interpreter printed.
    """
    package_parent = site or pathlib.Path(sw.__file__).parents[1]
    bindings = importlib.util.find_spec("mlir_swage._mlir_libs")
    bindings_parent = pathlib.Path(
        list(bindings.submodule_search_locations)[0]
    ).parents[1]
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("SWAGE_DUMP_")
    }
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(package_parent), str(bindings_parent)]
    )
    environment["SWAGE_CACHE_DIR"] = str(cache)
    completed = subprocess.run(
        [
            sys.executable,
            __file__,
            mode,
            "touch" if touch else "keep",
            *[str(block) for block in blocks],
        ],
        env=environment,
        cwd=cache.parent,
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout.splitlines()[-1])
    assert report["package"] == str(pathlib.Path(package_parent) / "swage")
    return report


def _assert_published_once_then_reused(cache, first, second):
    """Check one native compile and one entry, then a pure disk hit."""
    (key,) = first["keys"]
    assert first["compiles"] == 1
    assert first["entries"] == [key]
    assert first["warnings"] == []
    assert sorted(path.name for path in (cache / key).iterdir()) == (
        _ENTRY_FILES
    )

    assert second["compiles"] == 0
    assert second["keys"] == [key]
    assert second["ptx"] == first["ptx"]
    assert second["entries"] == [key]
    assert second["warnings"] == []

    # The key and the entry carry the real identity, in both processes.
    assert len(first["frontend"]) == 64
    assert first["native"]
    assert second["frontend"] == first["frontend"]
    assert second["native"] == first["native"]
    recorded = json.loads((cache / key / "metadata.json").read_text())
    assert recorded["key"] == key
    assert recorded["specialization"]["frontend"] == first["frontend"]
    assert recorded["specialization"]["native"] == first["native"]


def test_second_process_reads_the_entry_the_first_published(tmp_path):
    """Compile once in one process and zero times in the next."""
    cache = tmp_path / "cache"

    first = _run(cache)
    second = _run(cache)

    _assert_published_once_then_reused(cache, first, second)


def test_second_process_launch_reuses_the_published_kernel(tmp_path):
    """Launch on the device from the entry another process published."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    cache = tmp_path / "cache"

    first = _run(cache, mode="launch")
    second = _run(cache, mode="launch")

    _assert_published_once_then_reused(cache, first, second)


def test_frontend_touched_after_start_is_neither_read_nor_published(
    tmp_path,
):
    """Keep a process off the disk cache once its frontend looks newer."""
    site = tmp_path / "site"
    shutil.copytree(
        pathlib.Path(sw.__file__).parent,
        site / "swage",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    cache = tmp_path / "cache"
    # The process start time has a resolution of one clock tick (10 ms),
    # so let the copy, and later the touch, age before a process starts.
    time.sleep(0.05)
    warm = _run(cache, site=site)
    (key,) = warm["keys"]
    assert warm["compiles"] == 1
    assert warm["entries"] == [key]

    touched = _run(cache, site=site, touch=True, blocks=(128, 64))

    assert touched["keys"][0] == key
    assert touched["compiles"] == 2
    assert touched["entries"] == [key]
    assert len(touched["warnings"]) == 1
    assert (
        "_frontend.py is not older than this process"
        in (touched["warnings"][0])
    )

    time.sleep(0.05)
    later = _run(cache, site=site, blocks=(128, 64))

    assert later["keys"] == touched["keys"]
    assert later["compiles"] == 1
    assert later["entries"] == sorted(later["keys"])
    assert later["warnings"] == []


if __name__ == "__main__":
    _main(sys.argv[1], sys.argv[2], sys.argv[3:])
