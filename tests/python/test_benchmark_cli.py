# tests/python/test_benchmark_cli.py
"""Fixed-only CLI parsing, dependency-free help, and evidence exit behavior."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from swage import _benchmark, bench


@pytest.mark.parametrize(
    "argv",
    (
        [],
        ["unknown"],
        ["segmented-sum"],
        ["vector-multiply"],
        ["vector-add"],
        ["vector-add", "--output", "unused.json", "--suite", "segmented-sum"],
        ["--cold-child"],
    ),
)
def test_rejected_commands_never_measure(argv, monkeypatch, capsys):
    """Invalid selectors and missing output fail at the argparse boundary."""

    def unexpected(**kwargs):
        pytest.fail("invalid command reached measurement")

    monkeypatch.setattr(_benchmark, "run_fixed_vector_add", unexpected)
    with pytest.raises(SystemExit) as caught:
        bench.main(argv)
    assert caught.value.code == 2
    assert "usage:" in capsys.readouterr().err


@pytest.mark.parametrize("argv", (["--help"], ["vector-add", "--help"]))
def test_help_without_optional_runtime_imports(argv):
    """Both help pages and the core import work with runtime imports blocked."""
    code = """
import builtins
import runpy
import sys

original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'mlir_swage'):
        raise AssertionError('optional runtime imported: ' + name)
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
from swage import _benchmark
sys.argv = ['swage.bench', *sys.argv[1:]]
runpy.run_module('swage.bench', run_name='__main__')
"""
    process = subprocess.run(
        [sys.executable, "-c", code, *argv],
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    assert "vector-add" in process.stdout
    assert "segmented" not in process.stdout
    assert "--cold-child" not in process.stdout
    assert not process.stderr


@pytest.mark.parametrize("enforce", (False, True))
def test_vector_add_retains_failed_evidence(
    enforce, monkeypatch, tmp_path, capsys
):
    """The selected command writes partial schema-v1 evidence on failure."""

    def fail(record, enforce):
        record["measurements"]["warm_host_us"].append(12.5)
        raise ValueError("fixed vector-add correctness mismatch")

    monkeypatch.setattr(_benchmark, "_measure", fail)
    output = tmp_path / "nested" / "evidence.json"
    argv = ["vector-add", "--output", str(output)]
    if enforce:
        argv.append("--enforce")
    assert bench.main(argv) == 1
    record = json.loads(output.read_text())
    assert record["schema_version"] == 1
    assert record["benchmark"] == "fixed-runtime"
    assert record["enforced"] is enforce
    assert record["measurements"]["warm_host_us"] == [12.5]
    assert not record["valid"] and not record["passed"]
    summary = {"output": str(output), "valid": False, "passed": False}
    assert capsys.readouterr().out == json.dumps(summary, sort_keys=True) + "\n"


def test_cold_child_needs_no_output(monkeypatch, capsys):
    """The internal fresh-process command emits only its raw cold sample."""
    sample = {
        "elapsed_ms": 12.5,
        "rss_before_bytes": 100,
        "rss_after_bytes": 200,
        "rss_delta_bytes": 100,
        "correct": True,
    }
    monkeypatch.setattr(_benchmark, "_cold_child", lambda: sample)
    assert bench.main(["vector-add", "--cold-child"]) == 0
    assert capsys.readouterr().out == json.dumps(sample, sort_keys=True) + "\n"


def test_module_entrypoint_propagates_failed_measurement(tmp_path):
    """The actual module exits nonzero while preserving invalid raw evidence."""
    output = tmp_path / "evidence.json"
    source = Path(__file__).resolve().parents[2] / "python"
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "swage.bench",
            "vector-add",
            "--output",
            str(output),
        ],
        env={**os.environ, "PYTHONPATH": str(source)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 1, process.stderr
    record = json.loads(output.read_text())
    assert record["schema_version"] == 1
    assert record["benchmark"] == "fixed-runtime"
    assert record["error"]["type"] == "ValueError"
    assert not record["valid"] and not record["passed"]
    assert json.loads(process.stdout) == {
        "output": str(output),
        "valid": False,
        "passed": False,
    }
