# tests/python/test_env.py
"""Independent prerequisite probes and fail-closed environment health checks."""

import json
import os
import subprocess
import sys
import types

import pytest
from swage import _native, env
from swage._errors import BackendUnavailableError


@pytest.fixture
def probes(monkeypatch):
    """Make prerequisite probes deterministic without native code or a GPU."""
    info = {
        "schema_version": 1,
        "package_version": "0.5.2",
        "source_revision": "a" * 40,
        "source_clean": True,
        "llvm_version": "llvmorg-22.1.8",
        "build_type": "Release",
    }
    monkeypatch.setattr(_native, "build_info", lambda: info)
    monkeypatch.setattr(_native, "load_ir", lambda: None)
    monkeypatch.setattr(_native, "load_extension", lambda: None)
    monkeypatch.setattr(env, "_driver_info", lambda: ("13.0", None))
    torch = types.SimpleNamespace(
        __version__="2.6.0",
        version=types.SimpleNamespace(cuda="12.4"),
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            get_device_capability=lambda: (8, 6),
            get_device_name=lambda: "NVIDIA RTX A6000",
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch, info


def test_report_identity_and_driver_build_separation(probes):
    """Packaged identity wins and CUDA build version is not the driver."""
    result = env.report()
    assert result["llvm_pin"] == "llvmorg-22.1.8"
    assert result["native"]["source_revision"] == "a" * 40
    assert result["native"]["source_clean"] is True
    assert result["torch_cuda_build"] == "12.4"
    assert result["cuda_driver"] == "13.0"
    assert result["backends"]["cpu"]["available"]
    assert result["backends"]["cuda"] == {
        "available": True,
        "qualified": True,
        "target": "sm_86",
        "reason": None,
    }


def test_no_torch_keeps_native_and_driver_facts(probes, monkeypatch):
    """Missing PyTorch cannot hide successful native or driver probes."""
    monkeypatch.setitem(sys.modules, "torch", None)
    result = env.report()
    assert result["torch"] is None
    assert result["native"]["available"]
    assert result["cuda_driver"] == "13.0"
    assert not result["backends"]["cpu"]["available"]
    assert not result["backends"]["cuda"]["available"]


def test_no_native_keeps_torch_and_driver_facts(probes, monkeypatch):
    """Missing bindings disable both adapters, not independent GPU facts."""

    def unavailable():
        raise BackendUnavailableError(
            "missing",
            code="native-unavailable",
            backend="native",
            remediation="install wheel",
        )

    monkeypatch.setattr(_native, "load_ir", unavailable)
    result = env.report()
    assert not result["native"]["available"]
    assert result["torch"] == "2.6.0"
    assert result["gpu"]["compute_capability"] == "8.6"
    assert not result["backends"]["cpu"]["available"]
    assert not result["backends"]["cuda"]["available"]


def test_invalid_metadata_is_visible_without_disabling_compilation(
    probes,
    monkeypatch,
):
    """Malformed provenance is diagnosable while native imports still work."""

    def invalid():
        raise ValueError("invalid native build metadata: source_revision")

    monkeypatch.setattr(_native, "build_info", invalid)
    result = env.report()
    assert result["native"]["available"]
    assert result["native"]["source_revision"] is None
    assert "source_revision" in result["native"]["error"]
    assert result["backends"]["cpu"]["available"]


def test_cuda_target_admission_is_not_qualification(probes, monkeypatch):
    """Admitted non-sm86 targets remain available but are not qualified."""
    torch, _ = probes
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))
    result = env.report()["backends"]["cuda"]
    assert result == {
        "available": True,
        "qualified": False,
        "target": "sm_90",
        "reason": None,
    }
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (7, 5))
    result = env.report()["backends"]["cuda"]
    assert not result["available"]
    assert not result["qualified"]
    assert "not admitted" in result["reason"]


def test_same_sm_does_not_qualify_another_gpu(probes, monkeypatch):
    """An admitted sm86 device is usable but not the qualified A6000."""
    torch, _ = probes
    monkeypatch.setattr(
        torch.cuda, "get_device_name", lambda: "NVIDIA GeForce RTX 3090"
    )
    cuda = env.report()["backends"]["cuda"]
    assert cuda["available"] and cuda["target"] == "sm_86"
    assert not cuda["qualified"]
    assert env.main(["--json", "--check", "cuda"]) == 0


def test_driver_failure_does_not_disable_cpu(probes, monkeypatch):
    """A driver problem is never a CPU prerequisite."""
    monkeypatch.setattr(
        env, "_driver_info", lambda: (None, "driver unavailable")
    )
    result = env.report()
    assert result["backends"]["cpu"]["available"]
    assert not result["backends"]["cuda"]["available"]
    assert result["torch_cuda_build"] == "12.4"


def test_probe_exceptions_do_not_disclose_payloads(probes, monkeypatch):
    """Diagnostics catch arbitrary probe failures without printing secrets."""

    def broken():
        raise RuntimeError("tensor([1234]); secret=do-not-log")

    torch, _ = probes
    monkeypatch.setattr(torch.cuda, "get_device_capability", broken)
    result = env.report()
    assert not result["backends"]["cuda"]["available"]
    assert "do-not-log" not in json.dumps(result)
    assert "RuntimeError" in result["backends"]["cuda"]["reason"]


@pytest.mark.parametrize("check", (None, "native", "cpu", "cuda"))
def test_json_health_exit_contract(probes, monkeypatch, capsys, check):
    """A complete sorted JSON report is printed even when checks fail."""
    monkeypatch.setitem(sys.modules, "torch", None)
    args = ["--json"] + ([] if check is None else ["--check", check])
    assert env.main(args) == (1 if check in ("cpu", "cuda") else 0)
    out = capsys.readouterr()
    result = json.loads(out.out)
    assert out.out == json.dumps(result, sort_keys=True) + "\n"
    assert not out.err
    assert result["schema_version"] == 1
    assert "implementation" in result and "machine" in result


def test_module_entrypoint_json_without_check():
    """The real CLI emits exactly one JSON object and exits zero by default."""
    proc = subprocess.run(
        [sys.executable, "-m", "swage.env", "--json"],
        env={**os.environ},
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(proc.stdout)
    assert proc.stdout == json.dumps(result, sort_keys=True) + "\n"
    assert result["schema_version"] == 1


def test_metadata_resource_validation_is_fail_closed(tmp_path, monkeypatch):
    """Absent metadata differs from malformed bytes and invalid field types."""
    monkeypatch.setattr(
        _native.importlib.resources, "files", lambda _: tmp_path
    )
    assert _native.build_info() is None
    resource = tmp_path / "_build_info.json"
    resource.write_text('{"secret": "never-print-this"}')
    with pytest.raises(ValueError) as caught:
        _native.build_info()
    assert "never-print-this" not in str(caught.value)
    valid = {
        "schema_version": 1,
        "package_version": "0.5.2",
        "source_revision": "a" * 40,
        "source_clean": True,
        "llvm_version": "llvmorg-22.1.8",
        "build_type": "Release",
    }
    resource.write_text(json.dumps(valid))
    assert _native.build_info()["source_revision"] == "a" * 40
    resource.write_text(json.dumps({**valid, "source_clean": "true"}))
    with pytest.raises(ValueError, match="source_clean"):
        _native.build_info()
    resource.write_text(json.dumps({**valid, "schema_version": True}))
    with pytest.raises(ValueError, match="schema_version"):
        _native.build_info()
