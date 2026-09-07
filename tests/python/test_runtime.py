# tests/python/test_runtime.py
"""LLVM-free tests for the fixed vector-add CUDA launch boundary."""

import ctypes
import gc
import hashlib
import json
import logging
import pathlib
import stat
import sys
import threading
import types
import weakref
from concurrent.futures import Future, ThreadPoolExecutor
from unittest import mock

import pytest
import swage as sw
import swage.language as sl
from swage import _abi, _cuda_backend


def _fixed_contract_json(kernel_name="add_kernel", block=128):
    return json.dumps(
        {
            "version": 2,
            "backend": "cuda",
            "entry": kernel_name,
            "launch": {"model": "spmd-grid", "block": [block, 1, 1]},
            "arguments": [
                {
                    "kind": "ptr",
                    "origin": "user",
                    "source_index": 0,
                    "access": "read",
                },
                {
                    "kind": "ptr",
                    "origin": "user",
                    "source_index": 1,
                    "access": "read",
                },
                {
                    "kind": "ptr",
                    "origin": "user",
                    "source_index": 2,
                    "access": "write",
                },
                {"kind": "i32", "origin": "user", "source_index": 3},
            ],
        },
        separators=(",", ":"),
    )


def _cpu_contract_json(kernel_name="add_kernel"):
    return json.dumps(
        {
            "version": 2,
            "backend": "cpu",
            "entry": kernel_name,
            "launch": {"model": "host-call"},
            "arguments": [
                {
                    "kind": "ptr",
                    "origin": "user",
                    "source_index": 0,
                    "access": "read",
                },
                {
                    "kind": "ptr",
                    "origin": "user",
                    "source_index": 1,
                    "access": "read",
                },
                {
                    "kind": "ptr",
                    "origin": "user",
                    "source_index": 2,
                    "access": "write",
                },
                {"kind": "i32", "origin": "user", "source_index": 3},
            ],
        },
        separators=(",", ":"),
    )


def _fixed_specialization(kernel_name="add_kernel", block=128):
    return {
        "kernel": kernel_name,
        "backend": "cuda",
        "format": "ptx",
        "target": "sm_86",
        "descriptors": ["ptr<f32>", "ptr<f32>", "ptr<f32>", "i32"],
        "codegen": {
            "lowering": "fixed",
            "block_size": block,
            "options": [],
        },
    }


def _fixed_artifact(key="cache-key", kernel_name="add_kernel", block=128):
    from swage import _runtime

    contract_json = _fixed_contract_json(kernel_name, block)
    contract = _abi.parse_kernel_contract(contract_json)
    return _runtime._make_artifact(
        key,
        _cuda_backend.CUDA_BACKEND,
        "sm_86",
        "lowered",
        "ptx",
        contract_json,
        contract,
    )


def _fixed_compile_artifact(
    _adapter, _specialization, kernel_name, block_size, *_args, **_kwargs
):
    return _fixed_artifact(kernel_name=kernel_name, block=block_size)


@sw.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n, BLOCK: sl.constexpr):  # noqa: D103
    pid = sl.program_id(0)
    offsets = pid * BLOCK + sl.arange(0, BLOCK)
    mask = offsets < n
    x = sl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = sl.load(y_ptr + offsets, mask=mask, other=0.0)
    sl.store(output_ptr + offsets, x + y, mask=mask)


@sw.jit
def renamed_kernel(left, right, destination, length, TILE: sl.constexpr):
    """Canonical vector add with arbitrary diagnostic parameter labels."""
    pid = sl.program_id(0)
    offsets = pid * TILE + sl.arange(0, TILE)
    mask = offsets < length
    x = sl.load(left + offsets, mask=mask, other=0.0)
    y = sl.load(right + offsets, mask=mask, other=0.0)
    sl.store(destination + offsets, x + y, mask=mask)


class _Device:
    def __init__(self, device_type="cuda", index=0):
        self.type = device_type
        self.index = index


class _Tensor:
    def __init__(
        self,
        torch,
        *,
        size=129,
        device_type="cuda",
        device_index=0,
        dtype=None,
        rank=1,
        contiguous=True,
        pointer=0x1000,
    ):
        self.layout = torch.strided
        self.dtype = torch.float32 if dtype is None else dtype
        self.device = _Device(device_type, device_index)
        self._size = size
        self._rank = rank
        self._contiguous = contiguous
        self._pointer = pointer
        self.recorded_streams = []

    def dim(self):
        return self._rank

    def is_contiguous(self):
        return self._contiguous

    def numel(self):
        return self._size

    def data_ptr(self):
        return self._pointer

    def record_stream(self, stream):
        self.recorded_streams.append(stream)


class _Driver:
    def __init__(self):
        self.loads = []
        self.launches = []
        self.launch_kinds = []

    def current_context(self):
        return 0xCAFE

    def load(self, ptx, kernel_name):
        self.loads.append((ptx, kernel_name))
        return 0xBEEF, 0xF00D

    def event_create(self):
        return 0xE001

    def is_stream_capturing(self, _stream):
        return False

    def event_record(self, _event, _stream):
        return None

    def event_query(self, _event):
        return True

    def event_destroy(self, _event):
        return None

    def module_unload(self, _module):
        return None

    def launch_entry(self, function, contract, bindings, grid, stream):
        kinds, arguments = bindings
        self.launch_kinds.append(kinds)
        self.launches.append(
            (function, grid, contract.launch.block[0], stream, arguments)
        )


class _RecordingLease:
    def __init__(self, entry):
        self.entry = entry
        self.released = False

    def release(self):
        self.released = True


class _RecordingBackend:
    def __init__(self, name, *, compile_error=None):
        self.name = name
        self.artifact_format = "ptx" if name == "cuda" else "llvm-jit"
        self.persistent_cache = False
        self.compile_error = compile_error
        self.calls = []

    def compile(
        self,
        _module,
        kernel_name,
        block_size,
        target,
        lowering_kind,
        lowering_options,
    ):
        self.calls.append(
            (
                "compile",
                kernel_name,
                block_size,
                target,
                lowering_kind,
                lowering_options,
            )
        )
        if self.compile_error is not None:
            raise self.compile_error
        contract = (
            _fixed_contract_json(kernel_name, block_size)
            if self.name == "cuda"
            else _cpu_contract_json(kernel_name)
        )
        image = "ptx" if self.name == "cuda" else object()
        return "lowered", image, contract

    def lease(self, artifact, *, capturing=False):
        self.calls.append(("lease", artifact.backend))
        return _RecordingLease(artifact.image)

    def launch(
        self,
        lease,
        contract,
        bindings,
        *,
        grid,
        stream,
        capturing,
    ):
        self.calls.append(
            (
                "launch",
                contract.backend,
                bindings,
                grid,
                stream,
                capturing,
            )
        )
        assert not lease.released

    def release(self, lease):
        self.calls.append(("release",))
        lease.release()


def _fake_torch(
    *, available=True, current_device=0, max_threads=1024, low_precision=False
):
    torch = types.ModuleType("torch")
    torch.float32 = object()
    if low_precision:
        torch.float16 = object()
        torch.float8_e4m3fn = object()
        torch.float8_e5m2 = object()
    torch.strided = object()
    stream = types.SimpleNamespace(cuda_stream=0xABCD)
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: available,
        current_device=lambda: current_device,
        current_stream=lambda: stream,
        get_device_capability=lambda _device=None: (8, 6),
        get_device_properties=lambda _device=None: types.SimpleNamespace(
            max_threads_per_block=max_threads
        ),
        is_current_stream_capturing=lambda: False,
    )
    torch.version = types.SimpleNamespace(cuda="13.0")
    torch.Tensor = _Tensor
    return torch, stream


def _arguments(torch, *, n=129, **tensor_overrides):
    return {
        "x_ptr": _Tensor(torch, pointer=0x1000, **tensor_overrides),
        "y_ptr": _Tensor(torch, pointer=0x2000, **tensor_overrides),
        "output_ptr": _Tensor(torch, pointer=0x3000, **tensor_overrides),
        "n": n,
    }


def _install_launch_fakes(monkeypatch, torch):
    from swage import _runtime

    driver = _Driver()
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(add_kernel, "emit_mlir", lambda **_kwargs: object())
    monkeypatch.setattr(
        _runtime,
        "_compile_cached",
        _fixed_compile_artifact,
    )
    monkeypatch.setattr(_cuda_backend, "_get_driver", lambda: driver)
    _cuda_backend._loaded_functions.clear()
    _cuda_backend._retired_loaded.clear()
    return driver


@pytest.mark.parametrize(
    ("requested_backend", "device_type"),
    [(None, "cuda"), ("cuda", "cuda"), ("cpu", "cpu")],
)
def test_public_launch_routes_only_to_selected_backend(
    monkeypatch,
    requested_backend,
    device_type,
):
    """Route default CUDA, explicit CUDA, and explicit CPU without fallback."""
    from swage import _runtime

    adapters = {name: _RecordingBackend(name) for name in ("cuda", "cpu")}
    torch, _ = _fake_torch()
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(_runtime, "get_backend", adapters.__getitem__)
    monkeypatch.setattr(
        _runtime,
        "_compiler_identity",
        lambda: {"revision": None, "clean": False, "llvm": None},
    )
    monkeypatch.setattr(add_kernel, "emit_mlir", lambda **_kwargs: object())
    _runtime._identity_cache = None
    _runtime._artifact_cache.clear()
    _runtime._compilations.clear()
    add_kernel.__dict__.pop("_specialization_memo", None)
    arguments = _arguments(torch, device_type=device_type)
    launch_kwargs = {
        "arguments": arguments,
        "constexprs": {"BLOCK": 128},
        "grid": (2,),
    }
    if requested_backend is not None:
        launch_kwargs["backend"] = requested_backend

    add_kernel.launch(**launch_kwargs)

    selected = requested_backend or "cuda"
    other = "cpu" if selected == "cuda" else "cuda"
    assert [call[0] for call in adapters[selected].calls] == [
        "compile",
        "lease",
        "launch",
        "release",
    ]
    assert adapters[selected].calls[0][3] == (
        "native" if selected == "cpu" else "sm_86"
    )
    physical = adapters[selected].calls[2]
    assert physical[3] == (None if selected == "cpu" else (2, 1, 1))
    assert adapters[other].calls == []


def test_unknown_backend_fails_before_torch_or_pointer_access(monkeypatch):
    """Reject an unknown backend before importing torch or reading inputs."""
    real_import = __import__

    def reject_torch(name, *args, **kwargs):
        if name == "torch":
            pytest.fail("unknown backend imported torch")
        return real_import(name, *args, **kwargs)

    with mock.patch("builtins.__import__", side_effect=reject_torch):
        with pytest.raises(
            ValueError,
            match="unknown execution backend 'rocm'; expected 'cpu' or 'cuda'",
        ):
            add_kernel.launch(
                arguments=object(),
                constexprs=object(),
                grid=object(),
                backend="rocm",
            )


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
@pytest.mark.parametrize("unavailable", [False, True])
def test_selected_backend_compile_failure_does_not_consult_other(
    monkeypatch, backend, unavailable
):
    """Never probe the other backend after compilation or availability fails."""
    from swage import _runtime

    error = (
        sw.BackendUnavailableError(
            "native missing",
            code="native-unavailable",
            backend=backend,
            remediation="Install a supported wheel.",
        )
        if unavailable
        else RuntimeError("compile failed")
    )
    other = "cuda" if backend == "cpu" else "cpu"
    adapters = {
        backend: _RecordingBackend(backend, compile_error=error),
        other: _RecordingBackend(other),
    }
    torch, _ = _fake_torch()
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(_runtime, "get_backend", adapters.__getitem__)
    monkeypatch.setattr(
        _runtime,
        "_compiler_identity",
        lambda: {"revision": None, "clean": False, "llvm": None},
    )
    monkeypatch.setattr(add_kernel, "emit_mlir", lambda **_kwargs: object())
    _runtime._identity_cache = None
    _runtime._artifact_cache.clear()
    _runtime._compilations.clear()
    add_kernel.__dict__.pop("_specialization_memo", None)

    with pytest.raises(RuntimeError) as caught:
        add_kernel.launch(
            arguments=_arguments(torch, device_type=backend),
            constexprs={"BLOCK": 128},
            grid=(2,),
            backend=backend,
        )

    assert caught.value is error
    assert [call[0] for call in adapters[backend].calls] == ["compile"]
    assert adapters[other].calls == []


def test_launch_uses_current_stream_and_raw_abi(monkeypatch):
    """Launch asynchronously and preserve tensor storage on the stream."""
    torch, stream = _fake_torch()
    arguments = _arguments(torch)
    driver = _install_launch_fakes(monkeypatch, torch)

    result = add_kernel.launch(
        arguments=arguments,
        constexprs={"BLOCK": 128},
        grid=(2,),
    )

    assert result is None
    assert driver.loads == [("ptx", "add_kernel")]
    assert driver.launches == [
        (0xF00D, (2, 1, 1), 128, 0xABCD, (0x1000, 0x2000, 0x3000, 129))
    ]
    assert driver.launch_kinds == [("ptr", "ptr", "ptr", "i32")]
    for tensor in tuple(arguments.values())[:3]:
        assert tensor.recorded_streams == [stream]


def test_launch_accepts_renamed_canonical_parameters(monkeypatch):
    """Use source order and compiler metadata instead of fixed labels."""
    torch, _ = _fake_torch()
    driver = _install_launch_fakes(monkeypatch, torch)
    monkeypatch.setattr(renamed_kernel, "emit_mlir", lambda **_kwargs: object())
    arguments = {
        "left": _Tensor(torch, pointer=0x1000),
        "right": _Tensor(torch, pointer=0x2000),
        "destination": _Tensor(torch, pointer=0x3000),
        "length": 129,
    }

    renamed_kernel.launch(
        arguments=arguments,
        constexprs={"TILE": 128},
        grid=(2,),
    )

    assert driver.loads == [("ptx", "renamed_kernel")]
    assert driver.launches == [
        (0xF00D, (2, 1, 1), 128, 0xABCD, (0x1000, 0x2000, 0x3000, 129))
    ]


def test_repeated_launch_reuses_loaded_function_without_retaining_tensors(
    monkeypatch,
):
    """Cache only module handles, never tensor objects or data pointers."""
    torch, _ = _fake_torch()
    driver = _install_launch_fakes(monkeypatch, torch)
    first = _arguments(torch)
    reference = weakref.ref(first["x_ptr"])

    for arguments in (first, _arguments(torch)):
        add_kernel.launch(
            arguments=arguments,
            constexprs={"BLOCK": 128},
            grid=(2,),
        )
    del arguments
    del first
    gc.collect()

    assert len(driver.loads) == 1
    assert len(driver.launches) == 2
    assert reference() is None


def test_empty_launch_is_a_validated_noop(monkeypatch):
    """Avoid native compilation and driver loading for the empty grid."""
    torch, _ = _fake_torch()
    monkeypatch.setitem(sys.modules, "torch", torch)
    with mock.patch.object(add_kernel, "emit_mlir") as emit:
        assert (
            add_kernel.launch(
                arguments=_arguments(torch, n=0, size=0),
                constexprs={"BLOCK": 128},
                grid=(0,),
            )
            is None
        )
    emit.assert_not_called()


@pytest.mark.parametrize(
    ("arguments", "constexprs", "grid", "reason"),
    [
        ("bad", {"BLOCK": 128}, (2,), "arguments must be a mapping"),
        (None, "bad", (2,), "constexprs must be a mapping"),
        (None, {"BLOCK": 0}, (2,), "BLOCK.*positive"),
        (None, {"BLOCK": 128}, [2], "grid must be a one-element tuple"),
        (None, {"BLOCK": 128}, (1,), "grid must equal"),
        (None, {"BLOCK": 2048}, (1,), "exceeds device limit"),
    ],
)
def test_launch_rejects_invalid_mapping_block_and_grid(
    monkeypatch, arguments, constexprs, grid, reason
):
    """Fail before compilation for invalid launch geometry."""
    torch, _ = _fake_torch()
    monkeypatch.setitem(sys.modules, "torch", torch)
    if arguments is None:
        arguments = _arguments(torch)
    with pytest.raises((TypeError, ValueError), match=reason):
        add_kernel.launch(
            arguments=arguments,
            constexprs=constexprs,
            grid=grid,
        )


@pytest.mark.parametrize(
    ("overrides", "n", "reason"),
    [
        ({"device_type": "cpu"}, 1, "must be a CUDA tensor"),
        ({"device_index": 1}, 1, "current CUDA device"),
        ({"rank": 2}, 1, "rank one"),
        ({"contiguous": False}, 1, "contiguous"),
        ({"dtype": object()}, 1, "torch.float32"),
        ({"size": 3}, 4, "exceeds tensor length"),
        ({}, -1, "nonnegative"),
        ({}, 1 << 31, "nonnegative i32"),
    ],
)
def test_launch_rejects_invalid_tensor_metadata(
    monkeypatch, overrides, n, reason
):
    """Reject metadata that cannot satisfy the raw-pointer ABI."""
    torch, _ = _fake_torch()
    monkeypatch.setitem(sys.modules, "torch", torch)
    with pytest.raises((TypeError, ValueError), match=reason):
        add_kernel.launch(
            arguments=_arguments(torch, n=n, **overrides),
            constexprs={"BLOCK": 128},
            grid=(1,),
        )


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
@pytest.mark.parametrize("n", [0, 1])
@pytest.mark.parametrize("dtype_name", ["int32", "bfloat16", "float64"])
def test_unsupported_dtypes_fail_before_pointer_acquisition(
    monkeypatch, backend, n, dtype_name
):
    """Reject unsupported storage even for a zero-length public launch."""
    torch, _ = _fake_torch(low_precision=True)
    dtype = object()
    setattr(torch, dtype_name, dtype)
    monkeypatch.setitem(sys.modules, "torch", torch)

    def reject_pointer_read(self):
        raise AssertionError("validation must precede data_ptr")

    monkeypatch.setattr(_Tensor, "data_ptr", reject_pointer_read)
    with pytest.raises(TypeError, match="argument 'x_ptr' must have dtype"):
        add_kernel.launch(
            arguments=_arguments(torch, n=n, dtype=dtype, device_type=backend),
            constexprs={"BLOCK": 128},
            grid=(n,),
            backend=backend,
        )


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
@pytest.mark.parametrize("n", [0, 1])
@pytest.mark.parametrize(
    ("base_dtype", "mixed_dtype", "parameter"),
    [
        ("float32", "float16", "y_ptr"),
        ("float8_e4m3fn", "float8_e5m2", "output_ptr"),
    ],
)
def test_mixed_dtypes_fail_before_pointer_acquisition(
    monkeypatch, backend, n, base_dtype, mixed_dtype, parameter
):
    """Reject mixed input/output types, including equal-width FP8 formats."""
    torch, _ = _fake_torch(low_precision=True)
    monkeypatch.setitem(sys.modules, "torch", torch)
    arguments = _arguments(
        torch, n=n, dtype=getattr(torch, base_dtype), device_type=backend
    )
    arguments[parameter].dtype = getattr(torch, mixed_dtype)

    def reject_pointer_read(self):
        raise AssertionError("validation must precede data_ptr")

    monkeypatch.setattr(_Tensor, "data_ptr", reject_pointer_read)
    with pytest.raises(
        TypeError,
        match=f"argument '{parameter}' must have the same dtype",
    ):
        add_kernel.launch(
            arguments=arguments,
            constexprs={"BLOCK": 128},
            grid=(n,),
            backend=backend,
        )


def test_cache_round_trip_and_corruption_rejection(tmp_path, monkeypatch):
    """Reuse verified PTX and never return corrupted cache contents."""
    from swage import _runtime

    calls = []
    monkeypatch.setenv("SWAGE_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(
        _runtime,
        "_compiler_identity",
        lambda: {"revision": "abc", "clean": True, "llvm": "llvmorg-test"},
    )
    monkeypatch.setattr(
        _cuda_backend.CUDA_BACKEND,
        "compile",
        lambda *_args: calls.append(True)
        or ("lowered", "ptx", _fixed_contract_json()),
    )
    key_data = _fixed_specialization()

    first = _runtime._compile_cached(
        _cuda_backend.CUDA_BACKEND, key_data, "add_kernel", 128, object
    )
    _runtime._artifact_cache.clear()
    second = _runtime._compile_cached(
        _cuda_backend.CUDA_BACKEND, key_data, "add_kernel", 128, object
    )

    assert first == second
    assert len(calls) == 1
    entry = tmp_path / first.key
    assert stat.S_IMODE(entry.stat().st_mode) == 0o700
    assert stat.S_IMODE((entry / "kernel.ptx").stat().st_mode) == 0o600
    metadata = json.loads((entry / "metadata.json").read_text())
    assert metadata["contract"] == _fixed_contract_json()
    assert metadata["version"] == 3
    assert (
        metadata["digests"]["contract"]
        == hashlib.sha256(_fixed_contract_json().encode()).hexdigest()
    )
    assert first.contract_json == _fixed_contract_json()
    assert first.contract == second.contract
    assert first.contract.arguments[0].access == "read"
    with pytest.raises(AttributeError):
        first.contract.entry = "changed"
    (entry / "kernel.ptx").write_text("corrupt")
    _runtime._artifact_cache.clear()
    with pytest.raises(RuntimeError, match="digest mismatch"):
        _runtime._compile_cached(
            _cuda_backend.CUDA_BACKEND, key_data, "add_kernel", 128, object
        )


@pytest.mark.parametrize(
    ("contract_json", "reason"),
    [
        ("not-json", "not valid JSON"),
        (
            _fixed_contract_json().replace('"version":2', '"version":1'),
            "unsupported kernel contract version",
        ),
        (
            _fixed_contract_json().replace('"kind":"i32"', '"kind":"u32"'),
            "unknown kind",
        ),
        (_fixed_contract_json() + " ", "not canonical"),
    ],
)
def test_contract_parser_rejects_unknown_or_malformed_schema(
    contract_json, reason
):
    """Accept only canonical immutable v2 contracts."""
    with pytest.raises((TypeError, ValueError), match=reason):
        _abi.parse_kernel_contract(contract_json)


def test_contract_parser_accepts_sparse_user_source_indexes():
    """Preserve semantic source positions for entries using a subset."""
    contract_json = json.dumps(
        {
            "version": 2,
            "backend": "cuda",
            "entry": "split_merge",
            "launch": {"model": "spmd-grid", "block": [128, 1, 1]},
            "arguments": [
                {
                    "kind": "ptr",
                    "origin": "user",
                    "source_index": 2,
                    "access": "write",
                }
            ],
        },
        separators=(",", ":"),
    )

    contract = _abi.parse_kernel_contract(contract_json)

    assert contract.arguments[0].source_index == 2


def test_cache_rejects_contract_corruption_and_old_metadata(
    tmp_path, monkeypatch
):
    """Fail closed on contract bytes and reject v2 cache entries."""
    from swage import _runtime

    monkeypatch.setenv("SWAGE_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(
        _runtime,
        "_compiler_identity",
        lambda: {"revision": "abc", "clean": True, "llvm": "llvmorg-test"},
    )
    monkeypatch.setattr(
        _cuda_backend.CUDA_BACKEND,
        "compile",
        lambda *_args: ("lowered", "ptx", _fixed_contract_json()),
    )
    _runtime._artifact_cache.clear()
    specialization = _fixed_specialization()
    artifact = _runtime._compile_cached(
        _cuda_backend.CUDA_BACKEND, specialization, "add_kernel", 128, object
    )
    entry = tmp_path / artifact.key
    metadata_path = entry / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["contract"] = _fixed_contract_json("other_kernel")
    metadata_path.write_text(
        json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    )
    _runtime._artifact_cache.clear()
    with pytest.raises(RuntimeError, match="contract digest mismatch"):
        _runtime._compile_cached(
            _cuda_backend.CUDA_BACKEND,
            specialization,
            "add_kernel",
            128,
            object,
        )

    metadata = json.loads(metadata_path.read_text())
    metadata["digests"]["contract"] = hashlib.sha256(
        metadata["contract"].encode()
    ).hexdigest()
    metadata_path.write_text(
        json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    )
    with pytest.raises(RuntimeError, match="entry does not match"):
        _runtime._compile_cached(
            _cuda_backend.CUDA_BACKEND,
            specialization,
            "add_kernel",
            128,
            object,
        )

    metadata["version"] = 2
    del metadata["backend"]
    del metadata["contract"]
    del metadata["digests"]["contract"]
    metadata_path.write_text(
        json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    )
    with pytest.raises(RuntimeError, match="cache metadata mismatch"):
        _runtime._compile_cached(
            _cuda_backend.CUDA_BACKEND,
            specialization,
            "add_kernel",
            128,
            object,
        )


def test_launch_rejects_contract_mismatch_before_module_load(monkeypatch):
    """Validate compiler metadata before loading PTX into CUDA."""
    from swage import _runtime

    torch, _ = _fake_torch()
    driver = _Driver()
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(add_kernel, "emit_mlir", lambda **_kwargs: object())
    monkeypatch.setattr(
        _cuda_backend.CUDA_BACKEND,
        "compile",
        lambda *_args: (
            "lowered",
            "ptx",
            _fixed_contract_json(block=64),
        ),
    )
    monkeypatch.setattr(
        _runtime,
        "_compiler_identity",
        lambda: {"revision": None, "clean": False, "llvm": None},
    )
    monkeypatch.setattr(_cuda_backend, "_get_driver", lambda: driver)
    _runtime._identity_cache = None
    _runtime._artifact_cache.clear()
    _cuda_backend._loaded_functions.clear()

    with pytest.raises(RuntimeError, match="block does not match"):
        add_kernel.launch(
            arguments=_arguments(torch),
            constexprs={"BLOCK": 128},
            grid=(2,),
        )
    assert driver.loads == []


def test_cache_rejects_unsafe_entries(tmp_path, monkeypatch):
    """Reject symlinked and world-writable cache content."""
    from swage import _runtime

    monkeypatch.setenv("SWAGE_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(
        _runtime,
        "_compiler_identity",
        lambda: {"revision": "abc", "clean": True, "llvm": "llvmorg-test"},
    )
    key_data = _fixed_specialization()
    key = _runtime._cache_key(key_data)
    entry = tmp_path / key
    entry.mkdir()
    target = tmp_path / "outside.json"
    target.write_text(json.dumps({}))
    (entry / "metadata.json").symlink_to(target)

    with pytest.raises(RuntimeError, match="symlink"):
        _runtime._compile_cached(
            _cuda_backend.CUDA_BACKEND, key_data, "add_kernel", 128, object
        )

    (entry / "metadata.json").unlink()
    (entry / "metadata.json").write_text("{}")
    (entry / "metadata.json").chmod(0o606)
    with pytest.raises(RuntimeError, match="world-writable"):
        _runtime._compile_cached(
            _cuda_backend.CUDA_BACKEND, key_data, "add_kernel", 128, object
        )


def test_dirty_build_uses_only_process_cache(tmp_path, monkeypatch):
    """Do not persist artifacts for dirty or unidentified compiler builds."""
    from swage import _runtime

    calls = []
    monkeypatch.setenv("SWAGE_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(
        _runtime,
        "_compiler_identity",
        lambda: {"revision": "abc", "clean": False, "llvm": "llvmorg-test"},
    )
    monkeypatch.setattr(
        _cuda_backend.CUDA_BACKEND,
        "compile",
        lambda *_args: calls.append(True)
        or ("lowered", "ptx", _fixed_contract_json("add")),
    )

    specialization = _fixed_specialization("add")
    first = _runtime._compile_cached(
        _cuda_backend.CUDA_BACKEND, specialization, "add", 128, object
    )
    second = _runtime._compile_cached(
        _cuda_backend.CUDA_BACKEND, specialization, "add", 128, object
    )

    assert first == second
    assert len(calls) == 1
    assert list(tmp_path.iterdir()) == []


def test_dump_switches_write_requested_artifacts(tmp_path, monkeypatch):
    """Write deterministic debug artifacts only when explicitly requested."""
    from swage import _runtime

    monkeypatch.setenv("SWAGE_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("SWAGE_DUMP_MLIR", "1")
    monkeypatch.setenv("SWAGE_DUMP_PTX", "1")
    artifact = _fixed_artifact(key="key")

    _runtime._write_dumps(artifact)

    assert (tmp_path / "key.mlir").read_text() == "lowered"
    assert (tmp_path / "key.ptx").read_text() == "ptx"


def test_cache_path_defaults_to_user_cache(monkeypatch):
    """Keep cache placement predictable without creating it during import."""
    from swage import _runtime

    monkeypatch.delenv("SWAGE_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", "/tmp/user-cache")
    assert _runtime._cache_dir() == pathlib.Path("/tmp/user-cache/swage")


def test_driver_marshals_pointer_and_i32_parameters():
    """Pass raw pointers and an i32 through the CUDA Driver ABI."""
    driver = object.__new__(_cuda_backend._CudaDriver)
    calls = []
    driver._call = lambda name, *args: calls.append((name, args))

    contract = _fixed_artifact().contract
    driver.launch_entry(
        0xF00D,
        contract,
        (("ptr", "ptr", "ptr", "i32"), (0x10, 0x20, 0x30, 129)),
        (2, 1, 1),
        0xABCD,
    )

    name, call = calls[0]
    parameters = call[-2]
    pointer_values = [
        ctypes.cast(
            parameters[index], ctypes.POINTER(ctypes.c_void_p)
        ).contents.value
        for index in range(3)
    ]
    scalar = ctypes.cast(
        parameters[3], ctypes.POINTER(ctypes.c_int32)
    ).contents.value
    assert name == "cuLaunchKernel"
    assert pointer_values == [0x10, 0x20, 0x30]
    assert scalar == 129


def test_driver_error_contains_stable_name_code_and_text():
    """Preserve actionable CUDA Driver diagnostics."""

    def set_text(_result, output, value):
        ctypes.cast(output, ctypes.POINTER(ctypes.c_char_p))[0] = value
        return 0

    driver = object.__new__(_cuda_backend._CudaDriver)
    driver.library = types.SimpleNamespace(
        cuBad=lambda: 1,
        cuGetErrorName=lambda result, output: set_text(
            result, output, b"CUDA_ERROR_INVALID_VALUE"
        ),
        cuGetErrorString=lambda result, output: set_text(
            result, output, b"invalid argument"
        ),
    )

    with pytest.raises(
        RuntimeError,
        match=(
            r"cuBad failed: CUDA_ERROR_INVALID_VALUE \(1\): "
            "invalid argument"
        ),
    ):
        driver._call("cuBad")


def test_compiler_identity_is_cached_per_process(monkeypatch):
    """Spawn the git subprocesses once, not twice per launch."""
    import subprocess

    from swage import _runtime

    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(
            command, 0, stdout="abc\n", stderr=""
        )

    monkeypatch.setattr(_runtime.subprocess, "run", fake_run)
    monkeypatch.setattr(_runtime._native, "build_info", lambda: None)
    _runtime._identity_cache = None
    results = [_runtime._cached_identity() for _ in range(3)]
    assert results[0] == results[1] == results[2]
    assert len(commands) == 2


def test_warm_launch_emits_mlir_only_once(monkeypatch):
    """Skip AST-to-MLIR emission entirely on a specialization-cache hit."""
    from swage import _runtime

    torch, _ = _fake_torch()
    driver = _Driver()
    emissions = []
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(
        add_kernel,
        "emit_mlir",
        lambda **_kwargs: emissions.append(1) or object(),
    )
    monkeypatch.setattr(
        _cuda_backend.CUDA_BACKEND,
        "compile",
        lambda *_args, **_kwargs: (
            "lowered",
            "ptx",
            _fixed_contract_json(),
        ),
    )
    fake = {"revision": None, "clean": False, "llvm": None}
    monkeypatch.setattr(_runtime, "_compiler_identity", lambda: fake)
    monkeypatch.setattr(_cuda_backend, "_get_driver", lambda: driver)
    _runtime._identity_cache = None
    _runtime._artifact_cache.clear()
    _cuda_backend._loaded_functions.clear()

    for _ in range(3):
        add_kernel.launch(
            arguments=_arguments(torch),
            constexprs={"BLOCK": 128},
            grid=(2,),
        )

    assert len(emissions) == 1
    assert len(driver.launches) == 3
    artifact = next(iter(_runtime._artifact_cache.values()))
    assert artifact.contract_json == _fixed_contract_json()
    assert tuple(
        argument.source_index for argument in artifact.contract.arguments
    ) == (0, 1, 2, 3)


def test_device_fact_cache_is_isolated_per_torch_module(monkeypatch):
    """Never let one process's device cache leak across torch modules."""
    from swage import _runtime

    driver = _Driver()
    monkeypatch.setattr(_cuda_backend, "_get_driver", lambda: driver)
    monkeypatch.setattr(
        _runtime,
        "_compile_cached",
        _fixed_compile_artifact,
    )
    _cuda_backend._loaded_functions.clear()

    big_torch, _ = _fake_torch(max_threads=1024)
    monkeypatch.setitem(sys.modules, "torch", big_torch)
    monkeypatch.setattr(add_kernel, "emit_mlir", lambda **_kwargs: object())
    add_kernel.launch(
        arguments=_arguments(big_torch),
        constexprs={"BLOCK": 1024},
        grid=(1,),
    )

    small_torch, _ = _fake_torch(max_threads=512)
    monkeypatch.setitem(sys.modules, "torch", small_torch)
    with pytest.raises(ValueError, match="exceeds device limit"):
        add_kernel.launch(
            arguments=_arguments(small_torch),
            constexprs={"BLOCK": 1024},
            grid=(1,),
        )


def _stub_cuda_library(monkeypatch):
    monkeypatch.setattr(
        _cuda_backend.ctypes,
        "CDLL",
        lambda _name: mock.MagicMock(**{"cuLaunchKernel.return_value": 0}),
    )


def test_segmented_artifact_compile_and_context_load_reuse(monkeypatch):
    """Reuse one compiled artifact and one loaded module per CUDA context."""
    from swage import _runtime

    identity = {"revision": None, "clean": False, "llvm": "llvm-pin"}
    monkeypatch.setattr(_runtime, "_cached_identity", lambda: identity)
    specialization = _runtime.segmented_specialization(
        "module { func.func @segmented_sum() }",
        kernel_name="segmented_sum",
        target="sm_86",
        block_size=32,
        lowering_kind="segmented",
        lowering_options={"use_task_ids": True},
        schedule={"warp_max_elements": 32, "cta_chunk_elements": 4096},
        adapter=_cuda_backend.CUDA_BACKEND,
    )
    contract_json = json.dumps(
        {
            "version": 2,
            "backend": "cuda",
            "entry": "segmented_sum",
            "launch": {"model": "spmd-grid", "block": [32, 1, 1]},
            "arguments": [
                {
                    "kind": "ptr",
                    "origin": "plan",
                    "key": "task_ids",
                    "access": "read",
                },
                {"kind": "i32", "origin": "derived", "key": "task_count"},
            ],
        },
        separators=(",", ":"),
    )
    compiles = []
    emits = []

    def compile_native(module, kernel, block, target, kind, options):
        compiles.append((module, kernel, block, target, kind, options))
        return "lowered", "ptx", contract_json

    monkeypatch.setattr(_cuda_backend.CUDA_BACKEND, "compile", compile_native)
    _runtime._artifact_cache.clear()
    _cuda_backend._loaded_functions.clear()
    first = _runtime._compile_cached(
        _cuda_backend.CUDA_BACKEND,
        specialization,
        "segmented_sum",
        32,
        lambda: emits.append(True) or object(),
        lowering_kind="segmented",
        lowering_options={"use_task_ids": True},
    )
    second = _runtime._compile_cached(
        _cuda_backend.CUDA_BACKEND,
        specialization,
        "segmented_sum",
        32,
        lambda: pytest.fail("warm artifact must not emit MLIR"),
        lowering_kind="segmented",
        lowering_options={"use_task_ids": True},
    )

    class _ContextDriver:
        def __init__(self):
            self.context = 1
            self.loads = []

        def current_context(self):
            return self.context

        def load(self, ptx, entry):
            loaded = (100 + self.context, 200 + self.context)
            self.loads.append((self.context, ptx, entry, loaded))
            return loaded

    driver = _ContextDriver()
    assert first is second
    assert len(emits) == 1
    assert compiles[0][4:] == ("segmented", {"use_task_ids": True})
    monkeypatch.setattr(_cuda_backend, "_get_driver", lambda: driver)
    first_lease = _cuda_backend.CUDA_BACKEND.lease(first)
    second_lease = _cuda_backend.CUDA_BACKEND.lease(first)
    assert (first_lease.entry.module, first_lease.entry.function) == (101, 201)
    assert second_lease.entry is first_lease.entry
    driver.context = 2
    third_lease = _cuda_backend.CUDA_BACKEND.lease(first)
    assert (third_lease.entry.module, third_lease.entry.function) == (102, 202)
    for lease in (first_lease, second_lease, third_lease):
        _cuda_backend.CUDA_BACKEND.release(lease)
    assert len(driver.loads) == 2


def test_compiled_artifact_does_not_retain_emitted_module(monkeypatch):
    """Keep compiler artifacts free of semantic modules and caller storage."""
    from swage import _runtime

    class _Module:
        pass

    module = _Module()
    reference = weakref.ref(module)
    holder = [module]
    specialization = _fixed_specialization()
    monkeypatch.setattr(
        _runtime,
        "_cached_identity",
        lambda: {"revision": None, "clean": False, "llvm": "llvm-pin"},
    )
    monkeypatch.setattr(
        _cuda_backend.CUDA_BACKEND,
        "compile",
        lambda *_args: ("lowered", "ptx", _fixed_contract_json()),
    )
    _runtime._artifact_cache.clear()
    artifact = _runtime._compile_cached(
        _cuda_backend.CUDA_BACKEND,
        specialization,
        "add_kernel",
        128,
        lambda: holder[0],
        lowering_kind="fixed",
    )
    holder.clear()
    del module
    gc.collect()

    assert artifact.image == "ptx"
    assert reference() is None


def test_driver_falls_back_to_ctypes_without_the_bindings(monkeypatch):
    """Keep the ctypes path working when mlir_swage is absent."""
    _stub_cuda_library(monkeypatch)
    # Blocking the parent is not enough: a fully dotted module already in
    # sys.modules is returned without consulting the parent, so drop any
    # cached binding modules for the duration of the test as well.
    monkeypatch.setitem(sys.modules, "mlir_swage", None)
    monkeypatch.delitem(
        sys.modules,
        "mlir_swage._mlir_libs._swageDialectsNanobind",
        raising=False,
    )
    monkeypatch.delitem(sys.modules, "mlir_swage._mlir_libs", raising=False)

    driver = _cuda_backend._CudaDriver()
    assert driver._native_launch is None
    driver.launch_entry(
        7,
        _fixed_artifact().contract,
        (("ptr", "ptr", "ptr", "i32"), (0x1, 0x2, 0x3, 129)),
        (3, 1, 1),
        9,
    )
    assert driver.library.cuLaunchKernel.called


def _reset_process_runtime(runtime):
    runtime._artifact_cache.clear()
    runtime._compilations.clear()
    runtime._memory_cache_entries = None
    _cuda_backend._loaded_functions.clear()
    _cuda_backend._retired_loaded.clear()
    _cuda_backend._memory_cache_entries = None
    _cuda_backend._stream_objects = weakref.WeakKeyDictionary()


def _install_fake_compiler(monkeypatch, runtime, callback=None):
    monkeypatch.setattr(
        runtime,
        "_cached_identity",
        lambda: {"revision": None, "clean": False, "llvm": "llvm-pin"},
    )

    def compile_native(*_arguments):
        if callback is not None:
            callback()
        return "lowered", "ptx", _fixed_contract_json()

    monkeypatch.setattr(_cuda_backend.CUDA_BACKEND, "compile", compile_native)


def test_memory_artifact_lru_is_bounded_and_refreshes_recency(monkeypatch):
    """Bound complete artifacts and refresh a warm specialization."""
    from swage import _runtime

    monkeypatch.setenv("SWAGE_MEMORY_CACHE_ENTRIES", "2")
    _reset_process_runtime(_runtime)
    _install_fake_compiler(monkeypatch, _runtime)
    specialization = _fixed_specialization()

    first = _runtime._compile_cached(
        _cuda_backend.CUDA_BACKEND,
        specialization,
        "add_kernel",
        128,
        object,
        key="first",
    )
    _runtime._compile_cached(
        _cuda_backend.CUDA_BACKEND,
        specialization,
        "add_kernel",
        128,
        object,
        key="second",
    )
    assert (
        _runtime._compile_cached(
            _cuda_backend.CUDA_BACKEND,
            specialization,
            "add_kernel",
            128,
            lambda: pytest.fail("warm hit emitted MLIR"),
            key="first",
        )
        is first
    )
    _runtime._compile_cached(
        _cuda_backend.CUDA_BACKEND,
        specialization,
        "add_kernel",
        128,
        object,
        key="third",
    )

    assert list(_runtime._artifact_cache) == ["first", "third"]
    assert all(
        artifact.lowered == "lowered"
        and artifact.image == "ptx"
        and artifact.contract.entry == "add_kernel"
        for artifact in _runtime._artifact_cache.values()
    )


@pytest.mark.parametrize(
    "configured", ["", "0", "-1", "+", "+1", "1.5", "many"]
)
def test_memory_cache_entries_rejects_invalid_values(monkeypatch, configured):
    """Reject every noncanonical or nonpositive process-cache capacity."""
    from swage import _runtime

    monkeypatch.setenv("SWAGE_MEMORY_CACHE_ENTRIES", configured)
    _runtime._memory_cache_entries = None
    with pytest.raises(
        ValueError,
        match="SWAGE_MEMORY_CACHE_ENTRIES must be a positive integer",
    ):
        _runtime._memory_cache_limit()


def test_same_specialization_compilation_is_coalesced(monkeypatch):
    """Make concurrent callers share one in-flight specialization future."""
    from swage import _runtime

    _reset_process_runtime(_runtime)
    started = threading.Event()
    release = threading.Event()
    calls = []

    def callback():
        calls.append(True)
        started.set()
        assert release.wait(5)

    _install_fake_compiler(monkeypatch, _runtime, callback)
    specialization = _fixed_specialization()
    with ThreadPoolExecutor(max_workers=8) as pool:
        first = pool.submit(
            _runtime._compile_cached,
            _cuda_backend.CUDA_BACKEND,
            specialization,
            "add_kernel",
            128,
            object,
            key="shared",
        )
        assert started.wait(5)
        others = [
            pool.submit(
                _runtime._compile_cached,
                _cuda_backend.CUDA_BACKEND,
                specialization,
                "add_kernel",
                128,
                object,
                key="shared",
            )
            for _ in range(7)
        ]
        release.set()
        results = [first.result(), *(future.result() for future in others)]

    assert len(calls) == 1
    assert all(result is results[0] for result in results)


def test_same_specialization_remains_coalesced_until_settlement(monkeypatch):
    """Keep the in-flight future visible through result publication."""
    from swage import _runtime

    class _DelayedFuture(Future):
        def set_result(self, result):
            settling.set()
            assert release.wait(5)
            super().set_result(result)

    _reset_process_runtime(_runtime)
    settling = threading.Event()
    release = threading.Event()
    calls = []
    _install_fake_compiler(
        monkeypatch, _runtime, lambda: calls.append(threading.get_ident())
    )
    monkeypatch.setattr(_runtime, "Future", _DelayedFuture)
    specialization = _fixed_specialization()

    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(
            _runtime._compile_cached,
            _cuda_backend.CUDA_BACKEND,
            specialization,
            "add_kernel",
            128,
            object,
            key="settling",
        )
        assert settling.wait(5)
        with _runtime._cache_lock:
            _runtime._artifact_cache.clear()
        timer = threading.Timer(0.05, release.set)
        timer.start()
        second = _runtime._compile_cached(
            _cuda_backend.CUDA_BACKEND,
            specialization,
            "add_kernel",
            128,
            object,
            key="settling",
        )
        timer.join()
        first = first.result()

    assert len(calls) == 1
    assert second is first


def test_different_specializations_compile_concurrently(monkeypatch):
    """Never hold the cache lock while compiling independent keys."""
    from swage import _runtime

    _reset_process_runtime(_runtime)
    barrier = threading.Barrier(2)
    entered = []

    def callback():
        entered.append(threading.get_ident())
        barrier.wait(5)

    _install_fake_compiler(monkeypatch, _runtime, callback)
    specialization = _fixed_specialization()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [
            pool.submit(
                _runtime._compile_cached,
                _cuda_backend.CUDA_BACKEND,
                specialization,
                "add_kernel",
                128,
                object,
                key=key,
            )
            for key in ("left", "right")
        ]
        [future.result() for future in results]

    assert len(set(entered)) == 2


class _LifecycleDriver:
    """Model CUDA events as snapshots of submitted stream work."""

    def __init__(self):
        self.context = 1
        self.next_handle = 0
        self.queries = []
        self.submitted = {}
        self.completed = {}
        self.closed_streams = set()
        self.fences = {}
        self.destroyed = []
        self.unloaded = []
        self.launch_error = None
        self.record_error = None

    def current_context(self):
        return self.context

    def load(self, _ptx, _entry):
        self.next_handle += 1
        return 100 + self.next_handle, 200 + self.next_handle

    def launch_entry(self, _function, _contract, _bindings, _grid, stream):
        self.submitted[stream] = self.submitted.get(stream, 0) + 1
        if self.launch_error is not None:
            raise self.launch_error

    def complete_stream(self, stream):
        self.completed[stream] = self.submitted[stream]

    def close_stream(self, stream):
        assert stream != 0
        self.closed_streams.add(stream)

    def is_stream_capturing(self, stream):
        assert stream not in self.closed_streams
        return False

    def event_create(self):
        self.next_handle += 1
        event = 300 + self.next_handle
        # CUDA considers a newly created, unrecorded event complete.
        self.fences[event] = None
        return event

    def event_record(self, event, stream):
        assert stream not in self.closed_streams
        if self.record_error is not None:
            raise self.record_error
        self.fences[event] = (stream, self.submitted.get(stream, 0))

    def event_query(self, event):
        self.queries.append((self.context, event))
        fence = self.fences[event]
        if fence is None:
            return True
        stream, sequence = fence
        return self.completed.get(stream, 0) >= sequence

    def event_destroy(self, event):
        self.destroyed.append(event)

    def module_unload(self, module):
        self.unloaded.append((self.context, module))


def _launch_fake_loaded(lease, stream, *, capturing=False):
    artifact = _fixed_artifact()
    _cuda_backend.CUDA_BACKEND.launch(
        lease,
        artifact.contract,
        (artifact.argument_kinds, (1, 2, 3, 4)),
        grid=(1, 1, 1),
        stream=stream,
        capturing=capturing,
    )


def test_deferred_unload_waits_for_all_streams_in_same_context(monkeypatch):
    """Fence every stream in its own context, then wait for all completions."""
    from swage import _runtime

    monkeypatch.setenv("SWAGE_MEMORY_CACHE_ENTRIES", "1")
    _reset_process_runtime(_runtime)
    driver = _LifecycleDriver()
    monkeypatch.setattr(_cuda_backend, "_get_driver", lambda: driver)
    first = _fixed_artifact("first")
    lease = _cuda_backend.CUDA_BACKEND.lease(first)
    _launch_fake_loaded(lease, 0)
    _launch_fake_loaded(lease, 22)
    events = tuple(lease.entry.events.values())
    driver.complete_stream(22)
    lease.release()
    replacement = _cuda_backend.CUDA_BACKEND.lease(_fixed_artifact("second"))
    replacement.release()

    driver.context = 2
    _cuda_backend._poll_deferred(driver, 2)
    assert driver.queries == []
    driver.context = 1
    _cuda_backend._poll_deferred(driver, 1)
    assert driver.unloaded == []
    assert {event for _context, event in driver.queries} == set(events)
    driver.complete_stream(0)
    _cuda_backend._poll_deferred(driver, 1)

    assert set(driver.destroyed) == set(events)
    assert driver.unloaded == [(1, lease.entry.module)]


def test_loaded_module_lease_blocks_unload_until_final_launch(monkeypatch):
    """A retired lease may enqueue more work before its final fence."""
    from swage import _runtime

    monkeypatch.setenv("SWAGE_MEMORY_CACHE_ENTRIES", "1")
    _reset_process_runtime(_runtime)
    driver = _LifecycleDriver()
    monkeypatch.setattr(_cuda_backend, "_get_driver", lambda: driver)
    lease = _cuda_backend.CUDA_BACKEND.lease(_fixed_artifact("leased"))
    _launch_fake_loaded(lease, 0)
    event = lease.entry.events[0]
    assert driver.event_query(event) is True
    replacement = _cuda_backend.CUDA_BACKEND.lease(
        _fixed_artifact("replacement")
    )
    replacement.release()
    driver.complete_stream(0)
    _cuda_backend._poll_deferred(driver, 1)
    assert driver.unloaded == []

    _launch_fake_loaded(lease, 0)
    lease.release()
    _cuda_backend._poll_deferred(driver, 1)
    assert driver.unloaded == []
    driver.complete_stream(0)
    _cuda_backend._poll_deferred(driver, 1)
    assert driver.unloaded == [(1, lease.entry.module)]


def test_revived_module_requires_a_fence_covering_its_new_launch(monkeypatch):
    """An old completed fence must not authorize unloading revived work."""
    from swage import _runtime

    monkeypatch.setenv("SWAGE_MEMORY_CACHE_ENTRIES", "1")
    _reset_process_runtime(_runtime)
    driver = _LifecycleDriver()
    monkeypatch.setattr(_cuda_backend, "_get_driver", lambda: driver)
    artifact = _fixed_artifact("revived")
    lease = _cuda_backend.CUDA_BACKEND.lease(artifact)
    _launch_fake_loaded(lease, 0)
    event = lease.entry.events[0]
    lease.release()
    replacement = _cuda_backend.CUDA_BACKEND.lease(
        _fixed_artifact("replacement")
    )
    replacement.release()
    _cuda_backend._poll_deferred(driver, 1)
    assert driver.unloaded == []
    assert driver.event_query(event) is False

    revived = _cuda_backend.CUDA_BACKEND.lease(artifact)
    assert revived.entry is lease.entry
    driver.complete_stream(0)
    _launch_fake_loaded(revived, 0)
    # The first fence is complete even though the later launch is not.
    assert driver.event_query(event) is True
    revived.release()
    replacement = _cuda_backend.CUDA_BACKEND.lease(_fixed_artifact("third"))
    replacement.release()
    _cuda_backend._poll_deferred(driver, 1)
    assert (1, lease.entry.module) not in driver.unloaded
    assert driver.event_query(event) is False

    driver.complete_stream(0)
    _cuda_backend._poll_deferred(driver, 1)
    assert event in driver.destroyed
    assert (1, lease.entry.module) in driver.unloaded


@pytest.mark.parametrize("stream", [0, 11], ids=["legacy", "external"])
def test_enqueue_error_still_fences_possibly_submitted_work(
    monkeypatch, stream
):
    """Fence failed legacy work and retain unsafe external-stream work."""
    from swage import _runtime

    monkeypatch.setenv("SWAGE_MEMORY_CACHE_ENTRIES", "1")
    _reset_process_runtime(_runtime)
    driver = _LifecycleDriver()
    monkeypatch.setattr(_cuda_backend, "_get_driver", lambda: driver)
    lease = _cuda_backend.CUDA_BACKEND.lease(_fixed_artifact("failed-launch"))
    driver.launch_error = RuntimeError("launch failed")
    with pytest.raises(RuntimeError):
        _launch_fake_loaded(lease, stream)
    if stream:
        driver.close_stream(stream)
    lease.release()
    replacement = _cuda_backend.CUDA_BACKEND.lease(
        _fixed_artifact("replacement")
    )
    replacement.release()
    _cuda_backend._poll_deferred(driver, 1)
    assert driver.unloaded == []

    driver.complete_stream(stream)
    _cuda_backend._poll_deferred(driver, 1)
    if stream:
        assert driver.unloaded == []
    else:
        assert driver.unloaded == [(1, lease.entry.module)]


def test_module_retirement_does_not_reuse_destroyed_stream(monkeypatch):
    """A caller may destroy its stream while its recorded work completes."""
    from swage import _runtime

    monkeypatch.setenv("SWAGE_MEMORY_CACHE_ENTRIES", "1")
    _reset_process_runtime(_runtime)
    driver = _LifecycleDriver()
    monkeypatch.setattr(_cuda_backend, "_get_driver", lambda: driver)
    lease = _cuda_backend.CUDA_BACKEND.lease(_fixed_artifact("external"))
    _launch_fake_loaded(lease, 11)
    driver.close_stream(11)
    lease.release()
    replacement = _cuda_backend.CUDA_BACKEND.lease(
        _fixed_artifact("replacement")
    )
    replacement.release()
    _cuda_backend._poll_deferred(driver, 1)
    assert driver.unloaded == []

    driver.complete_stream(11)
    _cuda_backend._poll_deferred(driver, 1)
    assert driver.unloaded == [(1, lease.entry.module)]


def test_capture_pinned_module_is_never_explicitly_unloaded(monkeypatch):
    """Leave graph-captured module and event ownership to context teardown."""
    from swage import _runtime

    monkeypatch.setenv("SWAGE_MEMORY_CACHE_ENTRIES", "1")
    _reset_process_runtime(_runtime)
    driver = _LifecycleDriver()
    monkeypatch.setattr(_cuda_backend, "_get_driver", lambda: driver)
    lease = _cuda_backend.CUDA_BACKEND.lease(_fixed_artifact("captured"))
    _launch_fake_loaded(lease, 11, capturing=True)
    driver.complete_stream(11)
    lease.release()
    replacement = _cuda_backend.CUDA_BACKEND.lease(
        _fixed_artifact("replacement")
    )
    replacement.release()
    _cuda_backend._poll_deferred(driver, 1)

    assert driver.destroyed == []
    assert driver.unloaded == []


def test_event_record_failure_permanently_blocks_explicit_unload(monkeypatch):
    """Never treat an unrecorded completion event as proof of completion."""
    from swage import _runtime

    monkeypatch.setenv("SWAGE_MEMORY_CACHE_ENTRIES", "1")
    _reset_process_runtime(_runtime)
    driver = _LifecycleDriver()
    monkeypatch.setattr(_cuda_backend, "_get_driver", lambda: driver)
    lease = _cuda_backend.CUDA_BACKEND.lease(_fixed_artifact("failed-record"))
    _launch_fake_loaded(lease, 0)
    independent = _cuda_backend.CUDA_BACKEND.lease(
        _fixed_artifact("independent")
    )
    _launch_fake_loaded(independent, 0)
    independent_events = tuple(independent.entry.events.values())
    replacement = _cuda_backend.CUDA_BACKEND.lease(
        _fixed_artifact("replacement")
    )
    replacement.release()
    lease.release()
    independent.release()
    driver.record_error = RuntimeError("record failed")
    with pytest.raises(RuntimeError):
        _cuda_backend._poll_deferred(driver, 1)

    driver.record_error = None
    driver.complete_stream(0)
    _cuda_backend._poll_deferred(driver, 1)

    assert set(driver.destroyed) == set(independent_events)
    assert driver.unloaded == [(1, independent.entry.module)]


def test_stream_wrapper_cache_is_module_scoped_and_bounded():
    """Bound wrappers and release them with their injected torch module."""

    class _Stream:
        pass

    handle = [77]
    streams = {}
    torch = types.ModuleType("stream-cache-torch")
    torch._C = types.SimpleNamespace(
        _cuda_getCurrentRawStream=lambda _index: handle[0]
    )
    torch.cuda = types.SimpleNamespace(
        current_stream=lambda _index: streams.setdefault(handle[0], _Stream())
    )
    _cuda_backend._stream_objects = weakref.WeakKeyDictionary()

    first = _cuda_backend.current_stream(torch, 0)
    assert _cuda_backend.current_stream(torch, 0) is first
    for next_handle in range(78, 78 + 128):
        handle[0] = next_handle
        _cuda_backend.current_stream(torch, 0)
    per_torch = _cuda_backend._stream_objects[torch]
    assert len(per_torch) == 128
    assert (0, 77) not in per_torch

    module_reference = weakref.ref(torch)
    del torch
    gc.collect()
    assert module_reference() is None
    assert not _cuda_backend._stream_objects


def test_driver_event_unload_surface_and_not_ready_mapping():
    """Marshal event/module handles and map only CUDA error 600 to pending."""
    driver = object.__new__(_cuda_backend._CudaDriver)
    calls = []

    def create(pointer, flags):
        calls.append(("create", flags))
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_void_p))[0] = 41
        return 0

    library = types.SimpleNamespace(
        cuEventCreate=create,
        cuEventRecord=lambda event, stream: calls.append(
            ("record", event.value, stream.value)
        )
        or 0,
        cuEventQuery=mock.Mock(side_effect=[600, 0, 700]),
        cuModuleUnload=lambda module: calls.append(("unload", module.value))
        or 0,
        cuGetErrorName=lambda *_args: 0,
        cuGetErrorString=lambda *_args: 0,
    )
    driver.library = library
    driver._event_destroy = (
        lambda event: calls.append(("destroy", event.value)) or 0
    )

    event = driver.event_create()
    driver.event_record(event, 51)
    assert driver.event_query(event) is False
    assert driver.event_query(event) is True
    with pytest.raises(RuntimeError, match=r"cuEventQuery.*\(700\)"):
        driver.event_query(event)
    driver.event_destroy(event)
    driver.module_unload(61)

    assert calls == [
        ("create", 2),
        ("record", 41, 51),
        ("destroy", 41),
        ("unload", 61),
    ]


def _packaged_build_info():
    return {
        "schema_version": 1,
        "package_version": "0.5.2",
        "source_revision": "a" * 40,
        "source_clean": True,
        "llvm_version": "llvmorg-22.1.8",
        "build_type": "Release",
    }


def test_packaged_identity_precedes_checkout_and_specializes(monkeypatch):
    """Use wheel provenance, not the checkout, in artifact specialization."""
    from swage import _runtime

    info = _packaged_build_info()
    monkeypatch.setattr(_runtime._native, "build_info", lambda: info)
    monkeypatch.setattr(_runtime, "_identity_cache", None)
    monkeypatch.setattr(
        _runtime.subprocess,
        "run",
        mock.Mock(side_effect=AssertionError("packaged identity ran git")),
    )
    assert _runtime._compiler_identity() == {
        "revision": info["source_revision"],
        "clean": True,
        "llvm": info["llvm_version"],
    }

    def specialization_key():
        return _runtime._cache_key(
            _runtime._specialization_data(
                add_kernel,
                descriptors=("ptr<f32>",) * 3 + ("i32",),
                constexprs={"BLOCK": 128},
                target="sm_86",
                adapter=_cuda_backend.CUDA_BACKEND,
            )
        )

    original = specialization_key()
    info["source_revision"] = "b" * 40
    monkeypatch.setattr(_runtime, "_identity_cache", None)
    revised = specialization_key()
    assert revised != original
    info["llvm_version"] = "llvmorg-other"
    monkeypatch.setattr(_runtime, "_identity_cache", None)
    assert specialization_key() not in (original, revised)


def test_absent_packaged_identity_uses_checkout(tmp_path, monkeypatch):
    """Retain source-build identity when no packaged resource exists."""
    from swage import _runtime

    (tmp_path / ".git").mkdir()
    (tmp_path / "cmake").mkdir()
    (tmp_path / "cmake" / "llvm-version.txt").write_text("llvmorg-source\n")
    monkeypatch.setattr(
        _runtime, "__file__", str(tmp_path / "python/swage/_runtime.py")
    )
    monkeypatch.setattr(_runtime._native, "build_info", lambda: None)
    monkeypatch.setattr(
        _runtime.subprocess,
        "run",
        mock.Mock(
            side_effect=[
                types.SimpleNamespace(stdout="c" * 40 + "\n"),
                types.SimpleNamespace(stdout=""),
            ]
        ),
    )
    assert _runtime._compiler_identity() == {
        "revision": "c" * 40,
        "clean": True,
        "llvm": "llvmorg-source",
    }


@pytest.mark.parametrize("malformed", [False, True])
def test_untrusted_packaged_identity_keeps_only_process_artifacts(
    tmp_path, monkeypatch, malformed
):
    """Dirty or invalid wheel provenance permits compilation, never disk IO."""
    from swage import _runtime

    info = _packaged_build_info()
    info["source_clean"] = False

    def build_info():
        if malformed:
            raise ValueError("invalid native build metadata: source_revision")
        return info

    _reset_process_runtime(_runtime)
    monkeypatch.setenv("SWAGE_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("SWAGE_DUMP_MLIR", raising=False)
    monkeypatch.delenv("SWAGE_DUMP_PTX", raising=False)
    monkeypatch.setattr(_runtime._native, "build_info", build_info)
    monkeypatch.setattr(_runtime, "_identity_cache", None)
    monkeypatch.setattr(
        _runtime.subprocess,
        "run",
        mock.Mock(side_effect=AssertionError("untrusted identity ran git")),
    )
    monkeypatch.setattr(
        _runtime,
        "_read_cache_entry",
        mock.Mock(side_effect=AssertionError("untrusted cache read")),
    )
    compile_native = mock.Mock(
        return_value=("lowered", "ptx", _fixed_contract_json())
    )
    monkeypatch.setattr(_cuda_backend.CUDA_BACKEND, "compile", compile_native)
    first = _runtime._compile_cached(
        _cuda_backend.CUDA_BACKEND,
        _fixed_specialization(),
        "add_kernel",
        128,
        object,
    )
    second = _runtime._compile_cached(
        _cuda_backend.CUDA_BACKEND,
        _fixed_specialization(),
        "add_kernel",
        128,
        object,
    )
    assert first is second
    assert first.image == "ptx"
    assert compile_native.call_count == 1
    assert list(tmp_path.iterdir()) == []


def _assert_unavailable(error, code, backend):
    assert isinstance(error, sw.SwageError)
    assert isinstance(error, RuntimeError)
    assert error.code == code
    assert error.backend == backend
    assert isinstance(error.remediation, str) and error.remediation


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
@pytest.mark.parametrize("failure_type", [ImportError, OSError])
def test_torch_import_link_failure_is_selected_backend_unavailable(
    monkeypatch, backend, failure_type
):
    """Classify import/link failures and preserve their original cause."""
    real_import = __import__
    cause = failure_type("torch dependency failed")

    def unavailable(name, *args, **kwargs):
        if name == "torch":
            raise cause
        return real_import(name, *args, **kwargs)

    with mock.patch("builtins.__import__", side_effect=unavailable):
        with pytest.raises(sw.BackendUnavailableError) as caught:
            add_kernel.launch(
                arguments={}, constexprs={}, grid=(1,), backend=backend
            )
    _assert_unavailable(caught.value, "pytorch-unavailable", backend)
    assert caught.value.__cause__ is cause


def test_torch_initialization_bug_is_not_unavailability():
    """Do not relabel arbitrary exceptions during optional dependency import."""
    from swage import _runtime

    real_import = __import__
    cause = RuntimeError("initialization bug")

    def broken(name, *args, **kwargs):
        if name == "torch":
            raise cause
        return real_import(name, *args, **kwargs)

    with mock.patch("builtins.__import__", side_effect=broken):
        with pytest.raises(RuntimeError) as caught:
            _runtime._import_torch()
    assert caught.value is cause


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
@pytest.mark.parametrize("failure_type", [ImportError, OSError])
def test_native_compile_import_failure_is_backend_unavailable(
    backend, failure_type
):
    """Compilation requires native bindings regardless of selected backend."""
    from swage import _backends

    real_import = __import__
    cause = failure_type("native dependency missing")

    def unavailable(name, *args, **kwargs):
        if name == "mlir_swage" or name.startswith("mlir_swage."):
            raise cause
        return real_import(name, *args, **kwargs)

    with mock.patch("builtins.__import__", side_effect=unavailable):
        with pytest.raises(sw.BackendUnavailableError) as caught:
            _backends.get_backend(backend).compile(
                object(),
                "add_kernel",
                128,
                "native" if backend == "cpu" else "sm_86",
                "fixed",
                {},
            )
    _assert_unavailable(caught.value, "native-unavailable", backend)
    assert caught.value.__cause__ is cause


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_native_compiler_failure_is_not_unavailability(monkeypatch, backend):
    """Propagate compiler failures unchanged after native loading succeeds."""
    from swage import _backends, _native

    cause = RuntimeError("compiler failure")
    native = types.SimpleNamespace(
        _compile_fixed_host=mock.Mock(side_effect=cause),
        _compile_ptx=mock.Mock(side_effect=cause),
    )
    monkeypatch.setattr(_native, "load_extension", lambda **_kwargs: native)
    with pytest.raises(RuntimeError) as caught:
        _backends.get_backend(backend).compile(
            object(),
            "add_kernel",
            128,
            "native" if backend == "cpu" else "sm_86",
            "fixed",
            {},
        )
    assert caught.value is cause


def test_cuda_unavailable_never_probes_cpu(monkeypatch):
    """Unavailable explicit CUDA fails without native compilation or CPU use."""
    from swage import _cpu_backend

    torch, _ = _fake_torch(available=False)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(
        _cpu_backend.CPU_BACKEND,
        "compile",
        mock.Mock(side_effect=AssertionError("CUDA probed CPU")),
    )
    with mock.patch.object(
        add_kernel,
        "emit_mlir",
        side_effect=AssertionError("unavailable CUDA compiled"),
    ):
        with pytest.raises(sw.BackendUnavailableError) as caught:
            add_kernel.launch(
                arguments=_arguments(torch),
                constexprs={"BLOCK": 128},
                grid=(2,),
                backend="cuda",
            )
    _assert_unavailable(caught.value, "cuda-unavailable", "cuda")


def test_cuda_driver_load_failure_preserves_cause(monkeypatch):
    """Expose driver installation failure separately from CUDA build support."""
    cause = OSError("libcuda.so.1 not found")
    monkeypatch.setattr(
        _cuda_backend.ctypes, "CDLL", mock.Mock(side_effect=cause)
    )
    with pytest.raises(sw.BackendUnavailableError) as caught:
        _cuda_backend._CudaDriver()
    _assert_unavailable(caught.value, "cuda-driver-unavailable", "cuda")
    assert caught.value.__cause__ is cause


def test_missing_cuda_context_is_unavailable():
    """A successful driver query returning no context has a distinct code."""
    driver = object.__new__(_cuda_backend._CudaDriver)
    driver.library = types.SimpleNamespace(
        cuCtxGetCurrent=mock.Mock(return_value=0)
    )
    with pytest.raises(sw.BackendUnavailableError) as caught:
        driver.current_context()
    _assert_unavailable(caught.value, "cuda-context-unavailable", "cuda")


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_debug_events_report_cache_and_launch_without_payload(
    tmp_path, monkeypatch, caplog, backend
):
    """Log cache decisions and successful launch status, never runtime data."""
    from swage import _runtime

    _reset_process_runtime(_runtime)
    adapter = _RecordingBackend(backend)
    adapter.persistent_cache = backend == "cuda"
    torch, _ = _fake_torch()
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setenv("SWAGE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("SWAGE_TEST_SECRET", "secret-must-not-be-logged")
    monkeypatch.delenv("SWAGE_DUMP_MLIR", raising=False)
    monkeypatch.delenv("SWAGE_DUMP_PTX", raising=False)
    monkeypatch.setattr(_runtime, "get_backend", lambda _name: adapter)
    monkeypatch.setattr(_runtime._native, "build_info", _packaged_build_info)
    monkeypatch.setattr(_runtime, "_identity_cache", None)
    monkeypatch.setattr(
        add_kernel, "emit_mlir", lambda **_kwargs: "secret-semantic-module"
    )
    arguments = _arguments(torch, device_type=backend)
    arguments["x_ptr"]._pointer = 987654321
    caplog.set_level(logging.DEBUG, logger="swage.runtime")
    launches = 3 if backend == "cuda" else 2
    for index in range(launches):
        if index == 2:
            _runtime._artifact_cache.clear()
        add_kernel.launch(
            arguments=arguments,
            constexprs={"BLOCK": 128},
            grid=(2,),
            backend=backend,
        )

    records = [
        record for record in caplog.records if record.name == "swage.runtime"
    ]
    messages = [record.getMessage() for record in records]
    status = "launch complete" if backend == "cpu" else "launch enqueued"
    events = [message.split(" backend=", 1)[0] for message in messages]
    expected = ["compile", status, "memory-hit", status]
    if backend == "cuda":
        expected += ["persistent-hit", status]
    assert events == expected
    key = next(iter(_runtime._artifact_cache))
    for record, message in zip(records, messages):
        assert record.levelno == logging.DEBUG
        assert record.args  # Formatting remains lazy until a handler reads it.
        assert f"backend={backend}" in message
        assert f"target={'native' if backend == 'cpu' else 'sm_86'}" in message
        assert "kernel=add_kernel" in message
        assert f"key={key[:12]}" in message
        assert key not in message
        fields = message.split(" backend=", 1)[1].split()
        expected_fields = {"target", "kernel", "key"}
        if message.startswith(status):
            expected_fields.add("grid")
            assert "grid=(2,)" in message
        assert {field.split("=")[0] for field in fields[1:]} == (
            expected_fields
        )
    for secret in (
        "secret-must-not-be-logged",
        "secret-semantic-module",
        "987654321",
        str(tmp_path),
    ):
        assert secret not in "\n".join(messages)


def test_same_name_arithmetic_uses_distinct_disk_cache_entries(
    tmp_path, monkeypatch
):
    """The AST digest separates add/multiply with every other field equal."""
    from swage import _runtime

    @sw.jit
    def elementwise(x_ptr, y_ptr, output_ptr, n, BLOCK: sl.constexpr):
        offsets = sl.program_id(0) * BLOCK + sl.arange(0, BLOCK)
        mask = offsets < n
        x = sl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = sl.load(y_ptr + offsets, mask=mask, other=0.0)
        sl.store(output_ptr + offsets, x + y, mask=mask)

    addition = elementwise

    @sw.jit
    def elementwise(x_ptr, y_ptr, output_ptr, n, BLOCK: sl.constexpr):
        offsets = sl.program_id(0) * BLOCK + sl.arange(0, BLOCK)
        mask = offsets < n
        x = sl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = sl.load(y_ptr + offsets, mask=mask, other=0.0)
        sl.store(output_ptr + offsets, x * y, mask=mask)

    adapter = _RecordingBackend("cuda")
    adapter.persistent_cache = True
    monkeypatch.setenv("SWAGE_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(_runtime, "_artifact_cache", _runtime.OrderedDict())
    monkeypatch.setattr(
        _runtime,
        "_compiler_identity",
        lambda: {"revision": "a" * 40, "clean": True, "llvm": "llvmorg-test"},
    )
    specializations = [
        _runtime._specialization_data(
            kernel,
            descriptors=("ptr<f32>",) * 3 + ("i32",),
            constexprs={"BLOCK": 128},
            target="sm_86",
            adapter=adapter,
        )
        for kernel in (addition, elementwise)
    ]
    assert specializations[0]["source"] != specializations[1]["source"]
    assert {k: v for k, v in specializations[0].items() if k != "source"} == {
        k: v for k, v in specializations[1].items() if k != "source"
    }
    first = [
        _runtime._compile_cached(adapter, spec, "elementwise", 128, object)
        for spec in specializations
    ]
    assert len(adapter.calls) == 2
    assert first[0].key != first[1].key
    assert len(list(tmp_path.glob("*/metadata.json"))) == 2
    _runtime._artifact_cache.clear()
    monkeypatch.setattr(
        adapter,
        "compile",
        mock.Mock(side_effect=AssertionError("disk hit recompiled")),
    )
    for spec, expected in zip(reversed(specializations), reversed(first)):
        actual = _runtime._compile_cached(
            adapter, spec, "elementwise", 128, object
        )
        assert actual == expected
