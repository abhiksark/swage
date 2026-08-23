# tests/python/test_runtime.py
"""LLVM-free tests for the M3 CUDA launch boundary."""

import ctypes
import gc
import json
import pathlib
import stat
import sys
import types
import weakref
from unittest import mock

import pytest
import swage as sw
import swage.language as sl


@sw.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n, BLOCK: sl.constexpr):  # noqa: D103
    pid = sl.program_id(0)
    offsets = pid * BLOCK + sl.arange(0, BLOCK)
    mask = offsets < n
    x = sl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = sl.load(y_ptr + offsets, mask=mask, other=0.0)
    sl.store(output_ptr + offsets, x + y, mask=mask)


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

    def current_context(self):
        return 0xCAFE

    def load(self, ptx, kernel_name):
        self.loads.append((ptx, kernel_name))
        return 0xBEEF, 0xF00D

    def launch(self, function, grid, block, stream, arguments):
        self.launches.append((function, grid, block, stream, arguments))


def _fake_torch(*, available=True, current_device=0, max_threads=1024):
    torch = types.ModuleType("torch")
    torch.float32 = object()
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
        lambda *_args, **_kwargs: _runtime._Artifact(
            key="cache-key", lowered="lowered", ptx="ptx"
        ),
    )
    monkeypatch.setattr(_runtime, "_get_driver", lambda: driver)
    _runtime._loaded_functions.clear()
    return driver


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
        (0xF00D, (2,), 128, 0xABCD, (0x1000, 0x2000, 0x3000, 129))
    ]
    for tensor in tuple(arguments.values())[:3]:
        assert tensor.recorded_streams == [stream]


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


def test_launch_requires_pytorch_and_cuda(monkeypatch):
    """Report missing optional dependencies before importing native code."""
    from swage import _runtime

    real_import = __import__

    def import_without_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    with mock.patch("builtins.__import__", side_effect=import_without_torch):
        with pytest.raises(RuntimeError, match="requires PyTorch"):
            _runtime._import_torch()

    torch, _ = _fake_torch(available=False)
    monkeypatch.setitem(sys.modules, "torch", torch)
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        add_kernel.launch(
            arguments=_arguments(torch),
            constexprs={"BLOCK": 128},
            grid=(2,),
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
        _runtime,
        "_compile_native",
        lambda *_args: calls.append(True) or ("lowered", "ptx"),
    )
    key_data = {"kernel": "add_kernel", "target": "sm_86"}

    first = _runtime._compile_cached(object(), key_data, "add_kernel", 128)
    _runtime._ptx_cache.clear()
    second = _runtime._compile_cached(object(), key_data, "add_kernel", 128)

    assert first == second
    assert len(calls) == 1
    entry = tmp_path / first.key
    assert stat.S_IMODE(entry.stat().st_mode) == 0o700
    assert stat.S_IMODE((entry / "kernel.ptx").stat().st_mode) == 0o600
    (entry / "kernel.ptx").write_text("corrupt")
    _runtime._ptx_cache.clear()
    with pytest.raises(RuntimeError, match="digest mismatch"):
        _runtime._compile_cached(object(), key_data, "add_kernel", 128)


def test_cache_rejects_unsafe_entries(tmp_path, monkeypatch):
    """Reject symlinked and world-writable cache content."""
    from swage import _runtime

    monkeypatch.setenv("SWAGE_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(
        _runtime,
        "_compiler_identity",
        lambda: {"revision": "abc", "clean": True, "llvm": "llvmorg-test"},
    )
    key_data = {"kernel": "add_kernel"}
    key = _runtime._cache_key(key_data)
    entry = tmp_path / key
    entry.mkdir()
    target = tmp_path / "outside.json"
    target.write_text(json.dumps({}))
    (entry / "metadata.json").symlink_to(target)

    with pytest.raises(RuntimeError, match="symlink"):
        _runtime._compile_cached(object(), key_data, "add_kernel", 128)

    (entry / "metadata.json").unlink()
    (entry / "metadata.json").write_text("{}")
    (entry / "metadata.json").chmod(0o606)
    with pytest.raises(RuntimeError, match="world-writable"):
        _runtime._compile_cached(object(), key_data, "add_kernel", 128)


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
        _runtime,
        "_compile_native",
        lambda *_args: calls.append(True) or ("lowered", "ptx"),
    )

    first = _runtime._compile_cached(object(), {"kernel": "add"}, "add", 128)
    second = _runtime._compile_cached(object(), {"kernel": "add"}, "add", 128)

    assert first == second
    assert len(calls) == 1
    assert list(tmp_path.iterdir()) == []


def test_dump_switches_write_requested_artifacts(tmp_path, monkeypatch):
    """Write deterministic debug artifacts only when explicitly requested."""
    from swage import _runtime

    monkeypatch.setenv("SWAGE_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("SWAGE_DUMP_MLIR", "1")
    monkeypatch.setenv("SWAGE_DUMP_PTX", "1")
    artifact = _runtime._Artifact("key", "lowered", "ptx")

    _runtime._write_dumps(artifact)

    assert (tmp_path / "key.mlir").read_text() == "lowered"
    assert (tmp_path / "key.ptx").read_text() == "ptx"


def test_compiler_key_contains_every_specialization_input(monkeypatch):
    """Keep cache identity complete and deterministic."""
    from swage import _runtime

    monkeypatch.setattr(
        _runtime,
        "_compiler_identity",
        lambda: {"revision": "abc", "clean": True, "llvm": "llvmorg-test"},
    )
    data = _runtime._specialization_data(
        add_kernel,
        descriptors=("ptr<f32>", "ptr<f32>", "ptr<f32>", "i32"),
        constexprs={"BLOCK": 128},
        target="sm_86",
    )

    assert data == {
        "source": mock.ANY,
        "kernel": "add_kernel",
        "descriptors": ["ptr<f32>", "ptr<f32>", "ptr<f32>", "i32"],
        "constexprs": [["BLOCK", 128]],
        "compute_capability": "sm_86",
        "codegen": {"block_size": 128, "index_bits": 64},
        "swage_revision": "abc",
        "dialect_version": 1,
        "llvm_version": "llvmorg-test",
    }
    assert len(data["source"]) == 64
    assert _runtime._cache_key(data) == _runtime._cache_key(data)


def test_cache_path_defaults_to_user_cache(monkeypatch):
    """Keep cache placement predictable without creating it during import."""
    from swage import _runtime

    monkeypatch.delenv("SWAGE_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", "/tmp/user-cache")
    assert _runtime._cache_dir() == pathlib.Path("/tmp/user-cache/swage")


def test_driver_marshals_pointer_and_i32_parameters():
    """Pass raw pointers and an i32 through the CUDA Driver ABI."""
    from swage import _runtime

    driver = object.__new__(_runtime._CudaDriver)
    calls = []
    driver._call = lambda name, *args: calls.append((name, args))

    driver.launch(0xF00D, (2,), 128, 0xABCD, (0x10, 0x20, 0x30, 129))

    name, call = calls[0]
    parameters = call[-2]
    pointer_values = [
        ctypes.cast(parameters[index], ctypes.POINTER(ctypes.c_void_p))
        .contents.value
        for index in range(3)
    ]
    scalar = ctypes.cast(
        parameters[3], ctypes.POINTER(ctypes.c_int32)
    ).contents.value
    assert name == "cuLaunchKernel"
    assert pointer_values == [0x10, 0x20, 0x30]
    assert scalar == 129


def test_driver_marshals_four_pointer_segmented_task_abi():
    """Pass four pointers and two i32 counts through the CUDA Driver ABI."""
    from swage import _runtime

    driver = object.__new__(_runtime._CudaDriver)
    calls = []
    driver._call = lambda name, *args: calls.append((name, args))

    driver.launch_segmented_tasks(
        0xF00D,
        (7,),
        32,
        0xABCD,
        (0x10, 0x20, 0x30, 0x40, 4096, 7),
    )

    name, call = calls[0]
    parameters = call[-2]
    pointer_values = [
        ctypes.cast(parameters[index], ctypes.POINTER(ctypes.c_void_p))
        .contents.value
        for index in range(4)
    ]
    scalar_values = [
        ctypes.cast(parameters[index], ctypes.POINTER(ctypes.c_int32))
        .contents.value
        for index in range(4, 6)
    ]
    assert name == "cuLaunchKernel"
    assert pointer_values == [0x10, 0x20, 0x30, 0x40]
    assert scalar_values == [4096, 7]


def test_driver_error_contains_stable_name_code_and_text():
    """Preserve actionable CUDA Driver diagnostics."""
    from swage import _runtime

    def set_text(_result, output, value):
        ctypes.cast(output, ctypes.POINTER(ctypes.c_char_p))[0] = value
        return 0

    driver = object.__new__(_runtime._CudaDriver)
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
