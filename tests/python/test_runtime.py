# tests/python/test_runtime.py
"""LLVM-free tests for the fixed elementwise launch and its runtime."""

import ctypes
import gc
import hashlib
import json
import logging
import multiprocessing
import os
import pathlib
import re
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time
import types
import warnings
import weakref
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import pytest
import swage as sw
import swage.language as sl
from swage import _abi, _cuda_backend


@sw.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n, BLOCK: sl.constexpr):  # noqa: D103
    pid = sl.program_id(0)
    offsets = pid * BLOCK + sl.arange(0, BLOCK)
    mask = offsets < n
    x = sl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = sl.load(y_ptr + offsets, mask=mask, other=0.0)
    sl.store(output_ptr + offsets, x + y, mask=mask)


@sw.jit
def multiply_kernel(x_ptr, y_ptr, output_ptr, n, BLOCK: sl.constexpr):
    """Canonical vector multiply used to exercise launch admission."""
    pid = sl.program_id(0)
    offsets = pid * BLOCK + sl.arange(0, BLOCK)
    mask = offsets < n
    x = sl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = sl.load(y_ptr + offsets, mask=mask, other=0.0)
    sl.store(output_ptr + offsets, x * y, mask=mask)


@sw.jit
def renamed_kernel(left, right, destination, length, TILE: sl.constexpr):
    """Canonical vector add with arbitrary diagnostic parameter labels."""
    pid = sl.program_id(0)
    offsets = pid * TILE + sl.arange(0, TILE)
    mask = offsets < length
    x = sl.load(left + offsets, mask=mask, other=0.0)
    y = sl.load(right + offsets, mask=mask, other=0.0)
    sl.store(destination + offsets, x + y, mask=mask)


@sw.jit
def defaulted_kernel(  # noqa: D103
    x_ptr, y_ptr, output_ptr, n, BLOCK: sl.constexpr = 128
):
    pid = sl.program_id(0)
    offsets = pid * BLOCK + sl.arange(0, BLOCK)
    mask = offsets < n
    x = sl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = sl.load(y_ptr + offsets, mask=mask, other=0.0)
    sl.store(output_ptr + offsets, x + y, mask=mask)


@sw.jit
def annotated_count_kernel(  # noqa: D103
    x_ptr, y_ptr, output_ptr, n: int, BLOCK: sl.constexpr
):
    pid = sl.program_id(0)
    offsets = pid * BLOCK + sl.arange(0, BLOCK)
    mask = offsets < n
    x = sl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = sl.load(y_ptr + offsets, mask=mask, other=0.0)
    sl.store(output_ptr + offsets, x + y, mask=mask)


@sw.jit
def annotated_pointer_kernel(  # noqa: D103
    x_ptr: float, y_ptr, output_ptr, n, BLOCK: sl.constexpr
):
    pid = sl.program_id(0)
    offsets = pid * BLOCK + sl.arange(0, BLOCK)
    mask = offsets < n
    x = sl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = sl.load(y_ptr + offsets, mask=mask, other=0.0)
    sl.store(output_ptr + offsets, x + y, mask=mask)


@sw.jit
def annotated_block_kernel(  # noqa: D103
    x_ptr, y_ptr, output_ptr, n, BLOCK: int
):
    pid = sl.program_id(0)
    offsets = pid * BLOCK + sl.arange(0, BLOCK)
    mask = offsets < n
    x = sl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = sl.load(y_ptr + offsets, mask=mask, other=0.0)
    sl.store(output_ptr + offsets, x + y, mask=mask)


_IMPORTED_NS = time.time_ns()


def _user(kind, index, access=None):
    """Return a contract argument bound to source parameter `index`."""
    return _abi.KernelArgument(kind, "user", source_index=index, access=access)


def _keyed(kind, origin, key, access=None):
    """Return a derived, plan, or scratch contract argument named `key`."""
    return _abi.KernelArgument(kind, origin, key=key, access=access)


# The fixed elementwise kernel: two inputs, the output, and the count.
_FIXED_ARGUMENTS = (
    _user("ptr", 0, "read"),
    _user("ptr", 1, "read"),
    _user("ptr", 2, "write"),
    _user("i32", 3),
)
# A segmented kernel binds the parameters of its segment function by
# position (values, offsets, output, value_count, segment_count), and its
# plan records, plan counts, and scratch buffers by key.
_SEGMENT_BUFFERS = (
    _user("ptr", 0, "read"),
    _user("ptr", 1, "read"),
    _user("ptr", 2, "write"),
)
_DIRECT_ARGUMENTS = (*_SEGMENT_BUFFERS, _user("i32", 3), _user("i32", 4))
_TASK_ARGUMENTS = (
    *_SEGMENT_BUFFERS,
    _keyed("ptr", "plan", "task_ids", "read"),
    _user("i32", 3),
    _keyed("i32", "derived", "task_count"),
    _user("i32", 4),
)
_FUSED_ARGUMENTS = (
    *_SEGMENT_BUFFERS,
    _keyed("ptr", "plan", "task_ids", "read"),
    _user("i32", 3),
    _keyed("i32", "derived", "warp_task_count"),
    _keyed("i32", "derived", "cta_task_count"),
    _user("i32", 4),
)
_PERSISTENT_ARGUMENTS = (
    *_SEGMENT_BUFFERS,
    *(
        _keyed("ptr", "plan", key, "read")
        for key in (
            "warp_ids",
            "cta_ids",
            "partial_ranges",
            "partial_merge_ids",
            "merge_records",
        )
    ),
    _keyed("ptr", "scratch", "scratch", "readwrite"),
    _keyed("ptr", "scratch", "counters", "readwrite"),
    _user("i32", 3),
    *(
        _keyed("i32", "derived", key)
        for key in (
            "warp_task_count",
            "cta_task_count",
            "partial_count",
            "merge_count",
        )
    ),
    _user("i32", 4),
)


def _contract(
    entry="add_kernel", block=128, arguments=_FIXED_ARGUMENTS, backend="cuda"
):
    """Return the launch contract a compiler gives one kernel."""
    if backend == "cuda":
        launch = _abi.KernelLaunch("spmd-grid", (block, 1, 1))
    else:
        launch = _abi.KernelLaunch("host-call")
    return _abi.KernelContract(
        _abi._VERSION, backend, entry, launch, tuple(arguments)
    )


def _contract_json(entry="add_kernel", block=128, **options):
    """Return the canonical JSON of `_contract(entry, block, **options)`."""
    return _abi.serialize_kernel_contract(_contract(entry, block, **options))


def _spec(kernel="add_kernel", block=128, **fields):
    """Return a fixed CUDA specialization of `kernel`, plus `fields`."""
    return {
        "kernel": kernel,
        "backend": "cuda",
        "format": "ptx",
        "target": "sm_86",
        "descriptors": ["ptr<f32>", "ptr<f32>", "ptr<f32>", "i32"],
        "codegen": {"lowering": "fixed", "block_size": block, "options": []},
        **fields,
    }


_SHARED_KEY = _spec()


def _artifact(
    key="cache-key",
    kernel_name="add_kernel",
    block=128,
    *,
    image="ptx",
    lowered="lowered",
):
    """Return the verified CUDA artifact a fixed compile would produce."""
    from swage import _runtime

    contract_json = _contract_json(kernel_name, block)
    return _runtime._make_artifact(
        key,
        _cuda_backend.CUDA_BACKEND,
        "sm_86",
        lowered,
        image,
        contract_json,
        _abi.parse_kernel_contract(contract_json),
    )


def _fixed_compile_artifact(
    _adapter, _specialization, kernel_name, block_size, *_args, **_kwargs
):
    """Stand in for `_runtime._compile_cached` without any cache."""
    return _artifact(kernel_name=kernel_name, block=block_size)


def _compiler(calls=None, *, ptx="ptx", lowered="lowered"):
    """Return a stand-in for `_cuda_backend._compile_native`.

    It answers every compile with `lowered`, `ptx`, and the canonical fixed
    contract of the requested kernel and block, and records it in `calls`.
    """

    def compile_native(_module, kernel_name, block_size, *_options):
        if calls is not None:
            calls.append(True)
        return lowered, ptx, _contract_json(kernel_name, block_size)

    return compile_native


def _compile(specialization=None, emit=object, **options):
    """Compile one fixed CUDA specialization through both caches."""
    from swage import _runtime

    if specialization is None:
        specialization = _SHARED_KEY
    return _runtime._compile_cached(
        _cuda_backend.CUDA_BACKEND,
        specialization,
        specialization["kernel"],
        specialization["codegen"]["block_size"],
        emit,
        **options,
    )


def _identity(**overrides):
    """Return a fully identified compiler, adjusted by `overrides`."""
    identity = {
        "revision": "abc",
        "clean": True,
        "llvm": "llvmorg-test",
        "frontend": "f" * 64,
        "native": [["_swageDialectsNanobind.so", 1, 2]],
    }
    identity.update(overrides)
    return identity


def _unpersisted_identity():
    """Return an identity without native libraries, which nothing persists."""
    return _identity(revision=None, clean=False, llvm=None, native=None)


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """Keep every test away from the user's cache and from earlier tests.

    Each test also starts with empty process caches of artifacts and of
    loaded modules, whose bounds are read again, and leaks neither.
    """
    from swage import _runtime

    monkeypatch.setenv("SWAGE_CACHE_DIR", str(tmp_path / "isolated-cache"))
    monkeypatch.setattr(_runtime, "_cache_off", {}, raising=False)
    monkeypatch.setattr(_runtime, "_artifact_cache", OrderedDict())
    monkeypatch.setattr(_runtime, "_memory_cache_entries", None)
    monkeypatch.setattr(_cuda_backend, "_loaded_functions", OrderedDict())
    monkeypatch.setattr(_cuda_backend, "_retired_loaded", {})
    monkeypatch.setattr(_cuda_backend, "_memory_cache_entries", None)
    for name in (
        "SWAGE_ARTIFACT_DIR",
        "SWAGE_CACHE_MAX_ENTRIES",
        "SWAGE_CACHE_READ_ONLY",
        "SWAGE_DUMP_MLIR",
        "SWAGE_DUMP_PTX",
        "SWAGE_MEMORY_CACHE_ENTRIES",
        "SWAGE_NO_COMPILE",
    ):
        monkeypatch.delenv(name, raising=False)


def _use_shared_cache(cache_dir):
    """Point a spawned process at one cache with a stubbed identity."""
    from swage import _runtime

    os.environ["SWAGE_CACHE_DIR"] = cache_dir
    _runtime._compiler_identity = _identity
    _runtime._stale_identity = lambda _identity: None
    return _runtime


def _cold_start(cache_dir, results_dir, barrier, rounds):
    """Cold-start one key per round, in step with the other processes."""
    _use_shared_cache(cache_dir)
    _cuda_backend._compile_native = _compiler(
        ptx=f"ptx from process {os.getpid()}"
    )
    try:
        for round_id in range(rounds):
            barrier.wait(timeout=60)
            artifact = _compile(dict(_SHARED_KEY, round=round_id))
            result = pathlib.Path(results_dir) / f"{round_id}-{os.getpid()}"
            result.write_text(artifact.image)
    except BaseException:
        barrier.abort()  # Release the other processes instead of timing out.
        raise


def _churn(cache_dir, barrier, rounds, bound):
    """Publish a contended key and a private key per round, bounded."""
    os.environ["SWAGE_CACHE_MAX_ENTRIES"] = str(bound)
    _use_shared_cache(cache_dir)
    _cuda_backend._compile_native = _compiler()
    # A cache that gives up warns, and here a warning fails the process.
    warnings.simplefilter("error")
    try:
        for round_id in range(rounds):
            barrier.wait(timeout=60)
            for owner in ("every process", os.getpid()):
                artifact = _compile(
                    dict(_SHARED_KEY, round=round_id, owner=owner)
                )
                assert artifact.image == "ptx"
    except BaseException:
        barrier.abort()  # Release the other processes instead of timing out.
        raise


def _stop(workers):
    """Terminate and reap every worker that is still running.

    A worker that outlives its test would otherwise block interpreter exit.
    """
    for worker in workers:
        if worker.is_alive():
            worker.terminate()
    for worker in workers:
        if worker.pid is not None:
            worker.join(timeout=30)
            if worker.is_alive():
                worker.kill()
                worker.join()


def _hang(_seconds):
    """Sleep long enough to outlive any test that forgets to stop it."""
    time.sleep(_seconds)


def _killed_writer(cache_dir):
    """Die from SIGKILL after writing one of the three entry files."""
    _runtime = _use_shared_cache(cache_dir)
    _cuda_backend._compile_native = _compiler()
    write = _runtime._atomic_write

    def write_then_die(path, contents):
        write(path, contents)
        os.kill(os.getpid(), signal.SIGKILL)

    _runtime._atomic_write = write_then_die
    _compile()


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
        negative=False,
        conjugate=False,
        requires_grad=False,
    ):
        self.layout = torch.strided
        self.dtype = torch.float32 if dtype is None else dtype
        self.device = _Device(device_type, device_index)
        self._size = size
        self._rank = rank
        self._contiguous = contiguous
        self._pointer = pointer
        self._negative = negative
        self._conjugate = conjugate
        self.requires_grad = requires_grad
        self._element_size = 1
        if self.dtype is torch.float32:
            self._element_size = 4
        elif self.dtype is getattr(torch, "float16", None):
            self._element_size = 2
        self.recorded_streams = []

    def dim(self):
        return self._rank

    def is_contiguous(self):
        return self._contiguous

    def is_neg(self):
        return self._negative

    def is_conj(self):
        return self._conjugate

    def numel(self):
        return self._size

    def element_size(self):
        return self._element_size

    def data_ptr(self):
        return self._pointer

    def record_stream(self, stream):
        self.recorded_streams.append(stream)


class _Driver:
    """A CUDA driver that records loads and launches and completes all."""

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
    """A backend adapter that records every call it receives."""

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
        if self.name == "cuda":
            return "lowered", "ptx", _contract_json(kernel_name, block_size)
        return "lowered", object(), _contract_json(kernel_name, backend="cpu")

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
    *,
    available=True,
    current_device=0,
    max_threads=1024,
    version="2.6.0",
    low_precision=False,
):
    torch = types.ModuleType("torch")
    torch.__version__ = version
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
    # Each entry is a tensor whose version a launch advanced, with the
    # streams that had retained it by then.
    torch.advanced = []
    torch.autograd = types.SimpleNamespace(
        graph=types.SimpleNamespace(
            increment_version=lambda tensor: torch.advanced.append(
                (tensor, list(tensor.recorded_streams))
            )
        )
    )
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
    monkeypatch.setattr(_runtime, "_compile_cached", _fixed_compile_artifact)
    monkeypatch.setattr(_cuda_backend, "_get_driver", lambda: driver)
    return driver


def _install_recording_backends(monkeypatch, torch, *adapters):
    """Route launches to `adapters` by name, with nothing persisted.

    A CUDA launch makes a context current before it leases its module, so
    a fake driver answers for the context.
    """
    from swage import _runtime

    by_name = {adapter.name: adapter for adapter in adapters}
    driver = _Driver()
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(_runtime, "get_backend", by_name.__getitem__)
    monkeypatch.setattr(_runtime, "_compiler_identity", _unpersisted_identity)
    monkeypatch.setattr(_cuda_backend, "_get_driver", lambda: driver)
    _runtime._identity_cache = None
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
    adapters = {name: _RecordingBackend(name) for name in ("cuda", "cpu")}
    torch, _ = _fake_torch()
    _install_recording_backends(monkeypatch, torch, *adapters.values())
    monkeypatch.setattr(add_kernel, "emit_mlir", lambda **_kwargs: object())
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
    _install_recording_backends(monkeypatch, torch, *adapters.values())
    monkeypatch.setattr(add_kernel, "emit_mlir", lambda **_kwargs: object())
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


def test_launch_advances_the_version_of_the_output_only(monkeypatch):
    """Tell autograd the output was written, after it is retained."""
    torch, stream = _fake_torch()
    arguments = _arguments(torch)
    _install_launch_fakes(monkeypatch, torch)

    for _ in range(2):
        add_kernel.launch(
            arguments=arguments, constexprs={"BLOCK": 128}, grid=(2,)
        )

    output = arguments["output_ptr"]
    assert torch.advanced == [(output, [stream]), (output, [stream] * 2)]


def test_launch_without_work_advances_no_version(monkeypatch):
    """Leave the counter alone when nothing is enqueued or written."""
    torch, _ = _fake_torch()
    driver = _install_launch_fakes(monkeypatch, torch)

    add_kernel.launch(
        arguments=_arguments(torch, n=0), constexprs={"BLOCK": 128}, grid=(0,)
    )
    with pytest.raises(ValueError, match="grid must equal"):
        add_kernel.launch(
            arguments=_arguments(torch), constexprs={"BLOCK": 128}, grid=(3,)
        )

    assert driver.launches == []
    assert torch.advanced == []


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


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
@pytest.mark.parametrize("warm", [False, True], ids=["cold", "warm"])
@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("arguments-type", "mapping"),
        ("argument-count", "arguments must contain exactly"),
        ("constexpr-count", "constexprs must contain exactly"),
        ("zero-block", "positive integer"),
        ("negative-block", "positive integer"),
        ("boolean-block", "positive integer"),
        ("oversized-block", "1024|device limit"),
        ("grid-type", "one-element tuple"),
        ("boolean-grid", "one-element tuple"),
        ("negative-grid", "grid must equal"),
        ("multidimensional-grid", "one-element tuple"),
        ("grid-count", "grid must equal"),
        ("boolean-count", "nonnegative i32"),
        ("floating-count", "nonnegative i32"),
        ("negative-count", "nonnegative i32"),
        ("oversized-count", "nonnegative i32"),
        ("length", "exceeds tensor length"),
        ("rank", "rank one"),
        ("stride", "contiguous"),
        ("dtype", "must have dtype"),
        ("mixed-dtype", "same dtype"),
        ("device", "tensor"),
        ("zero-rank", "rank one"),
        ("zero-stride", "contiguous"),
        ("zero-dtype", "must have dtype"),
        ("zero-device", "tensor"),
    ],
)
def test_multiply_rejects_invalid_launches_before_backend_work(
    monkeypatch, backend, warm, case, reason
):
    """Apply multiplication admission before cold or cached backend work."""
    torch, _ = _fake_torch()
    adapter = _RecordingBackend(backend)
    _install_recording_backends(monkeypatch, torch, adapter)
    monkeypatch.setattr(
        multiply_kernel, "emit_mlir", lambda **_kwargs: object()
    )
    multiply_kernel.__dict__.pop("_specialization_memo", None)
    if warm:
        multiply_kernel.launch(
            arguments=_arguments(torch, n=1, device_type=backend),
            constexprs={"BLOCK": 128},
            grid=(1,),
            backend=backend,
        )
    calls_before = list(adapter.calls)
    arguments = _arguments(torch, n=1, device_type=backend)
    constexprs = {"BLOCK": 128}
    grid = (1,)
    if case == "arguments-type":
        arguments = []
    elif case == "argument-count":
        arguments.pop("y_ptr")
    elif case == "constexpr-count":
        constexprs["EXTRA"] = 1
    elif case == "zero-block":
        constexprs["BLOCK"] = 0
    elif case == "negative-block":
        constexprs["BLOCK"] = -1
    elif case == "boolean-block":
        constexprs["BLOCK"] = True
    elif case == "oversized-block":
        constexprs["BLOCK"] = 2048
        grid = (1,)
    elif case == "grid-type":
        grid = [1]
    elif case == "boolean-grid":
        grid = (True,)
    elif case == "negative-grid":
        grid = (-1,)
    elif case == "multidimensional-grid":
        grid = (1, 1)
    elif case == "grid-count":
        grid = (2,)
    elif case == "boolean-count":
        arguments["n"] = True
    elif case == "floating-count":
        arguments["n"] = 1.0
    elif case == "negative-count":
        arguments["n"] = -1
    elif case == "oversized-count":
        arguments["n"] = 1 << 31
    elif case == "length":
        arguments["x_ptr"] = _Tensor(torch, size=0, device_type=backend)
    elif case == "rank":
        arguments["y_ptr"] = _Tensor(torch, rank=2, device_type=backend)
    elif case == "stride":
        arguments["output_ptr"] = _Tensor(
            torch, contiguous=False, device_type=backend
        )
    elif case == "dtype":
        arguments["x_ptr"] = _Tensor(torch, dtype=object(), device_type=backend)
    elif case == "mixed-dtype":
        torch.float16 = object()
        arguments["y_ptr"] = _Tensor(
            torch, dtype=torch.float16, device_type=backend
        )
    elif case == "device":
        wrong_device = "cpu" if backend == "cuda" else "cuda"
        arguments["x_ptr"] = _Tensor(torch, device_type=wrong_device)
    elif case.startswith("zero-"):
        arguments = _arguments(torch, n=0, size=0, device_type=backend)
        grid = (0,)
        parameter = {
            "zero-rank": "x_ptr",
            "zero-stride": "y_ptr",
            "zero-dtype": "output_ptr",
            "zero-device": "x_ptr",
        }[case]
        overrides = {"size": 0, "device_type": backend}
        if case == "zero-rank":
            overrides["rank"] = 2
        elif case == "zero-stride":
            overrides["contiguous"] = False
        elif case == "zero-dtype":
            overrides["dtype"] = object()
        else:
            overrides["device_type"] = "cpu" if backend == "cuda" else "cuda"
        arguments[parameter] = _Tensor(torch, **overrides)

    with pytest.raises((TypeError, ValueError), match=reason):
        multiply_kernel.launch(
            arguments=arguments,
            constexprs=constexprs,
            grid=grid,
            backend=backend,
        )

    assert adapter.calls == calls_before


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_empty_multiply_is_a_validated_noop_for_every_backend(
    monkeypatch, backend
):
    """Validate zero work without compilation, pointer access, or writes."""
    torch, _ = _fake_torch()
    adapter = _RecordingBackend(backend)
    _install_recording_backends(monkeypatch, torch, adapter)
    monkeypatch.setattr(
        _Tensor,
        "data_ptr",
        lambda _self: pytest.fail("zero work acquired a raw pointer"),
    )
    with mock.patch.object(multiply_kernel, "emit_mlir") as emit:
        multiply_kernel.launch(
            arguments=_arguments(torch, n=0, size=0, device_type=backend),
            constexprs={"BLOCK": 128},
            grid=(0,),
            backend=backend,
        )

    emit.assert_not_called()
    assert adapter.calls == []
    assert torch.advanced == []


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


@pytest.mark.parametrize("flag", ["_negative", "_conjugate"])
def test_lazy_metadata_fails_before_any_pointer_access(monkeypatch, flag):
    """Reject unresolved metadata in every position before pointer access."""
    from swage import _runtime

    torch, _ = _fake_torch()
    monkeypatch.setattr(
        _Tensor, "data_ptr", lambda _self: pytest.fail("pointer read")
    )
    for name in ("x_ptr", "y_ptr", "output_ptr"):
        for n in (0, 1):
            arguments = _arguments(torch, n=n)
            setattr(arguments[name], flag, True)
            with pytest.raises(
                ValueError, match=f"{name}.*(negation|conjugate)"
            ):
                _runtime._validate_runtime_arguments(
                    arguments, tuple(arguments), torch, "cuda"
                )


@pytest.mark.parametrize(
    ("view", "reason"),
    [
        ({"negative": True}, "lazy negation view"),
        ({"conjugate": True}, "lazy conjugate view"),
    ],
)
@pytest.mark.parametrize("name", ["x_ptr", "y_ptr", "output_ptr"])
def test_launch_rejects_lazy_views(monkeypatch, name, view, reason):
    """Reject a view whose storage does not hold the values it shows."""
    torch, _ = _fake_torch()
    driver = _install_launch_fakes(monkeypatch, torch)
    arguments = _arguments(torch)
    arguments[name] = _Tensor(torch, pointer=arguments[name].data_ptr(), **view)

    with pytest.raises(ValueError, match=f"'{name}' must not be a {reason}"):
        add_kernel.launch(
            arguments=arguments, constexprs={"BLOCK": 128}, grid=(2,)
        )

    assert driver.loads == driver.launches == []


@pytest.mark.parametrize("name", ["x_ptr", "y_ptr", "output_ptr"])
def test_launch_rejects_tensors_that_require_grad(monkeypatch, name):
    """Reject a tensor autograd tracks, because a launch records nothing."""
    torch, _ = _fake_torch()
    driver = _install_launch_fakes(monkeypatch, torch)
    arguments = _arguments(torch)
    arguments[name] = _Tensor(
        torch, pointer=arguments[name].data_ptr(), requires_grad=True
    )

    with pytest.raises(
        ValueError,
        match=f"'{name}' must not require grad.*pass tensor.detach\\(\\)",
    ):
        add_kernel.launch(
            arguments=arguments, constexprs={"BLOCK": 128}, grid=(2,)
        )

    assert driver.loads == driver.launches == []


# A launch of 129 four-byte elements covers 0x204 bytes of each tensor.
@pytest.mark.parametrize(
    ("pointers", "overlapped"),
    [
        ({"output_ptr": 0x1004}, "x_ptr"),
        ({"output_ptr": 0x1000 - 0x200}, "x_ptr"),
        ({"output_ptr": 0x1000 + 0x200}, "x_ptr"),
        ({"output_ptr": 0x2000 - 4}, "y_ptr"),
        ({"x_ptr": 0x3000 + 0x100}, "x_ptr"),
    ],
)
def test_launch_rejects_an_output_that_overlaps_an_input(
    monkeypatch, pointers, overlapped
):
    """Reject an output that shares only part of an input's active range."""
    torch, _ = _fake_torch()
    driver = _install_launch_fakes(monkeypatch, torch)
    arguments = _arguments(torch)
    for name, pointer in pointers.items():
        arguments[name] = _Tensor(torch, pointer=pointer)

    with pytest.raises(
        ValueError,
        match=(
            f"argument 'output_ptr' must not partially overlap argument "
            f"'{overlapped}' in their active ranges"
        ),
    ):
        add_kernel.launch(
            arguments=arguments, constexprs={"BLOCK": 128}, grid=(2,)
        )

    assert driver.loads == driver.launches == []


@pytest.mark.parametrize(
    "pointers",
    [
        {"output_ptr": 0x1000 + 0x204},
        {"output_ptr": 0x1000 - 0x204},
        {"output_ptr": 0x1000},
        {"output_ptr": 0x2000},
        {"y_ptr": 0x1000},
        {"y_ptr": 0x1004},
    ],
    ids=[
        "after-x",
        "before-x",
        "output-is-x",
        "output-is-y",
        "same-inputs",
        "overlapping-inputs",
    ],
)
def test_launch_accepts_adjacent_aliased_and_overlapping_inputs(
    monkeypatch, pointers
):
    """Allow touching buffers, an output that is an input, shared inputs.

    Every lane reads and writes one element index, so an output exactly
    equal to an input is safe, and only the output is written.
    """
    torch, stream = _fake_torch()
    driver = _install_launch_fakes(monkeypatch, torch)
    arguments = _arguments(torch)
    for name, pointer in pointers.items():
        arguments[name] = _Tensor(torch, pointer=pointer)

    add_kernel.launch(arguments=arguments, constexprs={"BLOCK": 128}, grid=(2,))

    assert len(driver.launches) == 1
    assert torch.advanced == [(arguments["output_ptr"], [stream])]


def test_empty_launch_has_no_active_range_to_overlap(monkeypatch):
    """Admit any placement at zero work, where no element is active."""
    torch, _ = _fake_torch()
    driver = _install_launch_fakes(monkeypatch, torch)
    arguments = _arguments(torch, n=0)
    arguments["output_ptr"] = _Tensor(torch, pointer=0x1004)

    add_kernel.launch(arguments=arguments, constexprs={"BLOCK": 128}, grid=(0,))

    assert driver.loads == driver.launches == []
    assert torch.advanced == []


@pytest.mark.parametrize(
    "version", ["2.5.1", "2.5.1+cu124", "1.13.1", "2", "nightly", None]
)
def test_launch_rejects_a_pytorch_below_the_floor(monkeypatch, version):
    """Fail before compiling or enqueueing on an unsupported PyTorch."""
    torch, _ = _fake_torch(version=version)
    driver = _install_launch_fakes(monkeypatch, torch)
    reason = (
        "requires PyTorch 2.6 or newer; found PyTorch "
        f"{re.escape(str(version))}; install PyTorch 2.6 or newer$"
    )

    for _ in range(2):
        with pytest.raises(sw.BackendUnavailableError, match=reason) as caught:
            add_kernel.launch(
                arguments=_arguments(torch),
                constexprs={"BLOCK": 128},
                grid=(2,),
            )
        assert caught.value.code == "pytorch-unsupported"

    assert driver.loads == driver.launches == []


def test_launch_rejects_a_pytorch_without_record_stream(monkeypatch):
    """Fail before the enqueue when submitted tensors cannot be retained."""

    class TensorWithoutRecordStream:
        """The tensor type of a PyTorch that lacks `record_stream`."""

    torch, _ = _fake_torch(version="2.6.0")
    driver = _install_launch_fakes(monkeypatch, torch)
    arguments = _arguments(torch)
    torch.Tensor = TensorWithoutRecordStream

    with pytest.raises(
        RuntimeError,
        match="requires torch.Tensor.record_stream.*found PyTorch 2.6.0",
    ):
        add_kernel.launch(
            arguments=arguments, constexprs={"BLOCK": 128}, grid=(2,)
        )

    assert driver.loads == driver.launches == []


@pytest.mark.parametrize("missing", ["autograd", "graph", "function"])
def test_launch_rejects_a_pytorch_without_increment_version(
    monkeypatch, missing
):
    """Fail before the enqueue when the output cannot be marked written."""
    torch, _ = _fake_torch(version="2.6.0")
    driver = _install_launch_fakes(monkeypatch, torch)
    if missing == "autograd":
        del torch.autograd
    elif missing == "graph":
        torch.autograd = types.SimpleNamespace()
    else:
        torch.autograd.graph = types.SimpleNamespace()

    with pytest.raises(
        RuntimeError,
        match=(
            "requires torch.autograd.graph.increment_version.*found "
            "PyTorch 2.6.0"
        ),
    ):
        add_kernel.launch(
            arguments=_arguments(torch), constexprs={"BLOCK": 128}, grid=(2,)
        )

    assert driver.loads == driver.launches == []


@pytest.mark.parametrize(
    "version",
    ["2.6.0", "2.6.0a0+git0123abc", "2.12.0+cu130", "2.13.0+cpu", "3.0.0.dev1"],
)
def test_launch_accepts_a_supported_pytorch(monkeypatch, version):
    """Admit the declared floor and every later release, with no ceiling."""
    torch, _ = _fake_torch(version=version)
    driver = _install_launch_fakes(monkeypatch, torch)

    add_kernel.launch(
        arguments=_arguments(torch), constexprs={"BLOCK": 128}, grid=(2,)
    )

    assert len(driver.launches) == 1


def test_minimum_pytorch_matches_the_declared_dependency():
    """Keep the launch-time floor equal to the one `pyproject.toml` states."""
    from swage import _runtime

    project = pathlib.Path(__file__).parents[2] / "pyproject.toml"
    # No closing quote: a patch level or an upper bound may follow.
    declared = re.search(r'"torch>=(\d+)\.(\d+)', project.read_text())

    assert declared is not None
    assert _runtime._MIN_TORCH == (int(declared[1]), int(declared[2]))


def _launch_location(kernel):
    """Return the suffix a launch error carries for `kernel`."""
    line = kernel.source_line + kernel.function.lineno - 1
    return (
        f" (in launch of kernel '{kernel.__name__}', defined at "
        f"{kernel.filename}:{line})"
    )


@pytest.mark.parametrize(
    ("kernel", "reason"),
    [
        (
            defaulted_kernel,
            "parameter 'BLOCK' has a default value",
        ),
        (
            annotated_count_kernel,
            "unsupported annotation 'int' on parameter 'n'",
        ),
        (
            annotated_pointer_kernel,
            "unsupported annotation 'float' on parameter 'x_ptr'",
        ),
        (
            annotated_block_kernel,
            "unsupported annotation 'int' on parameter 'BLOCK'",
        ),
    ],
)
def test_launch_rejects_defaults_and_foreign_annotations(
    monkeypatch, kernel, reason
):
    """Refuse a parameter list outside the ABI before any compile.

    The launch asks the frontend about the parameters before it checks
    their shape, so each form is named with its source location.
    """
    from swage import _runtime

    torch, _ = _fake_torch()
    driver = _Driver()
    emissions = []
    compiles = []
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(
        kernel, "emit_mlir", lambda **_kwargs: emissions.append(1)
    )
    for module, name in (
        (_runtime, "_compile_cached"),
        (_cuda_backend, "_compile_native"),
    ):
        monkeypatch.setattr(
            module, name, lambda *_args, **_kwargs: compiles.append(1)
        )
    monkeypatch.setattr(_cuda_backend, "_get_driver", lambda: driver)
    arguments = _arguments(torch)

    with pytest.raises(sw.CompilationError) as rejection:
        kernel.launch(arguments=arguments, constexprs={"BLOCK": 128}, grid=(2,))

    message = str(rejection.value)
    assert type(rejection.value) is sw.CompilationError
    assert message.startswith(f"{__file__}:")
    assert f": {kernel.__name__}: {reason}" in message
    assert emissions == compiles == []
    assert driver.loads == driver.launches == []
    for tensor in tuple(arguments.values())[:3]:
        assert tensor.recorded_streams == []


@pytest.mark.parametrize(
    ("arguments", "constexprs", "grid", "error", "reason"),
    [
        ("bad", {"BLOCK": 128}, (2,), TypeError, "arguments must be a mapping"),
        (None, {"BLOCK": 0}, (2,), ValueError, "constexpr BLOCK must be"),
        (None, {"BLOCK": 128}, (1,), ValueError, "grid must equal"),
        (None, {"BLOCK": 2048}, (1,), ValueError, "BLOCK 2048 exceeds device"),
        (
            {"device_type": "cpu"},
            {"BLOCK": 128},
            (2,),
            TypeError,
            "argument 'x_ptr' must be a CUDA tensor",
        ),
        (
            {"contiguous": False},
            {"BLOCK": 128},
            (2,),
            ValueError,
            "argument 'x_ptr' must be contiguous",
        ),
        (
            {"negative": True},
            {"BLOCK": 128},
            (2,),
            ValueError,
            "argument 'x_ptr' must not be a lazy negation view",
        ),
    ],
)
def test_launch_errors_name_the_kernel(
    monkeypatch, arguments, constexprs, grid, error, reason
):
    """Keep each message and add the kernel and where it is defined."""
    torch, _ = _fake_torch()
    driver = _install_launch_fakes(monkeypatch, torch)
    if arguments is None:
        arguments = _arguments(torch)
    elif isinstance(arguments, dict):
        arguments = _arguments(torch, **arguments)

    with pytest.raises(error) as rejection:
        add_kernel.launch(arguments=arguments, constexprs=constexprs, grid=grid)

    message = str(rejection.value)
    assert type(rejection.value) is error
    assert message.startswith(reason)
    assert message.endswith(_launch_location(add_kernel))
    assert message.count("in launch of kernel") == 1
    # The original raise stays the last frame, with no chained exception.
    assert rejection.value.__cause__ is None
    assert rejection.value.__suppress_context__
    assert re.match(r"_?validate_", rejection.traceback[-1].name)
    assert driver.loads == driver.launches == []


def test_launch_names_the_kernel_for_a_compiler_rejection(monkeypatch):
    """Name the kernel when the native compiler refuses the target."""
    from swage import _runtime

    torch, _ = _fake_torch()
    driver = _install_launch_fakes(monkeypatch, torch)

    def refuse(*_args, **_kwargs):
        raise ValueError("unsupported target 'sm_75'")

    monkeypatch.setattr(_runtime, "_compile_cached", refuse)

    with pytest.raises(ValueError) as rejection:
        add_kernel.launch(
            arguments=_arguments(torch), constexprs={"BLOCK": 128}, grid=(2,)
        )

    assert str(rejection.value) == (
        "unsupported target 'sm_75'" + _launch_location(add_kernel)
    )
    assert driver.loads == driver.launches == []


def test_launch_leaves_other_exception_types_unchanged(monkeypatch):
    """Do not rebuild an error whose type takes more than a message."""
    from swage import _runtime

    torch, _ = _fake_torch()
    _install_launch_fakes(monkeypatch, torch)
    original = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

    def fail(*_args, **_kwargs):
        raise original

    monkeypatch.setattr(_runtime, "_compile_cached", fail)

    with pytest.raises(UnicodeDecodeError) as rejection:
        add_kernel.launch(
            arguments=_arguments(torch), constexprs={"BLOCK": 128}, grid=(2,)
        )

    assert rejection.value is original


def test_missing_bindings_error_names_the_installation_page(monkeypatch):
    """Point a launch without the native package at the build instructions."""
    from swage import _frontend

    monkeypatch.setitem(sys.modules, "mlir_swage", None)

    with pytest.raises(sw.BackendUnavailableError) as failure:
        _cuda_backend._compile_native(
            object(), "add_kernel", 128, "sm_86", "fixed", {}
        )

    message = str(failure.value)
    assert message.startswith("Swage launch requires the mlir_swage bindings")
    assert "which this installation does not have" in message
    assert "kernel 'add_kernel' was not compiled" in message
    assert message.endswith(
        f"; see {_frontend._INSTALLATION} for the native build"
    )
    assert "docs/getting-started/installation.md" in message
    assert failure.value.code == "native-unavailable"


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


def _stub_compiler(monkeypatch, identity=_identity):
    """Count native compiles, with a stubbed identity unless it is None.

    A stubbed identity describes no files, so the check that ties the
    identity to the loaded code is stubbed with it. `identity=None` keeps
    the real identity and the real check.
    """
    from swage import _runtime

    calls = []
    if identity is not None:
        monkeypatch.setattr(_runtime, "_compiler_identity", identity)
        monkeypatch.setattr(
            _runtime, "_stale_identity", lambda _identity: None, raising=False
        )
    _runtime._identity_cache = None
    monkeypatch.setattr(_cuda_backend, "_compile_native", _compiler(calls))
    _runtime._artifact_cache.clear()
    return _runtime, calls


def _compile_recording_warnings(specialization=None):
    """Compile one key and return the artifact with the warnings it raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        artifact = _compile(specialization)
    return artifact, [str(warning.message) for warning in caught]


def _assert_complete_entry(entry):
    """Check one published entry holds exactly the three private files."""
    assert stat.S_IMODE(entry.stat().st_mode) == 0o700
    assert sorted(path.name for path in entry.iterdir()) == [
        "kernel.ptx",
        "lowered.mlir",
        "metadata.json",
    ]
    for path in entry.iterdir():
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_cache_round_trip_and_corruption_rejection(tmp_path, monkeypatch):
    """Reuse verified PTX and never return corrupted cache contents."""
    from swage import _runtime

    monkeypatch.setenv("SWAGE_CACHE_DIR", str(tmp_path))
    _, calls = _stub_compiler(monkeypatch)

    first = _compile()
    _runtime._artifact_cache.clear()
    second = _compile()

    assert first == second
    assert len(calls) == 1
    assert [path.name for path in tmp_path.iterdir()] == [first.key]
    entry = tmp_path / first.key
    _assert_complete_entry(entry)
    metadata = json.loads((entry / "metadata.json").read_text())
    assert metadata["version"] == _runtime._CACHE_VERSION == 4
    assert (metadata["backend"], metadata["format"], metadata["target"]) == (
        "cuda",
        "ptx",
        "sm_86",
    )
    assert metadata["contract"] == _contract_json()
    assert (
        metadata["digests"]["contract"]
        == hashlib.sha256(_contract_json().encode()).hexdigest()
    )
    assert first.contract_json == _contract_json()
    assert first.contract == second.contract == _contract()
    assert first.contract.arguments[0].access == "read"
    with pytest.raises(AttributeError):
        first.contract.entry = "changed"
    (entry / "kernel.ptx").write_text("corrupt")
    _runtime._artifact_cache.clear()
    with pytest.raises(RuntimeError, match="digest mismatch"):
        _compile()


@pytest.mark.parametrize(
    ("contract_json", "reason"),
    [
        ("not-json", "not valid JSON"),
        (
            _contract_json().replace('"version":2', '"version":1'),
            "unsupported kernel contract version",
        ),
        (
            _contract_json().replace('"kind":"i32"', '"kind":"u32"'),
            "unknown kind",
        ),
        (_contract_json() + " ", "not canonical"),
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
    contract_json = _contract_json(
        "split_merge", arguments=(_user("ptr", 2, "write"),)
    )

    contract = _abi.parse_kernel_contract(contract_json)

    assert contract.arguments[0].source_index == 2


def test_cache_rejects_contract_corruption_and_old_metadata(
    tmp_path, monkeypatch
):
    """Fail closed on contract bytes and reject an older cache format."""
    from swage import _runtime

    monkeypatch.setenv("SWAGE_CACHE_DIR", str(tmp_path))
    _stub_compiler(monkeypatch)
    artifact = _compile()
    entry = tmp_path / artifact.key
    metadata_path = entry / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["contract"] = _contract_json("other_kernel")
    metadata_path.write_text(
        json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    )
    _runtime._artifact_cache.clear()
    with pytest.raises(RuntimeError, match="contract digest mismatch"):
        _compile()

    metadata = json.loads(metadata_path.read_text())
    metadata["digests"]["contract"] = hashlib.sha256(
        metadata["contract"].encode()
    ).hexdigest()
    metadata_path.write_text(
        json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    )
    with pytest.raises(RuntimeError, match="entry does not match"):
        _compile()

    # The format before this one had the same fields.
    metadata["version"] = 3
    metadata["contract"] = _contract_json()
    metadata["digests"]["contract"] = hashlib.sha256(
        metadata["contract"].encode()
    ).hexdigest()
    metadata_path.write_text(
        json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    )
    with pytest.raises(
        RuntimeError, match=_rejects("cache metadata mismatch", entry)
    ):
        _compile()


def test_launch_rejects_contract_mismatch_before_module_load(monkeypatch):
    """Validate compiler metadata before loading PTX into CUDA."""
    from swage import _runtime

    torch, _ = _fake_torch()
    driver = _Driver()
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(add_kernel, "emit_mlir", lambda **_kwargs: object())
    monkeypatch.setattr(
        _cuda_backend,
        "_compile_native",
        lambda *_args: ("lowered", "ptx", _contract_json(block=64)),
    )
    monkeypatch.setattr(_runtime, "_compiler_identity", _unpersisted_identity)
    monkeypatch.setattr(_cuda_backend, "_get_driver", lambda: driver)
    _runtime._identity_cache = None

    with pytest.raises(RuntimeError, match="block does not match"):
        add_kernel.launch(
            arguments=_arguments(torch),
            constexprs={"BLOCK": 128},
            grid=(2,),
        )
    assert driver.loads == []


def _rejects(reason, path):
    """Match a rejection that gives `reason` and names exactly `path`."""
    return rf"{reason}.*: {re.escape(str(path))}$"


def test_cache_rejects_unsafe_entries(tmp_path, monkeypatch):
    """Reject symlinked and world-writable files and a foreign-owned root."""
    monkeypatch.setenv("SWAGE_CACHE_DIR", str(tmp_path))
    _runtime, calls = _stub_compiler(monkeypatch)
    key = _runtime._cache_key(_SHARED_KEY)
    entry = tmp_path / key
    entry.mkdir()
    target = tmp_path / "outside.json"
    target.write_text(json.dumps({}))
    (entry / "metadata.json").symlink_to(target)

    with pytest.raises(RuntimeError, match="symlink"):
        _compile()

    (entry / "metadata.json").unlink()
    (entry / "metadata.json").write_text("{}")
    (entry / "metadata.json").chmod(0o606)
    with pytest.raises(RuntimeError, match="world-writable"):
        _compile()

    (entry / "metadata.json").chmod(0o600)
    other_user = os.geteuid() + 1
    # Every path is foreign to another user, and the root is checked first.
    with monkeypatch.context() as patch:
        patch.setattr(_runtime.os, "geteuid", lambda: other_user)
        with pytest.raises(RuntimeError, match=_rejects("not owned", tmp_path)):
            _compile()

    assert calls == []
    assert (entry / "metadata.json").read_text() == "{}"


def _make_foreign(monkeypatch, foreign):
    """Report one path, and no other, as owned by another user."""
    lstat = pathlib.Path.lstat

    def lstat_with_one_foreign_path(path):
        details = lstat(path)
        if path != foreign:
            return details
        fields = list(details)
        fields[stat.ST_UID] = details.st_uid + 1
        return os.stat_result(fields)

    monkeypatch.setattr(pathlib.Path, "lstat", lstat_with_one_foreign_path)


def _published_entry(tmp_path, monkeypatch):
    """Publish one valid entry and forget it in the process."""
    monkeypatch.setenv("SWAGE_CACHE_DIR", str(tmp_path))
    _runtime, calls = _stub_compiler(monkeypatch)
    artifact = _compile()
    _runtime._artifact_cache.clear()
    calls.clear()
    return _runtime, calls, tmp_path / artifact.key


def test_cache_rejects_a_foreign_owned_entry_file(tmp_path, monkeypatch):
    """Check ownership of every file, not only of the cache root."""
    _, calls, entry = _published_entry(tmp_path, monkeypatch)
    _make_foreign(monkeypatch, entry / "kernel.ptx")

    with pytest.raises(
        RuntimeError, match=_rejects("not owned", entry / "kernel.ptx")
    ):
        _compile()
    assert calls == []


@pytest.mark.parametrize(
    ("unsafe", "reason"),
    [
        ("symlink", "is a symlink"),
        ("world-writable", "is world-writable"),
        ("foreign owner", "not owned"),
    ],
)
def test_cache_rejects_an_unsafe_entry_directory(
    tmp_path, monkeypatch, unsafe, reason
):
    """Check the entry directory itself, even when its files are safe."""
    _, calls, entry = _published_entry(tmp_path, monkeypatch)
    if unsafe == "symlink":
        moved = tmp_path / "moved-entry"
        entry.rename(moved)
        entry.symlink_to(moved)
    elif unsafe == "world-writable":
        entry.chmod(0o707)
    else:
        _make_foreign(monkeypatch, entry)

    with pytest.raises(RuntimeError, match=_rejects(reason, entry)):
        _compile()
    assert calls == []


def test_cache_rejects_a_corrupt_lowered_module(tmp_path, monkeypatch):
    """Verify the lowered MLIR digest, not only the PTX digest."""
    _, calls, entry = _published_entry(tmp_path, monkeypatch)
    (entry / "lowered.mlir").write_text("corrupt")

    with pytest.raises(
        RuntimeError, match=_rejects("lowered MLIR digest mismatch", entry)
    ):
        _compile()
    assert calls == []


def test_cache_rejects_an_entry_for_another_specialization(
    tmp_path, monkeypatch
):
    """Never serve an entry whose recorded specialization differs."""
    monkeypatch.setenv("SWAGE_CACHE_DIR", str(tmp_path))
    _runtime, calls = _stub_compiler(monkeypatch)
    key = _runtime._cache_key(_SHARED_KEY)
    other = dict(_SHARED_KEY, source="another kernel body")
    fixed = ("add_kernel", 128, "fixed", _cuda_backend.CUDA_BACKEND)
    _runtime._write_cache_entry(_artifact(key), other, *fixed)
    assert _runtime._read_cache_entry(key, other, *fixed, "sm_86").image == (
        "ptx"
    )

    with pytest.raises(
        RuntimeError, match=_rejects("specialization mismatch", tmp_path / key)
    ):
        _compile()
    assert calls == []


@pytest.mark.parametrize("field", ["version", "key"])
def test_cache_rejects_metadata_of_another_format_or_key(
    tmp_path, monkeypatch, field
):
    """Verify the metadata version and that the entry belongs to its key."""
    _runtime, calls, entry = _published_entry(tmp_path, monkeypatch)
    key = entry.name
    if field == "version":
        metadata = json.loads((entry / "metadata.json").read_text())
        metadata["version"] = 2
        (entry / "metadata.json").write_text(json.dumps(metadata))
    else:
        # A valid entry found under a name that is not its own key.
        key = _runtime._cache_key(_spec("another"))
        entry = entry.rename(tmp_path / key)

    with pytest.raises(
        RuntimeError, match=_rejects("metadata mismatch", entry)
    ):
        _compile(key=key)
    assert calls == []


@pytest.mark.parametrize(
    ("identity", "warned"),
    [
        (_identity(native=None), False),
        (_identity(frontend=None), True),
        (
            _identity(revision=None, clean=False, frontend=None, native=None),
            False,
        ),
    ],
)
def test_unidentified_compiler_uses_only_process_cache(
    tmp_path, monkeypatch, identity, warned
):
    """Do not persist artifacts when the loaded compiler is unidentified."""
    monkeypatch.setenv("SWAGE_CACHE_DIR", str(tmp_path))
    _, calls = _stub_compiler(monkeypatch, lambda: identity)

    first, messages = _compile_recording_warnings(_spec("add"))
    second, later = _compile_recording_warnings(_spec("add"))
    _, other_key = _compile_recording_warnings(_spec("sub"))

    assert first == second
    assert len(calls) == 2
    assert list(tmp_path.iterdir()) == []
    assert len(messages) == int(warned)
    assert all("persistent cache is off" in message for message in messages)
    assert later == other_key == []


@pytest.mark.parametrize(
    "identity",
    [
        _identity(clean=False),
        _identity(revision=None, clean=False, llvm=None),
    ],
)
def test_identified_compiler_persists_without_a_clean_checkout(
    tmp_path, monkeypatch, identity
):
    """Persist for dirty checkouts and for installs outside a checkout."""
    monkeypatch.setenv("SWAGE_CACHE_DIR", str(tmp_path))
    _runtime, calls = _stub_compiler(monkeypatch, lambda: identity)

    first = _compile(_spec("add"))
    _runtime._artifact_cache.clear()
    second = _compile(_spec("add"))

    assert first == second
    assert len(calls) == 1
    _assert_complete_entry(tmp_path / first.key)


@pytest.mark.parametrize(
    "leftover",
    [(), ("lowered.mlir",), ("lowered.mlir", "kernel.ptx"), ("metadata.json",)],
)
def test_incomplete_entry_is_a_miss_and_is_replaced(
    tmp_path, monkeypatch, leftover
):
    """Recover from a writer that died after creating the entry directory."""
    monkeypatch.setenv("SWAGE_CACHE_DIR", str(tmp_path))
    _runtime, calls = _stub_compiler(monkeypatch)
    entry = tmp_path / _runtime._cache_key(_SHARED_KEY)
    entry.mkdir(mode=0o700)
    for name in leftover:
        (entry / name).write_text("stale")

    first = _compile()
    _runtime._artifact_cache.clear()
    second = _compile()

    assert first == second == _artifact(entry.name)
    assert len(calls) == 1
    assert [path.name for path in tmp_path.iterdir()] == [entry.name]
    _assert_complete_entry(entry)
    assert (entry / "lowered.mlir").read_text() == "lowered"


def test_leftover_staging_directories_are_ignored(tmp_path, monkeypatch):
    """Never read, publish, or fail on another writer's staging directory."""
    monkeypatch.setenv("SWAGE_CACHE_DIR", str(tmp_path))
    _runtime, calls = _stub_compiler(monkeypatch)
    leftover = pathlib.Path(
        _runtime.tempfile.mkdtemp(dir=tmp_path, prefix=_runtime._STAGING_PREFIX)
    )
    (leftover / "lowered.mlir").write_text("stale")

    first = _compile()
    _runtime._artifact_cache.clear()
    second = _compile()

    assert first == second
    assert len(calls) == 1
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(
        [leftover.name, first.key]
    )
    assert [path.name for path in leftover.iterdir()] == ["lowered.mlir"]
    _assert_complete_entry(tmp_path / first.key)


def test_losing_writer_uses_the_published_entry(tmp_path, monkeypatch):
    """Discard the staged copy when another writer published the key."""
    monkeypatch.setenv("SWAGE_CACHE_DIR", str(tmp_path))
    _runtime, _ = _stub_compiler(monkeypatch)
    winner = _compile()
    loser = _artifact(winner.key, image="ptx from the loser")

    used = _runtime._write_cache_entry(
        loser,
        _SHARED_KEY,
        "add_kernel",
        128,
        "fixed",
        _cuda_backend.CUDA_BACKEND,
    )

    assert used == winner
    assert [path.name for path in tmp_path.iterdir()] == [winner.key]
    assert (tmp_path / winner.key / "kernel.ptx").read_text() == "ptx"


def test_failed_publish_degrades_to_process_reuse(tmp_path, monkeypatch):
    """Keep the artifact, warn once, and clean up when publishing fails."""
    monkeypatch.setenv("SWAGE_CACHE_DIR", str(tmp_path))
    _runtime, calls = _stub_compiler(monkeypatch)
    renames = []

    def refuse(source, _destination):
        renames.append(source)
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(_runtime.os, "rename", refuse)
    first, messages = _compile_recording_warnings()
    second, repeated = _compile_recording_warnings()
    _, other_key = _compile_recording_warnings(_spec("other"))

    assert first == second == _artifact(first.key)
    assert len(calls) == 2
    assert len(renames) == 1
    assert len(messages) == 1
    assert str(tmp_path) in messages[0]
    assert "Permission denied" in messages[0]
    assert repeated == other_key == []
    assert list(tmp_path.iterdir()) == []


def _unwritable_root(tmp_path, monkeypatch, how):
    """Return a cache root that cannot be written, and undo the damage."""
    from swage import _runtime

    root = tmp_path / "cache"
    if how == "read-only root":
        root.mkdir(mode=0o500)
    elif how == "read-only parent":
        root = tmp_path / "locked" / "cache"
        root.parent.mkdir(mode=0o500)
    else:
        root.mkdir(mode=0o700)

        def no_space(*_args, **_kwargs):
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(_runtime.tempfile, "mkdtemp", no_space)
    return root


@pytest.mark.parametrize(
    "how",
    [
        pytest.param(
            "read-only root",
            marks=pytest.mark.skipif(
                os.geteuid() == 0, reason="root ignores directory modes"
            ),
        ),
        pytest.param(
            "read-only parent",
            marks=pytest.mark.skipif(
                os.geteuid() == 0, reason="root ignores directory modes"
            ),
        ),
        "no space left",
    ],
)
def test_unwritable_cache_never_fails_a_launch(tmp_path, monkeypatch, how):
    """Launch from the retained artifact when the cache cannot be written."""
    torch, _ = _fake_torch()
    driver = _Driver()
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(add_kernel, "emit_mlir", lambda **_kwargs: object())
    monkeypatch.setattr(_cuda_backend, "_get_driver", lambda: driver)
    _runtime, calls = _stub_compiler(monkeypatch)
    monkeypatch.delitem(
        add_kernel.__dict__, "_specialization_memo", raising=False
    )
    root = _unwritable_root(tmp_path, monkeypatch, how)
    monkeypatch.setenv("SWAGE_CACHE_DIR", str(root))

    def launch(n, block):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            add_kernel.launch(
                arguments=_arguments(torch, n=n),
                constexprs={"BLOCK": block},
                grid=((n + block - 1) // block,),
            )
        return [str(warning.message) for warning in caught]

    try:
        first = launch(129, 128)
        second = launch(129, 128)
        other_specialization = launch(129, 64)
        leftovers = list(root.iterdir()) if root.exists() else []
    finally:
        root.parent.chmod(0o700)
        if root.exists():
            root.chmod(0o700)
        _runtime._identity_cache = None

    assert len(driver.launches) == 3
    assert len(calls) == 2
    assert len(first) == 1
    assert str(root) in first[0]
    assert "persistent cache is not written by this process" in first[0]
    assert second == other_specialization == []
    assert leftovers == []


def test_read_only_cache_still_serves_published_entries(tmp_path, monkeypatch):
    """Keep reading a warm cache after a write to it has failed."""
    monkeypatch.setenv("SWAGE_CACHE_DIR", str(tmp_path))
    _runtime, calls = _stub_compiler(monkeypatch)
    warm = _compile()
    _runtime._artifact_cache.clear()

    def no_space(*_args, **_kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(_runtime.tempfile, "mkdtemp", no_space)
    _, messages = _compile_recording_warnings(_spec("cold"))
    reread, later = _compile_recording_warnings()

    assert len(messages) == 1
    assert "published entries are still read" in messages[0]
    assert reread == warm
    assert later == []
    assert len(calls) == 2


@pytest.mark.parametrize(
    "how",
    [
        pytest.param(
            "read-only root",
            marks=pytest.mark.skipif(
                os.geteuid() == 0, reason="root ignores directory modes"
            ),
        ),
        "read-only file system",
    ],
)
def test_debris_in_a_read_only_cache_is_a_miss_for_its_key_only(
    tmp_path, monkeypatch, how
):
    """Keep reading other keys when an incomplete entry cannot be removed."""
    monkeypatch.setenv("SWAGE_CACHE_DIR", str(tmp_path))
    _runtime, calls = _stub_compiler(monkeypatch)
    warm = _compile()
    _runtime._artifact_cache.clear()
    debris = tmp_path / _runtime._cache_key(_spec("debris"))
    debris.mkdir(mode=0o700)
    (debris / "lowered.mlir").write_text("stale")
    if how == "read-only root":
        tmp_path.chmod(0o500)
    else:

        def read_only(*_args, **_kwargs):
            raise OSError(30, "Read-only file system")

        monkeypatch.setattr(_runtime.tempfile, "mkdtemp", read_only)

    try:
        missed, messages = _compile_recording_warnings(_spec("debris"))
        again, repeated = _compile_recording_warnings(_spec("debris"))
        reread, later = _compile_recording_warnings()
        _, cold = _compile_recording_warnings(_spec("cold"))
    finally:
        tmp_path.chmod(0o700)

    assert missed == again
    assert missed.image == "ptx"
    assert len(messages) == 1
    assert str(tmp_path) in messages[0]
    assert "published entries are still read" in messages[0]
    assert reread == warm
    assert repeated == later == cold == []
    assert len(calls) == 3
    assert set(_runtime._cache_off) == {"write"}
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(
        [warm.key, debris.name]
    )
    assert [path.name for path in debris.iterdir()] == ["lowered.mlir"]


def test_inaccessible_cache_root_degrades_to_process_reuse(
    tmp_path, monkeypatch
):
    """Treat a cache root that cannot be inspected as no cache at all."""
    root = tmp_path / "file" / "cache"
    root.parent.write_text("a file where a directory is expected")
    monkeypatch.setenv("SWAGE_CACHE_DIR", str(root))
    _, calls = _stub_compiler(monkeypatch)

    first, messages = _compile_recording_warnings()
    second, repeated = _compile_recording_warnings()
    _, other_key = _compile_recording_warnings(_spec("other"))

    assert first == second
    assert len(calls) == 2
    assert len(messages) == 1
    assert str(root) in messages[0]
    assert "persistent cache is off for this process" in messages[0]
    assert repeated == other_key == []


@pytest.mark.parametrize(
    ("unsafe", "reason"),
    [("symlink", "is a symlink"), ("world-writable", "is world-writable")],
)
def test_tampering_found_while_publishing_still_raises(
    tmp_path, monkeypatch, unsafe, reason
):
    """Raise on an unsafe root at publish, but keep the compiled artifact."""
    root = tmp_path / "cache"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir(mode=0o700)
    monkeypatch.setenv("SWAGE_CACHE_DIR", str(root))
    _, calls = _stub_compiler(monkeypatch)
    compile_native = _compiler(calls)

    def compile_while_the_root_appears(*arguments):
        # The lookup found no root, so only the publish can notice this one.
        if unsafe == "symlink":
            root.symlink_to(elsewhere)
        else:
            root.mkdir()
            root.chmod(0o707)
        return compile_native(*arguments)

    monkeypatch.setattr(
        _cuda_backend, "_compile_native", compile_while_the_root_appears
    )
    with pytest.raises(RuntimeError, match=_rejects(reason, root)):
        _compile()
    retained = _compile()

    assert retained.image == "ptx"
    assert len(calls) == 1
    assert list(elsewhere.iterdir()) == []
    assert list(root.iterdir()) == []


def test_removing_an_incomplete_entry_keeps_a_concurrent_publish(
    tmp_path, monkeypatch
):
    """Do not delete an entry that was published after the reader looked."""
    monkeypatch.setenv("SWAGE_CACHE_DIR", str(tmp_path))
    _runtime, calls = _stub_compiler(monkeypatch)
    published = _compile()
    entry = tmp_path / published.key

    # A reader that saw debris reaches removal after the publish above.
    _runtime._remove_incomplete_entry(entry)

    _runtime._artifact_cache.clear()
    assert [path.name for path in tmp_path.iterdir()] == [published.key]
    _assert_complete_entry(entry)
    assert _compile() == published
    assert len(calls) == 1

    (entry / "metadata.json").unlink()
    _runtime._remove_incomplete_entry(entry)
    _runtime._remove_incomplete_entry(entry)
    assert list(tmp_path.iterdir()) == []


def test_a_file_in_place_of_an_entry_is_rejected(tmp_path, monkeypatch):
    """Refuse to treat a regular file at an entry name as debris."""
    monkeypatch.setenv("SWAGE_CACHE_DIR", str(tmp_path))
    _runtime, calls = _stub_compiler(monkeypatch)
    entry = tmp_path / _runtime._cache_key(_SHARED_KEY)
    entry.write_text("not an entry")

    with pytest.raises(RuntimeError, match="not a directory"):
        _compile()

    assert calls == []
    assert [path.name for path in tmp_path.iterdir()] == [entry.name]


def _snapshot(root):
    """Return every path under `root` with its change time and contents."""
    return {
        str(path.relative_to(root)): (
            path.lstat().st_mtime_ns,
            path.read_text() if path.is_file() else None,
        )
        for path in sorted(root.rglob("*"))
    }


def test_read_only_mode_reads_entries_and_writes_nothing(tmp_path, monkeypatch):
    """Serve published entries and keep new kernels in the process only."""
    monkeypatch.setenv("SWAGE_CACHE_DIR", str(tmp_path))
    _runtime, calls = _stub_compiler(monkeypatch)
    warm = _compile()
    debris = tmp_path / _runtime._cache_key(_spec("debris"))
    debris.mkdir(mode=0o700)
    (debris / "lowered.mlir").write_text("stale")
    before = _snapshot(tmp_path)
    _runtime._artifact_cache.clear()
    monkeypatch.setenv("SWAGE_CACHE_READ_ONLY", "1")

    reread, on_hit = _compile_recording_warnings()
    cold, on_miss = _compile_recording_warnings(_spec("cold"))
    again, repeated = _compile_recording_warnings(_spec("cold"))
    missed, on_debris = _compile_recording_warnings(_spec("debris"))

    assert reread == warm
    assert cold == again
    assert cold.image == missed.image == "ptx"
    assert len(calls) == 3
    assert on_hit == on_miss == repeated == on_debris == []
    assert _snapshot(tmp_path) == before
    assert not _runtime._cache_off


def test_read_only_mode_does_not_create_the_cache_root(tmp_path, monkeypatch):
    """Leave a missing cache root missing."""
    root = tmp_path / "absent" / "cache"
    monkeypatch.setenv("SWAGE_CACHE_DIR", str(root))
    monkeypatch.setenv("SWAGE_CACHE_READ_ONLY", "1")
    _, calls = _stub_compiler(monkeypatch)

    artifact, messages = _compile_recording_warnings()

    assert artifact.image == "ptx"
    assert len(calls) == 1
    assert messages == []
    assert not root.parent.exists()


def test_read_only_mode_still_rejects_unsafe_entries(tmp_path, monkeypatch):
    """Keep treating a corrupt entry as tamper evidence."""
    monkeypatch.setenv("SWAGE_CACHE_DIR", str(tmp_path))
    _runtime, calls = _stub_compiler(monkeypatch)
    warm = _compile()
    (tmp_path / warm.key / "kernel.ptx").write_text("corrupt")
    _runtime._artifact_cache.clear()
    monkeypatch.setenv("SWAGE_CACHE_READ_ONLY", "1")

    with pytest.raises(RuntimeError, match="digest mismatch"):
        _compile()

    assert len(calls) == 1


def test_no_compile_mode_serves_entries_and_refuses_a_miss(
    tmp_path, monkeypatch
):
    """Launch from the cache and raise instead of compiling on a miss."""
    monkeypatch.setenv("SWAGE_CACHE_DIR", str(tmp_path))
    _runtime, calls = _stub_compiler(monkeypatch)
    warm = _compile()
    before = _snapshot(tmp_path)
    _runtime._artifact_cache.clear()
    monkeypatch.setenv("SWAGE_NO_COMPILE", "1")
    emissions = []
    cold = _spec("cold_kernel")

    reread, on_hit = _compile_recording_warnings()
    in_process, _ = _compile_recording_warnings()
    for _ in range(2):
        with pytest.raises(RuntimeError) as refusal:
            _compile(cold, lambda: emissions.append(1))

    assert reread == in_process == warm
    assert on_hit == []
    message = str(refusal.value)
    assert "SWAGE_NO_COMPILE=1 refuses to compile kernel 'cold_kernel'" in (
        message
    )
    assert f"no entry {_runtime._cache_key(cold)} in {tmp_path}" in message
    assert len(calls) == 1
    assert emissions == []
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize(
    ("identity", "reason"),
    [
        (_identity(native=None), "native compiler libraries are not found"),
        (_identity(frontend=None), "frontend sources are not identified"),
    ],
)
def test_no_compile_mode_says_why_the_cache_was_not_read(
    tmp_path, monkeypatch, identity, reason
):
    """Name the cause when the refusal follows from an unusable cache."""
    monkeypatch.setenv("SWAGE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("SWAGE_NO_COMPILE", "1")
    _, calls = _stub_compiler(monkeypatch, lambda: identity)
    for _ in range(2):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(RuntimeError) as refusal:
                _compile()
        assert "SWAGE_NO_COMPILE=1 refuses to compile" in str(refusal.value)
        assert "the persistent cache is off for this process" in str(
            refusal.value
        )
        assert reason in str(refusal.value)

    assert calls == []


def test_no_compile_mode_says_when_the_cache_root_is_unreadable(
    tmp_path, monkeypatch
):
    """Keep the cause of an unreadable root in every later refusal."""
    root = tmp_path / "file" / "cache"
    root.parent.write_text("a file where a directory is expected")
    monkeypatch.setenv("SWAGE_CACHE_DIR", str(root))
    monkeypatch.setenv("SWAGE_NO_COMPILE", "1")
    _, calls = _stub_compiler(monkeypatch)

    for _ in range(2):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(RuntimeError) as refusal:
                _compile()
        assert f"cannot use {root}" in str(refusal.value)
        assert "Not a directory" in str(refusal.value)

    assert calls == []


def test_no_compile_mode_refuses_a_launch_before_any_driver_work(
    tmp_path, monkeypatch
):
    """Raise from the public launch with nothing loaded or enqueued."""
    torch, _ = _fake_torch()
    driver = _Driver()
    emissions = []
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(
        add_kernel, "emit_mlir", lambda **_kwargs: emissions.append(1)
    )
    monkeypatch.setattr(_cuda_backend, "_get_driver", lambda: driver)
    _runtime, calls = _stub_compiler(monkeypatch)
    monkeypatch.delitem(
        add_kernel.__dict__, "_specialization_memo", raising=False
    )
    monkeypatch.setenv("SWAGE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("SWAGE_NO_COMPILE", "1")
    arguments = _arguments(torch)

    try:
        with pytest.raises(RuntimeError, match="SWAGE_NO_COMPILE=1 refuses"):
            add_kernel.launch(
                arguments=arguments, constexprs={"BLOCK": 128}, grid=(2,)
            )
    finally:
        _runtime._identity_cache = None

    assert calls == emissions == []
    assert driver.loads == driver.launches == []
    assert arguments["output_ptr"].recorded_streams == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("name", ["SWAGE_CACHE_READ_ONLY", "SWAGE_NO_COMPILE"])
@pytest.mark.parametrize("value", ["true", "yes", "2", " 1", "on"])
def test_cache_switches_reject_unknown_values(
    tmp_path, monkeypatch, name, value
):
    """Fail instead of guessing what a mistyped safety switch meant."""
    monkeypatch.setenv("SWAGE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv(name, value)
    _, calls = _stub_compiler(monkeypatch)

    with pytest.raises(
        ValueError, match=rf"{name} must be 0 or 1; found '{value}'"
    ):
        _compile()

    assert calls == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("name", ["SWAGE_CACHE_READ_ONLY", "SWAGE_NO_COMPILE"])
@pytest.mark.parametrize("value", ["", "0"])
def test_cache_switches_are_off_when_empty_or_zero(
    tmp_path, monkeypatch, name, value
):
    """Treat an empty switch and `0` like an unset switch."""
    monkeypatch.setenv("SWAGE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv(name, value)
    _, calls = _stub_compiler(monkeypatch)

    artifact = _compile()

    assert len(calls) == 1
    _assert_complete_entry(tmp_path / artifact.key)


def _publish(name, age_seconds):
    """Publish one entry and date it `age_seconds` before now."""
    from swage import _runtime

    artifact = _compile(_spec(name))
    entry = _runtime._cache_dir() / artifact.key
    published = time.time() - age_seconds
    os.utime(entry, (published, published))
    return artifact.key


def test_cache_keeps_the_newest_entries_within_the_bound(tmp_path, monkeypatch):
    """Remove the entries published longest ago once the bound is passed."""
    monkeypatch.setenv("SWAGE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("SWAGE_CACHE_MAX_ENTRIES", "3")
    _runtime, calls = _stub_compiler(monkeypatch)

    # Published out of age order, so the age decides, not the name or the
    # order of arrival.
    middle = _publish("middle", 200)
    oldest = _publish("oldest", 400)
    newer = _publish("newer", 100)
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(
        [middle, oldest, newer]
    )
    # This publish is not dated back, so it is the newest of the four.
    latest = _compile()
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(
        [middle, newer, latest.key]
    )
    one_more = _publish("one more", 0)

    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(
        [newer, latest.key, one_more]
    )
    for key in (newer, latest.key, one_more):
        _assert_complete_entry(tmp_path / key)

    # An evicted key is a plain miss and is published again.
    _runtime._artifact_cache.clear()
    compiles = len(calls)
    assert _publish("oldest", 0) == oldest
    assert len(calls) == compiles + 1
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(
        [latest.key, one_more, oldest]
    )


def test_cache_is_bounded_without_configuration(tmp_path, monkeypatch):
    """Apply the default bound when the variable is unset or empty."""
    from swage import _runtime

    monkeypatch.delenv("SWAGE_CACHE_MAX_ENTRIES", raising=False)
    assert _runtime._max_entries() == 1024
    monkeypatch.setenv("SWAGE_CACHE_MAX_ENTRIES", "")
    assert _runtime._max_entries() == 1024
    monkeypatch.setenv("SWAGE_CACHE_MAX_ENTRIES", "7")
    assert _runtime._max_entries() == 7


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "many", " 8", "0x10"])
def test_cache_bound_rejects_values_that_are_not_positive_integers(
    tmp_path, monkeypatch, value
):
    """Fail instead of guessing a bound, before compiling or writing."""
    monkeypatch.setenv("SWAGE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("SWAGE_CACHE_MAX_ENTRIES", value)
    _, calls = _stub_compiler(monkeypatch)

    with pytest.raises(
        ValueError,
        match=(
            "SWAGE_CACHE_MAX_ENTRIES must be a positive integer; "
            f"found '{re.escape(value)}'"
        ),
    ):
        _compile()

    assert calls == []
    assert list(tmp_path.iterdir()) == []


def test_trimming_leaves_everything_that_is_not_an_old_entry(
    tmp_path, monkeypatch
):
    """Remove only entry directories and dead staging directories."""
    monkeypatch.setenv("SWAGE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("SWAGE_CACHE_MAX_ENTRIES", "1")
    _runtime, _ = _stub_compiler(monkeypatch)
    long_ago = time.time() - 30 * 24 * 3600
    outside = tmp_path.parent / "outside"
    outside.mkdir()
    (outside / "kept.txt").write_text("kept")

    def old(path):
        os.utime(path, (long_ago, long_ago), follow_symlinks=False)
        return path

    live_staging = pathlib.Path(
        _runtime.tempfile.mkdtemp(dir=tmp_path, prefix=_runtime._STAGING_PREFIX)
    )
    dead_staging = pathlib.Path(
        _runtime.tempfile.mkdtemp(dir=tmp_path, prefix=_runtime._STAGING_PREFIX)
    )
    (dead_staging / "lowered.mlir").write_text("stale")
    old(dead_staging)
    kept = [live_staging]
    for name in ("notes", "a" * 63, "A" * 64, "g" * 64, "a" * 65):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "data").write_text("unrelated")
        kept.append(old(directory))
    file_named_like_an_entry = tmp_path / ("b" * 64)
    file_named_like_an_entry.write_text("not a directory")
    kept.append(old(file_named_like_an_entry))
    for name in ("c" * 64, f"{_runtime._STAGING_PREFIX}link"):
        link = tmp_path / name
        link.symlink_to(outside, target_is_directory=True)
        kept.append(old(link))
    evicted = _publish("evicted", 3600)

    latest = _compile()

    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(
        [path.name for path in kept] + [latest.key]
    )
    assert evicted not in os.listdir(tmp_path)
    assert (outside / "kept.txt").read_text() == "kept"
    assert (tmp_path / "notes" / "data").read_text() == "unrelated"


def test_trimming_skips_directories_of_another_user(tmp_path, monkeypatch):
    """Leave a foreign entry for the lookup that will reject it."""
    monkeypatch.setenv("SWAGE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("SWAGE_CACHE_MAX_ENTRIES", "1")
    _stub_compiler(monkeypatch)
    foreign = tmp_path / _publish("foreign", 3600)
    _make_foreign(monkeypatch, foreign)
    latest = _compile()

    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(
        [foreign.name, latest.key]
    )


def test_an_entry_evicted_by_another_process_is_not_an_error(
    tmp_path, monkeypatch
):
    """Treat an entry that vanished during trimming as already removed."""
    monkeypatch.setenv("SWAGE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("SWAGE_CACHE_MAX_ENTRIES", "1")
    _runtime, _ = _stub_compiler(monkeypatch)
    raced = tmp_path / _publish("raced", 3600)
    rename = os.rename

    def rename_after_another_process_removed_it(source, destination):
        if pathlib.Path(source) == raced:
            shutil.rmtree(raced)
        return rename(source, destination)

    monkeypatch.setattr(
        _runtime.os, "rename", rename_after_another_process_removed_it
    )
    latest, messages = _compile_recording_warnings()

    assert messages == []
    assert [path.name for path in tmp_path.iterdir()] == [latest.key]


def test_failed_trimming_stops_publishing_with_one_warning(
    tmp_path, monkeypatch
):
    """Stop adding entries when the bound cannot be kept."""
    monkeypatch.setenv("SWAGE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("SWAGE_CACHE_MAX_ENTRIES", "1")
    _runtime, calls = _stub_compiler(monkeypatch)
    stuck = _publish("stuck", 3600)
    rename = os.rename

    def refuse_to_move_the_old_entry(source, destination):
        if pathlib.Path(source).name == stuck:
            raise PermissionError(13, "Permission denied")
        return rename(source, destination)

    monkeypatch.setattr(_runtime.os, "rename", refuse_to_move_the_old_entry)
    latest, messages = _compile_recording_warnings()
    later, repeated = _compile_recording_warnings(_spec("later"))

    assert latest.image == later.image == "ptx"
    assert len(calls) == 3
    assert len(messages) == 1
    assert "persistent cache is not written by this process" in messages[0]
    assert "Permission denied" in messages[0]
    assert repeated == []
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(
        [stuck, latest.key]
    )


def test_cache_status_describes_an_active_cache_without_touching_it(
    tmp_path, monkeypatch
):
    """Report the root, the entry count, and the modes, and change nothing."""
    root = tmp_path / "cache"
    monkeypatch.setenv("SWAGE_CACHE_DIR", str(root))
    _runtime, _ = _stub_compiler(monkeypatch)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        absent = _runtime._cache_status()
    assert not root.exists()
    assert absent == _runtime._CacheStatus(
        directory=root,
        problem=None,
        rejected=False,
        writes=True,
        entries=0,
        max_entries=1024,
        compiles=True,
    )

    _publish("first", 10)
    _publish("second", 0)
    (root / "notes").mkdir()
    (root / ("b" * 64)).write_text("a file named like an entry")
    before = _snapshot(root)
    monkeypatch.setenv("SWAGE_CACHE_MAX_ENTRIES", "5")
    monkeypatch.setenv("SWAGE_CACHE_READ_ONLY", "1")
    monkeypatch.setenv("SWAGE_NO_COMPILE", "1")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        status = _runtime._cache_status()

    assert status == _runtime._CacheStatus(
        directory=root,
        problem=None,
        rejected=False,
        writes=False,
        entries=2,
        max_entries=5,
        compiles=False,
    )
    assert _snapshot(root) == before
    assert not _runtime._cache_off


@pytest.mark.parametrize(
    ("identity", "reason"),
    [
        (_identity(native=None), "native compiler libraries are not found"),
        (_identity(frontend=None), "frontend sources are not identified"),
    ],
)
def test_cache_status_says_why_the_cache_is_off(
    tmp_path, monkeypatch, identity, reason
):
    """Give the reason a launch would warn about, without warning."""
    monkeypatch.setenv("SWAGE_CACHE_DIR", str(tmp_path))
    _runtime, _ = _stub_compiler(monkeypatch, lambda: identity)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        status = _runtime._cache_status()

    assert reason in status.problem
    assert not status.rejected
    assert not _runtime._cache_off


def test_cache_status_reports_what_this_process_gave_up(tmp_path, monkeypatch):
    """Describe the process that asks, including a cache it turned off."""
    root = tmp_path / "file" / "cache"
    root.parent.write_text("a file where a directory is expected")
    monkeypatch.setenv("SWAGE_CACHE_DIR", str(root))
    _runtime, _ = _stub_compiler(monkeypatch)

    fresh = _runtime._cache_status()
    _compile_recording_warnings()
    after = _runtime._cache_status()

    assert f"cannot use {root}" in fresh.problem
    assert after.problem == fresh.problem
    assert not after.rejected


def test_cache_status_reports_an_unsafe_root_as_rejected(tmp_path, monkeypatch):
    """Say that lookups raise, which is not the same as the cache being off."""
    monkeypatch.setenv("SWAGE_CACHE_DIR", str(tmp_path))
    _runtime, _ = _stub_compiler(monkeypatch)
    tmp_path.chmod(0o707)

    try:
        status = _runtime._cache_status()
    finally:
        tmp_path.chmod(0o700)

    assert status.rejected
    assert status.problem == f"cache entry is world-writable: {tmp_path}"


def test_concurrent_cold_starts_publish_one_entry(tmp_path, monkeypatch):
    """Let eight processes cold-start one key on one cache directory."""
    from swage import _runtime

    processes, rounds = 8, 5
    cache = tmp_path / "shared"
    results = tmp_path / "results"
    results.mkdir()
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(processes)
    workers = [
        context.Process(
            target=_cold_start,
            args=(str(cache), str(results), barrier, rounds),
        )
        for _ in range(processes)
    ]
    try:
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=120)
        exit_codes = [worker.exitcode for worker in workers]
    finally:
        _stop(workers)

    assert exit_codes == [0] * processes
    keys = {
        _runtime._cache_key(dict(_SHARED_KEY, round=round_id)): round_id
        for round_id in range(rounds)
    }
    assert sorted(path.name for path in cache.iterdir()) == sorted(keys)
    monkeypatch.setenv("SWAGE_CACHE_DIR", str(cache))
    for key, round_id in keys.items():
        _assert_complete_entry(cache / key)
        published = _runtime._read_cache_entry(
            key,
            dict(_SHARED_KEY, round=round_id),
            "add_kernel",
            128,
            "fixed",
            _cuda_backend.CUDA_BACKEND,
            "sm_86",
        )
        used = [path.read_text() for path in results.glob(f"{round_id}-*")]
        assert used == [published.image] * processes


def test_concurrent_publishers_keep_the_cache_within_its_bound(tmp_path):
    """Let eight processes publish and evict on one small cache directory."""
    processes, rounds, bound = 8, 6, 4
    cache = tmp_path / "shared"
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(processes)
    workers = [
        context.Process(
            target=_churn, args=(str(cache), barrier, rounds, bound)
        )
        for _ in range(processes)
    ]
    try:
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=120)
        exit_codes = [worker.exitcode for worker in workers]
    finally:
        _stop(workers)

    assert exit_codes == [0] * processes
    entries = list(cache.iterdir())
    assert 1 <= len(entries) <= bound
    for entry in entries:
        _assert_complete_entry(entry)


def test_hung_workers_are_terminated_and_reaped():
    """Stop workers that are still running, started or not."""
    context = multiprocessing.get_context("spawn")
    hung = context.Process(target=_hang, args=(600,))
    never_started = context.Process(target=_hang, args=(600,))
    hung.start()

    _stop([hung, never_started])

    assert not hung.is_alive()
    assert hung.exitcode == -signal.SIGTERM
    assert never_started.exitcode is None


def test_writer_killed_mid_entry_does_not_poison_the_key(tmp_path, monkeypatch):
    """Recompile after a writer was killed between entry files."""
    monkeypatch.setenv("SWAGE_CACHE_DIR", str(tmp_path))
    _runtime, calls = _stub_compiler(monkeypatch)
    key = _runtime._cache_key(_SHARED_KEY)
    writer = multiprocessing.get_context("spawn").Process(
        target=_killed_writer, args=(str(tmp_path),)
    )
    try:
        writer.start()
        writer.join(timeout=120)
        exit_code = writer.exitcode
    finally:
        _stop([writer])

    assert exit_code == -signal.SIGKILL
    assert key not in [path.name for path in tmp_path.iterdir()]

    first = _compile()
    _runtime._artifact_cache.clear()
    second = _compile()

    assert first == second == _artifact(key)
    assert len(calls) == 1
    _assert_complete_entry(tmp_path / key)


def test_dump_switches_write_requested_artifacts(tmp_path, monkeypatch):
    """Write deterministic debug artifacts only when explicitly requested."""
    from swage import _runtime

    monkeypatch.setenv("SWAGE_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("SWAGE_DUMP_MLIR", "1")
    monkeypatch.setenv("SWAGE_DUMP_PTX", "1")
    artifact = _artifact("key")

    _runtime._write_dumps(artifact)

    assert (tmp_path / "key.mlir").read_text() == "lowered"
    assert (tmp_path / "key.ptx").read_text() == "ptx"


def test_compiler_key_contains_every_specialization_input(monkeypatch):
    """Keep cache identity complete and deterministic."""
    from swage import _runtime

    monkeypatch.setattr(_runtime, "_compiler_identity", _identity)
    data = _runtime._specialization_data(
        add_kernel,
        descriptors=("ptr<f32>", "ptr<f32>", "ptr<f32>", "i32"),
        constexprs={"BLOCK": 128},
        target="sm_86",
        adapter=_cuda_backend.CUDA_BACKEND,
    )

    assert data == {
        "source": mock.ANY,
        "kernel": "add_kernel",
        "backend": "cuda",
        "format": "ptx",
        "target": "sm_86",
        "descriptors": ["ptr<f32>", "ptr<f32>", "ptr<f32>", "i32"],
        "constexprs": [["BLOCK", 128]],
        "codegen": {
            "lowering": "fixed",
            "block_size": 128,
            "options": [],
            "index_bits": 64,
        },
        "frontend": "f" * 64,
        "native": [["_swageDialectsNanobind.so", 1, 2]],
        "dialect_version": 1,
    }
    assert len(data["source"]) == 64
    assert _runtime._cache_key(data) == _runtime._cache_key(data)
    assert json.loads(json.dumps(data)) == data


def _fake_package(tmp_path, monkeypatch):
    """Point the frontend identity at a package outside any checkout."""
    from swage import _runtime

    package = tmp_path / "site" / "python" / "swage"
    (package / "nested").mkdir(parents=True)
    (package / "__init__.py").write_text("VERSION = 1\n")
    (package / "_frontend.py").write_text("LOWERING = 1\n")
    (package / "nested" / "helper.py").write_text("HELPER = 1\n")
    (package / "notes.txt").write_text("not a source file\n")
    monkeypatch.setattr(_runtime, "_package_dir", lambda: package)
    return package


_LONG_AFTER_NOW_NS = 10**15


def _start_process_after_every_file(monkeypatch):
    """Pretend the process started after the files the test just wrote."""
    from swage import _runtime

    monkeypatch.setattr(
        _runtime,
        "_PROCESS_START_NS",
        time.time_ns() + _LONG_AFTER_NOW_NS,
        raising=False,
    )


def _changed_ns(path):
    """Return when `path` last changed, as the runtime measures it."""
    details = path.stat()
    return max(details.st_mtime_ns, details.st_ctime_ns)


def _fresh_key(_runtime):
    """Recompute the launch cache key from a newly derived identity."""
    _runtime._identity_cache = None
    data = _runtime._specialization_data(
        add_kernel,
        descriptors=("ptr<f32>", "ptr<f32>", "ptr<f32>", "i32"),
        constexprs={"BLOCK": 128},
        target="sm_86",
        adapter=_cuda_backend.CUDA_BACKEND,
    )
    return _runtime._cache_key(data)


def test_cache_key_tracks_the_frontend_and_native_identity(
    tmp_path, monkeypatch
):
    """Change the key when any frontend byte or native library changes."""
    from swage import _runtime

    package = _fake_package(tmp_path, monkeypatch)
    native = [["_swageDialectsNanobind.so", 10, 20]]
    monkeypatch.setattr(_runtime, "_native_identity", lambda: native)
    original = _fresh_key(_runtime)
    assert _fresh_key(_runtime) == original

    native[0][2] = 21
    rebuilt = _fresh_key(_runtime)
    native[0][2] = 20
    assert rebuilt != original
    assert _fresh_key(_runtime) == original

    (package / "notes.txt").write_text("still not a source file\n")
    assert _fresh_key(_runtime) == original
    (package / "_frontend.py").write_text("LOWERING = 2\n")
    edited = _fresh_key(_runtime)
    (package / "_frontend.py").write_text("LOWERING = 1\n")
    assert edited not in (original, rebuilt)
    assert _fresh_key(_runtime) == original

    (package / "nested" / "helper.py").write_text("HELPER = 2\n")
    assert _fresh_key(_runtime) != original
    (package / "nested" / "helper.py").write_text("HELPER = 1\n")
    (package / "_frontend.py").rename(package / "_lowering.py")
    assert _fresh_key(_runtime) != original
    _runtime._identity_cache = None


def test_frontend_identity_is_absent_without_sources(tmp_path):
    """Leave a package without Python sources unidentified."""
    from swage import _runtime

    (tmp_path / "module.pyc").write_bytes(b"bytecode")
    assert _runtime._frontend_digest(tmp_path) is None


def test_persistence_does_not_need_a_git_checkout(tmp_path, monkeypatch):
    """Persist from an sdist or installed tree that has no `.git`."""
    from swage import _runtime

    _fake_package(tmp_path, monkeypatch)
    _start_process_after_every_file(monkeypatch)
    native = [["_swageDialectsNanobind.so", 10, 20]]
    monkeypatch.setattr(_runtime, "_native_identity", lambda: native)
    monkeypatch.setattr(
        _runtime, "_native_libraries", lambda: [], raising=False
    )

    def no_git(*_args, **_kwargs):
        raise AssertionError("git must not run outside a checkout")

    monkeypatch.setattr(_runtime.subprocess, "run", no_git)
    monkeypatch.setattr(_cuda_backend, "_compile_native", _compiler())
    cache = tmp_path / "cache"
    monkeypatch.setenv("SWAGE_CACHE_DIR", str(cache))
    _runtime._identity_cache = None

    identity = _runtime._cached_identity()
    artifact = _compile()
    _runtime._identity_cache = None

    assert identity == {
        "revision": None,
        "clean": False,
        "llvm": None,
        "frontend": mock.ANY,
        "native": native,
    }
    assert len(identity["frontend"]) == 64
    _assert_complete_entry(cache / artifact.key)


def test_frontend_digest_skips_what_python_cannot_import(tmp_path, monkeypatch):
    """Ignore lock files, dangling links, hidden names, and directories."""
    from swage import _runtime

    package = _fake_package(tmp_path, monkeypatch)
    clean = _runtime._frontend_digest(package)
    assert len(clean) == 64

    (package / ".#_frontend.py").symlink_to("user@host.1234:5678")
    (package / "dangling.py").symlink_to("missing.py")
    (package / "weird.py").mkdir()
    (package / ".backup.py").write_text("LOWERING = 0\n")
    (package / ".checkpoints").mkdir()
    (package / ".checkpoints" / "_frontend.py").write_text("LOWERING = 0\n")

    assert _runtime._frontend_digest(package) == clean
    _runtime._identity_cache = None
    assert _runtime._cached_identity()["frontend"] == clean
    _runtime._identity_cache = None

    (package / "linked.py").symlink_to("_frontend.py")
    assert _runtime._frontend_digest(package) not in (clean, None)


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads any file")
def test_unreadable_frontend_file_turns_persistence_off(tmp_path, monkeypatch):
    """Lose the frontend identity, not the launch, to an unreadable file."""
    package = _fake_package(tmp_path, monkeypatch)
    _start_process_after_every_file(monkeypatch)
    cache = tmp_path / "cache"
    monkeypatch.setenv("SWAGE_CACHE_DIR", str(cache))
    _runtime, calls = _stub_compiler(monkeypatch, identity=None)
    native = [["_swageDialectsNanobind.so", 10, 20]]
    monkeypatch.setattr(_runtime, "_native_identity", lambda: native)
    monkeypatch.setattr(_runtime, "_native_libraries", lambda: [])
    (package / "_frontend.py").chmod(0)

    try:
        digest = _runtime._frontend_digest(package)
        identity = _runtime._cached_identity()
        first, messages = _compile_recording_warnings()
        second, repeated = _compile_recording_warnings()
    finally:
        (package / "_frontend.py").chmod(0o600)
        _runtime._identity_cache = None

    assert digest is None
    assert identity["frontend"] is None
    assert identity["native"] == native
    assert first == second
    assert len(calls) == 1
    assert len(messages) == 1
    assert "_frontend.py" in messages[0]
    assert "Permission denied" in messages[0]
    assert repeated == []
    assert not cache.exists()


def _kernel_k_arguments():
    """Return the specialization and contract a script compiles `k` with.

    The scripts below run in a fresh interpreter and read both from their
    last two arguments.
    """
    return [json.dumps(_spec("k")), _contract_json("k")]


_STALE_FRONTEND_SCRIPT = """
import json
import os
import pathlib
import sys
import warnings

import swage

sampled_at_import = "swage._runtime" in sys.modules
package = pathlib.Path(swage.__file__).parent
# The script edits the package it imported, so it must be the copy.
if package.parent != pathlib.Path(os.environ["PYTHONPATH"]).resolve():
    sys.exit(f"imported {package}, not the copy on PYTHONPATH")
if sys.argv[1] == "edit after import":
    with open(package / "_frontend.py", "ab") as source:
        source.write(b"# edited after this process imported swage")

from swage import _cuda_backend, _runtime

specialization = json.loads(sys.argv[-2])
contract = sys.argv[-1]
native = [["_swageDialectsNanobind.so", 1, 2]]
compiles = []
_runtime._native_identity = lambda: native
_runtime._native_libraries = lambda: []
_cuda_backend._compile_native = lambda *_args: compiles.append(1) or (
    "lowered",
    "ptx from the frontend this process loaded",
    contract,
)
identity = _runtime._cached_identity()
data = dict(specialization, frontend=identity["frontend"], native=native)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    first, second = (
        _runtime._compile_cached(
            _cuda_backend.CUDA_BACKEND, data, "k", 128, object
        )
        for _ in range(2)
    )
cache = pathlib.Path(os.environ["SWAGE_CACHE_DIR"])
on_disk = dict(data, frontend=_runtime._frontend_digest(package))
# A package outside a checkout reads the build record of `mlir_swage`,
# which imports that namespace package but never the native extension.
heavy = ("torch", "mlir_swage._mlir_libs")
print(json.dumps({
    "sampled_at_import": sampled_at_import,
    "heavy": [name for name in heavy if name in sys.modules],
    "same": first == second,
    "compiles": len(compiles),
    "warnings": [str(warning.message) for warning in caught],
    "entries": sorted(path.name for path in cache.glob("*")),
    "key": first.key,
    "key_of_the_files_on_disk": _runtime._cache_key(on_disk),
    "started": getattr(_runtime, "_PROCESS_START_NS", None),
}))
"""


def _copied_package(tmp_path):
    """Return a directory to import a private copy of `swage` from."""
    site = tmp_path / "site"
    if not site.exists():
        shutil.copytree(
            pathlib.Path(sw.__file__).parent,
            site / "swage",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    # The process start time has a resolution of one clock tick (10 ms),
    # so let the files age before a process that must trust them starts.
    time.sleep(0.05)
    return site


def _run_with_copied_frontend(tmp_path, mode):
    """Run the stale-frontend script in a process that owns its package."""
    site = _copied_package(tmp_path)
    before = time.time_ns()
    completed = subprocess.run(
        [
            sys.executable,
            # Without site-packages, an editable install of swage cannot
            # stand in for the copy on PYTHONPATH.
            "-S",
            "-c",
            _STALE_FRONTEND_SCRIPT,
            mode,
            *_kernel_k_arguments(),
        ],
        env=dict(
            os.environ,
            PYTHONPATH=str(site),
            SWAGE_CACHE_DIR=str(tmp_path / "cache"),
        ),
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["sampled_at_import"]
    assert result["heavy"] == []
    assert result["same"]
    assert result["compiles"] == 1
    tick_ns = 20_000_000
    assert before - tick_ns <= result["started"] <= time.time_ns()
    return result


def test_frontend_edited_after_import_publishes_nothing(tmp_path):
    """Never publish old code's PTX under the key of the edited files."""
    result = _run_with_copied_frontend(tmp_path, "edit after import")

    assert result["entries"] == []
    assert len(result["warnings"]) == 1
    assert "_frontend.py" in result["warnings"][0]
    assert "is not older than this process" in result["warnings"][0]


def test_frontend_unchanged_since_start_is_published(tmp_path):
    """Publish, and reuse from a second process, when nothing changed."""
    result = _run_with_copied_frontend(tmp_path, "no edit")

    assert result["entries"] == [result["key"]]
    assert result["key"] == result["key_of_the_files_on_disk"]
    assert result["warnings"] == []

    with open(tmp_path / "site" / "swage" / "_frontend.py", "ab") as source:
        source.write(b"# edited before the next process starts")
    edited = _run_with_copied_frontend(tmp_path, "no edit")

    assert edited["key"] == edited["key_of_the_files_on_disk"]
    assert edited["key"] != result["key"]
    assert edited["entries"] == sorted([result["key"], edited["key"]])
    assert edited["warnings"] == []


_FORKED_CHILD_SCRIPT = """
import json
import multiprocessing
import os
import pathlib
import sys
import time
import traceback
import warnings

import swage

package = pathlib.Path(swage.__file__).parent
# The script edits the package it imported, so it must be the copy.
if package.parent != pathlib.Path(os.environ["PYTHONPATH"]).resolve():
    sys.exit(f"imported {package}, not the copy on PYTHONPATH")


def compile_in_the_child():
    from swage import _cuda_backend, _runtime

    specialization = json.loads(sys.argv[-2])
    contract = sys.argv[-1]
    native = [["_swageDialectsNanobind.so", 1, 2]]
    compiles = []
    _runtime._native_identity = lambda: native
    _runtime._native_libraries = lambda: []
    _cuda_backend._compile_native = lambda *_args: compiles.append(1) or (
        "lowered",
        "ptx from the frontend the parent loaded",
        contract,
    )
    identity = _runtime._cached_identity()
    data = dict(specialization, frontend=identity["frontend"], native=native)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        first, second = (
            _runtime._compile_cached(
                _cuda_backend.CUDA_BACKEND, data, "k", 128, object
            )
            for _ in range(2)
        )
    cache = pathlib.Path(os.environ["SWAGE_CACHE_DIR"])
    on_disk = dict(data, frontend=_runtime._frontend_digest(package))
    pathlib.Path(sys.argv[2]).write_text(json.dumps({
        "same": first == second,
        "compiles": len(compiles),
        "warnings": [str(warning.message) for warning in caught],
        "entries": sorted(path.name for path in cache.glob("*")),
        "key": first.key,
        "key_of_the_files_on_disk": _runtime._cache_key(on_disk),
        "started": _runtime._PROCESS_START_NS,
        "forked": _runtime._process_start_ns(),
    }))


with open(package / "_frontend.py", "ab") as source:
    source.write(b"# edited after the parent imported swage")
# Let the edit age past one clock tick: the child then starts later than
# every file, and only the parent's start time shows the edit came after
# the code was loaded.
time.sleep(0.05)
if sys.argv[1] == "os.fork":
    child = os.fork()
    if child == 0:
        try:
            compile_in_the_child()
        except BaseException:
            traceback.print_exc()
            os._exit(1)
        os._exit(0)
    _, status = os.waitpid(child, 0)
    sys.exit(os.waitstatus_to_exitcode(status))
worker = multiprocessing.get_context("fork").Process(
    target=compile_in_the_child
)
worker.start()
worker.join()
sys.exit(worker.exitcode)
"""


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork")
@pytest.mark.parametrize("how", ["os.fork", "multiprocessing fork"])
def test_forked_child_does_not_publish_code_its_parent_loaded_earlier(
    tmp_path, how
):
    """Judge a forked child by the start of the process that loaded swage."""
    site = _copied_package(tmp_path)
    report = tmp_path / "report.json"
    completed = subprocess.run(
        [
            sys.executable,
            # Without site-packages, an editable install of swage cannot
            # stand in for the copy on PYTHONPATH.
            "-S",
            "-c",
            _FORKED_CHILD_SCRIPT,
            how,
            str(report),
            *_kernel_k_arguments(),
        ],
        env=dict(
            os.environ,
            PYTHONPATH=str(site),
            SWAGE_CACHE_DIR=str(tmp_path / "cache"),
        ),
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(report.read_text())

    assert result["entries"] == []
    assert result["key"] == result["key_of_the_files_on_disk"]
    assert result["started"] < result["forked"]
    assert len(result["warnings"]) == 1
    assert (
        "_frontend.py is not older than this process" in (result["warnings"][0])
    )
    assert result["same"]
    assert result["compiles"] == 1


_NO_START_TIME_SCRIPT = """
import builtins
import io
import json
import os
import sys
import time
import warnings

real_open = builtins.open
failure = sys.argv[1]


def open_stat(file, *arguments, **keywords):
    if str(file) != "/proc/self/stat":
        return real_open(file, *arguments, **keywords)
    if failure == "no /proc":
        raise FileNotFoundError(2, "No such file or directory", str(file))
    return io.StringIO("1 (python) garbled")


if failure in ("no /proc", "garbled stat"):
    builtins.open = open_stat
elif failure == "no boot clock":
    del time.CLOCK_BOOTTIME
elif failure == "zero clock tick":
    os.sysconf = lambda _name: 0
elif failure == "undefined clock tick":
    os.sysconf = lambda _name: -1

import swage  # Runs under -W error: a warning here is a failure.
from swage import _cuda_backend, _runtime

assert _runtime._PROCESS_START_NS is None
assert not _runtime._cache_off

specialization = json.loads(sys.argv[-2])
contract = sys.argv[-1]
compiles = []
_runtime._native_identity = lambda: [["_swageDialectsNanobind.so", 1, 2]]
_runtime._native_libraries = lambda: []
_cuda_backend._compile_native = lambda *_args: compiles.append(1) or (
    "l",
    "p",
    contract,
)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    for _ in range(2):
        _runtime._compile_cached(
            _cuda_backend.CUDA_BACKEND, specialization, "k", 128, object
        )
(warning,) = caught
assert "start time is unavailable" in str(warning.message), warning.message
assert compiles == [1]
assert not os.path.exists(os.environ["SWAGE_CACHE_DIR"])
"""


def _run_cold(arguments, tmp_path):
    """Run a fresh interpreter that turns every warning into an error."""
    return subprocess.run(
        [sys.executable, "-W", "error", *arguments],
        env=dict(
            os.environ,
            PYTHONPATH=str(pathlib.Path(sw.__file__).parents[1]),
            SWAGE_CACHE_DIR=str(tmp_path / "cache"),
        ),
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    "failure",
    [
        "no /proc",
        "garbled stat",
        "no boot clock",
        "zero clock tick",
        "undefined clock tick",
    ],
)
def test_import_is_silent_without_a_process_start_time(tmp_path, failure):
    """Import cleanly without a start time and warn only at first use."""
    completed = _run_cold(
        ["-c", _NO_START_TIME_SCRIPT, failure, *_kernel_k_arguments()],
        tmp_path,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""


@pytest.mark.parametrize(
    "arguments",
    [
        ["-c", "import swage"],
        ["-c", "import swage.language"],
        ["-c", "import swage.env"],
        ["-c", "import swage._runtime"],
        ["-c", "from swage import _runtime, env, language, jit"],
    ],
)
def test_package_imports_from_a_cold_interpreter(tmp_path, arguments):
    """Import each entry point first, without cycles or heavy modules."""
    check = (
        "; import sys"
        "; assert 'swage._runtime' in sys.modules"
        "; started = sys.modules['swage._runtime']._PROCESS_START_NS"
        "; assert started or sys.platform != 'linux'"
        "; assert 'torch' not in sys.modules"
        "; assert 'mlir_swage' not in sys.modules"
    )
    completed = _run_cold([arguments[0], arguments[1] + check], tmp_path)

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""


def test_environment_report_runs_with_warnings_as_errors(tmp_path):
    """Run `python -m swage.env` cleanly now that it loads the runtime."""
    completed = _run_cold(["-m", "swage.env"], tmp_path)

    assert completed.returncode == 0, completed.stderr
    assert "swage:" in completed.stdout


def _identified_process(tmp_path, monkeypatch):
    """Derive a real identity from a fake package and fake bindings."""
    package = _fake_package(tmp_path, monkeypatch)
    extension, versioned, _ = _fake_bindings(tmp_path, monkeypatch)
    _start_process_after_every_file(monkeypatch)
    cache = tmp_path / "cache"
    monkeypatch.setenv("SWAGE_CACHE_DIR", str(cache))
    _runtime, calls = _stub_compiler(monkeypatch, identity=None)
    identity = _runtime._cached_identity()
    assert identity["frontend"] is not None
    assert identity["native"] is not None
    return _runtime, calls, identity, package, versioned, cache


def test_identity_of_files_older_than_the_process_is_current(
    tmp_path, monkeypatch
):
    """Accept an identity when nothing changed since the process started."""
    _runtime, calls, identity, _, _, cache = _identified_process(
        tmp_path, monkeypatch
    )

    assert _runtime._stale_identity(identity) is None
    artifact, messages = _compile_recording_warnings()
    _runtime._identity_cache = None

    assert messages == []
    assert len(calls) == 1
    _assert_complete_entry(cache / artifact.key)


def test_identity_is_stale_without_a_process_start_time(tmp_path, monkeypatch):
    """Do not trust files on disk when the start time is unknown."""
    _runtime, calls, identity, _, _, cache = _identified_process(
        tmp_path, monkeypatch
    )
    monkeypatch.setattr(_runtime, "_PROCESS_START_NS", None)

    assert "start time" in _runtime._stale_identity(identity)
    _, messages = _compile_recording_warnings()
    _runtime._identity_cache = None

    assert len(messages) == 1
    assert len(calls) == 1
    assert not cache.exists()


@pytest.mark.parametrize("changed", ["frontend", "native"])
def test_identity_is_stale_when_a_file_changed_after_start(
    tmp_path, monkeypatch, changed
):
    """Distrust a file as new as the process, even with unchanged bytes."""
    _runtime, calls, identity, package, versioned, cache = _identified_process(
        tmp_path, monkeypatch
    )
    path = package / "_frontend.py" if changed == "frontend" else versioned
    if changed == "native":
        # File times are coarse: make the library newer than the frontend
        # by more than a timer tick, without changing its identity.
        time.sleep(0.05)
        os.utime(versioned, ns=(1_000, 2_000))
    monkeypatch.setattr(_runtime, "_PROCESS_START_NS", _changed_ns(path))

    problem = _runtime._stale_identity(identity)
    _, messages = _compile_recording_warnings()
    _runtime._identity_cache = None

    assert str(path.parent) in problem
    assert "is not older than this process" in problem
    assert len(messages) == 1
    assert problem in messages[0]
    assert len(calls) == 1
    assert not cache.exists()


@pytest.mark.parametrize("changed", ["frontend", "native"])
def test_identity_changed_during_a_compile_is_not_published(
    tmp_path, monkeypatch, changed
):
    """Keep the artifact in the process when the disk moved under it."""
    _runtime, calls, identity, package, versioned, cache = _identified_process(
        tmp_path, monkeypatch
    )
    compile_native = _compiler(
        calls, ptx="ptx from the compiler that was loaded"
    )

    def compile_while_the_disk_changes(*arguments):
        if changed == "frontend":
            (package / "_frontend.py").write_text("LOWERING = 2\n")
        else:
            versioned.write_bytes(b"another library")
        return compile_native(*arguments)

    monkeypatch.setattr(
        _cuda_backend, "_compile_native", compile_while_the_disk_changes
    )
    first, messages = _compile_recording_warnings()
    second, repeated = _compile_recording_warnings()
    _, other_key = _compile_recording_warnings(_spec("other"))
    problem = _runtime._stale_identity(identity)
    _runtime._identity_cache = None

    assert first == second
    assert first.image == "ptx from the compiler that was loaded"
    assert len(calls) == 2
    assert len(messages) == 1
    assert changed in problem
    assert "after the cache key was derived" in problem
    assert problem in messages[0]
    assert repeated == other_key == []
    assert list(cache.glob("*")) == []


def test_process_start_time_precedes_this_test():
    """Read a start time that is earlier than anything the process did."""
    from swage import _runtime

    started = _runtime._process_start_ns()

    assert started == pytest.approx(_runtime._PROCESS_START_NS, abs=10**7)
    assert started < _IMPORTED_NS < time.time_ns()


def _fake_bindings(tmp_path, monkeypatch):
    """Stand in for the directory `mlir_swage._mlir_libs` is found in."""
    from swage import _runtime

    libraries = tmp_path / "mlir_swage" / "_mlir_libs"
    libraries.mkdir(parents=True)
    extension = libraries / "_swageDialectsNanobind.cpython-313-x86_64.so"
    extension.write_bytes(b"extension")
    versioned = libraries / "libSwagePythonCAPI.so.22.1"
    versioned.write_bytes(b"compiler library")
    (libraries / "libSwagePythonCAPI.so").symlink_to(versioned.name)
    (libraries / "libnanobind-mlir_swage.so").write_bytes(b"support")
    (libraries / "__init__.py").write_text("")
    for path in (extension, versioned):
        os.utime(path, ns=(1_000, 2_000))
    spec = types.SimpleNamespace(submodule_search_locations=[str(libraries)])
    names = []
    monkeypatch.setattr(
        _runtime.importlib.util,
        "find_spec",
        lambda name: names.append(name) or spec,
    )
    return extension, versioned, names


def test_native_identity_describes_the_compiler_libraries(
    tmp_path, monkeypatch
):
    """Identify the native compiler by the contents of its libraries."""
    from swage import _runtime

    extension, versioned, names = _fake_bindings(tmp_path, monkeypatch)

    identity = _runtime._native_identity()

    assert names == ["mlir_swage._mlir_libs"]
    # The symlink counts once, as the library it leads to.
    assert identity == [
        [extension.name, f"sha256:{_sha256(b'extension')}"],
        [versioned.name, f"sha256:{_sha256(b'compiler library')}"],
    ]
    os.utime(versioned, ns=(1_000, 3_000))
    assert _runtime._native_identity() == identity
    extension.write_bytes(b"EXTENSION")
    os.utime(extension, ns=(1_000, 2_000))
    relinked = _runtime._native_identity()
    assert relinked[0] == [extension.name, f"sha256:{_sha256(b'EXTENSION')}"]
    assert relinked[1] == identity[1]

    extension.unlink()
    assert _runtime._native_identity() is None


def _sha256(contents):
    """Return the SHA-256 hex digest of `contents`."""
    return hashlib.sha256(contents).hexdigest()


@pytest.mark.parametrize("fake", [None, types.ModuleType("mlir_swage")])
def test_native_identity_is_absent_without_the_bindings(monkeypatch, fake):
    """Report no native identity, without raising, when bindings are gone."""
    from swage import _runtime

    monkeypatch.setitem(sys.modules, "mlir_swage", fake)
    monkeypatch.delitem(sys.modules, "mlir_swage._mlir_libs", raising=False)
    assert _runtime._native_identity() is None

    monkeypatch.setitem(
        sys.modules,
        "mlir_swage._mlir_libs",
        types.ModuleType("mlir_swage._mlir_libs"),
    )
    assert _runtime._native_identity() is None
    _runtime._identity_cache = None
    assert _runtime._cached_identity()["native"] is None
    _runtime._identity_cache = None


def test_cache_path_defaults_to_user_cache(monkeypatch):
    """Keep cache placement predictable without creating it during import."""
    from swage import _runtime

    monkeypatch.delenv("SWAGE_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", "/tmp/user-cache")
    assert _runtime._cache_dir() == pathlib.Path("/tmp/user-cache/swage")


def _ctypes_launch(arguments, values, grid, block=128):
    """Launch one contract through the ctypes path and decode the call.

    Returns:
        The driver function called, its grid and block, and the pointer and
        i32 parameter values that reached it, each in order.
    """
    driver = object.__new__(_cuda_backend._CudaDriver)
    calls = []
    driver._call = lambda name, *args: calls.append((name, args))
    kinds = tuple(argument.kind for argument in arguments)

    driver.launch_entry(
        0xF00D,
        _contract("kernel", block, arguments),
        (kinds, values),
        grid,
        0xABCD,
    )

    ((name, call),) = calls
    parameters = call[-2]
    decoded = {"ptr": [], "i32": []}
    for index, kind in enumerate(kinds):
        storage = ctypes.c_void_p if kind == "ptr" else ctypes.c_int32
        decoded[kind].append(
            ctypes.cast(
                parameters[index], ctypes.POINTER(storage)
            ).contents.value
        )
    return name, call[1:7], decoded["ptr"], decoded["i32"]


def test_driver_checks_a_contract_once_and_every_launch_against_it(
    monkeypatch,
):
    """Check a contract at its first launch, and each launch's own values.

    A later launch of the same contract skips the contract check but still
    refuses kinds the contract does not have and a malformed grid.
    """
    driver = object.__new__(_cuda_backend._CudaDriver)
    launches = []
    driver._call = lambda name, *args: launches.append(name)
    checks = []
    check = _cuda_backend._CudaDriver._check_contract
    monkeypatch.setattr(
        _cuda_backend._CudaDriver,
        "_check_contract",
        lambda self, contract: checks.append(contract) or check(self, contract),
    )
    monkeypatch.setattr(_cuda_backend, "_checked_contracts", {})
    contract = _contract("kernel", 128, _FIXED_ARGUMENTS)
    kinds = tuple(argument.kind for argument in _FIXED_ARGUMENTS)
    values = (0x10, 0x20, 0x30, 129)

    for _ in range(3):
        driver.launch_entry(0xF00D, contract, (kinds, values), (2, 1, 1), 0)
    with pytest.raises(ValueError, match="kinds do not match contract"):
        driver.launch_entry(
            0xF00D, contract, (kinds[:-1] + ("i64",), values), (2, 1, 1), 0
        )
    with pytest.raises(ValueError, match="grid must contain exactly three"):
        driver.launch_entry(0xF00D, contract, (kinds, values), (2, 0, 1), 0)

    assert checks == [contract]
    assert launches == ["cuLaunchKernel"] * 3


def test_driver_marshals_pointer_and_i32_parameters():
    """Pass raw pointers and an i32 through the CUDA Driver ABI."""
    name, geometry, pointers, scalars = _ctypes_launch(
        _FIXED_ARGUMENTS, (0x10, 0x20, 0x30, 129), (2, 1, 1)
    )

    assert name == "cuLaunchKernel"
    assert geometry == (2, 1, 1, 128, 1, 1)
    assert pointers == [0x10, 0x20, 0x30]
    assert scalars == [129]


def test_driver_marshals_four_pointer_segmented_task_abi():
    """Pass four pointers and three i32 counts through the CUDA Driver ABI."""
    name, geometry, pointers, scalars = _ctypes_launch(
        _TASK_ARGUMENTS,
        (0x10, 0x20, 0x30, 0x40, 4096, 7, 5),
        (7, 1, 1),
        block=32,
    )

    assert name == "cuLaunchKernel"
    assert geometry == (7, 1, 1, 32, 1, 1)
    assert pointers == [0x10, 0x20, 0x30, 0x40]
    assert scalars == [4096, 7, 5]


def test_driver_marshals_four_pointer_fused_segmented_abi():
    """Pass four pointers and four i32 counts through the CUDA Driver ABI."""
    name, geometry, pointers, scalars = _ctypes_launch(
        _FUSED_ARGUMENTS,
        (0x10, 0x20, 0x30, 0x40, 4096, 5, 7, 9),
        (9, 1, 1),
    )

    assert name == "cuLaunchKernel"
    assert geometry == (9, 1, 1, 128, 1, 1)
    assert pointers == [0x10, 0x20, 0x30, 0x40]
    assert scalars == [4096, 5, 7, 9]


def test_driver_marshals_ten_pointer_persistent_abi():
    """Pass queues, dependencies, scratch, and counts through the ABI."""
    buffers = (0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80, 0x90, 0xA0)
    name, geometry, pointers, scalars = _ctypes_launch(
        _PERSISTENT_ARGUMENTS, (*buffers, 4096, 5, 7, 11, 2, 13), (168, 1, 1)
    )

    assert name == "cuLaunchKernel"
    assert geometry == (168, 1, 1, 128, 1, 1)
    assert pointers == list(buffers)
    assert scalars == [4096, 5, 7, 11, 2, 13]


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


def test_compiler_identity_is_cached_per_process(tmp_path, monkeypatch):
    """Spawn the git subprocesses once, not twice per launch."""
    from swage import _runtime

    commands = []

    def fake_run(command, **kwargs):
        commands.append((command, kwargs["cwd"]))
        return subprocess.CompletedProcess(
            command, 0, stdout="abc\n", stderr=""
        )

    package = _fake_package(tmp_path, monkeypatch)
    checkout = package.parents[1]
    (checkout / ".git").mkdir()
    (checkout / "cmake").mkdir()
    (checkout / "cmake" / "llvm-version.txt").write_text("llvmorg-test\n")
    native = [["_swageDialectsNanobind.so", 10, 20]]
    monkeypatch.setattr(_runtime, "_native_identity", lambda: native)
    monkeypatch.setattr(_runtime.subprocess, "run", fake_run)
    _runtime._identity_cache = None
    results = [_runtime._cached_identity() for _ in range(3)]
    _runtime._identity_cache = None

    assert results[0] is results[1] is results[2]
    assert [cwd for _, cwd in commands] == [checkout, checkout]
    assert results[0] == {
        "revision": "abc",
        "clean": False,
        "llvm": "llvmorg-test",
        "frontend": mock.ANY,
        "native": native,
    }
    assert len(results[0]["frontend"]) == 64


def test_identity_cache_notices_a_monkeypatched_identity(monkeypatch):
    """Tests that fake the identity must see their fake, not stale cache."""
    from swage import _runtime

    _runtime._identity_cache = None
    _runtime._cached_identity()
    fake = _identity(revision="r")
    monkeypatch.setattr(_runtime, "_compiler_identity", lambda: fake)
    assert _runtime._cached_identity() == fake


def _packaged_build_info():
    """Return a valid build record of a packaged native compiler."""
    return {
        "schema_version": 2,
        "package_version": "0.5.2",
        "source_revision": "a" * 40,
        "source_clean": True,
        "frontend_digest": "d" * 64,
        "llvm_version": "llvmorg-22.1.8",
        "build_type": "Release",
    }


@pytest.mark.parametrize("malformed", [False, True])
def test_packaged_identity_describes_a_package_outside_a_checkout(
    tmp_path, monkeypatch, malformed
):
    """Take provenance from the build record, and never key on it.

    A package outside a Swage checkout reports the revision, cleanliness,
    and LLVM release its build recorded, and an invalid record gives no
    revision instead of a guess. Provenance is diagnostic only: the cache
    key follows the contents of the frontend and the native libraries.
    """
    from swage import _runtime

    info = _packaged_build_info()

    def build_info():
        if malformed:
            raise ValueError("invalid native build metadata: source_revision")
        return info

    _fake_package(tmp_path, monkeypatch)
    native = [["_swageDialectsNanobind.so", 10, 20]]
    monkeypatch.setattr(_runtime, "_native_identity", lambda: native)
    monkeypatch.setattr(_runtime._native, "build_info", build_info)
    monkeypatch.setattr(
        _runtime.subprocess,
        "run",
        mock.Mock(side_effect=AssertionError("packaged identity ran git")),
    )

    identity = _runtime._compiler_identity()

    if malformed:
        assert (identity["revision"], identity["clean"]) == (None, False)
        assert identity["llvm"] is None
    else:
        assert (identity["revision"], identity["clean"]) == ("a" * 40, True)
        assert identity["llvm"] == "llvmorg-22.1.8"
    assert identity["native"] == native
    assert len(identity["frontend"]) == 64

    original = _fresh_key(_runtime)
    info.update(
        source_revision="b" * 40,
        source_clean=False,
        llvm_version="llvmorg-23.1.0",
    )
    assert _fresh_key(_runtime) == original
    _runtime._identity_cache = None


def test_checkout_identity_precedes_the_packaged_record(tmp_path, monkeypatch):
    """Describe a Swage checkout by git, whatever record a build left."""
    from swage import _runtime

    (tmp_path / ".git").mkdir()
    (tmp_path / "cmake").mkdir()
    (tmp_path / "cmake" / "llvm-version.txt").write_text("llvmorg-source\n")
    monkeypatch.setattr(
        _runtime, "__file__", str(tmp_path / "python/swage/_runtime.py")
    )
    monkeypatch.setattr(
        _runtime._native,
        "build_info",
        mock.Mock(side_effect=AssertionError("a checkout read the record")),
    )
    monkeypatch.setattr(_runtime, "_native_identity", lambda: None)
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
        "frontend": None,
        "native": None,
    }


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


def _committed_package(repository, package, *, pin):
    """Commit a `swage` package at `package` inside a new git repository.

    Args:
        repository: Directory that becomes the root of the repository.
        package: Path of the package below `repository`.
        pin: Directory below `repository` that receives
            `cmake/llvm-version.txt`, or None for no pin file.

    Returns:
        The package directory and the HEAD of the repository.
    """
    directory = repository / package
    directory.mkdir(parents=True)
    (directory / "__init__.py").write_text("VERSION = 1\n")
    if pin is not None:
        (repository / pin / "cmake").mkdir(parents=True)
        (repository / pin / "cmake" / "llvm-version.txt").write_text(
            "llvmorg-test\n"
        )
    _git(repository, "init", "--quiet")
    _git(repository, "add", "--all")
    _git(repository, "commit", "--quiet", "--message", "initial")
    return directory, _git(repository, "rev-parse", "HEAD")


def _identity_of(monkeypatch, package):
    """Return the compiler identity of a package at `package`."""
    from swage import _runtime

    monkeypatch.setattr(_runtime, "_package_dir", lambda: package)
    monkeypatch.setattr(_runtime, "_native_identity", lambda: None)
    return _runtime._compiler_identity()


@pytest.mark.skipif(shutil.which("git") is None, reason="git unavailable")
def test_revision_names_the_swage_checkout_of_the_package(
    tmp_path, monkeypatch
):
    """Report HEAD for `python/swage` of a checkout that holds the pin."""
    package, head = _committed_package(
        tmp_path, pathlib.Path("python", "swage"), pin="."
    )

    clean = _identity_of(monkeypatch, package)
    (package / "__init__.py").write_text("VERSION = 2\n")
    dirty = _identity_of(monkeypatch, package)

    assert (clean["revision"], clean["clean"]) == (head, True)
    assert (dirty["revision"], dirty["clean"]) == (head, False)


@pytest.mark.skipif(shutil.which("git") is None, reason="git unavailable")
@pytest.mark.parametrize(
    ("package", "pin"),
    [
        pytest.param(("src", "swage"), None, id="vendored-under-src"),
        pytest.param(("src", "swage"), ".", id="vendored-beside-a-pin"),
        pytest.param(("python", "swage"), None, id="layout-without-pin"),
        pytest.param(("python", "other"), ".", id="another-package-name"),
        pytest.param(
            ("vendor", "swage", "python", "swage"),
            "vendor/swage",
            id="whole-tree-vendored",
        ),
    ],
)
def test_revision_is_absent_for_a_package_in_another_repository(
    tmp_path, monkeypatch, package, pin
):
    """Do not report the HEAD of a repository that is not Swage's own.

    A copy of the package vendored into an application repository sits
    under that repository's root. Its HEAD is the application's commit and
    would name the wrong project in a bug report.
    """
    directory, head = _committed_package(
        tmp_path, pathlib.Path(*package), pin=pin
    )

    identity = _identity_of(monkeypatch, directory)

    assert len(head) == 40
    assert (identity["revision"], identity["clean"]) == (None, False)


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
    monkeypatch.setattr(_cuda_backend, "_compile_native", _compiler())
    monkeypatch.setattr(_runtime, "_compiler_identity", _unpersisted_identity)
    monkeypatch.setattr(_cuda_backend, "_get_driver", lambda: driver)
    _runtime._identity_cache = None

    for _ in range(3):
        add_kernel.launch(
            arguments=_arguments(torch),
            constexprs={"BLOCK": 128},
            grid=(2,),
        )

    assert len(emissions) == 1
    assert len(driver.launches) == 3
    artifact = next(iter(_runtime._artifact_cache.values()))
    assert artifact.contract_json == _contract_json()
    assert tuple(
        argument.source_index for argument in artifact.contract.arguments
    ) == (0, 1, 2, 3)


def test_device_fact_cache_is_isolated_per_torch_module(monkeypatch):
    """Never let one process's device cache leak across torch modules."""
    from swage import _runtime

    driver = _Driver()
    monkeypatch.setattr(_cuda_backend, "_get_driver", lambda: driver)
    monkeypatch.setattr(_runtime, "_compile_cached", _fixed_compile_artifact)

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


def test_driver_prefers_the_native_launcher_when_available(monkeypatch):
    """Dispatch through the compiled launcher when the bindings exist."""
    from swage import _runtime

    _stub_cuda_library(monkeypatch)
    calls = []
    native = types.ModuleType("_swageDialectsNanobind")
    native.swage = types.SimpleNamespace(
        __version__=sw.__version__,
        __source_revision__="unknown",
        _launch_cuda_kernel=lambda *arguments: calls.append(arguments),
        _FixedCUDALaunch=object,
    )
    monkeypatch.setattr(_runtime, "_verified_bindings", None)
    libs = types.ModuleType("mlir_swage._mlir_libs")
    libs._swageDialectsNanobind = native
    monkeypatch.setitem(sys.modules, "mlir_swage._mlir_libs", libs)
    monkeypatch.setitem(
        sys.modules, "mlir_swage._mlir_libs._swageDialectsNanobind", native
    )
    launches = [
        (_FIXED_ARGUMENTS, 128, (3, 1, 1), (0x1, 0x2, 0x3, 129)),
        (_DIRECT_ARGUMENTS, 128, (4, 1, 1), (0x1, 0x2, 0x3, 60, 4)),
        (_TASK_ARGUMENTS, 32, (5, 1, 1), (1, 2, 3, 4, 60, 5, 4)),
        (_FUSED_ARGUMENTS, 128, (6, 1, 1), (1, 2, 3, 4, 60, 5, 1, 4)),
        (
            _PERSISTENT_ARGUMENTS,
            512,
            (7, 1, 1),
            (*range(1, 11), 60, 5, 1, 7, 2, 4),
        ),
    ]

    driver = _cuda_backend._CudaDriver()
    for arguments, block, grid, values in launches:
        kinds = tuple(argument.kind for argument in arguments)
        driver.launch_entry(
            7, _contract("kernel", block, arguments), (kinds, values), grid, 9
        )

    assert driver._native_fixed_launcher is object
    assert calls == [
        (
            tuple(argument.kind for argument in arguments),
            values,
            grid,
            (block, 1, 1),
            0,
            9,
            7,
        )
        for arguments, block, grid, values in launches
    ]
    assert not driver.library.cuLaunchKernel.called


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
        _contract(),
        (("ptr", "ptr", "ptr", "i32"), (0x1, 0x2, 0x3, 129)),
        (3, 1, 1),
        9,
    )
    assert driver.library.cuLaunchKernel.called


def test_segmented_artifact_compile_and_context_load_reuse(monkeypatch):
    """Reuse one compiled artifact and one loaded module per CUDA context."""
    from swage import _runtime

    identity = _identity(revision=None, clean=False, native=None)
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
    contract_json = _contract_json(
        "segmented_sum", 32, arguments=_TASK_ARGUMENTS
    )
    compiles = []
    emits = []

    def compile_native(module, kernel, block, target, kind, options):
        compiles.append((module, kernel, block, target, kind, options))
        return "lowered", "ptx", contract_json

    monkeypatch.setattr(_cuda_backend, "_compile_native", compile_native)
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


def _install_fake_compiler(monkeypatch, runtime, callback=None):
    monkeypatch.setattr(runtime, "_cached_identity", _unpersisted_identity)
    compile_native = _compiler()

    def compile_with_callback(*arguments):
        if callback is not None:
            callback()
        return compile_native(*arguments)

    monkeypatch.setattr(_cuda_backend, "_compile_native", compile_with_callback)


def test_compiled_artifact_does_not_retain_emitted_module(monkeypatch):
    """Keep compiler artifacts free of semantic modules and caller storage."""
    from swage import _runtime

    class _Module:
        pass

    module = _Module()
    reference = weakref.ref(module)
    holder = [module]
    _install_fake_compiler(monkeypatch, _runtime)
    artifact = _compile(emit=lambda: holder[0], lowering_kind="fixed")
    holder.clear()
    del module
    gc.collect()

    assert artifact.image == "ptx"
    assert reference() is None


def test_memory_artifact_lru_is_bounded_and_refreshes_recency(monkeypatch):
    """Bound complete artifacts and refresh a warm specialization."""
    from swage import _runtime

    monkeypatch.setenv("SWAGE_MEMORY_CACHE_ENTRIES", "2")
    _install_fake_compiler(monkeypatch, _runtime)

    first = _compile(key="first")
    _compile(key="second")
    assert (
        _compile(emit=lambda: pytest.fail("warm hit emitted MLIR"), key="first")
        is first
    )
    _compile(key="third")

    assert list(_runtime._artifact_cache) == ["first", "third"]
    assert all(
        artifact.lowered == "lowered"
        and artifact.image == "ptx"
        and artifact.contract.entry == "add_kernel"
        for artifact in _runtime._artifact_cache.values()
    )


@pytest.mark.parametrize("module", ["_runtime", "_cuda_backend"])
@pytest.mark.parametrize(
    "configured", ["", "0", "-1", "+", "+1", "1.5", "many"]
)
def test_memory_cache_entries_rejects_invalid_values(
    monkeypatch, module, configured
):
    """Reject every noncanonical or nonpositive process-cache capacity.

    The bound applies to the artifacts of `_runtime` and to the loaded
    modules of `_cuda_backend` alike.
    """
    import swage

    monkeypatch.setenv("SWAGE_MEMORY_CACHE_ENTRIES", configured)
    with pytest.raises(
        ValueError,
        match="SWAGE_MEMORY_CACHE_ENTRIES must be a positive integer",
    ):
        getattr(swage, module)._memory_cache_limit()


def test_same_specialization_compilation_is_coalesced(monkeypatch):
    """Make concurrent callers of one key share a single compile."""
    from swage import _runtime

    started = threading.Event()
    release = threading.Event()
    calls = []

    def callback():
        calls.append(True)
        started.set()
        assert release.wait(5)

    _install_fake_compiler(monkeypatch, _runtime, callback)
    with ThreadPoolExecutor(max_workers=8) as pool:
        first = pool.submit(_compile, key="shared")
        assert started.wait(5)
        others = [pool.submit(_compile, key="shared") for _ in range(7)]
        release.set()
        results = [first.result(), *(future.result() for future in others)]

    assert len(calls) == 1
    assert all(result is results[0] for result in results)


def test_different_specializations_compile_concurrently(monkeypatch):
    """Compile independent keys at once, and never make a hit wait.

    Each miss holds a compile place of the cold-path lock, not the lock, so
    a second key compiles while the first is in flight. A key the process
    holds is served meanwhile.
    """
    from swage import _runtime

    _install_fake_compiler(monkeypatch, _runtime)
    warm = _compile(key="warm")
    barrier = threading.Barrier(2)
    entered = []

    def callback():
        entered.append(threading.get_ident())
        # Both compiles must be inside the compiler at once to pass.
        barrier.wait(5)

    _install_fake_compiler(monkeypatch, _runtime, callback)
    with ThreadPoolExecutor(max_workers=3) as pool:
        left = pool.submit(_compile, key="left")
        right = pool.submit(_compile, key="right")
        hit = pool.submit(
            _compile,
            emit=lambda: pytest.fail("warm hit emitted MLIR"),
            key="warm",
        )
        assert hit.result(timeout=5) is warm
        compiled = [left.result(timeout=5), right.result(timeout=5)]

    assert len(set(entered)) == 2
    assert [artifact.key for artifact in compiled] == ["left", "right"]


def test_a_failed_compile_is_retried_by_a_waiting_caller(monkeypatch):
    """Let a caller that waited on a failed compile compile in its place."""
    from swage import _runtime

    started = threading.Event()
    release = threading.Event()
    calls = []

    def callback():
        calls.append(threading.get_ident())
        if len(calls) == 1:
            started.set()
            assert release.wait(5)
            raise RuntimeError("first compile failed")

    _install_fake_compiler(monkeypatch, _runtime, callback)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(_compile, key="retried")
        assert started.wait(5)
        second = pool.submit(_compile, key="retried")
        # Time enough for the second caller to find the key in flight.
        time.sleep(0.05)
        release.set()
        with pytest.raises(RuntimeError, match="first compile failed"):
            first.result(timeout=5)
        artifact = second.result(timeout=5)

    assert len(calls) == 2
    assert artifact.key == "retried"
    assert not _runtime._compile_flights


def test_exit_and_fork_wait_for_a_compile_in_flight(monkeypatch):
    """Hold the exit and fork handlers until every compile place is back."""
    from swage import _runtime

    lock = _runtime._ColdPathLock()
    monkeypatch.setattr(_runtime, "_compile_lock", lock)
    started = threading.Event()
    release = threading.Event()

    def callback():
        started.set()
        assert release.wait(5)

    _install_fake_compiler(monkeypatch, _runtime, callback)
    forked = threading.Event()

    def fork_handlers():
        lock.before_fork()
        forked.set()
        lock.after_fork()

    with ThreadPoolExecutor(max_workers=2) as pool:
        compiled = pool.submit(_compile, key="in-flight")
        assert started.wait(5)
        forking = pool.submit(fork_handlers)
        time.sleep(0.05)
        assert not forked.is_set()
        release.set()
        forking.result(timeout=5)
        assert compiled.result(timeout=5).key == "in-flight"

    # Exit waits for a place as well, and keeps the lock afterwards.
    holder = threading.Event()
    done = threading.Event()

    def hold_a_place():
        with lock.compiling():
            holder.set()
            assert done.wait(5)

    thread = threading.Thread(target=hold_a_place)
    thread.start()
    assert holder.wait(5)
    assert not lock.close(0.05)
    done.set()
    thread.join(5)


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
        self.record_failures = set()

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
        if event in self.record_failures:
            raise RuntimeError("record failed")
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


def _install_lifecycle_driver(monkeypatch):
    """Keep one loaded module resident and model its unload with a driver."""
    monkeypatch.setenv("SWAGE_MEMORY_CACHE_ENTRIES", "1")
    driver = _LifecycleDriver()
    monkeypatch.setattr(_cuda_backend, "_get_driver", lambda: driver)
    return driver


def _lease(key):
    """Lease the module of a fixed artifact named `key`."""
    return _cuda_backend.CUDA_BACKEND.lease(_artifact(key))


def _retire_all_but(key):
    """Lease and release another module, which retires the resident one."""
    _lease(key).release()


def _launch_fake_loaded(lease, stream, *, capturing=False):
    artifact = _artifact()
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
    driver = _install_lifecycle_driver(monkeypatch)
    lease = _lease("first")
    _launch_fake_loaded(lease, 0)
    _launch_fake_loaded(lease, 22)
    events = tuple(lease.entry.events.values())
    driver.complete_stream(22)
    lease.release()
    _retire_all_but("second")

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
    driver = _install_lifecycle_driver(monkeypatch)
    lease = _lease("leased")
    _launch_fake_loaded(lease, 0)
    event = lease.entry.events[0]
    assert driver.event_query(event) is True
    _retire_all_but("replacement")
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
    driver = _install_lifecycle_driver(monkeypatch)
    lease = _lease("revived")
    _launch_fake_loaded(lease, 0)
    event = lease.entry.events[0]
    lease.release()
    _retire_all_but("replacement")
    _cuda_backend._poll_deferred(driver, 1)
    assert driver.unloaded == []
    assert driver.event_query(event) is False

    revived = _lease("revived")
    assert revived.entry is lease.entry
    driver.complete_stream(0)
    _launch_fake_loaded(revived, 0)
    # The first fence is complete even though the later launch is not.
    assert driver.event_query(event) is True
    revived.release()
    _retire_all_but("third")
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
    driver = _install_lifecycle_driver(monkeypatch)
    lease = _lease("failed-launch")
    driver.launch_error = RuntimeError("launch failed")
    with pytest.raises(RuntimeError):
        _launch_fake_loaded(lease, stream)
    if stream:
        driver.close_stream(stream)
    lease.release()
    _retire_all_but("replacement")
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
    driver = _install_lifecycle_driver(monkeypatch)
    lease = _lease("external")
    _launch_fake_loaded(lease, 11)
    driver.close_stream(11)
    lease.release()
    _retire_all_but("replacement")
    _cuda_backend._poll_deferred(driver, 1)
    assert driver.unloaded == []

    driver.complete_stream(11)
    _cuda_backend._poll_deferred(driver, 1)
    assert driver.unloaded == [(1, lease.entry.module)]


def test_retired_modules_of_one_key_both_stay_queued(monkeypatch):
    """Unload an older module whose key a newer retired module shares.

    A module that another thread is fencing is not revived, so a lease of
    its key loads a new module. When that one retires too, the older module
    must stay queued: dropping it would leave it loaded for good.
    """
    driver = _install_lifecycle_driver(monkeypatch)
    first = _lease("same")
    _launch_fake_loaded(first, 11)
    older = first.entry
    first.release()
    _retire_all_but("other")
    # Another thread's poll is fencing the older module.
    older.unloading = True
    second = _lease("same")
    newer = second.entry
    older.unloading = False
    second.release()
    _retire_all_but("other")

    assert newer is not older
    assert newer.key == older.key
    assert {id(entry) for entry in _cuda_backend._retired_loaded.values()} >= {
        id(older),
        id(newer),
    }
    driver.complete_stream(11)
    _cuda_backend._poll_deferred(driver, 1)

    assert {(1, older.module), (1, newer.module)} <= set(driver.unloaded)
    assert older not in _cuda_backend._retired_loaded.values()
    assert newer not in _cuda_backend._retired_loaded.values()


def test_capture_pinned_module_is_never_explicitly_unloaded(monkeypatch):
    """Leave graph-captured module and event ownership to context teardown."""
    driver = _install_lifecycle_driver(monkeypatch)
    lease = _lease("captured")
    _launch_fake_loaded(lease, 11, capturing=True)
    driver.complete_stream(11)
    lease.release()
    _retire_all_but("replacement")
    _cuda_backend._poll_deferred(driver, 1)

    assert driver.destroyed == []
    assert driver.unloaded == []


def test_event_record_failure_permanently_blocks_explicit_unload(monkeypatch):
    """Never treat an unrecorded completion event as proof of completion.

    The failure is reported once as a warning, never raised, and the other
    retired modules still unload.
    """
    driver = _install_lifecycle_driver(monkeypatch)
    lease = _lease("failed-record")
    _launch_fake_loaded(lease, 0)
    independent = _lease("independent")
    _launch_fake_loaded(independent, 0)
    independent_events = tuple(independent.entry.events.values())
    _retire_all_but("replacement")
    lease.release()
    independent.release()
    driver.record_failures.add(lease.entry.events[0])
    with pytest.warns(
        RuntimeWarning, match="left 1 unused CUDA module loaded: record failed"
    ):
        _cuda_backend._poll_deferred(driver, 1)
    assert lease.entry.unload_blocked

    driver.record_failures.clear()
    driver.complete_stream(0)
    _cuda_backend._poll_deferred(driver, 1)
    _cuda_backend._poll_deferred(driver, 1)

    assert set(driver.destroyed) == set(independent_events)
    assert driver.unloaded == [(1, independent.entry.module)]


def test_stream_wrapper_cache_is_module_scoped_and_bounded(monkeypatch):
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
    monkeypatch.setattr(
        _cuda_backend, "_stream_objects", weakref.WeakKeyDictionary()
    )

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
        cuEventRecord=lambda event, stream: (
            calls.append(("record", event.value, stream.value)) or 0
        ),
        cuEventQuery=mock.Mock(side_effect=[600, 0, 700]),
        cuModuleUnload=lambda module: (
            calls.append(("unload", module.value)) or 0
        ),
        cuGetErrorName=lambda *_args: 0,
        cuGetErrorString=lambda *_args: 0,
    )
    driver.library = library
    driver._event_destroy = lambda event: (
        calls.append(("destroy", event.value)) or 0
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


def _assert_unavailable(error, code, backend):
    assert isinstance(error, sw.SwageError)
    assert isinstance(error, RuntimeError)
    assert error.code == code
    assert error.backend == backend
    assert isinstance(error.remediation, str) and error.remediation
    assert str(error).endswith(f"; {error.remediation}")


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


@pytest.mark.parametrize(
    ("backend", "reported"), [("cpu", "cpu"), ("cuda", "cuda")]
)
@pytest.mark.parametrize("failure_type", [ImportError, OSError])
def test_native_compile_import_failure_is_backend_unavailable(
    backend, reported, failure_type
):
    """Compilation requires native bindings regardless of selected backend.

    The CUDA adapter loads them through the binding version check, which
    reports the native bindings as what is unavailable.
    """
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
    _assert_unavailable(caught.value, "native-unavailable", reported)
    assert caught.value.__cause__ is cause


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_native_compiler_failure_is_not_unavailability(monkeypatch, backend):
    """Propagate compiler failures unchanged after native loading succeeds."""
    from swage import _backends, _native, _runtime

    cause = RuntimeError("compiler failure")
    native = types.SimpleNamespace(
        _compile_fixed_host=mock.Mock(side_effect=cause),
        _compile_ptx=mock.Mock(side_effect=cause),
    )
    monkeypatch.setattr(_native, "load_extension", lambda **_kwargs: native)
    monkeypatch.setattr(_runtime, "_native_bindings", lambda: native)
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
    driver._context_id = None
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

    adapter = _RecordingBackend(backend)
    adapter.persistent_cache = backend == "cuda"
    torch, _ = _fake_torch()
    _install_recording_backends(monkeypatch, torch, adapter)
    monkeypatch.setattr(_runtime, "_compiler_identity", _identity)
    monkeypatch.setattr(_runtime, "_stale_identity", lambda _identity: None)
    _runtime._identity_cache = None
    monkeypatch.setenv("SWAGE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("SWAGE_TEST_SECRET", "secret-must-not-be-logged")
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
    monkeypatch.setattr(_runtime, "_compiler_identity", _identity)
    monkeypatch.setattr(_runtime, "_stale_identity", lambda _identity: None)
    _runtime._identity_cache = None
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
    _runtime._identity_cache = None


def test_atomic_write_closes_its_descriptor_only_once(tmp_path, monkeypatch):
    """Leave a descriptor that reuses the number alone when a publish fails."""
    from swage import _runtime

    unrelated = {}

    def fail_after_another_open(_source, _target):
        # Another thread opens a file now. It receives the lowest free
        # number, which is the descriptor the write has just closed.
        unrelated["descriptor"] = os.open(
            tmp_path / "unrelated", os.O_CREAT | os.O_WRONLY, 0o600
        )
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(_runtime.os, "replace", fail_after_another_open)
    with pytest.raises(OSError, match="No space left on device"):
        _runtime._atomic_write(tmp_path / "kernel.ptx", "payload")

    try:
        assert os.write(unrelated["descriptor"], b"still open") == 10
    finally:
        try:
            os.close(unrelated["descriptor"])
        except OSError:
            pass  # The defect closed it; the assertion above reports that.
    assert [path.name for path in tmp_path.iterdir()] == ["unrelated"]


def test_atomic_write_closes_a_descriptor_it_could_not_wrap(
    tmp_path, monkeypatch
):
    """Close the raw descriptor when no file object took ownership of it."""
    from swage import _runtime

    opened = []
    mkstemp = _runtime.tempfile.mkstemp

    def recording_mkstemp(**options):
        descriptor, name = mkstemp(**options)
        opened.append(descriptor)
        return descriptor, name

    def refuse(_descriptor, _mode):
        raise MemoryError("no file object")

    monkeypatch.setattr(_runtime.tempfile, "mkstemp", recording_mkstemp)
    monkeypatch.setattr(_runtime.os, "fdopen", refuse)
    with pytest.raises(MemoryError, match="no file object"):
        _runtime._atomic_write(tmp_path / "kernel.ptx", "payload")

    with pytest.raises(OSError, match="Bad file descriptor"):
        os.fstat(opened[0])
    assert list(tmp_path.iterdir()) == []


def _no_current_context():
    """Raise what the driver raises on a thread without a CUDA context."""
    raise sw.BackendUnavailableError(
        "PyTorch has no current CUDA context",
        code="cuda-context-unavailable",
        backend="cuda",
        remediation="initialize PyTorch CUDA on the selected device",
    )


def test_launch_gives_a_thread_without_a_context_its_device_context(
    monkeypatch,
):
    """Make the validated device's context current and launch in it."""
    torch, _ = _fake_torch(current_device=0)
    driver = _install_launch_fakes(monkeypatch, torch)
    made_current = []
    torch.cuda.set_device = made_current.append

    def current_context():
        if not made_current:
            _no_current_context()
        return 0xCAFE

    driver.current_context = current_context
    arguments = _arguments(torch)

    add_kernel.launch(arguments=arguments, constexprs={"BLOCK": 128}, grid=(2,))

    assert made_current == [0]
    assert driver.loads == [("ptx", "add_kernel")]
    assert len(driver.launches) == 1


def test_launch_reports_a_context_it_could_not_make_current(monkeypatch):
    """Keep the driver's error when the device has no context to give."""
    torch, _ = _fake_torch()
    driver = _install_launch_fakes(monkeypatch, torch)
    made_current = []
    torch.cuda.set_device = made_current.append
    driver.current_context = _no_current_context

    with pytest.raises(
        sw.BackendUnavailableError, match="no current CUDA context"
    ):
        add_kernel.launch(
            arguments=_arguments(torch), constexprs={"BLOCK": 128}, grid=(2,)
        )

    assert made_current == [0]
    assert driver.loads == driver.launches == []


def test_launch_does_not_touch_a_context_that_is_current(monkeypatch):
    """Never set the device on a thread that already has a context."""
    torch, _ = _fake_torch()
    driver = _install_launch_fakes(monkeypatch, torch)
    made_current = []
    torch.cuda.set_device = made_current.append

    add_kernel.launch(
        arguments=_arguments(torch), constexprs={"BLOCK": 128}, grid=(2,)
    )

    assert made_current == []
    assert len(driver.launches) == 1


def _run_script(script, tmp_path, timeout=60, arguments=()):
    """Run `script` in a fresh interpreter with the package on its path."""
    return subprocess.run(
        [sys.executable, "-c", script, *arguments],
        env=dict(
            os.environ,
            PYTHONPATH=str(pathlib.Path(sw.__file__).parents[1]),
            SWAGE_CACHE_DIR=str(tmp_path / "cache"),
        ),
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )


_EXIT_DURING_A_COMPILE_SCRIPT = """
import sys
import threading
import time

from swage import _runtime

in_flight = threading.Event()


def compile_twice():
    with _runtime._compile_lock:
        in_flight.set()
        time.sleep(0.5)
        sys.stdout.write("first compile finished\\n")
        sys.stdout.flush()
    with _runtime._compile_lock:
        sys.stdout.write("second compile started\\n")
        sys.stdout.flush()


threading.Thread(target=compile_twice, daemon=True).start()
in_flight.wait()
# The main thread ends here, while the daemon thread holds the lock.
"""


def test_interpreter_exit_waits_for_the_compile_in_flight(tmp_path):
    """Let a compile finish before finalization and start no other."""
    completed = _run_script(_EXIT_DURING_A_COMPILE_SCRIPT, tmp_path)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "first compile finished\n"
    assert completed.stderr == ""


_EXIT_WITH_A_STUCK_COMPILE_SCRIPT = """
import threading
import time

from swage import _runtime

_runtime._EXIT_WAIT_SECONDS = 0.3
in_flight = threading.Event()


def never_finish():
    with _runtime._compile_lock:
        in_flight.set()
        time.sleep(600)


threading.Thread(target=never_finish, daemon=True).start()
in_flight.wait()
"""


def test_interpreter_exit_does_not_wait_forever(tmp_path):
    """Give up the wait at its bound, so a stuck thread cannot hold exit."""
    started = time.monotonic()
    completed = _run_script(_EXIT_WITH_A_STUCK_COMPILE_SCRIPT, tmp_path)

    assert completed.returncode == 0, completed.stderr
    assert time.monotonic() - started < 30


_STUCK_COMPILE_AND_A_LATER_HANDLER_SCRIPT = """
import atexit
import sys
import threading
import time


def use_the_cold_path():
    from swage import _runtime

    try:
        with _runtime._compile_lock:
            sys.stdout.write("took a lock that another thread holds\\n")
    except RuntimeError as error:
        sys.stdout.write(f"refused: {error}\\n")
    sys.stdout.flush()


atexit.register(use_the_cold_path)

from swage import _runtime  # noqa: E402

_runtime._EXIT_WAIT_SECONDS = 0.3
in_flight = threading.Event()


def never_finish():
    with _runtime._compile_lock:
        in_flight.set()
        time.sleep(600)


threading.Thread(target=never_finish, daemon=True).start()
in_flight.wait()
"""


def test_exit_does_not_hang_behind_a_stuck_compile_in_a_later_handler(
    tmp_path,
):
    """Refuse the cold path, not wait again, once the exit wait ran out."""
    completed = _run_script(
        _STUCK_COMPILE_AND_A_LATER_HANDLER_SCRIPT, tmp_path, timeout=30
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == (
        "refused: Swage cannot compile or load a kernel while the "
        "interpreter exits: another thread still holds the cold-path "
        "lock\n"
    )


def test_exit_wait_is_bounded_by_a_few_seconds():
    """Keep the bound above a slow compile and below a noticeable hang."""
    from swage import _runtime

    assert 1.0 <= _runtime._EXIT_WAIT_SECONDS <= 10.0


_LATER_EXIT_HANDLER_SCRIPT = """
import atexit
import sys


def use_the_cold_path():
    from swage import _runtime

    with _runtime._compile_lock:
        sys.stdout.write("later handler took the lock\\n")
        sys.stdout.flush()


# Registered before the package is imported, so it runs after the handler
# of the package, on the thread that then already keeps the lock.
atexit.register(use_the_cold_path)

import swage  # noqa: E402,F401
"""


def test_exit_handlers_that_run_later_can_still_use_the_cold_path(tmp_path):
    """Do not block the exiting thread on the lock it keeps."""
    completed = _run_script(_LATER_EXIT_HANDLER_SCRIPT, tmp_path, timeout=30)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "later handler took the lock\n"


_FORK_DURING_A_COMPILE_SCRIPT = """
import os
import signal
import sys
import threading
import time
import warnings

from swage import _runtime
from swage import _segmented_runtime as _execution

in_flight = threading.Event()
options = {"kernel_name": "segmented_sum", "target": "sm_86"}
contract = sys.argv[1]


def slow_compile(_module, **_options):
    in_flight.set()
    time.sleep(0.4)
    return "lowered", "ptx of the parent", contract


def fast_compile(_module, **_options):
    return "lowered", "ptx of the child", contract


thread = threading.Thread(
    target=_execution._compile_once,
    args=(slow_compile, "program"),
    kwargs=dict(options, module=object()),
)
thread.start()
in_flight.wait()
warnings.simplefilter("ignore", DeprecationWarning)
child = os.fork()
if child == 0:
    # SIGALRM ends a child that waits for a lock nobody will release.
    signal.alarm(5)
    kernel = _execution._compile_once(
        fast_compile, "another program", module=object(), **options
    )
    with _runtime._compile_lock:
        os._exit(0 if kernel.image == "ptx of the child" else 3)
_, status = os.waitpid(child, 0)
thread.join()
if os.WIFSIGNALED(status):
    print("child ended by signal", os.WTERMSIG(status))
else:
    print("child exited", os.WEXITSTATUS(status))
with _runtime._compile_lock:
    print("parent lock usable")
"""


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork")
def test_child_forked_during_a_compile_can_use_the_cold_path(tmp_path):
    """Do not hand a forked child a lock held by a thread it lacks."""
    completed = _run_script(
        _FORK_DURING_A_COMPILE_SCRIPT,
        tmp_path,
        arguments=(
            _contract_json("segmented_sum", arguments=_DIRECT_ARGUMENTS),
        ),
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "child exited 0\nparent lock usable\n"
