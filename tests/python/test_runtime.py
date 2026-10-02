# tests/python/test_runtime.py
"""LLVM-free tests for the fixed vector-add CUDA launch boundary."""

import ctypes
import gc
import json
import multiprocessing
import os
import pathlib
import re
import shutil
import signal
import stat
import subprocess
import sys
import time
import types
import warnings
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
_SHARED_KEY = {"kernel": "add_kernel", "target": "sm_86"}


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


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """Keep every test away from the user's cache and from earlier tests."""
    from swage import _runtime

    monkeypatch.setenv("SWAGE_CACHE_DIR", str(tmp_path / "isolated-cache"))
    monkeypatch.setattr(_runtime, "_cache_off", {}, raising=False)
    for name in (
        "SWAGE_CACHE_MAX_ENTRIES",
        "SWAGE_CACHE_READ_ONLY",
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
    _runtime = _use_shared_cache(cache_dir)
    ptx = f"ptx from process {os.getpid()}"
    _runtime._compile_native = lambda *_args: ("lowered", ptx)
    try:
        for round_id in range(rounds):
            barrier.wait(timeout=60)
            artifact = _runtime._compile_cached(
                dict(_SHARED_KEY, round=round_id), "add_kernel", 128, object
            )
            result = pathlib.Path(results_dir) / f"{round_id}-{os.getpid()}"
            result.write_text(artifact.ptx)
    except BaseException:
        barrier.abort()  # Release the other processes instead of timing out.
        raise


def _churn(cache_dir, barrier, rounds, bound):
    """Publish a contended key and a private key per round, bounded."""
    os.environ["SWAGE_CACHE_MAX_ENTRIES"] = str(bound)
    _runtime = _use_shared_cache(cache_dir)
    _runtime._compile_native = lambda *_args: ("lowered", "ptx")
    # A cache that gives up warns, and here a warning fails the process.
    warnings.simplefilter("error")
    try:
        for round_id in range(rounds):
            barrier.wait(timeout=60)
            for owner in ("every process", os.getpid()):
                artifact = _runtime._compile_cached(
                    dict(_SHARED_KEY, round=round_id, owner=owner),
                    "add_kernel",
                    128,
                    object,
                )
                assert artifact.ptx == "ptx"
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
    _runtime._compile_native = lambda *_args: ("lowered", "ptx")
    write = _runtime._atomic_write

    def write_then_die(path, contents):
        write(path, contents)
        os.kill(os.getpid(), signal.SIGKILL)

    _runtime._atomic_write = write_then_die
    _runtime._compile_cached(_SHARED_KEY, "add_kernel", 128, object)


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
        return 4

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


def _fake_torch(
    *, available=True, current_device=0, max_threads=1024, version="2.6.0"
):
    torch = types.ModuleType("torch")
    torch.__version__ = version
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
    arguments[name] = _Tensor(
        torch, pointer=arguments[name].data_ptr(), **view
    )

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


# Each fake tensor holds 129 four-byte elements: 0x204 bytes.
@pytest.mark.parametrize(
    ("pointers", "overlapped"),
    [
        ({"output_ptr": 0x1000}, "x_ptr"),
        ({"output_ptr": 0x1004}, "x_ptr"),
        ({"output_ptr": 0x1000 - 0x200}, "x_ptr"),
        ({"output_ptr": 0x1000 + 0x200}, "x_ptr"),
        ({"output_ptr": 0x2000}, "y_ptr"),
        ({"output_ptr": 0x2000 - 4}, "y_ptr"),
        ({"x_ptr": 0x3000 + 0x100}, "x_ptr"),
    ],
)
def test_launch_rejects_an_output_that_overlaps_an_input(
    monkeypatch, pointers, overlapped
):
    """Reject an output sharing any byte with a buffer the kernel reads."""
    torch, _ = _fake_torch()
    driver = _install_launch_fakes(monkeypatch, torch)
    arguments = _arguments(torch)
    for name, pointer in pointers.items():
        arguments[name] = _Tensor(torch, pointer=pointer)

    with pytest.raises(
        ValueError,
        match=f"'output_ptr' must not overlap argument '{overlapped}'",
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
        {"y_ptr": 0x1000},
        {"y_ptr": 0x1004},
    ],
)
def test_launch_accepts_adjacent_buffers_and_overlapping_inputs(
    monkeypatch, pointers
):
    """Allow buffers that only touch, and two inputs that share memory."""
    torch, _ = _fake_torch()
    driver = _install_launch_fakes(monkeypatch, torch)
    arguments = _arguments(torch)
    for name, pointer in pointers.items():
        arguments[name] = _Tensor(torch, pointer=pointer)

    add_kernel.launch(arguments=arguments, constexprs={"BLOCK": 128}, grid=(2,))

    assert len(driver.launches) == 1


def test_empty_launch_still_rejects_an_overlapping_output(monkeypatch):
    """Validate the whole boundary before the zero-work return."""
    torch, _ = _fake_torch()
    monkeypatch.setitem(sys.modules, "torch", torch)
    arguments = _arguments(torch, n=0)
    arguments["output_ptr"] = arguments["x_ptr"]

    with pytest.raises(ValueError, match="must not overlap"):
        add_kernel.launch(
            arguments=arguments, constexprs={"BLOCK": 128}, grid=(0,)
        )


@pytest.mark.parametrize(
    "version", ["2.5.1", "2.5.1+cu124", "1.13.1", "2", "nightly", None]
)
def test_launch_rejects_a_pytorch_below_the_floor(monkeypatch, version):
    """Fail before compiling or enqueueing on an unsupported PyTorch."""
    torch, _ = _fake_torch(version=version)
    driver = _install_launch_fakes(monkeypatch, torch)
    reason = (
        "requires PyTorch 2.6 or newer; found PyTorch "
        f"{re.escape(str(version))}$"
    )

    for _ in range(2):
        with pytest.raises(RuntimeError, match=reason):
            add_kernel.launch(
                arguments=_arguments(torch),
                constexprs={"BLOCK": 128},
                grid=(2,),
            )

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
        (defaulted_kernel, "parameter 'BLOCK' has a default value"),
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
    """Refuse a parameter list outside the ABI before any compile."""
    from swage import _runtime

    torch, _ = _fake_torch()
    driver = _Driver()
    emissions = []
    compiles = []
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(
        kernel, "emit_mlir", lambda **_kwargs: emissions.append(1)
    )
    for name in ("_compile_cached", "_compile_native"):
        monkeypatch.setattr(
            _runtime, name, lambda *_args, **_kwargs: compiles.append(1)
        )
    monkeypatch.setattr(_runtime, "_get_driver", lambda: driver)
    arguments = _arguments(torch)

    with pytest.raises(sw.CompilationError) as rejection:
        kernel.launch(
            arguments=arguments, constexprs={"BLOCK": 128}, grid=(2,)
        )

    message = str(rejection.value)
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
    assert rejection.traceback[-1].name.startswith("_validate_")
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
    from swage import _frontend, _runtime

    monkeypatch.setitem(sys.modules, "mlir_swage", None)

    with pytest.raises(RuntimeError) as failure:
        _runtime._compile_native(object(), "add_kernel", 128, "sm_86")

    message = str(failure.value)
    assert message.startswith(
        "Swage launch requires the build-tree mlir_swage bindings"
    )
    assert "which the swage-compiler wheel does not include" in message
    assert "kernel 'add_kernel' was not compiled" in message
    assert message.endswith(
        f"See {_frontend._INSTALLATION} for the native build"
    )
    assert "docs/getting-started/installation.md" in message


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
    monkeypatch.setattr(
        _runtime,
        "_compile_native",
        lambda *_args: calls.append(True) or ("lowered", "ptx"),
    )
    _runtime._ptx_cache.clear()
    return _runtime, calls


def _compile_recording_warnings(_runtime, key_data=None):
    """Compile one key and return the artifact with the warnings it raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        artifact = _runtime._compile_cached(
            _SHARED_KEY if key_data is None else key_data,
            "add_kernel",
            128,
            object,
        )
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
    monkeypatch.setenv("SWAGE_CACHE_DIR", str(tmp_path))
    _runtime, calls = _stub_compiler(monkeypatch)
    key_data = {"kernel": "add_kernel", "target": "sm_86"}

    first = _runtime._compile_cached(key_data, "add_kernel", 128, object)
    _runtime._ptx_cache.clear()
    second = _runtime._compile_cached(key_data, "add_kernel", 128, object)

    assert first == second
    assert len(calls) == 1
    assert [path.name for path in tmp_path.iterdir()] == [first.key]
    entry = tmp_path / first.key
    _assert_complete_entry(entry)
    (entry / "kernel.ptx").write_text("corrupt")
    _runtime._ptx_cache.clear()
    with pytest.raises(RuntimeError, match="digest mismatch"):
        _runtime._compile_cached(key_data, "add_kernel", 128, object)


def _rejects(reason, path):
    """Match a rejection that gives `reason` and names exactly `path`."""
    return rf"{reason}.*: {re.escape(str(path))}$"


def test_cache_rejects_unsafe_entries(tmp_path, monkeypatch):
    """Reject symlinked and world-writable files and a foreign-owned root."""
    monkeypatch.setenv("SWAGE_CACHE_DIR", str(tmp_path))
    _runtime, calls = _stub_compiler(monkeypatch)
    key_data = {"kernel": "add_kernel"}
    key = _runtime._cache_key(key_data)
    entry = tmp_path / key
    entry.mkdir()
    target = tmp_path / "outside.json"
    target.write_text(json.dumps({}))
    (entry / "metadata.json").symlink_to(target)

    with pytest.raises(RuntimeError, match="symlink"):
        _runtime._compile_cached(key_data, "add_kernel", 128, object)

    (entry / "metadata.json").unlink()
    (entry / "metadata.json").write_text("{}")
    (entry / "metadata.json").chmod(0o606)
    with pytest.raises(RuntimeError, match="world-writable"):
        _runtime._compile_cached(key_data, "add_kernel", 128, object)

    (entry / "metadata.json").chmod(0o600)
    other_user = os.geteuid() + 1
    # Every path is foreign to another user, and the root is checked first.
    with monkeypatch.context() as patch:
        patch.setattr(_runtime.os, "geteuid", lambda: other_user)
        with pytest.raises(RuntimeError, match=_rejects("not owned", tmp_path)):
            _runtime._compile_cached(key_data, "add_kernel", 128, object)

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
    artifact = _runtime._compile_cached(_SHARED_KEY, "add_kernel", 128, object)
    _runtime._ptx_cache.clear()
    calls.clear()
    return _runtime, calls, tmp_path / artifact.key


def test_cache_rejects_a_foreign_owned_entry_file(tmp_path, monkeypatch):
    """Check ownership of every file, not only of the cache root."""
    _runtime, calls, entry = _published_entry(tmp_path, monkeypatch)
    _make_foreign(monkeypatch, entry / "kernel.ptx")

    with pytest.raises(
        RuntimeError, match=_rejects("not owned", entry / "kernel.ptx")
    ):
        _runtime._compile_cached(_SHARED_KEY, "add_kernel", 128, object)
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
    _runtime, calls, entry = _published_entry(tmp_path, monkeypatch)
    if unsafe == "symlink":
        moved = tmp_path / "moved-entry"
        entry.rename(moved)
        entry.symlink_to(moved)
    elif unsafe == "world-writable":
        entry.chmod(0o707)
    else:
        _make_foreign(monkeypatch, entry)

    with pytest.raises(RuntimeError, match=_rejects(reason, entry)):
        _runtime._compile_cached(_SHARED_KEY, "add_kernel", 128, object)
    assert calls == []


def test_cache_rejects_a_corrupt_lowered_module(tmp_path, monkeypatch):
    """Verify the lowered MLIR digest, not only the PTX digest."""
    _runtime, calls, entry = _published_entry(tmp_path, monkeypatch)
    (entry / "lowered.mlir").write_text("corrupt")

    with pytest.raises(
        RuntimeError, match=_rejects("lowered MLIR digest mismatch", entry)
    ):
        _runtime._compile_cached(_SHARED_KEY, "add_kernel", 128, object)
    assert calls == []


def test_cache_rejects_an_entry_for_another_specialization(
    tmp_path, monkeypatch
):
    """Never serve an entry whose recorded specialization differs."""
    monkeypatch.setenv("SWAGE_CACHE_DIR", str(tmp_path))
    _runtime, calls = _stub_compiler(monkeypatch)
    key = _runtime._cache_key(_SHARED_KEY)
    other = dict(_SHARED_KEY, target="sm_80")
    artifact = _runtime._Artifact(key, "lowered", "ptx")
    _runtime._write_cache_entry(artifact, other)
    assert _runtime._read_cache_entry(key, other).ptx == "ptx"

    with pytest.raises(
        RuntimeError, match=_rejects("specialization mismatch", tmp_path / key)
    ):
        _runtime._compile_cached(_SHARED_KEY, "add_kernel", 128, object)
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
        key = _runtime._cache_key({"kernel": "another"})
        entry = entry.rename(tmp_path / key)

    with pytest.raises(
        RuntimeError, match=_rejects("metadata mismatch", entry)
    ):
        _runtime._compile_cached(
            _SHARED_KEY, "add_kernel", 128, object, key=key
        )
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
    _runtime, calls = _stub_compiler(monkeypatch, lambda: identity)

    first, messages = _compile_recording_warnings(_runtime, {"kernel": "add"})
    second, later = _compile_recording_warnings(_runtime, {"kernel": "add"})
    _, other_key = _compile_recording_warnings(_runtime, {"kernel": "sub"})

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

    first = _runtime._compile_cached({"kernel": "add"}, "add", 128, object)
    _runtime._ptx_cache.clear()
    second = _runtime._compile_cached({"kernel": "add"}, "add", 128, object)

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

    first = _runtime._compile_cached(_SHARED_KEY, "add_kernel", 128, object)
    _runtime._ptx_cache.clear()
    second = _runtime._compile_cached(_SHARED_KEY, "add_kernel", 128, object)

    assert first == second == _runtime._Artifact(entry.name, "lowered", "ptx")
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

    first = _runtime._compile_cached(_SHARED_KEY, "add_kernel", 128, object)
    _runtime._ptx_cache.clear()
    second = _runtime._compile_cached(_SHARED_KEY, "add_kernel", 128, object)

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
    winner = _runtime._compile_cached(_SHARED_KEY, "add_kernel", 128, object)
    loser = _runtime._Artifact(winner.key, "lowered", "ptx from the loser")

    used = _runtime._write_cache_entry(loser, _SHARED_KEY)

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
    first, messages = _compile_recording_warnings(_runtime)
    second, repeated = _compile_recording_warnings(_runtime)
    _, other_key = _compile_recording_warnings(_runtime, {"kernel": "other"})

    assert first == second == _runtime._Artifact(first.key, "lowered", "ptx")
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
    from swage import _runtime

    torch, _ = _fake_torch()
    driver = _Driver()
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(add_kernel, "emit_mlir", lambda **_kwargs: object())
    monkeypatch.setattr(_runtime, "_get_driver", lambda: driver)
    _runtime, calls = _stub_compiler(monkeypatch)
    _runtime._identity_cache = None
    _runtime._loaded_functions.clear()
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


def test_read_only_cache_still_serves_published_entries(
    tmp_path, monkeypatch
):
    """Keep reading a warm cache after a write to it has failed."""
    monkeypatch.setenv("SWAGE_CACHE_DIR", str(tmp_path))
    _runtime, calls = _stub_compiler(monkeypatch)
    warm = _runtime._compile_cached(_SHARED_KEY, "add_kernel", 128, object)
    _runtime._ptx_cache.clear()

    def no_space(*_args, **_kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(_runtime.tempfile, "mkdtemp", no_space)
    _, messages = _compile_recording_warnings(_runtime, {"kernel": "cold"})
    reread, later = _compile_recording_warnings(_runtime)

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
    warm = _runtime._compile_cached(_SHARED_KEY, "add_kernel", 128, object)
    _runtime._ptx_cache.clear()
    debris = tmp_path / _runtime._cache_key({"kernel": "debris"})
    debris.mkdir(mode=0o700)
    (debris / "lowered.mlir").write_text("stale")
    if how == "read-only root":
        tmp_path.chmod(0o500)
    else:

        def read_only(*_args, **_kwargs):
            raise OSError(30, "Read-only file system")

        monkeypatch.setattr(_runtime.tempfile, "mkdtemp", read_only)

    try:
        missed, messages = _compile_recording_warnings(
            _runtime, {"kernel": "debris"}
        )
        again, repeated = _compile_recording_warnings(
            _runtime, {"kernel": "debris"}
        )
        reread, later = _compile_recording_warnings(_runtime)
        _, cold = _compile_recording_warnings(_runtime, {"kernel": "cold"})
    finally:
        tmp_path.chmod(0o700)

    assert missed == again
    assert missed.ptx == "ptx"
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
    _runtime, calls = _stub_compiler(monkeypatch)

    first, messages = _compile_recording_warnings(_runtime)
    second, repeated = _compile_recording_warnings(_runtime)
    _, other_key = _compile_recording_warnings(_runtime, {"kernel": "other"})

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
    _runtime, calls = _stub_compiler(monkeypatch)

    def compile_while_the_root_appears(*_args):
        # The lookup found no root, so only the publish can notice this one.
        calls.append(True)
        if unsafe == "symlink":
            root.symlink_to(elsewhere)
        else:
            root.mkdir()
            root.chmod(0o707)
        return "lowered", "ptx"

    monkeypatch.setattr(
        _runtime, "_compile_native", compile_while_the_root_appears
    )
    with pytest.raises(RuntimeError, match=_rejects(reason, root)):
        _runtime._compile_cached(_SHARED_KEY, "add_kernel", 128, object)
    retained = _runtime._compile_cached(_SHARED_KEY, "add_kernel", 128, object)

    assert retained.ptx == "ptx"
    assert len(calls) == 1
    assert list(elsewhere.iterdir()) == []
    assert list(root.iterdir()) == []


def test_removing_an_incomplete_entry_keeps_a_concurrent_publish(
    tmp_path, monkeypatch
):
    """Do not delete an entry that was published after the reader looked."""
    monkeypatch.setenv("SWAGE_CACHE_DIR", str(tmp_path))
    _runtime, calls = _stub_compiler(monkeypatch)
    published = _runtime._compile_cached(_SHARED_KEY, "add_kernel", 128, object)
    entry = tmp_path / published.key

    # A reader that saw debris reaches removal after the publish above.
    _runtime._remove_incomplete_entry(entry)

    _runtime._ptx_cache.clear()
    assert [path.name for path in tmp_path.iterdir()] == [published.key]
    _assert_complete_entry(entry)
    assert (
        _runtime._compile_cached(_SHARED_KEY, "add_kernel", 128, object)
        == published
    )
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
        _runtime._compile_cached(_SHARED_KEY, "add_kernel", 128, object)

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
    warm = _runtime._compile_cached(_SHARED_KEY, "add_kernel", 128, object)
    debris = tmp_path / _runtime._cache_key({"kernel": "debris"})
    debris.mkdir(mode=0o700)
    (debris / "lowered.mlir").write_text("stale")
    before = _snapshot(tmp_path)
    _runtime._ptx_cache.clear()
    monkeypatch.setenv("SWAGE_CACHE_READ_ONLY", "1")

    reread, on_hit = _compile_recording_warnings(_runtime)
    cold, on_miss = _compile_recording_warnings(_runtime, {"kernel": "cold"})
    again, repeated = _compile_recording_warnings(_runtime, {"kernel": "cold"})
    missed, on_debris = _compile_recording_warnings(
        _runtime, {"kernel": "debris"}
    )

    assert reread == warm
    assert cold == again
    assert cold.ptx == missed.ptx == "ptx"
    assert len(calls) == 3
    assert on_hit == on_miss == repeated == on_debris == []
    assert _snapshot(tmp_path) == before
    assert not _runtime._cache_off


def test_read_only_mode_does_not_create_the_cache_root(tmp_path, monkeypatch):
    """Leave a missing cache root missing."""
    root = tmp_path / "absent" / "cache"
    monkeypatch.setenv("SWAGE_CACHE_DIR", str(root))
    monkeypatch.setenv("SWAGE_CACHE_READ_ONLY", "1")
    _runtime, calls = _stub_compiler(monkeypatch)

    artifact, messages = _compile_recording_warnings(_runtime)

    assert artifact.ptx == "ptx"
    assert len(calls) == 1
    assert messages == []
    assert not root.parent.exists()


def test_read_only_mode_still_rejects_unsafe_entries(tmp_path, monkeypatch):
    """Keep treating a corrupt entry as tamper evidence."""
    monkeypatch.setenv("SWAGE_CACHE_DIR", str(tmp_path))
    _runtime, calls = _stub_compiler(monkeypatch)
    warm = _runtime._compile_cached(_SHARED_KEY, "add_kernel", 128, object)
    (tmp_path / warm.key / "kernel.ptx").write_text("corrupt")
    _runtime._ptx_cache.clear()
    monkeypatch.setenv("SWAGE_CACHE_READ_ONLY", "1")

    with pytest.raises(RuntimeError, match="digest mismatch"):
        _runtime._compile_cached(_SHARED_KEY, "add_kernel", 128, object)

    assert len(calls) == 1


def test_no_compile_mode_serves_entries_and_refuses_a_miss(
    tmp_path, monkeypatch
):
    """Launch from the cache and raise instead of compiling on a miss."""
    monkeypatch.setenv("SWAGE_CACHE_DIR", str(tmp_path))
    _runtime, calls = _stub_compiler(monkeypatch)
    warm = _runtime._compile_cached(_SHARED_KEY, "add_kernel", 128, object)
    before = _snapshot(tmp_path)
    _runtime._ptx_cache.clear()
    monkeypatch.setenv("SWAGE_NO_COMPILE", "1")
    emissions = []
    cold = {"kernel": "cold"}

    reread, on_hit = _compile_recording_warnings(_runtime)
    in_process, _ = _compile_recording_warnings(_runtime)
    for _ in range(2):
        with pytest.raises(RuntimeError) as refusal:
            _runtime._compile_cached(
                cold, "cold_kernel", 128, lambda: emissions.append(1)
            )

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
    _runtime, calls = _stub_compiler(monkeypatch, lambda: identity)
    for _ in range(2):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(RuntimeError) as refusal:
                _runtime._compile_cached(_SHARED_KEY, "add_kernel", 128, object)
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
    _runtime, calls = _stub_compiler(monkeypatch)

    for _ in range(2):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(RuntimeError) as refusal:
                _runtime._compile_cached(_SHARED_KEY, "add_kernel", 128, object)
        assert f"cannot use {root}" in str(refusal.value)
        assert "Not a directory" in str(refusal.value)

    assert calls == []


def test_no_compile_mode_refuses_a_launch_before_any_driver_work(
    tmp_path, monkeypatch
):
    """Raise from the public launch with nothing loaded or enqueued."""
    from swage import _runtime

    torch, _ = _fake_torch()
    driver = _Driver()
    emissions = []
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(
        add_kernel, "emit_mlir", lambda **_kwargs: emissions.append(1)
    )
    monkeypatch.setattr(_runtime, "_get_driver", lambda: driver)
    _runtime, calls = _stub_compiler(monkeypatch)
    _runtime._loaded_functions.clear()
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
    _runtime, calls = _stub_compiler(monkeypatch)

    with pytest.raises(
        ValueError, match=rf"{name} must be 0 or 1; found '{value}'"
    ):
        _runtime._compile_cached(_SHARED_KEY, "add_kernel", 128, object)

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
    _runtime, calls = _stub_compiler(monkeypatch)

    artifact = _runtime._compile_cached(_SHARED_KEY, "add_kernel", 128, object)

    assert len(calls) == 1
    _assert_complete_entry(tmp_path / artifact.key)


def _publish(_runtime, name, age_seconds):
    """Publish one entry and date it `age_seconds` before now."""
    artifact = _runtime._compile_cached({"kernel": name}, name, 128, object)
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
    middle = _publish(_runtime, "middle", 200)
    oldest = _publish(_runtime, "oldest", 400)
    newer = _publish(_runtime, "newer", 100)
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(
        [middle, oldest, newer]
    )
    # This publish is not dated back, so it is the newest of the four.
    latest = _runtime._compile_cached(_SHARED_KEY, "add_kernel", 128, object)
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(
        [middle, newer, latest.key]
    )
    one_more = _publish(_runtime, "one more", 0)

    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(
        [newer, latest.key, one_more]
    )
    for key in (newer, latest.key, one_more):
        _assert_complete_entry(tmp_path / key)

    # An evicted key is a plain miss and is published again.
    _runtime._ptx_cache.clear()
    compiles = len(calls)
    assert _publish(_runtime, "oldest", 0) == oldest
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
    _runtime, calls = _stub_compiler(monkeypatch)

    with pytest.raises(
        ValueError,
        match=(
            "SWAGE_CACHE_MAX_ENTRIES must be a positive integer; "
            f"found '{re.escape(value)}'"
        ),
    ):
        _runtime._compile_cached(_SHARED_KEY, "add_kernel", 128, object)

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
    evicted = _publish(_runtime, "evicted", 3600)

    latest = _runtime._compile_cached(_SHARED_KEY, "add_kernel", 128, object)

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
    _runtime, _ = _stub_compiler(monkeypatch)
    foreign = tmp_path / _publish(_runtime, "foreign", 3600)
    _make_foreign(monkeypatch, foreign)
    latest = _runtime._compile_cached(_SHARED_KEY, "add_kernel", 128, object)

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
    raced = tmp_path / _publish(_runtime, "raced", 3600)
    rename = os.rename

    def rename_after_another_process_removed_it(source, destination):
        if pathlib.Path(source) == raced:
            shutil.rmtree(raced)
        return rename(source, destination)

    monkeypatch.setattr(
        _runtime.os, "rename", rename_after_another_process_removed_it
    )
    latest, messages = _compile_recording_warnings(_runtime)

    assert messages == []
    assert [path.name for path in tmp_path.iterdir()] == [latest.key]


def test_failed_trimming_stops_publishing_with_one_warning(
    tmp_path, monkeypatch
):
    """Stop adding entries when the bound cannot be kept."""
    monkeypatch.setenv("SWAGE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("SWAGE_CACHE_MAX_ENTRIES", "1")
    _runtime, calls = _stub_compiler(monkeypatch)
    stuck = _publish(_runtime, "stuck", 3600)
    rename = os.rename

    def refuse_to_move_the_old_entry(source, destination):
        if pathlib.Path(source).name == stuck:
            raise PermissionError(13, "Permission denied")
        return rename(source, destination)

    monkeypatch.setattr(_runtime.os, "rename", refuse_to_move_the_old_entry)
    latest, messages = _compile_recording_warnings(_runtime)
    later, repeated = _compile_recording_warnings(_runtime, {"kernel": "later"})

    assert latest.ptx == later.ptx == "ptx"
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

    _publish(_runtime, "first", 10)
    _publish(_runtime, "second", 0)
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
    _compile_recording_warnings(_runtime)
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
            key, dict(_SHARED_KEY, round=round_id)
        )
        used = [path.read_text() for path in results.glob(f"{round_id}-*")]
        assert used == [published.ptx] * processes


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


def test_writer_killed_mid_entry_does_not_poison_the_key(
    tmp_path, monkeypatch
):
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

    first = _runtime._compile_cached(_SHARED_KEY, "add_kernel", 128, object)
    _runtime._ptx_cache.clear()
    second = _runtime._compile_cached(_SHARED_KEY, "add_kernel", 128, object)

    assert first == second == _runtime._Artifact(key, "lowered", "ptx")
    assert len(calls) == 1
    _assert_complete_entry(tmp_path / key)


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

    monkeypatch.setattr(_runtime, "_compiler_identity", _identity)
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
        "frontend": "f" * 64,
        "native": [["_swageDialectsNanobind.so", 1, 2]],
        "dialect_version": 1,
        "llvm_version": "llvmorg-test",
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
    monkeypatch.setattr(
        _runtime,
        "_compile_native",
        lambda *_args: ("lowered", "ptx"),
    )
    cache = tmp_path / "cache"
    monkeypatch.setenv("SWAGE_CACHE_DIR", str(cache))
    _runtime._identity_cache = None
    _runtime._ptx_cache.clear()

    identity = _runtime._cached_identity()
    artifact = _runtime._compile_cached(_SHARED_KEY, "add_kernel", 128, object)
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


def test_frontend_digest_skips_what_python_cannot_import(
    tmp_path, monkeypatch
):
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
def test_unreadable_frontend_file_turns_persistence_off(
    tmp_path, monkeypatch
):
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
        first, messages = _compile_recording_warnings(_runtime)
        second, repeated = _compile_recording_warnings(_runtime)
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


_STALE_FRONTEND_SCRIPT = """
import json
import os
import pathlib
import sys
import warnings

import swage

sampled_at_import = "swage._runtime" in sys.modules
package = pathlib.Path(swage.__file__).parent
if sys.argv[1] == "edit after import":
    with open(package / "_frontend.py", "ab") as source:
        source.write(b"# edited after this process imported swage")

from swage import _runtime

native = [["_swageDialectsNanobind.so", 1, 2]]
compiles = []
_runtime._native_identity = lambda: native
_runtime._native_libraries = lambda: []
_runtime._compile_native = lambda *_args: compiles.append(1) or (
    "lowered",
    "ptx from the frontend this process loaded",
)
identity = _runtime._cached_identity()
data = {"kernel": "k", "frontend": identity["frontend"], "native": native}
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    first = _runtime._compile_cached(data, "k", 128, object)
    second = _runtime._compile_cached(data, "k", 128, object)
cache = pathlib.Path(os.environ["SWAGE_CACHE_DIR"])
on_disk = dict(data, frontend=_runtime._frontend_digest(package))
print(json.dumps({
    "sampled_at_import": sampled_at_import,
    "heavy": [name for name in ("torch", "mlir_swage") if name in sys.modules],
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
        [sys.executable, "-c", _STALE_FRONTEND_SCRIPT, mode],
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


def compile_in_the_child():
    from swage import _runtime

    native = [["_swageDialectsNanobind.so", 1, 2]]
    compiles = []
    _runtime._native_identity = lambda: native
    _runtime._native_libraries = lambda: []
    _runtime._compile_native = lambda *_args: compiles.append(1) or (
        "lowered",
        "ptx from the frontend the parent loaded",
    )
    identity = _runtime._cached_identity()
    data = {"kernel": "k", "frontend": identity["frontend"], "native": native}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        first = _runtime._compile_cached(data, "k", 128, object)
        second = _runtime._compile_cached(data, "k", 128, object)
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
        [sys.executable, "-c", _FORKED_CHILD_SCRIPT, how, str(report)],
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
    assert "_frontend.py is not older than this process" in (
        result["warnings"][0]
    )
    assert result["same"]
    assert result["compiles"] == 1


_NO_START_TIME_SCRIPT = """
import builtins
import io
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
from swage import _runtime

assert _runtime._PROCESS_START_NS is None
assert not _runtime._cache_off

compiles = []
_runtime._native_identity = lambda: [["_swageDialectsNanobind.so", 1, 2]]
_runtime._native_libraries = lambda: []
_runtime._compile_native = lambda *_args: compiles.append(1) or ("l", "p")
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    _runtime._compile_cached({"kernel": "k"}, "k", 128, object)
    _runtime._compile_cached({"kernel": "k"}, "k", 128, object)
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
    completed = _run_cold(["-c", _NO_START_TIME_SCRIPT, failure], tmp_path)

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
    artifact, messages = _compile_recording_warnings(_runtime)
    _runtime._identity_cache = None

    assert messages == []
    assert len(calls) == 1
    _assert_complete_entry(cache / artifact.key)


def test_identity_is_stale_without_a_process_start_time(
    tmp_path, monkeypatch
):
    """Do not trust files on disk when the start time is unknown."""
    _runtime, calls, identity, _, _, cache = _identified_process(
        tmp_path, monkeypatch
    )
    monkeypatch.setattr(_runtime, "_PROCESS_START_NS", None)

    assert "start time" in _runtime._stale_identity(identity)
    _, messages = _compile_recording_warnings(_runtime)
    _runtime._identity_cache = None

    assert len(messages) == 1
    assert len(calls) == 1
    assert not cache.exists()


@pytest.mark.parametrize("changed", ["frontend", "native"])
def test_identity_is_stale_when_a_file_changed_after_start(
    tmp_path, monkeypatch, changed
):
    """Distrust a file as new as the process, even with unchanged bytes."""
    _runtime, calls, identity, package, versioned, cache = (
        _identified_process(tmp_path, monkeypatch)
    )
    path = package / "_frontend.py" if changed == "frontend" else versioned
    if changed == "native":
        # File times are coarse: make the library newer than the frontend
        # by more than a timer tick, without changing its identity.
        time.sleep(0.05)
        os.utime(versioned, ns=(1_000, 2_000))
    monkeypatch.setattr(_runtime, "_PROCESS_START_NS", _changed_ns(path))

    problem = _runtime._stale_identity(identity)
    _, messages = _compile_recording_warnings(_runtime)
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
    _runtime, calls, identity, package, versioned, cache = (
        _identified_process(tmp_path, monkeypatch)
    )

    def compile_while_the_disk_changes(*_args):
        calls.append(True)
        if changed == "frontend":
            (package / "_frontend.py").write_text("LOWERING = 2\n")
        else:
            os.utime(versioned, ns=(1_000, 3_000))
        return "lowered", "ptx from the compiler that was loaded"

    monkeypatch.setattr(
        _runtime, "_compile_native", compile_while_the_disk_changes
    )
    first, messages = _compile_recording_warnings(_runtime)
    second, repeated = _compile_recording_warnings(_runtime)
    _, other_key = _compile_recording_warnings(_runtime, {"kernel": "other"})
    problem = _runtime._stale_identity(identity)
    _runtime._identity_cache = None

    assert first == second
    assert first.ptx == "ptx from the compiler that was loaded"
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
    """Identify the native compiler by name, size, and modification time."""
    from swage import _runtime

    extension, versioned, names = _fake_bindings(tmp_path, monkeypatch)

    identity = _runtime._native_identity()

    assert names == ["mlir_swage._mlir_libs"]
    assert identity == [
        [extension.name, 9, 2_000],
        ["libSwagePythonCAPI.so", 16, 2_000],
        [versioned.name, 16, 2_000],
    ]
    os.utime(versioned, ns=(1_000, 3_000))
    relinked = _runtime._native_identity()
    assert relinked != identity
    assert relinked[0] == identity[0]
    os.utime(versioned, ns=(1_000, 2_000))
    extension.write_bytes(b"new extension")
    os.utime(extension, ns=(1_000, 2_000))
    assert _runtime._native_identity()[0] == [extension.name, 13, 2_000]

    extension.unlink()
    assert _runtime._native_identity() is None


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


def test_driver_marshals_four_pointer_fused_segmented_abi():
    """Pass four pointers and three i32 counts through the CUDA Driver ABI."""
    from swage import _runtime

    driver = object.__new__(_runtime._CudaDriver)
    calls = []
    driver._call = lambda name, *args: calls.append((name, args))

    driver.launch_segmented_mixed(
        0xF00D,
        (9,),
        128,
        0xABCD,
        (0x10, 0x20, 0x30, 0x40, 4096, 5, 7),
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
        for index in range(4, 7)
    ]
    assert name == "cuLaunchKernel"
    assert pointer_values == [0x10, 0x20, 0x30, 0x40]
    assert scalar_values == [4096, 5, 7]


def test_driver_marshals_ten_pointer_persistent_abi():
    """Pass queues, dependencies, scratch, and counts through the ABI."""
    from swage import _runtime

    driver = object.__new__(_runtime._CudaDriver)
    calls = []
    driver._call = lambda name, *args: calls.append((name, args))

    driver.launch_persistent(
        0xF00D,
        (168,),
        128,
        0xABCD,
        (
            0x10,
            0x20,
            0x30,
            0x40,
            0x50,
            0x60,
            0x70,
            0x80,
            0x90,
            0xA0,
            4096,
            5,
            7,
            11,
            2,
        ),
    )

    name, call = calls[0]
    parameters = call[-2]
    pointer_values = [
        ctypes.cast(parameters[index], ctypes.POINTER(ctypes.c_void_p))
        .contents.value
        for index in range(10)
    ]
    scalar_values = [
        ctypes.cast(parameters[index], ctypes.POINTER(ctypes.c_int32))
        .contents.value
        for index in range(10, 15)
    ]
    assert name == "cuLaunchKernel"
    assert pointer_values == [
        0x10,
        0x20,
        0x30,
        0x40,
        0x50,
        0x60,
        0x70,
        0x80,
        0x90,
        0xA0,
    ]
    assert scalar_values == [4096, 5, 7, 11, 2]


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


def test_compiler_identity_is_cached_per_process(tmp_path, monkeypatch):
    """Spawn the git subprocesses once, not twice per launch."""
    import subprocess

    from swage import _runtime

    commands = []

    def fake_run(command, **kwargs):
        commands.append((command, kwargs["cwd"]))
        return subprocess.CompletedProcess(command, 0, stdout="abc\n",
                                           stderr="")

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
        _runtime,
        "_compile_native",
        lambda *_args, **_kwargs: ("lowered", "ptx"),
    )
    fake = _identity(revision=None, clean=False, llvm=None, native=None)
    monkeypatch.setattr(_runtime, "_compiler_identity", lambda: fake)
    monkeypatch.setattr(_runtime, "_get_driver", lambda: driver)
    _runtime._identity_cache = None
    _runtime._ptx_cache.clear()
    _runtime._loaded_functions.clear()

    for _ in range(3):
        add_kernel.launch(
            arguments=_arguments(torch),
            constexprs={"BLOCK": 128},
            grid=(2,),
        )

    assert len(emissions) == 1
    assert len(driver.launches) == 3


def test_device_fact_cache_is_isolated_per_torch_module(monkeypatch):
    """Never let one process's device cache leak across torch modules."""
    from swage import _runtime

    driver = _Driver()
    monkeypatch.setattr(_runtime, "_get_driver", lambda: driver)
    monkeypatch.setattr(
        _runtime,
        "_compile_cached",
        lambda *_args, **_kwargs: _runtime._Artifact(
            key="cache-key", lowered="lowered", ptx="ptx"
        ),
    )
    _runtime._loaded_functions.clear()

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
    from swage import _runtime

    monkeypatch.setattr(
        _runtime.ctypes,
        "CDLL",
        lambda _name: mock.MagicMock(
            **{"cuLaunchKernel.return_value": 0}
        ),
    )


def test_driver_prefers_the_native_launcher_when_available(monkeypatch):
    """Dispatch through the compiled launcher when the bindings exist."""
    from swage import _runtime

    _stub_cuda_library(monkeypatch)
    calls = []
    native = types.ModuleType("_swageDialectsNanobind")
    native.swage = types.SimpleNamespace(
        _launch_kernel=lambda *arguments: calls.append(arguments)
    )
    libs = types.ModuleType("mlir_swage._mlir_libs")
    libs._swageDialectsNanobind = native
    monkeypatch.setitem(sys.modules, "mlir_swage._mlir_libs", libs)
    monkeypatch.setitem(
        sys.modules, "mlir_swage._mlir_libs._swageDialectsNanobind", native
    )

    driver = _runtime._CudaDriver()
    driver.launch(7, (3,), 128, 9, (0x1, 0x2, 0x3, 129))
    driver.launch_segmented(7, (4,), 128, 9, (0x1, 0x2, 0x3, 60, 4))
    driver.launch_segmented_tasks(7, (5,), 32, 9, (1, 2, 3, 4, 60, 5))
    driver.launch_segmented_mixed(7, (6,), 128, 9, (1, 2, 3, 4, 60, 5, 1))
    driver.launch_persistent(
        7,
        (7,),
        512,
        9,
        (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 60, 5, 1, 7, 2),
    )

    assert calls == [
        (7, 3, 128, 9, (0x1, 0x2, 0x3), (129,)),
        (7, 4, 128, 9, (0x1, 0x2, 0x3), (60, 4)),
        (7, 5, 32, 9, (1, 2, 3, 4), (60, 5)),
        (7, 6, 128, 9, (1, 2, 3, 4), (60, 5, 1)),
        (7, 7, 512, 9, (1, 2, 3, 4, 5, 6, 7, 8, 9, 10), (60, 5, 1, 7, 2)),
    ]


def test_driver_falls_back_to_ctypes_without_the_bindings(monkeypatch):
    """Keep the ctypes path working when mlir_swage is absent."""
    from swage import _runtime

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

    driver = _runtime._CudaDriver()
    assert driver._native_launch is None
    driver.launch(7, (3,), 128, 9, (0x1, 0x2, 0x3, 129))
    assert driver.library.cuLaunchKernel.called


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
            raise RuntimeError("PyTorch has no current CUDA context")
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

    def no_context():
        raise RuntimeError("PyTorch has no current CUDA context")

    driver.current_context = no_context

    with pytest.raises(RuntimeError, match="no current CUDA context"):
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


def _run_script(script, tmp_path, timeout=60):
    """Run `script` in a fresh interpreter with the package on its path."""
    return subprocess.run(
        [sys.executable, "-c", script],
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
from swage import _segmented_qualification as qualification

in_flight = threading.Event()
options = {"kernel_name": "segmented_sum", "target": "sm_86"}


def slow_compile(_module, **_options):
    in_flight.set()
    time.sleep(0.4)
    return "lowered", "ptx of the parent"


def fast_compile(_module, **_options):
    return "lowered", "ptx of the child"


thread = threading.Thread(
    target=qualification._compile_once,
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
    ptx = qualification._compile_once(
        fast_compile, "another program", module=object(), **options
    )
    with _runtime._compile_lock:
        os._exit(0 if ptx == "ptx of the child" else 3)
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
    completed = _run_script(_FORK_DURING_A_COMPILE_SCRIPT, tmp_path)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "child exited 0\nparent lock usable\n"
