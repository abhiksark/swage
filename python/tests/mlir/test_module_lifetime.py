# python/tests/mlir/test_module_lifetime.py
"""Lifetime of loaded CUDA modules, cold-path locking, and launch guards.

A loaded module stays loaded while a lease holds it or while it is in the
LRU of loaded modules. Once it has left the LRU and no lease holds it, the
next poll in its context, run by the next load or launch there, unloads it
if the event recorded after its last launch on each stream has completed.
A module launched during a graph capture is never unloaded.
"""

import collections
import ctypes
import functools
import gc
import itertools
import threading
import warnings
from contextlib import contextmanager
from unittest import mock

import pytest
import swage as sw
import swage.language as sl
import torch
from swage import _abi, _cuda_backend, _runtime
from swage import _segmented_programs as _programs
from swage import _segmented_qualification as qualification
from swage import _segmented_runtime as _execution

_requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"
)
_NOT_FOUND = 500
_ILLEGAL_ADDRESS = 700
_SENTINEL = -5.0
# Long enough for a thread to finish one host-side call, short enough that a
# call wrongly blocked on a lock fails the test instead of hanging the run.
_PROMPT = 5.0
# How long a test keeps a thread inside a compile or a launch. It exceeds
# `_PROMPT`, so a blocked call is seen blocked before the hold ends.
_HOLD = 60.0
# The argument kinds of the fake kernels: three buffers and a count.
_KINDS = ("ptr", "ptr", "ptr", "i32")


@sw.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n, BLOCK: sl.constexpr):  # noqa: D103
    pid = sl.program_id(0)
    offsets = pid * BLOCK + sl.arange(0, BLOCK)
    mask = offsets < n
    x = sl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = sl.load(y_ptr + offsets, mask=mask, other=0.0)
    sl.store(output_ptr + offsets, x + y, mask=mask)


def _add(x, y, output, block):
    n = x.numel()
    add_kernel.launch(
        arguments={"x_ptr": x, "y_ptr": y, "output_ptr": output, "n": n},
        constexprs={"BLOCK": block},
        grid=((n + block - 1) // block,),
    )


def _contract_json(entry="kernel", block=32):
    """Return the canonical launch contract of a fake fixed kernel."""
    argument = _abi.KernelArgument
    contract = _abi.KernelContract(
        version=_abi._VERSION,
        backend="cuda",
        entry=entry,
        launch=_abi.KernelLaunch("spmd-grid", (block, 1, 1)),
        arguments=(
            argument("ptr", "user", source_index=0, access="read"),
            argument("ptr", "user", source_index=1, access="read"),
            argument("ptr", "user", source_index=2, access="write"),
            argument("i32", "user", source_index=3),
        ),
    )
    return _abi.serialize_kernel_contract(contract)


def _kernel(image):
    """Return a fake kernel whose PTX, and so its identity, is `image`."""
    contract_json = _contract_json()
    return _execution._Kernel(
        image, contract_json, _abi.parse_kernel_contract(contract_json)
    )


class _FakeCuda:
    """A stand-in for libcuda that records every driver call in order.

    Every event reports complete unless `busy` is set, which stands for a
    device that has not finished the work recorded before any event.
    """

    _NAMES = (
        "cuCtxGetId",
        "cuCtxGetCurrent",
        "cuModuleLoadData",
        "cuModuleGetFunction",
        "cuModuleUnload",
        "cuEventCreate",
        "cuEventRecord",
        "cuEventQuery",
        "cuEventDestroy_v2",
        "cuStreamIsCapturing",
        "cuGetErrorName",
        "cuGetErrorString",
        "cuLaunchKernel",
    )

    def __init__(self):
        self.context = 1
        self.busy = False
        self.failures = {}
        self.library = mock.MagicMock()
        self._modules = itertools.count(0x1000, 0x10)
        self._events = itertools.count(0x9000, 0x10)
        for name in self._NAMES:
            getattr(self.library, name).side_effect = functools.partial(
                self._call, name
            )

    def _call(self, name, *arguments):
        failure = self.failures.get(name)
        if failure:
            return failure
        return getattr(self, f"_{name}")(*arguments) or 0

    def _cuCtxGetId(self, _context, identifier):
        identifier._obj.value = self.context

    def _cuCtxGetCurrent(self, context):
        # Every context of the fake lives at one address, as the contexts
        # of the real driver do when one is destroyed and another created.
        context._obj.value = 0xC000

    def _cuModuleLoadData(self, module, _image):
        module._obj.value = next(self._modules)

    def _cuModuleGetFunction(self, function, module, _name):
        function._obj.value = module.value + 1

    def _cuModuleUnload(self, _module):
        return 0

    def _cuEventCreate(self, event, _flags):
        event._obj.value = next(self._events)

    def _cuEventRecord(self, _event, _stream):
        return 0

    def _cuEventQuery(self, _event):
        return _cuda_backend._NOT_READY if self.busy else 0

    def _cuEventDestroy_v2(self, _event):
        return 0

    def _cuStreamIsCapturing(self, _stream, status):
        status._obj.value = 0

    def _cuGetErrorName(self, _code, name):
        name._obj.value = b"CUDA_ERROR"

    def _cuGetErrorString(self, _code, text):
        text._obj.value = b"driver error"

    def _cuLaunchKernel(self, *_arguments):
        return 0

    def calls(self, *names):
        """Return the recorded calls to `names`, in order.

        A module unload is reported with the handle it was given.
        """
        recorded = []
        for name, arguments, _ in self.library.mock_calls:
            if name not in names:
                continue
            if name == "cuModuleUnload":
                recorded.append((name, arguments[0].value))
            else:
                recorded.append((name,))
        return recorded

    def unloaded(self):
        """Return the handles of the modules asked to unload, in order."""
        return [module for _, module in self.calls("cuModuleUnload")]


@pytest.fixture
def fake_cuda(monkeypatch):
    """Build a real `_CudaDriver` over a fake driver library.

    The LRU of loaded modules starts empty and holds one module.
    """
    cuda = _FakeCuda()
    monkeypatch.setattr(
        _cuda_backend.ctypes, "CDLL", lambda _name: cuda.library
    )
    monkeypatch.setattr(
        _cuda_backend, "_loaded_functions", collections.OrderedDict()
    )
    monkeypatch.setattr(_cuda_backend, "_retired_loaded", {})
    monkeypatch.setattr(_cuda_backend, "_memory_cache_entries", 1)
    cuda.driver = _cuda_backend._CudaDriver()
    cuda.driver._native_launch = None
    return cuda


def _load(cuda, name, *, capturing=False):
    """Lease the module of the fake kernel `name` in the current context."""
    return _cuda_backend._load_artifact(
        _kernel(name), cuda.driver, capturing=capturing
    )


def _launch(lease, stream, *, capturing=False):
    """Enqueue one launch of a leased fake kernel on `stream`."""
    _cuda_backend._launch_loaded(
        lease.entry,
        _abi.parse_kernel_contract(_contract_json()),
        (_KINDS, (1, 2, 3, 4)),
        (1, 1, 1),
        stream,
        capturing=capturing,
    )


def _poll(cuda):
    """Unload the idle retired modules of the current context."""
    _cuda_backend._poll_deferred(cuda.driver, cuda.driver.current_context())


def _retired_modules():
    """Return the handles of the modules that left the LRU and stay loaded."""
    return sorted(
        entry.module for entry in _cuda_backend._retired_loaded.values()
    )


class _ForbiddenLock:
    """A lock that fails the test when anything tries to take it."""

    def acquire(self, *_arguments, **_options):
        pytest.fail("a warm path took the cold-path lock")

    __enter__ = acquire

    def __exit__(self, *_error):
        return False

    def release(self):
        pass


class _RecordingLock:
    """A lock that records when it is taken and given back."""

    def __init__(self, events):
        self.events = events

    def __enter__(self):
        self.events.append("lock")
        return self

    def __exit__(self, *_error):
        self.events.append("unlock")
        return False


def _run_promptly(action):
    """Run `action` on another thread and report whether it ended promptly.

    Returns:
        Whether `action` returned within `_PROMPT` seconds, and the thread,
        which the caller joins once it has released whatever blocks it.
    """
    results = []
    thread = threading.Thread(target=lambda: results.append(action()))
    thread.start()
    thread.join(_PROMPT)
    return bool(results), thread


@pytest.mark.parametrize("limit", [1, 3])
def test_bounded_cache_forgets_its_oldest_entry(limit):
    """Keep the newest entries and drop the earliest stored one."""
    cache = _runtime._BoundedCache(limit)

    for index in range(5):
        cache[index] = str(index)
        assert len(cache) <= limit

    assert list(cache) == list(range(5 - limit, 5))
    cache[4] = "again"
    assert list(cache) == list(range(5 - limit, 5))
    assert cache.get(0) is None
    cache.clear()
    assert not cache


def test_in_process_kernel_caches_are_bounded(monkeypatch):
    """Give every in-process cache of compiled or loaded kernels a bound.

    The compiled artifacts of the public launch and the loaded modules are
    LRUs bounded by SWAGE_MEMORY_CACHE_ENTRIES; the PTX memo of the private
    segmented path keeps `_CACHE_LIMIT` kernels.
    """
    monkeypatch.delenv("SWAGE_MEMORY_CACHE_ENTRIES", raising=False)
    for module in (_runtime, _cuda_backend):
        monkeypatch.setattr(module, "_memory_cache_entries", None)
        assert (
            module._memory_cache_limit() == module._DEFAULT_MEMORY_CACHE_ENTRIES
        )

    assert isinstance(_execution._ptx_memo, _runtime._BoundedCache)
    assert _execution._ptx_memo.limit == _runtime._CACHE_LIMIT


def test_function_lookup_failure_unloads_the_module(fake_cuda):
    """Do not leave a module loaded when its kernel cannot be resolved."""
    fake_cuda.failures["cuModuleGetFunction"] = _NOT_FOUND

    with pytest.raises(RuntimeError, match="cuModuleGetFunction failed"):
        fake_cuda.driver.load("ptx", "missing")

    assert fake_cuda.calls("cuModuleUnload") == [("cuModuleUnload", 0x1000)]
    del fake_cuda.failures["cuModuleGetFunction"]
    kept = fake_cuda.driver.load("ptx", "kernel")
    assert fake_cuda.calls("cuModuleUnload") == [("cuModuleUnload", 0x1000)]
    assert kept == (0x1010, 0x1011)


def test_module_stays_loaded_while_a_lease_or_the_lru_holds_it(fake_cuda):
    """Unload a module only once it left the LRU and its last lease is gone."""
    first = _load(fake_cuda, "ptx")
    assert (first.entry.module, first.entry.function) == (0x1000, 0x1001)
    holder = [first]
    del first
    gc.collect()

    # The second module pushes the first out of the LRU, which holds one.
    second = _load(fake_cuda, "second")
    _poll(fake_cuda)
    assert fake_cuda.unloaded() == []

    holder.clear()
    gc.collect()
    assert fake_cuda.unloaded() == []

    # The next load unloads the first. The second has no lease left, but
    # stays loaded until the third pushes it out too.
    second.release()
    third = _load(fake_cuda, "third")
    assert fake_cuda.unloaded() == [0x1000]
    _poll(fake_cuda)
    assert fake_cuda.unloaded() == [0x1000, 0x1010]
    assert (second.entry.function, third.entry.function) == (0x1011, 0x1021)


def test_unload_waits_for_the_last_launch_of_each_module(fake_cuda):
    """Unload an idle module once the event after its last launch completed.

    Nothing waits for the device: a module whose launches have not completed
    stays loaded until a later poll finds them complete. A launch on a
    stream the caller owns is fenced right after it is enqueued; the legacy
    default stream is fenced when the module is about to unload.
    """
    leases = []
    for name, stream in (("a", 5), ("b", 6), ("c", 0), ("d", 5)):
        leases.append(_load(fake_cuda, name))
        _launch(leases[-1], stream)
    assert len(fake_cuda.calls("cuEventRecord")) == 3
    for lease in leases:
        lease.release()

    fake_cuda.busy = True
    _poll(fake_cuda)
    assert fake_cuda.unloaded() == []
    assert len(fake_cuda.calls("cuEventRecord")) == 4

    fake_cuda.busy = False
    _poll(fake_cuda)
    assert sorted(fake_cuda.unloaded()) == [0x1000, 0x1010, 0x1020]
    assert len(fake_cuda.calls("cuEventDestroy_v2")) == 3
    _poll(fake_cuda)
    assert len(fake_cuda.unloaded()) == 3


def test_no_module_is_unloaded_while_this_thread_captures(fake_cuda):
    """Leave idle modules loaded during a capture, then unload them."""
    _load(fake_cuda, "ptx").release()

    kept = _load(fake_cuda, "second", capturing=True)
    _launch(kept, 7, capturing=True)
    assert fake_cuda.unloaded() == []

    _poll(fake_cuda)
    assert fake_cuda.unloaded() == [0x1000]
    assert kept.entry.function == 0x1011


def test_no_module_is_unloaded_while_its_stream_captures(fake_cuda):
    """Neither fence nor unload a module whose stream captures a graph.

    While another stream captures in the global mode, the legacy default
    stream cannot be used, and the driver reports it as capturing.
    """
    lease = _load(fake_cuda, "ptx")
    _launch(lease, 0)
    lease.release()
    _load(fake_cuda, "second").release()
    fake_cuda.failures["cuStreamIsCapturing"] = _cuda_backend._CAPTURE_IMPLICIT

    _poll(fake_cuda)
    assert fake_cuda.calls("cuEventRecord", "cuEventQuery") == []
    assert fake_cuda.unloaded() == []

    del fake_cuda.failures["cuStreamIsCapturing"]
    _poll(fake_cuda)
    assert fake_cuda.calls("cuEventRecord", "cuEventQuery") == [
        ("cuEventRecord",),
        ("cuEventQuery",),
    ]
    assert fake_cuda.unloaded() == [0x1000]


def test_module_of_another_context_waits_for_its_context(fake_cuda):
    """Unload a module only in the context that loaded it."""
    _load(fake_cuda, "ptx").release()
    fake_cuda.context = 2
    # A load in the second context pushes the first module out of the LRU.
    _load(fake_cuda, "other").release()
    _poll(fake_cuda)
    assert fake_cuda.unloaded() == []

    fake_cuda.context = 1
    _poll(fake_cuda)
    assert fake_cuda.unloaded() == [0x1000]


def test_failed_event_query_warns_and_leaves_the_module_loaded(fake_cuda):
    """Never unload without the event, and never fail the load for it.

    A module whose last launch cannot be known to have completed stays
    loaded for the rest of the process and is reported once.
    """
    lease = _load(fake_cuda, "ptx")
    _launch(lease, 5)
    lease.release()
    second = _load(fake_cuda, "second")
    fake_cuda.failures["cuEventQuery"] = _ILLEGAL_ADDRESS

    with pytest.warns(RuntimeWarning, match="cuEventQuery failed"):
        kept = _load(fake_cuda, "third")
    assert kept.entry.function == 0x1021
    assert fake_cuda.unloaded() == []
    assert lease.entry.unload_blocked

    del fake_cuda.failures["cuEventQuery"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _poll(fake_cuda)
    assert fake_cuda.unloaded() == []
    # The second left the LRU too, but its lease keeps it loaded.
    assert second.entry.leases == 1


def test_failed_unload_warns_and_leaves_the_module_loaded(fake_cuda):
    """Report a module the driver refuses to unload, and never try again.

    A later load of the same kernel loads a new module instead.
    """
    _load(fake_cuda, "ptx").release()
    _load(fake_cuda, "second").release()
    fake_cuda.failures["cuModuleUnload"] = 400

    with pytest.warns(RuntimeWarning, match="cuModuleUnload failed"):
        _poll(fake_cuda)
    assert _retired_modules() == [0x1000]

    del fake_cuda.failures["cuModuleUnload"]
    _poll(fake_cuda)
    assert fake_cuda.unloaded() == [0x1000]
    assert _load(fake_cuda, "ptx").entry.module == 0x1020


def _retire_three_modules(fake_cuda):
    """Launch three kernels, push them out, and refuse to unload the first.

    A fourth kernel stays in the LRU.
    """
    # Held until the fourth is loaded, so that no poll unloads an earlier
    # one.
    leases = []
    for name in ("a", "b", "c", "d"):
        leases.append(_load(fake_cuda, name))
        _launch(leases[-1], 5)
    for lease in leases:
        lease.release()
    modules = [lease.entry.module for lease in leases[:3]]
    assert _retired_modules() == modules
    fake_cuda._cuModuleUnload = lambda module: (
        400 if module.value == modules[0] else 0
    )
    return modules


def test_one_failed_unload_does_not_lose_the_modules_after_it(fake_cuda):
    """Unload the rest, leave the refused module loaded, and warn once."""
    modules = _retire_three_modules(fake_cuda)

    with pytest.warns(RuntimeWarning) as caught:
        _poll(fake_cuda)

    assert [str(warning.message) for warning in caught] == [
        "Swage left 1 unused CUDA module loaded: CUDA Driver "
        "cuModuleUnload failed: CUDA_ERROR (400): driver error"
    ]
    assert sorted(fake_cuda.unloaded()) == modules
    assert _retired_modules() == modules[:1]


def test_interrupted_unload_keeps_every_module_it_did_not_unload(fake_cuda):
    """Leave the module in hand loaded and the ones not yet tried queued."""
    modules = _retire_three_modules(fake_cuda)
    unloads = []

    def interrupt_the_second(module):
        unloads.append(module.value)
        if len(unloads) == 2:
            raise KeyboardInterrupt

    fake_cuda._cuModuleUnload = interrupt_the_second

    with pytest.raises(KeyboardInterrupt):
        _poll(fake_cuda)

    assert _retired_modules() == sorted(set(modules) - {unloads[0]})
    # The interrupted unload may or may not have happened, so that module is
    # never touched again; the next poll unloads the one not yet tried.
    _poll(fake_cuda)
    assert sorted(unloads) == modules
    assert _retired_modules() == [unloads[1]]


@pytest.mark.parametrize("failing", ["cuModuleUnload", "cuEventQuery"])
def test_unload_never_raises_into_a_load_when_warnings_are_errors(
    fake_cuda, capsys, failing
):
    """Load the unrelated kernel and keep every module that stays loaded."""
    modules = _retire_three_modules(fake_cuda)
    if failing == "cuEventQuery":
        # Nothing is unloaded without its event, so every module stays.
        fake_cuda.failures["cuEventQuery"] = _ILLEGAL_ADDRESS
        left, tried = modules, []
    else:
        left, tried = modules[:1], modules

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lease = _load(fake_cuda, "unrelated")

    assert lease.entry.function == 0x1041
    # The unrelated kernel pushed the fourth one out of the LRU.
    assert _retired_modules() == [*left, 0x1030]
    assert sorted(fake_cuda.unloaded()) == tried
    # The report is not lost when it cannot be a warning.
    report = capsys.readouterr().err
    assert report.count("RuntimeWarning: Swage left") == 1
    assert f"{failing} failed" in report


def test_context_identity_is_the_driver_id_not_the_handle(fake_cuda):
    """Tell apart two contexts that the driver placed at one address."""
    driver = fake_cuda.driver

    first = driver.current_context()
    fake_cuda.context = 2
    second = driver.current_context()

    assert (first, second) == (1, 2)


def test_context_identity_falls_back_to_the_handle(monkeypatch):
    """Use the context handle on a driver that has no context ids."""
    cuda = _FakeCuda()
    del cuda.library.cuCtxGetId
    monkeypatch.setattr(
        _cuda_backend.ctypes, "CDLL", lambda _name: cuda.library
    )

    assert _cuda_backend._CudaDriver().current_context() == 0xC000


def test_no_current_context_is_reported(fake_cuda):
    """Keep the error for a thread that has no CUDA context."""
    fake_cuda.failures["cuCtxGetId"] = 201
    fake_cuda._cuCtxGetCurrent = lambda context: None

    with pytest.raises(RuntimeError, match="no current CUDA context"):
        fake_cuda.driver.current_context()


def test_function_launched_under_capture_is_never_unloaded(fake_cuda):
    """Keep a module that a CUDA graph may replay after the LRU drops it."""
    captured = _load(fake_cuda, "captured")
    _launch(captured, 7, capturing=True)
    plain = _load(fake_cuda, "plain")
    _launch(plain, 9)
    _launch(plain, 0)
    captured.release()
    plain.release()
    _load(fake_cuda, "third").release()
    _poll(fake_cuda)

    assert fake_cuda.unloaded() == [0x1010]
    assert captured.entry.capture_pinned
    assert _retired_modules() == [0x1000]
    assert len(fake_cuda.calls("cuLaunchKernel")) == 3
    # The pinned module is never fenced. The legacy stream of the other is
    # asked once, before its fence.
    assert len(fake_cuda.calls("cuStreamIsCapturing")) == 1


def test_get_driver_takes_no_lock_once_created(monkeypatch):
    """Return the existing driver without the cold-path lock."""
    driver = object()
    monkeypatch.setattr(_cuda_backend, "_driver", driver)
    monkeypatch.setattr(_runtime, "_compile_lock", _ForbiddenLock())

    assert _cuda_backend._get_driver() is driver


def test_cold_paths_hold_the_cold_path_lock(monkeypatch):
    """Create the driver and load a module under the cold-path lock.

    Interpreter exit and fork wait for that lock; a warm lease never takes
    it.
    """
    events = []

    class Driver:
        def current_context(self):
            return 1

        def load(self, _ptx, _name):
            events.append("load")
            return 0x1000, 0x1001

    monkeypatch.setattr(_cuda_backend, "_driver", None)
    monkeypatch.setattr(
        _cuda_backend,
        "_CudaDriver",
        lambda: events.append("create") or Driver(),
    )
    monkeypatch.setattr(
        _cuda_backend, "_loaded_functions", collections.OrderedDict()
    )
    monkeypatch.setattr(_cuda_backend, "_retired_loaded", {})
    monkeypatch.setattr(_runtime, "_compile_lock", _RecordingLock(events))

    driver = _cuda_backend._get_driver()
    lease = _cuda_backend._load_artifact(_kernel("ptx"), driver)

    assert events == ["lock", "create", "unlock", "lock", "load", "unlock"]
    monkeypatch.setattr(_runtime, "_compile_lock", _ForbiddenLock())
    assert _cuda_backend._get_driver() is driver
    warm = _cuda_backend._load_artifact(_kernel("ptx"), driver)
    assert warm.entry is lease.entry


def _specialization(block):
    """Return the specialization of a fake fixed kernel with `block`."""
    return {
        "kernel": "kernel",
        "backend": "cuda",
        "format": "ptx",
        "target": "sm_86",
        "descriptors": ["ptr<f32>", "ptr<f32>", "ptr<f32>", "i32"],
        "codegen": {"lowering": "fixed", "block_size": block, "options": []},
    }


def _stub_public_compiler(monkeypatch, compile_ptx):
    """Compile through `compile_ptx` with the disk cache out of play.

    Args:
        monkeypatch: The pytest fixture that undoes the stubs.
        compile_ptx: Called with the block size of each compile; returns
            the PTX of the kernel.
    """
    identity = {
        "revision": None,
        "clean": False,
        "llvm": None,
        "frontend": None,
        "native": None,
    }
    monkeypatch.setattr(_runtime, "_compiler_identity", lambda: identity)

    def compile_native(_module, kernel_name, block_size, *_options):
        return (
            "lowered",
            compile_ptx(block_size),
            _contract_json(kernel_name, block_size),
        )

    monkeypatch.setattr(_cuda_backend.CUDA_BACKEND, "compile", compile_native)
    monkeypatch.setattr(_runtime, "_artifact_cache", collections.OrderedDict())


def _compile_public(key, block=128):
    return _runtime._compile_cached(
        _cuda_backend.CUDA_BACKEND,
        _specialization(block),
        "kernel",
        block,
        object,
        key=key,
    )


def test_warm_public_artifact_takes_no_lock(monkeypatch):
    """Serve a compiled kernel again without the cold-path lock."""
    compiles = []

    def compile_ptx(block_size):
        compiles.append(block_size)
        return "ptx"

    _stub_public_compiler(monkeypatch, compile_ptx)
    first = _compile_public("warm")
    monkeypatch.setattr(_runtime, "_compile_lock", _ForbiddenLock())

    assert _compile_public("warm") is first
    assert len(compiles) == 1


def test_cold_public_compile_does_not_block_a_warm_artifact(monkeypatch):
    """Serve a compiled kernel while another thread compiles a new one."""
    compiling = threading.Event()
    finish = threading.Event()

    def compile_ptx(block_size):
        if block_size == 64:
            compiling.set()
            finish.wait(_HOLD)
        return f"ptx{block_size}"

    _stub_public_compiler(monkeypatch, compile_ptx)
    warm = _compile_public("warm")
    cold = threading.Thread(target=lambda: _compile_public("cold", 64))
    cold.start()
    try:
        started = compiling.wait(_PROMPT)
        served, reader = _run_promptly(lambda: _compile_public("warm"))
    finally:
        finish.set()
        cold.join()
        reader.join()

    assert started
    assert served
    assert _runtime._artifact_cache["warm"] is warm
    assert _runtime._artifact_cache["cold"].image == "ptx64"


def test_public_artifact_cache_recompiles_what_it_forgot(monkeypatch):
    """Compile a kernel again after the LRU bound pushed it out.

    A kernel served from the cache becomes its most recent entry, so the
    bound pushes out the kernel used least recently.
    """
    compiles = []

    def compile_ptx(block_size):
        compiles.append(block_size)
        return f"ptx{block_size}"

    _stub_public_compiler(monkeypatch, compile_ptx)
    monkeypatch.setattr(_runtime, "_memory_cache_entries", 2)

    for key in ("a", "b", "a", "c", "b"):
        _compile_public(key)

    assert len(compiles) == 4
    assert list(_runtime._artifact_cache) == ["c", "b"]


def _unload_idle(driver):
    """Unload every retired module of the current context that is idle.

    The first poll fences the legacy stream behind the work queued on it,
    and the second, after the device finished, finds every fence complete.
    """
    context = driver.current_context()
    _cuda_backend._poll_deferred(driver, context)
    torch.cuda.synchronize()
    _cuda_backend._poll_deferred(driver, context)


def _unload_every_idle_module(driver):
    """Push every module out of the LRU and unload the idle ones."""
    gc.collect()
    with _cuda_backend._cuda_lock:
        _cuda_backend._evict_loaded_locked(0)
    _unload_idle(driver)


@pytest.fixture
def driver_calls(monkeypatch):
    """Count the loads and unloads the real driver performs.

    The test starts with an empty LRU and no retired module, so that it
    counts only its own modules and never revives one loaded before it,
    such as a module that a captured graph keeps loaded. Afterwards its
    idle modules are unloaded; one that stays loaded, such as a captured
    one, is left to its context.

    Yields:
        The counts of successful loads and unloads by driver call name.
    """
    driver = _cuda_backend._get_driver()
    monkeypatch.setattr(
        _cuda_backend, "_loaded_functions", collections.OrderedDict()
    )
    monkeypatch.setattr(_cuda_backend, "_retired_loaded", {})
    calls = collections.Counter()
    # The counts are updated from every thread that loads or unloads.
    counting_lock = threading.Lock()

    def counting(name, original):
        def call(*arguments):
            result = original(*arguments)
            if result == 0:
                with counting_lock:
                    calls[name] += 1
            return result

        return call

    for name in ("cuModuleLoadData", "cuModuleUnload"):
        monkeypatch.setattr(
            driver.library, name, counting(name, getattr(driver.library, name))
        )
    yield calls
    _unload_every_idle_module(driver)


def _snapshot(calls):
    return calls["cuModuleLoadData"], calls["cuModuleUnload"]


def _case(lengths):
    """Build position-dependent integral values with exact f32 sums."""
    offsets = list(itertools.accumulate(lengths, initial=0))
    values = torch.tensor(
        [index % 7 + 1 for index in range(offsets[-1])], dtype=torch.float32
    )
    sums = []
    maxima = []
    for begin, end in itertools.pairwise(offsets):
        segment = values[begin:end].double()
        sums.append(float(segment.sum()))
        maxima.append(float(segment.max()) if end > begin else float("-inf"))
    return (
        values.cuda(),
        torch.tensor(offsets, device="cuda", dtype=torch.int32),
        {
            "sum": torch.tensor(sums, dtype=torch.float32),
            "max": torch.tensor(maxima, dtype=torch.float32),
        },
    )


def _prepare(kind, values, offsets, output):
    return qualification._prepare_planned_reduction(
        values,
        offsets,
        output,
        module_text=_programs._semantic_module(kind),
        kernel_name=f"segmented_{kind}",
    )


def _assert_policies(prepared, output, expected):
    for launch in prepared:
        output.fill_(float("nan"))
        launch()
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


@_requires_cuda
def test_preparations_load_once_and_unload_what_nothing_holds(
    driver_calls, monkeypatch
):
    """Count driver loads and unloads over the same and other programs.

    The LRU holds five modules, one planned program. A second program
    pushes the first out of it, and the first is unloaded only after its
    prepared launches are gone and the work they queued has completed.
    """
    monkeypatch.setattr(_cuda_backend, "_memory_cache_entries", 5)
    lengths = [[1, 33, 4097, 0, 32], [40, 2, 8193, 5]]
    cases = [_case(lengths[0]), _case(lengths[1])]
    outputs = [torch.empty(5, device="cuda"), torch.empty(4, device="cuda")]

    first = _prepare("sum", *cases[0][:2], outputs[0])
    assert _snapshot(driver_calls) == (5, 0)
    second = _prepare("sum", *cases[1][:2], outputs[1])
    assert _snapshot(driver_calls) == (5, 0)
    _assert_policies(first, outputs[0], cases[0][2]["sum"])
    _assert_policies(second, outputs[1], cases[1][2]["sum"])

    # Another program fills the LRU. The first is still held by its
    # prepared launches, so nothing is unloaded and it still launches.
    maximum = _prepare("max", *cases[1][:2], outputs[1])
    assert _snapshot(driver_calls) == (10, 0)
    _assert_policies(maximum, outputs[1], cases[1][2]["max"])
    _assert_policies(first, outputs[0], cases[0][2]["sum"])

    # Queue work on the kernels that are about to lose their last holder,
    # behind a spin that keeps it from completing for about a second.
    outputs[0].fill_(float("nan"))
    torch.cuda._sleep(2_000_000_000)
    for _ in range(64):
        first.mixed()
    del first, second
    gc.collect()
    assert _snapshot(driver_calls) == (10, 0)

    # The next launch fences the queued work and finds it still running.
    maximum.warp()
    assert _snapshot(driver_calls) == (10, 0)

    # Once it completed, the next load unloads the five idle modules and
    # loads the program again.
    torch.cuda.synchronize()
    again = _prepare("sum", *cases[0][:2], outputs[0])
    assert _snapshot(driver_calls) == (15, 5)
    torch.testing.assert_close(
        outputs[0].cpu(), cases[0][2]["sum"], rtol=0, atol=0
    )
    _assert_policies(again, outputs[0], cases[0][2]["sum"])
    _assert_policies(maximum, outputs[1], cases[1][2]["max"])
    assert _snapshot(driver_calls) == (15, 5)


@_requires_cuda
def test_real_lookup_failure_unloads_the_module(driver_calls):
    """Unload a real module whose kernel name does not resolve."""
    from mlir_swage._mlir_libs._swageDialectsNanobind import swage as native

    kernel = _execution._compile_once(
        native._compile_segmented_reduction_ptx,
        _programs._semantic_module("sum"),
        kernel_name="segmented_sum",
        block_size=128,
        target="sm_{}{}".format(*torch.cuda.get_device_capability()),
    )

    with pytest.raises(RuntimeError, match="cuModuleGetFunction failed"):
        _cuda_backend._get_driver().load(kernel.image, "no_such_kernel")

    assert _snapshot(driver_calls) == (1, 1)


@_requires_cuda
def test_direct_segmented_launch_cannot_be_captured():
    """Pin why direct launches need no capture guard.

    Their validation reads the offsets on the host, which PyTorch refuses
    while a graph is captured, so no graph can hold one of their functions.
    """
    values, offsets, _ = _case([1, 33, 257])
    output = torch.empty(3, device="cuda")
    qualification.launch_gpu(values, offsets, output, "sum")
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()

    with pytest.raises(RuntimeError):
        with torch.cuda.graph(graph):
            qualification.launch_gpu(values, offsets, output, "sum")
    torch.cuda.synchronize()


@contextmanager
def _second_context(driver):
    """Make a second CUDA context on the current device current."""
    library = driver.library
    pointer = ctypes.c_void_p
    for name, argtypes in (
        ("cuDeviceGet", [ctypes.POINTER(ctypes.c_int), ctypes.c_int]),
        (
            "cuCtxCreate_v2",
            [ctypes.POINTER(pointer), ctypes.c_uint, ctypes.c_int],
        ),
        ("cuCtxPopCurrent_v2", [ctypes.POINTER(pointer)]),
        ("cuCtxDestroy_v2", [pointer]),
    ):
        getattr(library, name).argtypes = argtypes
        getattr(library, name).restype = ctypes.c_int
    device = ctypes.c_int()
    assert not library.cuDeviceGet(
        ctypes.byref(device), torch.cuda.current_device()
    )
    context = pointer()
    if library.cuCtxCreate_v2(ctypes.byref(context), 0, device):
        pytest.skip("the device does not admit a second CUDA context")
    try:
        yield context.value
    finally:
        popped = pointer()
        assert not library.cuCtxPopCurrent_v2(ctypes.byref(popped))
        assert not library.cuCtxDestroy_v2(context)


def _prepared_launch(policy, values, offsets, output):
    if policy == "persistent":
        return qualification._prepare_persistent_sum(
            values, offsets, output, resident_blocks=3
        ).launch
    return getattr(
        qualification._prepare_planned_sum(values, offsets, output), policy
    )


@_requires_cuda
@pytest.mark.parametrize("policy", ["warp", "cta", "mixed", "persistent"])
def test_prepared_launch_rejects_another_cuda_context(policy):
    """Raise before launching under a context other than the prepared one.

    One device is enough for two contexts: the second is created through
    the driver on the same device, where the device check still passes.
    """
    values, offsets, expected = _case([1, 33, 4097, 2])
    output = torch.full((4,), _SENTINEL, device="cuda")
    launch = _prepared_launch(policy, values, offsets, output)
    torch.cuda.synchronize()
    driver = _cuda_backend._get_driver()
    prepared_context = driver.current_context()

    with _second_context(driver):
        assert driver.current_context() != prepared_context
        for _ in range(2):
            with pytest.raises(RuntimeError, match="prepared CUDA context"):
                launch()

    assert driver.current_context() == prepared_context
    torch.cuda.synchronize()
    assert output.cpu().tolist() == [_SENTINEL] * 4
    launch()
    torch.testing.assert_close(output.cpu(), expected["sum"], rtol=0, atol=0)


@_requires_cuda
def test_module_is_loaded_again_in_a_second_real_context(driver_calls):
    """Never serve a module of one real context in another.

    The module of the second context goes with that context, and is never
    unloaded from the first.
    """
    from mlir_swage._mlir_libs._swageDialectsNanobind import swage as native

    driver = _cuda_backend._get_driver()
    kernel = _execution._compile_once(
        native._compile_segmented_reduction_ptx,
        _programs._semantic_module("sum"),
        kernel_name="segmented_sum",
        block_size=128,
        target="sm_{}{}".format(*torch.cuda.get_device_capability()),
    )

    first = _cuda_backend._load_artifact(kernel, driver)
    with _second_context(driver):
        second = _cuda_backend._load_artifact(kernel, driver)
        warm = _cuda_backend._load_artifact(kernel, driver)
        assert warm.entry is second.entry
    again = _cuda_backend._load_artifact(kernel, driver)

    assert again.entry is first.entry
    assert second.entry is not first.entry
    assert _snapshot(driver_calls) == (2, 0)

    for lease in (first, second, warm, again):
        lease.release()
    _unload_every_idle_module(driver)
    assert _snapshot(driver_calls) == (2, 1)
    assert _retired_modules() == [second.entry.module]


@_requires_cuda
def test_context_identity_survives_a_reused_context_handle():
    """Tell a recreated context from the destroyed one at its address."""
    driver = _cuda_backend._get_driver()
    if driver._context_id is None:
        pytest.skip("the driver has no context ids")
    primary = driver.current_context()
    handles = []
    identities = []

    for _ in range(3):
        with _second_context(driver) as handle:
            handles.append(handle)
            identities.append(driver.current_context())

    # The driver usually places every one of them at the same address.
    assert len(set(handles)) <= 3
    assert len(set(identities)) == 3
    assert primary not in identities
    assert driver.current_context() == primary


@_requires_cuda
@pytest.mark.parametrize("policy", ["warp", "cta", "mixed", "persistent"])
def test_prepared_launch_works_on_a_thread_that_has_not_used_cuda(policy):
    """Make the prepared context current instead of rejecting the thread.

    A new thread has no current CUDA context until PyTorch gives it one,
    which is not a different context.
    """
    values, offsets, expected = _case([1, 33, 4097, 2])
    output = torch.full((4,), float("nan"), device="cuda")
    launch = _prepared_launch(policy, values, offsets, output)
    # Finish the task upload, so that the launches below make no PyTorch
    # call that would make the context current before the driver is used.
    launch()
    torch.cuda.synchronize()
    launch()
    torch.cuda.synchronize()
    output.fill_(float("nan"))
    torch.cuda.synchronize()
    errors = []

    def launch_twice():
        try:
            launch()
            launch()
        except Exception as error:  # Reported by the assertion below.
            errors.append(error)

    worker = threading.Thread(target=launch_twice)
    worker.start()
    worker.join()

    assert errors == []
    torch.testing.assert_close(output.cpu(), expected["sum"], rtol=0, atol=0)


def _before_each_launch(monkeypatch, hook):
    """Call `hook` before every launch of a loaded kernel enqueues."""
    launch_loaded = _cuda_backend._launch_loaded

    def launch(*arguments, **options):
        hook()
        return launch_loaded(*arguments, **options)

    monkeypatch.setattr(_cuda_backend, "_launch_loaded", launch)


@_requires_cuda
def test_persistent_launch_rejects_a_concurrent_launch(monkeypatch):
    """Reject a second thread that launches while the first still does."""
    values, offsets, expected = _case([1, 33, 4097, 2, 8193])
    output = torch.full((5,), float("nan"), device="cuda")
    inside = threading.Event()
    proceed = threading.Event()
    launches = []

    def hold_the_first():
        name = threading.current_thread().name
        launches.append(name)
        if name == "first":
            inside.set()
            proceed.wait(_HOLD)

    _before_each_launch(monkeypatch, hold_the_first)
    prepared = qualification._prepare_persistent_sum(
        values, offsets, output, resident_blocks=3
    )
    errors = []

    def first_launch():
        try:
            prepared.launch()
        except Exception as error:  # Reported by the assertion below.
            errors.append(error)

    worker = threading.Thread(target=first_launch, name="first")
    worker.start()
    try:
        assert inside.wait(_PROMPT)
        for _ in range(2):
            with pytest.raises(RuntimeError, match="already launching"):
                prepared.launch()
    finally:
        proceed.set()
        worker.join()

    assert errors == []
    assert launches == ["first"]
    torch.testing.assert_close(output.cpu(), expected["sum"], rtol=0, atol=0)
    # The rejected launches left the guard free for the next one.
    output.fill_(float("nan"))
    prepared.launch()
    torch.testing.assert_close(output.cpu(), expected["sum"], rtol=0, atol=0)
    assert launches == ["first", "MainThread"]


@_requires_cuda
def test_persistent_launch_rejects_a_launch_in_flight_on_another_stream(
    monkeypatch,
):
    """Reject a launch while an earlier one still runs on another stream."""
    values, offsets, expected = _case([1, 33, 4097, 2, 8193])
    output = torch.full((5,), float("nan"), device="cuda")
    launches = []

    _before_each_launch(
        monkeypatch,
        lambda: launches.append(threading.current_thread().name),
    )
    prepared = qualification._prepare_persistent_sum(
        values, offsets, output, resident_blocks=3
    )
    prepared.launch()
    torch.cuda.synchronize()
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    queued = threading.Event()
    errors = []

    def busy_launch():
        try:
            with torch.cuda.stream(streams[0]):
                # Keep the stream busy so the launch behind it stays queued.
                torch.cuda._sleep(2_000_000_000)
                prepared.launch()
        except Exception as error:  # Reported by the assertion below.
            errors.append(error)
        queued.set()

    worker = threading.Thread(target=busy_launch, name="busy")
    worker.start()
    assert queued.wait(_PROMPT)
    worker.join()
    assert errors == []

    with torch.cuda.stream(streams[1]):
        with pytest.raises(RuntimeError, match="in flight on another stream"):
            prepared.launch()
    # The stream that holds the launch may queue another one behind it.
    with torch.cuda.stream(streams[0]):
        prepared.launch()
    assert launches == ["MainThread", "busy", "MainThread"]

    streams[0].synchronize()
    torch.testing.assert_close(output.cpu(), expected["sum"], rtol=0, atol=0)
    output.fill_(float("nan"))
    with torch.cuda.stream(streams[1]):
        prepared.launch()
    streams[1].synchronize()
    torch.testing.assert_close(output.cpu(), expected["sum"], rtol=0, atol=0)


def _public_tensors(n=300):
    x = torch.arange(n, device="cuda", dtype=torch.float32)
    y = torch.full((n,), 0.5, device="cuda")
    return x, y, torch.full((n,), float("nan"), device="cuda")


def _declined_by_the_native_launcher(*_arguments, **_options):
    pytest.fail("the native launcher declined a warm launch")


@_requires_cuda
def test_warm_public_launch_takes_no_lock(monkeypatch):
    """Launch a loaded kernel without the cold-path lock.

    The native launcher that the first launch prepared serves a warm launch;
    without it, the Python path serves it. Neither takes the lock.
    """
    # The native launcher declines while any module waits to unload, and
    # earlier tests may leave modules that never unload, such as captured
    # ones.
    monkeypatch.setattr(_cuda_backend, "_retired_loaded", {})
    x, y, output = _public_tensors()
    _add(x, y, output, 128)
    torch.testing.assert_close(output, x + y, rtol=0, atol=0)
    monkeypatch.setattr(_runtime, "_compile_lock", _ForbiddenLock())

    with monkeypatch.context() as patch:
        patch.setattr(_runtime, "launch", _declined_by_the_native_launcher)
        output.fill_(float("nan"))
        _add(x, y, output, 128)
    torch.testing.assert_close(output, x + y, rtol=0, atol=0)

    add_kernel._cuda_fast_launch = None
    output.fill_(float("nan"))
    _add(x, y, output, 128)
    torch.testing.assert_close(output, x + y, rtol=0, atol=0)


@_requires_cuda
def test_cold_compile_does_not_block_a_warm_public_launch(monkeypatch):
    """Launch a loaded kernel while another thread compiles a new one."""
    x, y, output = _public_tensors()
    cold_output = torch.full_like(output, float("nan"))
    _add(x, y, output, 128)
    torch.cuda.synchronize()
    output.fill_(float("nan"))
    monkeypatch.setattr(_runtime, "_artifact_cache", collections.OrderedDict())
    _add(x, y, output, 128)
    compiling = threading.Event()
    finish = threading.Event()
    compile_native = _cuda_backend._compile_native

    def slow_compile(module, kernel_name, block_size, *options):
        if block_size == 64:
            compiling.set()
            finish.wait(_HOLD)
        return compile_native(module, kernel_name, block_size, *options)

    monkeypatch.setattr(_cuda_backend.CUDA_BACKEND, "compile", slow_compile)
    # Keep the cold kernel out of the disk cache so that it compiles.
    monkeypatch.setattr(_runtime, "_cache_usable", lambda _identity: False)
    errors = []

    def launch_on_this_thread(target, block):
        # A new thread has no CUDA context until PyTorch gives it one.
        torch.cuda.set_device(x.device)
        _add(x, y, target, block)

    def cold_launch():
        try:
            launch_on_this_thread(cold_output, 64)
        except Exception as error:  # Reported by the assertion below.
            errors.append(error)

    cold = threading.Thread(target=cold_launch)
    cold.start()
    try:
        started = compiling.wait(_PROMPT)
        output.fill_(float("nan"))
        launched, warm = _run_promptly(
            lambda: launch_on_this_thread(output, 128)
        )
    finally:
        finish.set()
        cold.join()
        warm.join()

    assert started
    assert launched
    assert errors == []
    torch.testing.assert_close(output, x + y, rtol=0, atol=0)
    torch.testing.assert_close(cold_output, x + y, rtol=0, atol=0)


@_requires_cuda
def test_public_cache_unloads_a_forgotten_kernel_and_loads_it_again(
    driver_calls, monkeypatch
):
    """Bound the loaded kernels of the public launch and stay correct.

    The launches run on a stream of their own, whose launches are fenced
    as they are enqueued, and each check waits for that stream.
    """
    monkeypatch.setattr(_cuda_backend, "_memory_cache_entries", 1)
    stream = torch.cuda.Stream()

    with torch.cuda.stream(stream):
        x, y, output = _public_tensors()
        for index, block in enumerate([128, 64, 128, 64]):
            output.fill_(float("nan"))
            _add(x, y, output, block)
            torch.testing.assert_close(output, x + y, rtol=0, atol=0)
            # Every launch loads its kernel, which pushes the kernel of the
            # launch before out of the LRU. That launch has completed, so
            # the poll before this launch unloads it.
            assert _snapshot(driver_calls) == (index + 1, index)
            assert len(_cuda_backend._loaded_functions) == 1


@_requires_cuda
def test_captured_public_launch_keeps_its_kernel_loaded(
    driver_calls, monkeypatch
):
    """Replay a captured launch after the LRU forgot its kernel."""
    monkeypatch.setattr(_cuda_backend, "_memory_cache_entries", 1)
    x, y, output = _public_tensors()
    _add(x, y, output, 256)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _add(x, y, output, 256)

    for block in (128, 64, 32):
        _add(x, y, output, block)
    gc.collect()
    _unload_idle(_cuda_backend._get_driver())
    # The kernels of blocks 128 and 64 left the LRU and were unloaded. The
    # captured kernel left it first and is still loaded.
    assert _snapshot(driver_calls) == (4, 2)

    output.fill_(float("nan"))
    graph.replay()
    torch.testing.assert_close(output, x + y, rtol=0, atol=0)


def _run_threads(count, work):
    """Run `work(index)` on `count` threads and return what they raised."""
    errors = []

    def guarded(index):
        try:
            # A new thread has no CUDA context until PyTorch gives it one.
            torch.cuda.set_device(torch.cuda.current_device())
            work(index)
        except Exception as error:  # Reported by the caller's assertion.
            errors.append(error)

    threads = [
        threading.Thread(target=guarded, args=(index,))
        for index in range(count)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return errors


@_requires_cuda
def test_concurrent_public_launches_survive_unloading(
    driver_calls, monkeypatch
):
    """Launch from several threads while kernels are forgotten and reloaded.

    The LRU holds one kernel and the threads use four, so nearly every
    launch loads a kernel and pushes out one that another thread just used.
    Once the device finished, only the kernel in the LRU stays loaded.
    """
    monkeypatch.setattr(_cuda_backend, "_memory_cache_entries", 1)
    blocks = (32, 64, 128, 256)
    mismatches = []

    def work(index):
        x, y, output = _public_tensors(300 + index)
        expected = x + y
        for step in range(40):
            block = blocks[(index + step) % len(blocks)]
            output.fill_(float("nan"))
            _add(x, y, output, block)
            if not torch.equal(output, expected):
                mismatches.append((index, step, block))

    assert _run_threads(4, work) == []
    assert mismatches == []
    gc.collect()
    _unload_idle(_cuda_backend._get_driver())
    loads, unloads = _snapshot(driver_calls)
    assert loads > len(blocks)
    assert loads - unloads == len(_cuda_backend._loaded_functions) == 1


@_requires_cuda
def test_concurrent_preparations_survive_unloading(driver_calls, monkeypatch):
    """Prepare, launch, and drop from several threads at a small bound.

    The LRU holds one planned program and the threads alternate between
    two, so prepared launches keep running kernels the LRU has forgotten.
    Once they are gone and the device finished, only the kernels in the
    LRU stay loaded.
    """
    monkeypatch.setattr(_cuda_backend, "_memory_cache_entries", 5)
    lengths = [1, 33, 4097, 0, 32]
    kinds = ("sum", "max")
    for kind in kinds:
        values, offsets, _ = _case(lengths)
        _prepare(kind, values, offsets, torch.empty(5, device="cuda"))
    torch.cuda.synchronize()

    def work(index):
        values, offsets, expected = _case(lengths)
        output = torch.empty(len(lengths), device="cuda")
        torch.cuda.synchronize()
        for step in range(12):
            kind = kinds[(index + step) % len(kinds)]
            prepared = _prepare(kind, values, offsets, output)
            _assert_policies(prepared, output, expected[kind])

    assert _run_threads(3, work) == []
    gc.collect()
    _unload_idle(_cuda_backend._get_driver())
    loads, unloads = _snapshot(driver_calls)
    assert unloads > 0
    assert loads - unloads == len(_cuda_backend._loaded_functions)
