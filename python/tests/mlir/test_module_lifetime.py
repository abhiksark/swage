# python/tests/mlir/test_module_lifetime.py
"""Lifetime of loaded CUDA modules, cold-path locking, and launch guards."""

import collections
import ctypes
import functools
import gc
import itertools
import threading
import warnings
import weakref
from contextlib import contextmanager
from unittest import mock

import pytest
import swage as sw
import swage.language as sl
import torch
from swage import _runtime
from swage import _segmented_qualification as qualification

_requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"
)
_NOT_FOUND = 500
_CAPTURE_UNSUPPORTED = 900
_SENTINEL = -5.0
# Long enough for a thread to finish one host-side call, short enough that a
# call wrongly blocked on a lock fails the test instead of hanging the run.
_PROMPT = 5.0
# How long a test keeps a thread inside a compile or a launch. It exceeds
# `_PROMPT`, so a blocked call is seen blocked before the hold ends.
_HOLD = 60.0


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


class _FakeCuda:
    """A stand-in for libcuda that records every driver call in order."""

    _NAMES = (
        "cuCtxGetId",
        "cuCtxGetCurrent",
        "cuModuleLoadData",
        "cuModuleGetFunction",
        "cuModuleUnload",
        "cuCtxSynchronize",
        "cuStreamIsCapturing",
        "cuGetErrorName",
        "cuGetErrorString",
        "cuLaunchKernel",
    )

    def __init__(self):
        self.context = 1
        self.capturing_streams = set()
        self.failures = {}
        self.library = mock.MagicMock()
        self._modules = itertools.count(0x1000, 0x10)
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

    def _cuCtxSynchronize(self):
        return 0

    def _cuStreamIsCapturing(self, stream, status):
        status._obj.value = int(stream.value in self.capturing_streams)

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

    def released(self):
        """Return the synchronize and unload calls, in order."""
        return self.calls("cuCtxSynchronize", "cuModuleUnload")


@pytest.fixture
def fake_cuda(monkeypatch):
    """Build a real `_CudaDriver` over a fake driver library."""
    cuda = _FakeCuda()
    monkeypatch.setattr(_runtime.ctypes, "CDLL", lambda _name: cuda.library)
    monkeypatch.setattr(_runtime, "_capturing", lambda: False)
    cuda.driver = _runtime._CudaDriver()
    cuda.driver._native_launch = None
    return cuda


class _ForbiddenLock:
    """A lock that fails the test when anything tries to take it."""

    def acquire(self, *_arguments, **_options):
        pytest.fail("a warm path took the cold-path lock")

    __enter__ = acquire

    def __exit__(self, *_error):
        return False

    def release(self):
        pass


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


def test_in_process_kernel_caches_are_bounded():
    """Give every in-process cache of compiled or loaded kernels a bound."""
    for cache in (
        _runtime._ptx_cache,
        _runtime._loaded_functions,
        qualification._ptx_memo,
    ):
        assert isinstance(cache, _runtime._BoundedCache)
        assert cache.limit == _runtime._CACHE_LIMIT


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


def test_module_stays_loaded_while_its_function_is_referenced(fake_cuda):
    """Unload a module only after the last holder of its function is gone."""
    driver = fake_cuda.driver
    module, function = driver.load("ptx", "kernel")
    assert (module, function) == (0x1000, 0x1001)
    holder = [function]
    del function
    gc.collect()

    second = driver.load("second", "kernel")
    assert fake_cuda.released() == []

    holder.clear()
    gc.collect()
    assert fake_cuda.released() == []

    third = driver.load("third", "kernel")
    assert fake_cuda.released() == [
        ("cuCtxSynchronize",),
        ("cuModuleUnload", 0x1000),
    ]
    assert (second[1], third[1]) == (0x1011, 0x1021)


def test_unload_follows_one_context_synchronize(fake_cuda):
    """Wait for every queued launch once, then unload each idle module."""
    driver = fake_cuda.driver
    held = [driver.load(name, "kernel") for name in ("a", "b", "c")]
    assert fake_cuda.released() == []
    del held
    gc.collect()

    driver.unload_retired()

    released = fake_cuda.released()
    assert released[0] == ("cuCtxSynchronize",)
    assert sorted(released[1:]) == [
        ("cuModuleUnload", 0x1000),
        ("cuModuleUnload", 0x1010),
        ("cuModuleUnload", 0x1020),
    ]
    driver.unload_retired()
    assert len(fake_cuda.released()) == 4


def test_no_module_is_unloaded_while_this_thread_captures(
    fake_cuda, monkeypatch
):
    """Leave idle modules loaded during a capture, then unload them."""
    driver = fake_cuda.driver
    driver.load("ptx", "kernel")
    gc.collect()

    with monkeypatch.context() as patch:
        patch.setattr(_runtime, "_capturing", lambda: True)
        kept = driver.load("second", "kernel")
        assert fake_cuda.released() == []

    driver.unload_retired()
    assert fake_cuda.released() == [
        ("cuCtxSynchronize",),
        ("cuModuleUnload", 0x1000),
    ]
    assert kept[1] == 0x1011


def test_module_of_another_context_waits_for_its_context(fake_cuda):
    """Synchronize and unload only in the context that loaded the module."""
    driver = fake_cuda.driver
    driver.load("ptx", "kernel")
    gc.collect()

    fake_cuda.context = 2
    driver.unload_retired()
    assert fake_cuda.released() == []

    fake_cuda.context = 1
    driver.unload_retired()
    assert fake_cuda.released() == [
        ("cuCtxSynchronize",),
        ("cuModuleUnload", 0x1000),
    ]


def test_failed_synchronize_warns_and_keeps_the_modules_queued(fake_cuda):
    """Never unload without the wait, and never fail the load for it."""
    driver = fake_cuda.driver
    driver.load("ptx", "kernel")
    gc.collect()
    fake_cuda.failures["cuCtxSynchronize"] = _CAPTURE_UNSUPPORTED

    with pytest.warns(RuntimeWarning, match="cuCtxSynchronize failed"):
        kept = driver.load("second", "kernel")
    assert kept[1] == 0x1011
    assert fake_cuda.calls("cuModuleUnload") == []

    del fake_cuda.failures["cuCtxSynchronize"]
    driver.unload_retired()
    assert fake_cuda.calls("cuModuleUnload") == [("cuModuleUnload", 0x1000)]


def test_failed_unload_warns_and_keeps_the_module_queued(fake_cuda):
    """Report a module the driver refuses to unload and try it again."""
    driver = fake_cuda.driver
    driver.load("ptx", "kernel")
    gc.collect()
    fake_cuda.failures["cuModuleUnload"] = 400

    with pytest.warns(RuntimeWarning, match="cuModuleUnload failed"):
        driver.unload_retired()
    assert list(driver._retired) == [(1, 0x1000)]

    del fake_cuda.failures["cuModuleUnload"]
    driver.unload_retired()
    assert fake_cuda.calls("cuModuleUnload") == [("cuModuleUnload", 0x1000)] * 2
    assert not driver._retired


def _retire_three_modules(fake_cuda):
    """Load three kernels, drop them, and refuse to unload the first."""
    driver = fake_cuda.driver
    # Held until all three are loaded, so that no load unloads an earlier one.
    loaded = [driver.load(name, "kernel") for name in ("a", "b", "c")]
    modules = [module for module, _ in loaded]
    del loaded
    gc.collect()
    assert sorted(module for _, module in driver._retired) == modules
    fake_cuda._cuModuleUnload = lambda module: (
        400 if module.value == modules[0] else 0
    )
    return driver, modules


def test_one_failed_unload_does_not_lose_the_modules_after_it(fake_cuda):
    """Unload the rest, keep the refused module queued, and warn once."""
    driver, modules = _retire_three_modules(fake_cuda)

    with pytest.warns(RuntimeWarning) as caught:
        driver.unload_retired()

    assert [str(warning.message) for warning in caught] == [
        "Swage left 1 unused CUDA module loaded: CUDA Driver "
        "cuModuleUnload failed: CUDA_ERROR (400): driver error"
    ]
    assert sorted(fake_cuda.calls("cuModuleUnload")) == [
        ("cuModuleUnload", module) for module in modules
    ]
    assert list(driver._retired) == [(1, modules[0])]


def test_interrupted_unload_keeps_every_module_it_did_not_unload(fake_cuda):
    """Requeue the module in hand and the ones not yet tried."""
    driver, modules = _retire_three_modules(fake_cuda)
    unloads = []

    def interrupt_the_second(module):
        unloads.append(module.value)
        if len(unloads) == 2:
            raise KeyboardInterrupt

    fake_cuda._cuModuleUnload = interrupt_the_second

    with pytest.raises(KeyboardInterrupt):
        driver.unload_retired()

    assert sorted(module for _, module in driver._retired) == sorted(
        set(modules) - {unloads[0]}
    )


@pytest.mark.parametrize("failing", ["cuModuleUnload", "cuCtxSynchronize"])
def test_unload_never_raises_into_a_load_when_warnings_are_errors(
    fake_cuda, capsys, failing
):
    """Load the unrelated kernel and keep every module that stays loaded."""
    driver, modules = _retire_three_modules(fake_cuda)
    if failing == "cuCtxSynchronize":
        # Nothing is unloaded without the wait, so every module stays.
        fake_cuda.failures["cuCtxSynchronize"] = _CAPTURE_UNSUPPORTED
        left, tried = modules, []
    else:
        left, tried = modules[:1], modules

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, function = driver.load("unrelated", "kernel")

    assert function == 0x1031
    assert sorted(module for _, module in driver._retired) == left
    assert sorted(fake_cuda.calls("cuModuleUnload")) == [
        ("cuModuleUnload", module) for module in tried
    ]
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
    monkeypatch.setattr(_runtime.ctypes, "CDLL", lambda _name: cuda.library)

    assert _runtime._CudaDriver().current_context() == 0xC000


def test_no_current_context_is_reported(fake_cuda):
    """Keep the error for a thread that has no CUDA context."""
    fake_cuda.failures["cuCtxGetId"] = 201
    fake_cuda._cuCtxGetCurrent = lambda context: None

    with pytest.raises(RuntimeError, match="no current CUDA context"):
        fake_cuda.driver.current_context()


def test_function_launched_under_capture_is_never_unloaded(fake_cuda):
    """Keep a module that a CUDA graph may replay after the cache drops it."""
    driver = fake_cuda.driver
    fake_cuda.capturing_streams.add(7)
    _, captured = driver.load("captured", "kernel")
    _, plain = driver.load("plain", "kernel")

    driver.launch(captured, (1,), 32, 7, (1, 2, 3, 4))
    driver.launch(plain, (1,), 32, 9, (1, 2, 3, 4))
    driver.launch(plain, (1,), 32, 0, (1, 2, 3, 4))
    del captured, plain
    gc.collect()
    driver.unload_retired()

    assert fake_cuda.calls("cuModuleUnload") == [("cuModuleUnload", 0x1010)]
    # The NULL stream cannot capture, so the default stream is never asked.
    assert len(fake_cuda.calls("cuStreamIsCapturing")) == 2
    assert len(fake_cuda.calls("cuLaunchKernel")) == 3


def test_get_driver_takes_no_lock_once_created(monkeypatch):
    """Return the existing driver without the cold-path lock."""
    driver = object()
    monkeypatch.setattr(_runtime, "_driver", driver)
    monkeypatch.setattr(_runtime, "_compile_lock", _ForbiddenLock())

    assert _runtime._get_driver() is driver


def _stub_public_compiler(monkeypatch, compile_native):
    """Compile through `compile_native` with the disk cache out of play."""
    identity = {
        "revision": None,
        "clean": False,
        "llvm": None,
        "frontend": None,
        "native": None,
    }
    monkeypatch.setattr(_runtime, "_compiler_identity", lambda: identity)
    monkeypatch.setattr(_runtime, "_compile_native", compile_native)
    monkeypatch.setattr(
        _runtime, "_ptx_cache", _runtime._BoundedCache(_runtime._CACHE_LIMIT)
    )


def _compile_public(key):
    return _runtime._compile_cached(
        {"compute_capability": "sm_86"}, "kernel", 128, object, key=key
    )


def test_warm_public_artifact_takes_no_lock(monkeypatch):
    """Serve a compiled kernel again without the cold-path lock."""
    compiles = []

    def compile_native(*arguments):
        compiles.append(arguments)
        return "lowered", "ptx"

    _stub_public_compiler(monkeypatch, compile_native)
    first = _compile_public("warm")
    monkeypatch.setattr(_runtime, "_compile_lock", _ForbiddenLock())

    assert _compile_public("warm") is first
    assert len(compiles) == 1


def test_cold_public_compile_does_not_block_a_warm_artifact(monkeypatch):
    """Serve a compiled kernel while another thread compiles a new one."""
    compiling = threading.Event()
    finish = threading.Event()

    def compile_native(_module, _kernel_name, block_size, _target):
        if block_size == 64:
            compiling.set()
            finish.wait(_HOLD)
        return "lowered", f"ptx{block_size}"

    _stub_public_compiler(monkeypatch, compile_native)
    warm = _compile_public("warm")
    cold = threading.Thread(
        target=lambda: _runtime._compile_cached(
            {"compute_capability": "sm_86"}, "kernel", 64, object, key="cold"
        )
    )
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
    assert _runtime._ptx_cache["warm"] is warm
    assert _runtime._ptx_cache["cold"].ptx == "ptx64"


def test_public_artifact_cache_recompiles_what_it_forgot(monkeypatch):
    """Compile a kernel again after the bound pushed it out."""
    compiles = []

    def compile_native(_module, _kernel_name, block_size, _target):
        compiles.append(block_size)
        return "lowered", f"ptx{block_size}"

    _stub_public_compiler(monkeypatch, compile_native)
    monkeypatch.setattr(_runtime, "_ptx_cache", _runtime._BoundedCache(2))

    for key in ("a", "b", "a", "c", "a"):
        _compile_public(key)

    assert len(compiles) == 4
    assert list(_runtime._ptx_cache) == ["c", "a"]


@pytest.fixture
def driver_calls(monkeypatch):
    """Count the loads, unloads, and waits the real driver performs."""
    driver = _runtime._get_driver()
    gc.collect()
    torch.cuda.synchronize()
    driver.unload_retired()
    calls = collections.Counter()

    def counting(name, original):
        def call(*arguments):
            result = original(*arguments)
            if result == 0:
                calls[name] += 1
            return result

        return call

    for name in ("cuModuleLoadData", "cuModuleUnload", "cuCtxSynchronize"):
        monkeypatch.setattr(
            driver.library, name, counting(name, getattr(driver.library, name))
        )
    return calls


def _snapshot(calls):
    return (
        calls["cuModuleLoadData"],
        calls["cuModuleUnload"],
        calls["cuCtxSynchronize"],
    )


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
        module_text=qualification._semantic_module(kind),
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

    The cache holds five modules, one planned program. A second program
    pushes the first out of the cache, and the first is unloaded only after
    its prepared launches are gone, after one wait for the context.
    """
    monkeypatch.setattr(_runtime, "_CACHE_LIMIT", 5)
    monkeypatch.setattr(
        qualification, "_load_memo", weakref.WeakKeyDictionary()
    )
    lengths = [[1, 33, 4097, 0, 32], [40, 2, 8193, 5]]
    cases = [_case(lengths[0]), _case(lengths[1])]
    outputs = [torch.empty(5, device="cuda"), torch.empty(4, device="cuda")]

    first = _prepare("sum", *cases[0][:2], outputs[0])
    assert _snapshot(driver_calls) == (5, 0, 0)
    second = _prepare("sum", *cases[1][:2], outputs[1])
    assert _snapshot(driver_calls) == (5, 0, 0)
    _assert_policies(first, outputs[0], cases[0][2]["sum"])
    _assert_policies(second, outputs[1], cases[1][2]["sum"])

    # Another program fills the cache. The first is still held by its
    # prepared launches, so nothing is unloaded and it still launches.
    maximum = _prepare("max", *cases[1][:2], outputs[1])
    assert _snapshot(driver_calls) == (10, 0, 0)
    _assert_policies(maximum, outputs[1], cases[1][2]["max"])
    _assert_policies(first, outputs[0], cases[0][2]["sum"])

    # Queue work on the kernels that are about to lose their last holder.
    outputs[0].fill_(float("nan"))
    for _ in range(64):
        first.mixed()
    del first, second
    gc.collect()
    assert _snapshot(driver_calls) == (10, 0, 0)

    # The next load waits for the queued launches, unloads the five idle
    # modules, and loads the program again.
    again = _prepare("sum", *cases[0][:2], outputs[0])
    assert _snapshot(driver_calls) == (15, 5, 1)
    torch.testing.assert_close(
        outputs[0].cpu(), cases[0][2]["sum"], rtol=0, atol=0
    )
    _assert_policies(again, outputs[0], cases[0][2]["sum"])
    _assert_policies(maximum, outputs[1], cases[1][2]["max"])
    assert _snapshot(driver_calls) == (15, 5, 1)


@_requires_cuda
def test_real_lookup_failure_unloads_the_module(driver_calls):
    """Unload a real module whose kernel name does not resolve."""
    from mlir_swage._mlir_libs._swageDialectsNanobind import swage as native

    ptx = qualification._compile_once(
        native._compile_segmented_reduction_ptx,
        qualification._semantic_module("sum"),
        kernel_name="segmented_sum",
        block_size=128,
        target="sm_{}{}".format(*torch.cuda.get_device_capability()),
    )

    with pytest.raises(RuntimeError, match="cuModuleGetFunction failed"):
        _runtime._get_driver().load(ptx, "no_such_kernel")

    assert _snapshot(driver_calls)[:2] == (1, 1)


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
        ("cuCtxCreate_v2", [ctypes.POINTER(pointer), ctypes.c_uint,
                            ctypes.c_int]),
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
    driver = _runtime._get_driver()
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
def test_load_memo_loads_again_in_a_second_real_context(
    driver_calls, monkeypatch
):
    """Never serve a kernel of one real context in another."""
    from mlir_swage._mlir_libs._swageDialectsNanobind import swage as native

    monkeypatch.setattr(
        qualification, "_load_memo", weakref.WeakKeyDictionary()
    )
    driver = _runtime._get_driver()
    ptx = qualification._compile_once(
        native._compile_segmented_reduction_ptx,
        qualification._semantic_module("sum"),
        kernel_name="segmented_sum",
        block_size=128,
        target="sm_{}{}".format(*torch.cuda.get_device_capability()),
    )

    first = qualification._load_once(driver, ptx, "segmented_sum")
    with _second_context(driver):
        second = qualification._load_once(driver, ptx, "segmented_sum")
        assert qualification._load_once(driver, ptx, "segmented_sum") is second
    again = qualification._load_once(driver, ptx, "segmented_sum")

    assert again is first
    assert second is not first
    assert _snapshot(driver_calls)[:2] == (2, 0)


@_requires_cuda
def test_context_identity_survives_a_reused_context_handle():
    """Tell a recreated context from the destroyed one at its address."""
    driver = _runtime._get_driver()
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


class _DelegatingDriver:
    """Forward to the real driver, with the persistent launch replaced."""

    def __init__(self, driver, launch_persistent):
        self._driver = driver
        self._launch_persistent = launch_persistent

    def launch_persistent(self, *arguments):
        return self._launch_persistent(self._driver, *arguments)

    def __getattr__(self, name):
        return getattr(self._driver, name)


@_requires_cuda
def test_persistent_launch_rejects_a_concurrent_launch(monkeypatch):
    """Reject a second thread that launches while the first still does."""
    values, offsets, expected = _case([1, 33, 4097, 2, 8193])
    output = torch.full((5,), float("nan"), device="cuda")
    inside = threading.Event()
    proceed = threading.Event()
    launches = []

    def launch_persistent(driver, *arguments):
        name = threading.current_thread().name
        launches.append(name)
        if name == "first":
            inside.set()
            proceed.wait(_HOLD)
        driver.launch_persistent(*arguments)

    driver = _DelegatingDriver(_runtime._get_driver(), launch_persistent)
    monkeypatch.setattr(_runtime, "_get_driver", lambda: driver)
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

    def launch_persistent(driver, *arguments):
        launches.append(threading.current_thread().name)
        driver.launch_persistent(*arguments)

    driver = _DelegatingDriver(_runtime._get_driver(), launch_persistent)
    monkeypatch.setattr(_runtime, "_get_driver", lambda: driver)
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


@_requires_cuda
def test_warm_public_launch_takes_no_lock(monkeypatch):
    """Launch a loaded kernel without the cold-path lock."""
    x, y, output = _public_tensors()
    _add(x, y, output, 128)
    torch.testing.assert_close(output, x + y, rtol=0, atol=0)
    output.fill_(float("nan"))
    monkeypatch.setattr(_runtime, "_compile_lock", _ForbiddenLock())

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
    monkeypatch.setattr(
        _runtime, "_ptx_cache", _runtime._BoundedCache(_runtime._CACHE_LIMIT)
    )
    _add(x, y, output, 128)
    compiling = threading.Event()
    finish = threading.Event()
    compile_native = _runtime._compile_native

    def slow_compile(module, kernel_name, block_size, target):
        if block_size == 64:
            compiling.set()
            finish.wait(_HOLD)
        return compile_native(module, kernel_name, block_size, target)

    monkeypatch.setattr(_runtime, "_compile_native", slow_compile)
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
    """Bound the loaded kernels of the public launch and stay correct."""
    monkeypatch.setattr(
        _runtime, "_loaded_functions", _runtime._BoundedCache(1)
    )
    x, y, output = _public_tensors()

    for index, block in enumerate([128, 64, 128, 64]):
        output.fill_(float("nan"))
        _add(x, y, output, block)
        torch.testing.assert_close(output, x + y, rtol=0, atol=0)
        # Every launch loads; from the third on, the load before it
        # unloads the kernel that the launch two steps back forgot.
        assert _snapshot(driver_calls)[:2] == (index + 1, max(index - 1, 0))
        assert len(_runtime._loaded_functions) == 1


@_requires_cuda
def test_captured_public_launch_keeps_its_kernel_loaded(
    driver_calls, monkeypatch
):
    """Replay a captured launch after the cache forgot its kernel."""
    monkeypatch.setattr(
        _runtime, "_loaded_functions", _runtime._BoundedCache(1)
    )
    x, y, output = _public_tensors()
    _add(x, y, output, 256)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _add(x, y, output, 256)

    for block in (128, 64, 32):
        _add(x, y, output, block)
    gc.collect()
    torch.cuda.synchronize()
    _runtime._get_driver().unload_retired()
    # The kernels of blocks 128 and 64 were forgotten and unloaded. The
    # captured kernel was forgotten first and is still loaded.
    assert _snapshot(driver_calls)[:2] == (4, 2)

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

    The cache holds one kernel and the threads use four, so nearly every
    launch loads a kernel and unloads one that another thread just used.
    """
    monkeypatch.setattr(
        _runtime, "_loaded_functions", _runtime._BoundedCache(1)
    )
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
    loads, unloads, _ = _snapshot(driver_calls)
    assert loads > len(blocks)
    assert loads - unloads <= len(blocks) + 1


@_requires_cuda
def test_concurrent_preparations_survive_unloading(driver_calls, monkeypatch):
    """Prepare, launch, and drop from several threads at a small bound.

    The memo holds one planned program and the threads alternate between
    two, so prepared launches keep running kernels the memo has forgotten.
    """
    monkeypatch.setattr(_runtime, "_CACHE_LIMIT", 5)
    monkeypatch.setattr(
        qualification, "_load_memo", weakref.WeakKeyDictionary()
    )
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
    loads, unloads, _ = _snapshot(driver_calls)
    assert unloads > 0
    assert loads >= unloads
