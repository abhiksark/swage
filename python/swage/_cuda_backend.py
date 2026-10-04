# python/swage/_cuda_backend.py
"""CUDA compiler, residency, and asynchronous launch adapter.

A compiled artifact is loaded once per CUDA context into a `_LoadedModule`.
A launch holds a `_LoadedModuleLease` on the module for as long as it may
use it; a prepared segmented execution holds its leases until it is
collected. The loaded modules form an LRU bounded by
`SWAGE_MEMORY_CACHE_ENTRIES`. A module that leaves it is retired, and it is
unloaded only after no lease holds it, no CUDA graph captured it, and an
event recorded on every stream that launched it has completed. Retired
modules are unloaded when a later lease or launch finds them idle, never
while a stream of the context captures a graph.

Locks: a warm lease takes no process-wide lock. It reads the loaded module
with one `get` and counts itself under the module's own lock. `_cuda_lock`
guards the residency fields and the two caches, and the cold path (loading
a module) also holds the cold-path lock of `_runtime`, which interpreter
exit and `fork` coordinate with. The lock order is module lock, then
`_cuda_lock`.
"""

import ctypes
import os
import sys
import threading
import warnings
import weakref
from collections import OrderedDict

from . import _abi
from ._errors import BackendUnavailableError

_DEFAULT_MEMORY_CACHE_ENTRIES = 128
_cuda_lock = threading.Lock()
_loaded_functions = OrderedDict()
_retired_loaded = {}
_driver = None
_memory_cache_entries = None
_device_facts_cache = weakref.WeakKeyDictionary()
_stream_objects = weakref.WeakKeyDictionary()
# CUDA_ERROR_NOT_READY from cuEventQuery.
_NOT_READY = 600
# CUDA_ERROR_STREAM_CAPTURE_IMPLICIT from cuStreamIsCapturing.
_CAPTURE_IMPLICIT = 906


class _LoadedModule:
    """One CUDA module whose lookup residency is separate from its lifetime."""

    __slots__ = (
        "key",
        "context",
        "module",
        "function",
        "driver",
        "events",
        "pending_streams",
        "leases",
        "cache_resident",
        "capture_pinned",
        "unload_queued",
        "unload_blocked",
        "unloading",
        "unloaded",
        "lock",
        "fixed_launcher",
        "__weakref__",
    )

    def __init__(self, key, context, module, function, driver):
        self.key = key
        self.context = context
        self.module = module
        self.function = function
        self.driver = driver
        self.events = {}
        self.pending_streams: set[int] = set()
        self.leases = 0
        self.cache_resident = True
        self.capture_pinned = False
        self.unload_queued = False
        self.unload_blocked = False
        self.unloading = False
        self.unloaded = False
        self.lock = threading.Lock()
        self.fixed_launcher = None


class _LoadedModuleLease:
    """A Python-only claim that keeps one loaded module from unloading.

    `entry.function` is the function handle `cuLaunchKernel` takes. It
    stays valid for as long as some lease on the module is unreleased.
    """

    __slots__ = ("entry", "released")

    def __init__(self, entry):
        self.entry = entry
        self.released = False

    def release(self):
        """Give the claim back once; a retired module may then unload."""
        if self.released:
            return
        self.released = True
        entry = self.entry
        with entry.lock:
            entry.leases -= 1
        if not entry.cache_resident:
            with _cuda_lock:
                _queue_unload_locked(entry)

    def __del__(self):
        self.release()


def _memory_cache_limit():
    """Return the strict positive capacity of the loaded-module LRU."""
    global _memory_cache_entries
    limit = _memory_cache_entries
    if limit is not None:
        return limit
    with _cuda_lock:
        if _memory_cache_entries is None:
            configured = os.environ.get("SWAGE_MEMORY_CACHE_ENTRIES")
            if configured is None:
                _memory_cache_entries = _DEFAULT_MEMORY_CACHE_ENTRIES
            elif (
                not configured
                or not configured.isascii()
                or not configured.isdigit()
                or int(configured) <= 0
            ):
                raise ValueError(
                    "SWAGE_MEMORY_CACHE_ENTRIES must be a positive integer"
                )
            else:
                _memory_cache_entries = int(configured)
        return _memory_cache_entries


def cuda_unavailable():
    """Return the error for a PyTorch that has no usable CUDA device."""
    return BackendUnavailableError(
        "CUDA is unavailable in PyTorch",
        code="cuda-unavailable",
        backend="cuda",
        remediation=(
            "Install a CUDA-enabled PyTorch build and verify that an "
            "NVIDIA GPU is accessible to this process."
        ),
    )


def validate_device(tensors, runtime_names, torch):
    """Require CUDA availability and tensors on the active device."""
    if not torch.cuda.is_available():
        raise cuda_unavailable()
    current_device = torch.cuda.current_device()
    for name, tensor in zip(runtime_names, tensors):
        if tensor.device.index != current_device:
            raise ValueError(
                f"argument '{name}' must be on the current CUDA device"
            )
    return current_device


def _device_facts(torch, index):
    """Cached (max threads, sm target) per device for this torch module.

    Device facts cannot change within a process, but tests inject fresh
    fake torch modules, so the cache is scoped to the torch module object.
    """
    per_torch = _device_facts_cache.get(torch)
    if per_torch is None:
        per_torch = {}
        _device_facts_cache[torch] = per_torch
    facts = per_torch.get(index)
    if facts is None:
        properties = torch.cuda.get_device_properties(index)
        major, minor = torch.cuda.get_device_capability(index)
        facts = (properties.max_threads_per_block, f"sm_{major}{minor}")
        per_torch[index] = facts
    return facts


def current_stream(torch, index):
    """Return the current stream without rebuilding an unchanged object.

    The stream objects are bounded per importing torch module.
    """
    raw_stream = getattr(
        getattr(torch, "_C", None), "_cuda_getCurrentRawStream", None
    )
    if raw_stream is None:
        return torch.cuda.current_stream()
    handle = raw_stream(index)
    per_torch = _stream_objects.get(torch)
    if per_torch is None:
        per_torch = OrderedDict()
        _stream_objects[torch] = per_torch
    key = (index, handle)
    cached = per_torch.get(key)
    if cached is None:
        cached = torch.cuda.current_stream(index)
        per_torch[key] = cached
        while len(per_torch) > _DEFAULT_MEMORY_CACHE_ENTRIES:
            per_torch.popitem(last=False)
    else:
        per_torch.move_to_end(key)
    return cached


def validate_geometry(block, n, grid, torch, current_device):
    """Validate a fixed launch against the active CUDA device."""
    max_threads, target = _device_facts(torch, current_device)
    if block > max_threads:
        raise ValueError(f"BLOCK {block} exceeds device limit {max_threads}")
    expected_grid = ((n + block - 1) // block,)
    if grid != expected_grid:
        raise ValueError(f"grid must equal {expected_grid} for n and BLOCK")
    return target, current_stream(torch, current_device)


def is_current_stream_capturing(torch):
    """Return whether this thread is capturing a CUDA graph in PyTorch."""
    predicate = getattr(torch.cuda, "is_current_stream_capturing", None)
    return False if predicate is None else bool(predicate())


def _capturing():
    """Return whether this thread captures a graph, with torch imported."""
    torch = sys.modules.get("torch")
    return torch is not None and is_current_stream_capturing(torch)


def ensure_context(torch, device_index):
    """Return the current context, making the device's context current.

    A thread that has not used CUDA has no current context, and PyTorch
    makes the device's context current with its first CUDA call there. A
    launch may make no such call before it reaches the driver, so the
    context is made current here. No context is created: the device holds
    validated tensors, so its context exists. This is the path of a
    thread's first launch only; a context that is already current is never
    replaced.

    Args:
        torch: The PyTorch module.
        device_index: Index of the device of the validated tensors, which
            the caller has checked to be the current device.

    Returns:
        The identifier of the context that is current now.

    Raises:
        BackendUnavailableError: The thread still has no current context.
    """
    driver = _get_driver()
    try:
        return driver.current_context()
    except RuntimeError:
        torch.cuda.set_device(device_index)
        return driver.current_context()


def _compile_native(
    module,
    kernel_name,
    block_size,
    target,
    lowering_kind,
    lowering_options,
):
    """Compile one kernel with the native bindings.

    Returns:
        The lowered MLIR, the PTX, and the launch contract as JSON.
    """
    from . import _runtime

    native_swage = _runtime._native_compiler(kernel_name, "cuda")
    compilers = {
        "fixed": "_compile_ptx",
        "segmented": "_compile_segmented_reduction_ptx",
        "segmented_fused": "_compile_fused_segmented_reduction_ptx",
        "segmented_split_partial": "_compile_split_partial_reduction_ptx",
        "segmented_split_merge": "_compile_split_merge_reduction_ptx",
        "segmented_persistent": "_compile_persistent_segmented_reduction_ptx",
    }
    try:
        compiler_name = compilers[lowering_kind]
    except KeyError as error:
        raise ValueError(f"unknown lowering kind {lowering_kind!r}") from error
    native_kernel_name = kernel_name
    if lowering_kind in {
        "segmented_split_partial",
        "segmented_split_merge",
    }:
        native_kernel_name = kernel_name.rsplit("__", 1)[0]
    arguments = {
        "kernel_name": native_kernel_name,
        "target": target,
        **lowering_options,
    }
    if lowering_kind in {"fixed", "segmented"}:
        arguments["block_size"] = block_size
    return getattr(native_swage, compiler_name)(module, **arguments)


def _queue_unload_locked(entry):
    if (
        not entry.cache_resident
        and not entry.unloaded
        and not entry.unload_queued
    ):
        entry.unload_queued = True
        # An older module of the same key may still wait for its last
        # launch; both stay queued, so neither is forgotten loaded.
        slot = entry.key
        if _retired_loaded.get(slot, entry) is not entry:
            slot = (entry.key, id(entry))
        _retired_loaded[slot] = entry


def _evict_loaded_locked(limit):
    while len(_loaded_functions) > limit:
        _, entry = _loaded_functions.popitem(last=False)
        entry.cache_resident = False
        _queue_unload_locked(entry)


def _report_modules_left_loaded(count, error):
    """Report modules that stay loaded, without ever raising.

    The caller is loading or launching an unrelated kernel and the modules
    stay queued, so nothing is lost by going on. When the warning filters
    turn the report into an exception, it is written to standard error
    instead.
    """
    noun = "module" if count == 1 else "modules"
    try:
        warnings.warn(
            f"Swage left {count} unused CUDA {noun} loaded: {error}",
            RuntimeWarning,
            stacklevel=3,
        )
    except RuntimeWarning as raised:
        print(f"RuntimeWarning: {raised}", file=sys.stderr)


def _poll_deferred(driver, context, *, capturing=False):
    """Fence, query, and unload the idle retired modules of one context.

    A retired module is unloaded once no lease holds it, no CUDA graph
    captured it, and the event recorded after its last launch on each
    stream has completed. The legacy default stream is fenced here with a
    fresh event record; any other stream was fenced right after its launch,
    because the caller may destroy it. Nothing is unloaded while this
    thread or a pending stream captures a graph: CUDA forbids both calls
    then, and an event recorded during capture is a graph node, not
    evidence that earlier work completed.

    A driver error leaves the module loaded for the rest of the process and
    is reported once per call, never raised, because the caller is working
    on an unrelated kernel.
    """
    if capturing or not _retired_loaded:
        return
    with _cuda_lock:
        candidates = [
            entry
            for entry in _retired_loaded.values()
            if entry.driver is driver
            and entry.context == context
            and not entry.cache_resident
            and entry.leases == 0
            and not entry.capture_pinned
            and not entry.unload_blocked
            and not entry.unloading
            and not entry.unloaded
        ]
        for entry in candidates:
            entry.unloading = True

    blocked = 0
    first_error = None
    for index, entry in enumerate(candidates):
        try:
            with entry.lock:
                with _cuda_lock:
                    eligible = (
                        not entry.cache_resident
                        and entry.leases == 0
                        and not entry.capture_pinned
                        and not entry.unload_blocked
                    )
                if not eligible:
                    continue
                if any(
                    driver.is_stream_capturing(stream)
                    for stream in entry.pending_streams
                ):
                    continue
                # Zero leases and the unloading reservation keep another
                # launch from racing these final per-stream fences.
                for stream in tuple(entry.pending_streams):
                    driver.event_record(entry.events[stream], stream)
                    entry.pending_streams.remove(stream)
                statuses = [
                    driver.event_query(event)
                    for event in tuple(entry.events.values())
                ]
                if not all(statuses):
                    continue
                for event in tuple(entry.events.values()):
                    driver.event_destroy(event)
                driver.module_unload(entry.module)
                with _cuda_lock:
                    entry.events.clear()
                    entry.unloaded = True
                    entry.unload_queued = False
                    for slot, queued in tuple(_retired_loaded.items()):
                        if queued is entry:
                            del _retired_loaded[slot]
        except RuntimeError as error:
            with _cuda_lock:
                entry.unload_blocked = True
            blocked += 1
            first_error = first_error or error
        except BaseException:
            with _cuda_lock:
                entry.unload_blocked = True
                for pending in candidates[index + 1 :]:
                    pending.unloading = False
            raise
        finally:
            with _cuda_lock:
                entry.unloading = False
    if blocked:
        _report_modules_left_loaded(blocked, first_error)


def _warm_lease(key):
    """Claim a resident module without any process-wide lock, or None."""
    loaded = _loaded_functions.get(key)
    if loaded is None:
        return None
    with loaded.lock:
        if (
            not loaded.cache_resident
            or loaded.unloading
            or loaded.unloaded
            or loaded.unload_blocked
        ):
            return None
        loaded.leases += 1
    # Recency is refreshed only when nobody else holds the residency lock,
    # so a warm lease never waits.
    if len(_loaded_functions) > 1 and _cuda_lock.acquire(blocking=False):
        try:
            if _loaded_functions.get(key) is loaded:
                _loaded_functions.move_to_end(key)
        finally:
            _cuda_lock.release()
    return _LoadedModuleLease(loaded)


def _load_artifact(artifact, driver, *, capturing=False):
    """Return a lease on the module of `artifact` in the current context."""
    limit = _memory_cache_limit()
    context = driver.current_context()
    _poll_deferred(driver, context, capturing=capturing)
    # The driver is part of the key because a handle is valid only with the
    # driver that loaded it; a process normally has one driver.
    key = (artifact.identity, context, artifact.contract.entry, driver)
    lease = _warm_lease(key)
    if lease is not None:
        return lease
    from . import _runtime

    # The cold path: the module may have to be loaded, which waits for the
    # cold-path lock that interpreter exit and fork coordinate with.
    with _runtime._compile_lock:
        lease = _warm_lease(key)
        if lease is not None:
            return lease
        with _cuda_lock:
            loaded = _loaded_functions.get(key)
            if loaded is None:
                retired = _retired_loaded.get(key)
                if (
                    retired is not None
                    and not retired.unloading
                    and not retired.unloaded
                    and not retired.unload_blocked
                ):
                    loaded = _retired_loaded.pop(key)
                    loaded.cache_resident = True
                    loaded.unload_queued = False
        if loaded is None:
            module, function = driver.load(
                artifact.image, artifact.contract.entry
            )
            loaded = _LoadedModule(key, context, module, function, driver)
        with loaded.lock:
            loaded.leases += 1
        with _cuda_lock:
            _loaded_functions[key] = loaded
            _loaded_functions.move_to_end(key)
            _evict_loaded_locked(limit)
    return _LoadedModuleLease(loaded)


def _launch_loaded(
    entry,
    contract,
    bindings,
    grid,
    stream,
    *,
    capturing=False,
):
    """Enqueue one launch of a leased module and fence its stream."""
    driver = entry.driver
    context = driver.current_context()
    if context != entry.context:
        raise RuntimeError(
            "loaded CUDA module must launch in its original current context"
        )
    _poll_deferred(driver, context, capturing=capturing)
    with entry.lock:
        if stream not in entry.events:
            entry.events[stream] = driver.event_create()
        if capturing:
            with _cuda_lock:
                entry.capture_pinned = True
        # Only the context-owned legacy stream is guaranteed to remain valid
        # until retirement. Fence caller-owned streams before returning.
        if stream == 0:
            entry.pending_streams.add(stream)
        try:
            driver.launch_entry(
                entry.function, contract, bindings, grid, stream
            )
            if stream != 0 and not capturing:
                driver.event_record(entry.events[stream], stream)
        except BaseException:
            if stream != 0:
                with _cuda_lock:
                    entry.unload_blocked = True
            raise


def _prepare_fixed_launch(kernel, artifact, lease, stream, torch, dtype):
    """Keep a weak kernel shortcut only while its module remains resident."""
    if not isinstance(lease, _LoadedModuleLease):
        return
    entry = lease.entry
    native_type = getattr(entry.driver, "_native_fixed_launcher", None)
    if native_type is None:
        return
    from . import _runtime

    with entry.lock:
        with _cuda_lock:
            if (
                not entry.cache_resident
                or entry.unloading
                or entry.unloaded
                or entry.unload_blocked
                or stream.cuda_stream not in entry.events
            ):
                return
            fixed_launch = native_type(
                weakref.ref(entry),
                artifact,
                kernel,
                torch,
                stream,
                _runtime,
                sys.modules[__name__],
                dtype,
            )
            entry.fixed_launcher = fixed_launch
            kernel._cuda_fast_launch = weakref.ref(fixed_launch)


class _CudaDriver:
    """Small lazy wrapper around the Linux CUDA Driver API."""

    _native_launch = None
    _split_launch = None
    _native_fixed_launcher = None

    def __init__(self):
        try:
            self.library = ctypes.CDLL("libcuda.so.1")
        except OSError as error:
            raise BackendUnavailableError(
                "CUDA Driver library libcuda.so.1 is unavailable",
                code="cuda-driver-unavailable",
                backend="cuda",
                remediation=(
                    "Install a compatible NVIDIA driver and make "
                    "libcuda.so.1 accessible to this process."
                ),
            ) from error
        pointer = ctypes.c_void_p
        self.library.cuDriverGetVersion.argtypes = [
            ctypes.POINTER(ctypes.c_int)
        ]
        self.library.cuDriverGetVersion.restype = ctypes.c_int
        self.library.cuCtxGetCurrent.argtypes = [ctypes.POINTER(pointer)]
        self.library.cuCtxGetCurrent.restype = ctypes.c_int
        self.library.cuModuleLoadData.argtypes = [
            ctypes.POINTER(pointer),
            pointer,
        ]
        self.library.cuModuleLoadData.restype = ctypes.c_int
        self.library.cuModuleGetFunction.argtypes = [
            ctypes.POINTER(pointer),
            pointer,
            ctypes.c_char_p,
        ]
        self.library.cuModuleGetFunction.restype = ctypes.c_int
        self.library.cuModuleUnload.argtypes = [pointer]
        self.library.cuModuleUnload.restype = ctypes.c_int
        self.library.cuEventCreate.argtypes = [
            ctypes.POINTER(pointer),
            ctypes.c_uint,
        ]
        self.library.cuEventCreate.restype = ctypes.c_int
        self.library.cuEventRecord.argtypes = [pointer, pointer]
        self.library.cuEventRecord.restype = ctypes.c_int
        self.library.cuEventQuery.argtypes = [pointer]
        self.library.cuEventQuery.restype = ctypes.c_int
        self.library.cuStreamIsCapturing.argtypes = [
            pointer,
            ctypes.POINTER(ctypes.c_int),
        ]
        self.library.cuStreamIsCapturing.restype = ctypes.c_int
        try:
            self._event_destroy = self.library.cuEventDestroy_v2
        except AttributeError:
            self._event_destroy = self.library.cuEventDestroy
        self._event_destroy.argtypes = [pointer]
        self._event_destroy.restype = ctypes.c_int
        # Context ids exist from CUDA 12. An older driver has only handles.
        self._context_id = getattr(self.library, "cuCtxGetId", None)
        if self._context_id is not None:
            self._context_id.argtypes = [
                pointer,
                ctypes.POINTER(ctypes.c_ulonglong),
            ]
            self._context_id.restype = ctypes.c_int
        self.library.cuLaunchKernel.argtypes = [
            pointer,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            pointer,
            ctypes.POINTER(pointer),
            ctypes.POINTER(pointer),
        ]
        self.library.cuLaunchKernel.restype = ctypes.c_int
        self.library.cuGetErrorName.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        self.library.cuGetErrorName.restype = ctypes.c_int
        self.library.cuGetErrorString.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        self.library.cuGetErrorString.restype = ctypes.c_int
        self._choose_launchers()

    def _choose_launchers(self):
        """Select the compiled launchers of this process, if any.

        A process that names an artifact launches through the runtime
        library of the artifact and never imports the bindings here, which
        would load LLVM into a process that selected an artifact to stay
        without it. A directory that cannot be used leaves the launcher
        open: the call that needs the artifact reports why, and a kernel
        of an artifact that loads later lends its launcher.

        Without an artifact, the compiled launcher of the bindings skips
        per-launch ctypes marshalling, and the native fixed launcher serves
        warm fixed launches; the ctypes path stays as the fallback when the
        bindings cannot be imported. Bindings built for another swage
        version are refused here, not replaced by the fallback.
        """
        # Imported here because these modules import this one.
        from . import _artifact, _runtime

        if os.environ.get(_artifact._ENVIRONMENT):
            try:
                self._split_launch = _artifact.selected()._launch_kernel
            except RuntimeError:
                self._split_launch = None
            return
        try:
            native_swage = _runtime._native_bindings()
        except ImportError:
            return
        self._native_launch = native_swage._launch_cuda_kernel
        self._native_fixed_launcher = native_swage._FixedCUDALaunch

    def _check_result(self, name, result):
        if result == 0:
            return
        error_name = ctypes.c_char_p()
        error_text = ctypes.c_char_p()
        self.library.cuGetErrorName(result, ctypes.byref(error_name))
        self.library.cuGetErrorString(result, ctypes.byref(error_text))
        stable_name = (
            error_name.value.decode() if error_name.value else "unknown"
        )
        stable_text = (
            error_text.value.decode() if error_text.value else "unknown"
        )
        raise RuntimeError(
            f"CUDA Driver {name} failed: {stable_name} ({result}): "
            f"{stable_text}"
        )

    def _call(self, name, *arguments):
        result = getattr(self.library, name)(*arguments)
        self._check_result(name, result)

    def driver_version(self):
        version = ctypes.c_int()
        self._call("cuDriverGetVersion", ctypes.byref(version))
        return f"{version.value // 1000}.{(version.value % 1000) // 10}"

    def current_context(self):
        """Identify the CUDA context that is current on this thread.

        Returns:
            The driver's id of the context, which is unique for the life of
            the process. A context handle is not: the driver places a new
            context at the address of a destroyed one, so by its handle a
            module loaded in the destroyed context would pass for a module
            of the new one. A driver without context ids returns the
            handle.

        Raises:
            BackendUnavailableError: No CUDA context is current on this
                thread.
        """
        if self._context_id is not None:
            identifier = ctypes.c_ulonglong()
            # A null context asks for the id of the current context.
            if not self._context_id(None, ctypes.byref(identifier)):
                return identifier.value
        context = ctypes.c_void_p()
        self._call("cuCtxGetCurrent", ctypes.byref(context))
        if not context.value:
            raise BackendUnavailableError(
                "PyTorch has no current CUDA context",
                code="cuda-context-unavailable",
                backend="cuda",
                remediation=(
                    "Initialize PyTorch CUDA on the selected device before "
                    "launching the kernel."
                ),
            )
        return context.value

    def load(self, ptx, kernel_name):
        """Load one module and resolve one kernel in it.

        A module whose kernel cannot be resolved is not left loaded.

        Returns:
            The module and function handles. They are plain handles: the
            caller decides when the module is unloaded.
        """
        module = ctypes.c_void_p()
        image = ctypes.create_string_buffer(ptx.encode())
        self._call(
            "cuModuleLoadData",
            ctypes.byref(module),
            ctypes.cast(image, ctypes.c_void_p),
        )
        function = ctypes.c_void_p()
        try:
            self._call(
                "cuModuleGetFunction",
                ctypes.byref(function),
                module,
                kernel_name.encode(),
            )
        except RuntimeError:
            try:
                self.module_unload(module.value)
            except RuntimeError as error:
                _report_modules_left_loaded(1, error)
            raise
        return module.value, function.value

    def is_stream_capturing(self, stream):
        status = ctypes.c_int()
        result = self.library.cuStreamIsCapturing(
            ctypes.c_void_p(stream), ctypes.byref(status)
        )
        if result == _CAPTURE_IMPLICIT:
            return True
        self._check_result("cuStreamIsCapturing", result)
        return status.value != 0

    def event_create(self):
        event = ctypes.c_void_p()
        # CU_EVENT_DISABLE_TIMING.
        self._call("cuEventCreate", ctypes.byref(event), 2)
        return event.value

    def event_record(self, event, stream):
        self._call(
            "cuEventRecord", ctypes.c_void_p(event), ctypes.c_void_p(stream)
        )

    def event_query(self, event):
        result = self.library.cuEventQuery(ctypes.c_void_p(event))
        if result == _NOT_READY:
            return False
        self._check_result("cuEventQuery", result)
        return True

    def event_destroy(self, event):
        result = self._event_destroy(ctypes.c_void_p(event))
        self._check_result("cuEventDestroy", result)

    def module_unload(self, module):
        self._call("cuModuleUnload", ctypes.c_void_p(module))

    def launch_entry(self, function, contract, bindings, grid, stream):
        """Launch one compiler-described ordered physical CUDA entry."""
        if not isinstance(contract, _abi.KernelContract):
            raise TypeError("contract must be a KernelContract")
        if contract.backend != "cuda" or contract.launch.model != "spmd-grid":
            raise ValueError("CUDA launch requires a CUDA spmd-grid contract")
        if contract.launch.block is None:
            raise ValueError("CUDA contract is missing block geometry")
        kinds, arguments = bindings
        expected_kinds = tuple(argument.kind for argument in contract.arguments)
        if tuple(kinds) != expected_kinds:
            raise ValueError("launch argument kinds do not match contract")
        grid = self._validate_geometry(grid, "grid")
        block = self._validate_geometry(contract.launch.block, "block")
        if block[0] * block[1] * block[2] > 1024:
            raise ValueError("block may contain at most 1024 threads")
        self._launch_ordered(
            function,
            grid,
            block,
            stream,
            tuple(kinds),
            tuple(arguments),
        )

    @staticmethod
    def _validate_geometry(value, label):
        if (
            not isinstance(value, tuple)
            or len(value) != 3
            or any(
                type(axis) is not int or not 0 < axis <= (1 << 32) - 1
                for axis in value
            )
        ):
            raise ValueError(
                f"{label} must contain exactly three positive u32 axes"
            )
        return value

    @staticmethod
    def _ctype_argument(kind, value, index):
        widths = {
            "i1": (1, ctypes.c_uint8),
            "i8": (8, ctypes.c_uint8),
            "i16": (16, ctypes.c_uint16),
            "f16": (16, ctypes.c_uint16),
            "bf16": (16, ctypes.c_uint16),
            "i32": (32, ctypes.c_uint32),
            "f32": (32, ctypes.c_uint32),
            "ptr": (64, ctypes.c_uint64),
            "i64": (64, ctypes.c_uint64),
            "f64": (64, ctypes.c_uint64),
        }
        try:
            width, storage = widths[kind]
        except KeyError as error:
            raise ValueError(
                f"kernel argument {index} has invalid kind {kind!r}"
            ) from error
        if type(value) is not int or not 0 <= value < 1 << width:
            raise ValueError(f"{kind} argument {index} is out of range")
        return storage(value)

    def _launch_ordered(self, function, grid, block, stream, kinds, arguments):
        if len(kinds) != len(arguments):
            raise ValueError("kernel argument kind/value counts differ")
        if self._native_launch is not None:
            self._native_launch(
                kinds, arguments, grid, block, 0, stream, function
            )
            return
        if self._split_launch is not None:
            self._launch_split(function, grid, block, stream, kinds, arguments)
            return
        values = [
            self._ctype_argument(kind, value, index)
            for index, (kind, value) in enumerate(zip(kinds, arguments))
        ]
        self._launch(function, grid, block, stream, values)

    def _launch_split(self, function, grid, block, stream, kinds, arguments):
        """Launch through a runtime library that takes pointers, then i32s.

        Every kernel the compiler emits takes its buffers first and its
        counts after them, so the argument list splits in two. A launch
        that does not fit that shape is refused instead of reordered.
        """
        pointers = sum(1 for kind in kinds if kind == "ptr")
        if (
            kinds[:pointers] != ("ptr",) * pointers
            or any(kind != "i32" for kind in kinds[pointers:])
            or grid[1:] != (1, 1)
            or block[1:] != (1, 1)
        ):
            raise ValueError(
                "the artifact runtime launches one-dimensional kernels whose "
                "pointer arguments precede their i32 arguments"
            )
        scalars = []
        for index, value in enumerate(arguments[pointers:], pointers):
            raw = self._ctype_argument("i32", value, index).value
            scalars.append(raw - (1 << 32) if raw >= 1 << 31 else raw)
        self._split_launch(
            function,
            grid[0],
            block[0],
            stream,
            tuple(arguments[:pointers]),
            tuple(scalars),
        )

    def _launch(self, function, grid, block, stream, values):
        parameter_pointers = (ctypes.c_void_p * len(values))(
            *[
                ctypes.cast(ctypes.pointer(value), ctypes.c_void_p)
                for value in values
            ]
        )
        self._call(
            "cuLaunchKernel",
            ctypes.c_void_p(function),
            grid[0],
            grid[1],
            grid[2],
            block[0],
            block[1],
            block[2],
            0,
            ctypes.c_void_p(stream),
            parameter_pointers,
            None,
        )


def _get_driver():
    """Return the process-wide driver, creating it on first use.

    Creating the driver may import the native bindings, so it holds the
    cold-path lock of `_runtime`, which a fork waits for.
    """
    global _driver
    driver = _driver
    if driver is None:
        from . import _runtime

        with _runtime._compile_lock:
            if _driver is None:
                _driver = _CudaDriver()
            driver = _driver
    return driver


def driver_version():
    """Return the actual CUDA driver version, or ``None`` when unavailable."""
    try:
        return _get_driver().driver_version()
    except RuntimeError:
        return None


class _CUDABackend:
    name = "cuda"
    artifact_format = "ptx"
    persistent_cache = True

    @staticmethod
    def compile(
        module,
        kernel_name,
        block_size,
        target,
        lowering_kind,
        lowering_options,
    ):
        return _compile_native(
            module,
            kernel_name,
            block_size,
            target,
            lowering_kind,
            lowering_options,
        )

    @staticmethod
    def lease(artifact, *, capturing=False):
        return _load_artifact(artifact, _get_driver(), capturing=capturing)

    @staticmethod
    def launch(
        lease,
        contract,
        bindings,
        *,
        grid,
        stream,
        capturing,
    ):
        _launch_loaded(
            lease.entry,
            contract,
            bindings,
            grid,
            stream,
            capturing=capturing,
        )

    @staticmethod
    def release(lease):
        lease.release()


CUDA_BACKEND = _CUDABackend()
