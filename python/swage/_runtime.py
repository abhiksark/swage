# python/swage/_runtime.py
"""Backend-neutral specialization, artifact, and launch orchestration.

`launch` validates one fixed elementwise launch, finds or compiles its
artifact through the selected backend adapter (`_backends`), and launches
it through the compiler's launch contract. The compiler identity, the
binding check, the process and disk caches, and the cold-path lock that
interpreter exit and `fork` coordinate with live here too; the private
segmented runner shares them.
"""

import ast
import atexit
import errno
import hashlib
import importlib.util
import json
import logging
import os
import pathlib
import re
import shutil
import stat
import struct
import subprocess
import tempfile
import threading
import time
import warnings
from collections import OrderedDict
from collections.abc import Mapping
from typing import NamedTuple

from . import _abi, _native, language
from ._backends import get_backend
from ._errors import BackendUnavailableError, SwageError

_DIALECT_VERSION = 1
# Entries an in-process kernel cache keeps before it forgets its oldest.
_CACHE_LIMIT = 128


class _BoundedCache(dict):
    """A dict that forgets its oldest entry once it holds more than `limit`.

    Entries leave in the order they were first stored, so a kernel that is
    still in use may leave; its next use compiles or loads it again. Every
    writer holds `_compile_lock`. A reader takes no lock, because a single
    `get` is atomic.
    """

    def __init__(self, limit):
        """Create an empty cache that keeps at most `limit` entries."""
        super().__init__()
        self.limit = limit

    def __setitem__(self, key, value):
        """Store one entry, then drop the oldest entries over the limit."""
        super().__setitem__(key, value)
        while len(self) > self.limit:
            del self[next(iter(self))]


class _ColdPathLock:
    """A reentrant lock that interpreter exit closes to every other thread.

    The thread that runs the exit handlers calls `close`. It waits for the
    holder and then keeps the lock. From that call on, any other thread
    that asks for the lock waits forever instead, so a thread that would
    retake a released lock ahead of the exiting thread starts nothing new.
    The exiting thread itself can still take the lock, because a later exit
    handler may reach a cold path. When the wait of `close` ran out, the
    exiting thread does not wait a second time behind the same holder: its
    next request fails at once.
    """

    def __init__(self):
        """Create an open lock that nothing holds."""
        self._lock = threading.RLock()
        self._closed_by = None
        self._wait_ran_out = False

    def acquire(self, timeout=-1):
        """Take the lock; return whether it was taken within `timeout`."""
        closed_by = self._closed_by
        if closed_by is not None:
            if closed_by != threading.get_ident():
                # Only a daemon thread gets here, and its process is ending.
                threading.Event().wait()
            if self._wait_ran_out:
                timeout = 0
        return self._lock.acquire(timeout=timeout)

    def release(self):
        """Give the lock back once."""
        self._lock.release()

    def close(self, timeout):
        """Close the lock to other threads and wait for it.

        Returns:
            Whether the caller holds the lock. When the wait timed out, the
            holder is still at work and later requests still wait forever.
        """
        self._closed_by = threading.get_ident()
        acquired = self._lock.acquire(timeout=timeout)
        self._wait_ran_out = not acquired
        return acquired

    def __enter__(self):
        """Take the lock for a `with` block.

        Raises:
            RuntimeError: The interpreter exits and the thread that held
                the lock when the exit wait ran out still holds it.
        """
        if not self.acquire():
            raise RuntimeError(
                "Swage cannot compile or load a kernel while the "
                "interpreter exits: another thread still holds the "
                "cold-path lock"
            )
        return self

    def __exit__(self, *_error):
        """Give the lock back at the end of a `with` block."""
        self.release()


# The cold-path lock: every native compile and every first load of a kernel
# in this process holds it, here and in the private segmented runner.
# Compiles are serialized on purpose. The native compiler admits concurrent
# compiles only on separate MLIR contexts, and an LLVM fatal error ends the
# process, so a lock per key would add risk for little gain. The compiler
# releases the GIL while it works, and a warm launch never takes this lock:
# it reads the caches below with one `get` each.
#
# The lock is reentrant for the two handlers registered below. Interpreter
# exit keeps it on the exiting thread, whose later exit handlers may still
# reach a cold path, and a fork takes it on a thread that may hold it.
_compile_lock = _ColdPathLock()
# How long interpreter exit waits for a cold path in flight. A compile takes
# 3 to 20 ms here and a module load about as long, so the wait normally ends
# within one of them. The lock is also held while `unload_retired` waits for
# the device, which nothing bounds, so the wait is cut off: a thread stuck
# there must not hold the process open. Past the bound the exit goes on and
# the risk described at `_wait_for_cold_path_at_exit` returns.
_EXIT_WAIT_SECONDS = 5.0


def _wait_for_cold_path_at_exit():
    """Keep the interpreter from finalizing under a native compile.

    The native compiler releases the GIL, so the interpreter can finalize
    while a daemon thread is inside LLVM, and the process then aborts or
    crashes. This handler closes the lock, which waits for the cold path in
    flight and lets no other thread start a compile or a load afterwards.
    Threads that are not daemons have ended before exit handlers run.

    A native compile that was not started under the lock is not waited for.
    """
    _compile_lock.close(_EXIT_WAIT_SECONDS)


atexit.register(_wait_for_cold_path_at_exit)
if hasattr(os, "register_at_fork"):
    # A fork copies the lock in the state it has. Taken by another thread,
    # it would stay taken in the child, where that thread does not exist,
    # and the child's first cold path would wait for it forever. The fork
    # therefore waits for the cold path in flight and both sides release.
    os.register_at_fork(
        before=_compile_lock.acquire,
        after_in_parent=_compile_lock.release,
        after_in_child=_compile_lock.release,
    )


# The version of the metadata of a persistent cache entry. Version 4 adds
# the backend, the artifact format, the target, and the launch contract to
# the entry, and the cache key covers the backend and format.
_CACHE_VERSION = 4
_DEFAULT_MEMORY_CACHE_ENTRIES = 128
_FIXED_DESCRIPTORS = {
    language.float32: ("ptr<f32>", "ptr<f32>", "ptr<f32>", "i32"),
    language.float16: ("ptr<f16>", "ptr<f16>", "ptr<f16>", "i32"),
    language.float8_e4m3fn: (
        "ptr<f8E4M3FN>",
        "ptr<f8E4M3FN>",
        "ptr<f8E4M3FN>",
        "i32",
    ),
    language.float8_e5m2: ("ptr<f8E5M2>", "ptr<f8E5M2>", "ptr<f8E5M2>", "i32"),
}
_logger = logging.getLogger("swage.runtime")
# The process cache of verified artifacts, an LRU bounded by
# SWAGE_MEMORY_CACHE_ENTRIES. A hit takes no lock: it reads one entry with
# one `get`, and refreshes its recency only when `_cache_lock` is free. A
# miss holds the cold-path lock, and every change holds `_cache_lock`.
_cache_lock = threading.Lock()
_artifact_cache = OrderedDict()
_memory_cache_entries = None
_identity_cache = None


class _Artifact(NamedTuple):
    """A verified backend specialization artifact."""

    key: str
    backend: str
    artifact_format: str
    target: str
    lowered: str
    image: object
    contract_json: str
    contract: _abi.KernelContract
    argument_kinds: tuple
    identity: str


class _LaunchSpec(NamedTuple):
    """Validated values needed after the Python trust boundary."""

    adapter: object
    tensors: tuple
    n: int
    block: int
    grid: tuple
    target: str
    stream: object
    descriptors: tuple


def _contract_error(reason):
    raise RuntimeError(f"invalid kernel contract: {reason}")


def _parse_compiler_contract(contract_json):
    try:
        return _abi.parse_kernel_contract(contract_json)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"invalid kernel contract: {error}") from error


def _validate_contract_specialization(
    contract,
    specialization,
    kernel_name,
    block_size,
    lowering_kind,
    adapter,
):
    """Ensure compiler metadata agrees with the requested specialization."""
    if contract.backend != adapter.name:
        _contract_error("backend does not match the selected adapter")
    if contract.entry != kernel_name:
        _contract_error("entry does not match the requested kernel")
    if adapter.name == "cuda":
        if contract.launch.model != "spmd-grid":
            _contract_error("CUDA contract does not use spmd-grid launch")
        if contract.launch.block != (block_size, 1, 1):
            _contract_error("block does not match the specialization")
    elif adapter.name == "cpu":
        if contract.launch.model != "host-call":
            _contract_error("CPU contract does not use host-call launch")
        if contract.launch.block is not None:
            _contract_error("CPU host-call contract defines a block")
    else:
        _contract_error("adapter backend is unsupported")

    if specialization.get("kernel") != kernel_name:
        _contract_error("kernel does not match the specialization")
    if specialization.get("backend") != adapter.name:
        _contract_error("backend does not match specialization metadata")
    if specialization.get("format") != adapter.artifact_format:
        _contract_error(
            "artifact format does not match specialization metadata"
        )
    codegen = specialization.get("codegen")
    if not isinstance(codegen, dict) or codegen.get("block_size") != block_size:
        _contract_error("block does not match specialization metadata")
    if codegen.get("lowering", "fixed") != lowering_kind:
        _contract_error("lowering does not match specialization metadata")

    if lowering_kind != "fixed":
        return
    descriptors = specialization.get("descriptors")
    if not isinstance(descriptors, list):
        _contract_error("specialization descriptors are missing")
    kinds = []
    for descriptor in descriptors:
        if isinstance(descriptor, str) and descriptor.startswith("ptr<"):
            kinds.append("ptr")
        elif descriptor == "i32":
            kinds.append("i32")
        else:
            _contract_error(
                f"unsupported specialization descriptor {descriptor!r}"
            )
    if tuple(argument.kind for argument in contract.arguments) != tuple(kinds):
        _contract_error("argument kinds do not match the specialization")
    if any(argument.origin != "user" for argument in contract.arguments):
        _contract_error("fixed-runtime arguments must have user origin")
    if tuple(argument.source_index for argument in contract.arguments) != tuple(
        range(len(contract.arguments))
    ):
        _contract_error("fixed-runtime source indexes are inconsistent")


def launch(kernel, *, arguments, constexprs, grid, backend="cuda"):
    """Compile and launch one canonical fixed vector elementwise kernel.

    On the `cuda` backend the kernel is enqueued on the current stream and
    the call returns without waiting. On the `cpu` backend it runs to
    completion first. Either way the output's version counter advances
    once the kernel is enqueued or has run.

    A `TypeError` or `ValueError` raised on the way names the kernel and
    where it is defined, after the message of the check that failed.
    """
    try:
        adapter = get_backend(backend)
        torch = _import_torch(adapter.name)
        spec = _validate_launch(
            kernel,
            arguments,
            constexprs,
            grid,
            torch,
            adapter,
        )
        if spec.n == 0:
            return None

        memo = kernel.__dict__.setdefault("_specialization_memo", {})
        identity = _cached_identity()
        memo_key = (
            adapter.name,
            adapter.artifact_format,
            spec.block,
            spec.target,
            spec.descriptors,
        )
        entry = memo.get(memo_key)
        if entry is None or entry[2] is not identity:
            specialization = _specialization_data(
                kernel,
                descriptors=spec.descriptors,
                constexprs=constexprs,
                target=spec.target,
                adapter=adapter,
            )
            entry = (specialization, _cache_key(specialization), identity)
            memo[memo_key] = entry
        specialization, key, _ = entry
        artifact = _compile_cached(
            adapter,
            specialization,
            kernel.__name__,
            spec.block,
            lambda: kernel.emit_mlir(
                arguments=arguments, constexprs=constexprs
            ),
            key=key,
            lowering_kind="fixed",
        )
        launch_arguments = _bind_fixed_artifact(artifact, spec.tensors, spec.n)
        if adapter.name == "cuda":
            from . import _cuda_backend

            _cuda_backend.ensure_context(torch, spec.tensors[0].device.index)
            physical_grid = (spec.grid[0], 1, 1)
            stream_handle = spec.stream.cuda_stream
            capturing = _is_current_stream_capturing(torch)
        else:
            physical_grid = None
            stream_handle = None
            capturing = False
        lease = adapter.lease(artifact, capturing=capturing)
        try:
            adapter.launch(
                lease,
                artifact.contract,
                launch_arguments,
                grid=physical_grid,
                stream=stream_handle,
                capturing=capturing,
            )
        finally:
            adapter.release(lease)
            if adapter.name == "cuda":
                for tensor in spec.tensors:
                    tensor.record_stream(spec.stream)
        _advance_version(torch, spec.tensors[2])
        if adapter.name == "cuda" and not capturing:
            _cuda_backend._prepare_fixed_launch(
                kernel,
                artifact,
                lease,
                spec.stream,
                torch,
                spec.tensors[0].dtype,
            )
        if _logger.isEnabledFor(logging.DEBUG):
            _logger.debug(
                "%s backend=%s target=%s kernel=%s grid=%s key=%s",
                "launch complete"
                if adapter.name == "cpu"
                else "launch enqueued",
                adapter.name,
                spec.target,
                kernel.__name__,
                spec.grid,
                key[:12],
            )
        return None
    except (TypeError, ValueError) as error:
        # A subclass may take more than a message, so it is left as it is.
        if type(error) not in (TypeError, ValueError):
            raise
        raise type(error)(f"{error}{_launch_location(kernel)}").with_traceback(
            error.__traceback__
        ) from None


def _advance_version(torch, tensor):
    """Tell PyTorch that a launch wrote `tensor` through its raw pointer.

    Autograd saves tensors for a backward pass together with their version
    counters and raises when a counter moved in between. PyTorch advances
    the counter for its own in-place operations and cannot see a kernel
    store, so a launch advances the counter of its output.

    This changes host metadata only: it enqueues nothing, never waits for
    the device, and is safe while a stream captures a CUDA graph. A
    replayed graph runs no host code, so a replay does not advance the
    counter. An inference tensor has no counter and is skipped by PyTorch.
    """
    torch.autograd.graph.increment_version(tensor)


def _launch_location(kernel):
    """Return the suffix that ties a launch error to its kernel."""
    line = kernel.source_line + kernel.function.lineno - 1
    return (
        f" (in launch of kernel '{kernel.__name__}', defined at "
        f"{kernel.filename}:{line})"
    )


# The oldest PyTorch release a launch accepts, as (major, minor). It equals
# the `pytorch` extra in pyproject.toml; no newer release is refused.
_MIN_TORCH = (2, 6)
# The module that last passed `_require_supported_torch`. Tests inject fresh
# fake modules, so the verdict belongs to the module object.
_supported_torch = None


def _import_torch(backend="cuda"):
    """Return PyTorch, or raise when it is missing or unsupported.

    Raises:
        BackendUnavailableError: PyTorch cannot be imported, or its
            release or one of the methods a launch needs is missing. An
            error raised while PyTorch initializes is not unavailability
            and propagates unchanged.
    """
    global _supported_torch
    try:
        import torch
    except (ImportError, OSError) as error:
        raise BackendUnavailableError(
            "Swage launch requires PyTorch",
            code="pytorch-unavailable",
            backend=backend,
            remediation="install 'swage-compiler[pytorch]'",
        ) from error
    if torch is not _supported_torch:
        _require_supported_torch(torch, backend)
        _supported_torch = torch
    return torch


def _require_supported_torch(torch, backend):
    """Reject a PyTorch that a launch cannot use, before any work starts.

    A launch enqueues the kernel and then retains each tensor on the stream
    through `Tensor.record_stream`. Finding that method missing after the
    enqueue would leave a kernel running on storage PyTorch may reuse, so
    the release and the method are both checked here. So is
    `torch.autograd.graph.increment_version`, which marks the output as
    written after the enqueue.
    """
    found = getattr(torch, "__version__", None)
    release = re.match(r"(\d+)\.(\d+)", str(found))
    required = ".".join(str(part) for part in _MIN_TORCH)
    if release is None or (int(release[1]), int(release[2])) < _MIN_TORCH:
        raise BackendUnavailableError(
            f"Swage launch requires PyTorch {required} or newer; "
            f"found PyTorch {found}",
            code="pytorch-unsupported",
            backend=backend,
            remediation=f"install PyTorch {required} or newer",
        )
    if not callable(getattr(torch.Tensor, "record_stream", None)):
        raise BackendUnavailableError(
            "Swage launch requires torch.Tensor.record_stream to retain "
            f"submitted tensors; found PyTorch {found} without it",
            code="pytorch-unsupported",
            backend=backend,
            remediation=f"install PyTorch {required} or newer",
        )
    graph = getattr(getattr(torch, "autograd", None), "graph", None)
    if not callable(getattr(graph, "increment_version", None)):
        raise BackendUnavailableError(
            "Swage launch requires torch.autograd.graph.increment_version "
            "to mark the output as written; found PyTorch "
            f"{found} without it",
            code="pytorch-unsupported",
            backend=backend,
            remediation=f"install PyTorch {required} or newer",
        )


def _validate_launch_call(kernel, arguments, constexprs, grid):
    """Validate launch mappings and the canonical ordered parameter shape."""
    if not isinstance(arguments, Mapping):
        raise TypeError("arguments must be a mapping")
    if not isinstance(constexprs, Mapping):
        raise TypeError("constexprs must be a mapping")
    if (
        not isinstance(grid, tuple)
        or len(grid) != 1
        or type(grid[0]) is not int
    ):
        raise TypeError("grid must be a one-element tuple of integers")

    # The frontend names a parameter form outside the language precisely,
    # with its source location, so it is asked before the shape check.
    kernel._require_plain_parameters()
    parameter_names = [argument.arg for argument in kernel.function.args.args]
    if len(parameter_names) != 5:
        raise ValueError(
            "launch requires four runtime parameters followed by one "
            "constexpr block parameter"
        )
    runtime_names = parameter_names[:4]
    block_name = parameter_names[4]
    if kernel.constexpr_names != {block_name}:
        raise ValueError(
            "launch requires its final parameter to be the constexpr block"
        )
    if set(arguments) != set(runtime_names):
        expected = ", ".join(runtime_names)
        raise ValueError(f"arguments must contain exactly {expected}")
    if set(constexprs) != {block_name}:
        raise ValueError(f"constexprs must contain exactly {block_name}")
    block = constexprs[block_name]
    if type(block) is not int or block <= 0:
        raise ValueError(f"constexpr {block_name} must be a positive integer")
    return runtime_names, block


def _validate_launch_tensor(name, tensor, torch, backend):
    """Validate one tensor before its raw pointer crosses the ABI.

    Returns:
        The `swage.language` element type of the tensor.
    """
    if not isinstance(tensor, torch.Tensor) or tensor.device.type != backend:
        raise TypeError(f"argument '{name}' must be a {backend.upper()} tensor")
    element_type = language._torch_float_type(tensor.dtype, torch)
    if element_type is None:
        raise TypeError(
            f"argument '{name}' must have dtype torch.float32, torch.float16, "
            "torch.float8_e4m3fn, or torch.float8_e5m2"
        )
    if tensor.dim() != 1:
        raise TypeError(f"argument '{name}' must have rank one")
    if not tensor.is_contiguous():
        raise ValueError(f"argument '{name}' must be contiguous")
    # A lazy view shares the storage of its base but shows other values.
    # The kernel reads and writes the storage, so it would see the base.
    if tensor.is_neg():
        raise ValueError(
            f"argument '{name}' must not be a lazy negation view; pass "
            "tensor.resolve_neg()"
        )
    if tensor.is_conj():
        raise ValueError(
            f"argument '{name}' must not be a lazy conjugate view; pass "
            "tensor.resolve_conj()"
        )
    # The kernel reads and writes storage through raw pointers and records
    # no gradient, so the result would silently be cut from the graph.
    if tensor.requires_grad:
        raise ValueError(
            f"argument '{name}' must not require grad; a launch records no "
            "gradient, so pass tensor.detach()"
        )
    return element_type


def _validate_runtime_arguments(arguments, runtime_names, torch, backend):
    """Validate tensor metadata, scalar bounds, and buffer lengths."""
    tensors = tuple(arguments[name] for name in runtime_names[:3])
    element_type = None
    for name, tensor in zip(runtime_names, tensors):
        tensor_type = _validate_launch_tensor(name, tensor, torch, backend)
        if element_type is None:
            element_type = tensor_type
        elif tensor_type is not element_type:
            raise TypeError(
                f"argument '{name}' must have the same dtype as "
                f"argument '{runtime_names[0]}'"
            )

    count_name = runtime_names[3]
    n = arguments[count_name]
    if type(n) is not int or not 0 <= n < (1 << 31):
        raise ValueError(f"{count_name} must be a nonnegative i32")
    for name, tensor in zip(runtime_names, tensors):
        if n > tensor.numel():
            raise ValueError(
                f"{count_name} exceeds tensor length for argument '{name}'"
            )
    return tensors, n, element_type


def _validate_output_alias(names, tensors, n):
    """Reject an output that partially overlaps an input in the active range.

    Every lane reads its inputs and writes its output at one element index,
    so an output that is exactly an input, or that shares no byte of the
    first `n` elements with it, is safe. Any other overlap would let one
    lane store over an element another lane still has to read. Every tensor
    is known contiguous and rank one by this point, so the byte distance of
    the two starts decides. It cannot see two virtual mappings of one
    physical allocation, nor aliasing created after this returns.

    Args:
        names: Argument names, the inputs first and the output last.
        tensors: The tensors in the same order.
        n: The number of elements the launch covers.
    """
    if not n:
        return
    *inputs, output = tensors
    active_bytes = n * output.element_size()
    output_start = output.data_ptr()
    for name, tensor in zip(names, inputs):
        if 0 < abs(tensor.data_ptr() - output_start) < active_bytes:
            raise ValueError(
                f"argument '{names[-1]}' must not partially overlap argument "
                f"'{name}' in their active ranges"
            )


def _launch_descriptors(kernel, element_type):
    """Return descriptors for the validated public fixed semantic shape."""
    kernel._require_plain_parameters()
    return _FIXED_DESCRIPTORS[element_type]


def _validate_launch(kernel, arguments, constexprs, grid, torch, adapter):
    runtime_names, block = _validate_launch_call(
        kernel, arguments, constexprs, grid
    )
    tensors, n, element_type = _validate_runtime_arguments(
        arguments,
        runtime_names,
        torch,
        adapter.name,
    )
    expected_grid = ((n + block - 1) // block,)
    if grid != expected_grid:
        raise ValueError(f"grid must equal {expected_grid} for n and BLOCK")
    descriptors = _launch_descriptors(kernel, element_type)
    if adapter.name == "cuda":
        from . import _cuda_backend

        current_device = _cuda_backend.validate_device(
            tensors,
            runtime_names,
            torch,
        )
        target, stream = _cuda_backend.validate_geometry(
            block,
            n,
            grid,
            torch,
            current_device,
        )
    else:
        if block > 1024:
            raise ValueError("BLOCK must be at most 1024 for the CPU backend")
        target = "native"
        stream = None
    _validate_output_alias(runtime_names[:3], tensors, n)
    return _LaunchSpec(
        adapter,
        tensors,
        n,
        block,
        grid,
        target,
        stream,
        descriptors,
    )


def _is_current_stream_capturing(torch):
    from ._cuda_backend import is_current_stream_capturing

    return is_current_stream_capturing(torch)


def _specialization_data(
    kernel,
    *,
    descriptors,
    constexprs,
    target,
    adapter,
):
    """Return the cache identity of one fixed launch specialization."""
    identity = _cached_identity()
    source_digest = getattr(kernel, "source_digest", None)
    if source_digest is None:
        normalized_source = ast.dump(kernel.function, include_attributes=False)
        source_digest = hashlib.sha256(normalized_source.encode()).hexdigest()
    block = next(iter(constexprs.values()))
    return {
        "source": source_digest,
        "kernel": kernel.__name__,
        "backend": adapter.name,
        "format": adapter.artifact_format,
        "target": target,
        "descriptors": list(descriptors),
        "constexprs": [[key, constexprs[key]] for key in sorted(constexprs)],
        "codegen": {
            "lowering": "fixed",
            "block_size": block,
            "options": [],
            "index_bits": 64,
        },
        "frontend": identity["frontend"],
        # The native identity is derived from the contents of the compiler
        # libraries, so it also covers the LLVM they link. The LLVM pin is
        # not a field: it is read from a checkout, so the same libraries
        # installed from a wheel would have another key.
        "native": identity["native"],
        "dialect_version": _DIALECT_VERSION,
    }


def _package_dir():
    """Return the directory of the loaded `swage` package."""
    return pathlib.Path(__file__).resolve().parent


def _frontend_sources(package):
    """Return `(relative name, path)` for each frontend source, sorted.

    Only regular files that Python can import count: a dangling symlink such
    as an editor lock file, a directory named like a module, and anything
    under a name that starts with a dot are skipped.
    """
    sources = []
    for path in package.rglob("*.py"):
        relative = path.relative_to(package)
        hidden = any(part.startswith(".") for part in relative.parts)
        if not hidden and path.is_file():
            sources.append((relative.as_posix(), path))
    return sorted(sources)


def _changed_ns(path):
    """Return when `path` last changed, in nanoseconds since the epoch.

    Both the content time and the inode time count, for the file and for a
    symlink leading to it, so a rewritten, replaced, or retargeted file is
    seen whatever its modification time was set to.
    """
    return max(
        changed
        for details in (path.stat(), path.lstat())
        for changed in (details.st_mtime_ns, details.st_ctime_ns)
    )


def _too_new(path):
    """Describe a file that cannot be told apart from the loaded code."""
    return (
        f"{path} is not older than this process, so it may differ from the "
        "code that was loaded"
    )


def _hash_frontend(package, started=None):
    """Hash every Python source file of the package in sorted name order.

    Args:
        package: Directory of the `swage` package.
        started: The process start time in nanoseconds, or None to hash the
            files whatever their age. A source changed at or after `started`
            may differ from the code the process already loaded.

    Returns:
        `(digest, None)`, where the SHA-256 hex digest covers one line per
        file, `<relative name> <SHA-256 of its bytes>` and a newline, or
        `(None, problem)` when a file cannot be read, is too new, or the
        package holds no source. The build computes the same digest of the
        sources it was built with and records it as `frontend_digest` in
        the `_build_info.json` of the bindings; see
        cmake/SwageBuildIdentity.cmake.
    """
    try:
        sources = _frontend_sources(package)
        digest = hashlib.sha256()
        for name, path in sources:
            contents = path.read_bytes()
            if started is not None and _changed_ns(path) >= started:
                return None, _too_new(path)
            digest.update(
                f"{name} {hashlib.sha256(contents).hexdigest()}\n".encode()
            )
    except OSError as error:
        return None, f"cannot read the frontend sources: {error}"
    if not sources:
        return None, f"{package} holds no Python source files"
    return digest.hexdigest(), None


def _frontend_digest(package):
    """Return the frontend digest of the package, or None without one."""
    return _hash_frontend(package)[0]


_NATIVE_EXTENSION = "_swageDialectsNanobind"
_NATIVE_LIBRARY_PATTERNS = (
    f"{_NATIVE_EXTENSION}*.so",
    "libSwagePythonCAPI.so*",
)


def _native_libraries():
    """Return the paths of the native compiler libraries.

    Importing the bindings loads all of LLVM, so the nanobind extension and
    the C API library in `mlir_swage/_mlir_libs` are located through the
    import system instead. The list is empty when the bindings are not
    importable.
    """
    try:
        spec = importlib.util.find_spec("mlir_swage._mlir_libs")
    except (ImportError, ValueError):
        return []
    if spec is None:
        return []
    return [
        path
        for location in spec.submodule_search_locations or ()
        for pattern in _NATIVE_LIBRARY_PATTERNS
        for path in pathlib.Path(location).glob(pattern)
    ]


_ELF_HEADER = struct.Struct("<16sHHIQQQIHHHHHH")
_ELF_PROGRAM_HEADER = struct.Struct("<IIQQQQQQ")
_ELF_NOTE_HEADER = struct.Struct("<III")
_ELF_NOTE_SEGMENT = 4
_ELF_BUILD_ID_NOTE = 3
# No note segment of a real library comes near this size, so a larger one
# is a damaged file and is not read.
_ELF_NOTES_LIMIT = 1 << 20


def _elf_build_id(path):
    """Return the GNU build id of an ELF file in hex, or None without one.

    The linker derives the id from the linked contents and records it in a
    note. No later tool changes it, so a library that was stripped, copied,
    or given another modification time keeps its id. Only 64-bit
    little-endian files are read, which covers every platform the bindings
    are built for. A file of any other kind, a file without the note, and a
    damaged file have no id.

    Raises:
        OSError: The file cannot be read.
    """
    with open(path, "rb") as library:
        header = library.read(_ELF_HEADER.size)
        if len(header) < _ELF_HEADER.size:
            return None
        fields = _ELF_HEADER.unpack(header)
        magic, table, entry_size, entries = (
            fields[0],
            fields[5],
            fields[9],
            fields[10],
        )
        if magic[:6] != b"\x7fELF\x02\x01":
            return None
        if entry_size < _ELF_PROGRAM_HEADER.size:
            return None
        for index in range(entries):
            library.seek(table + index * entry_size)
            entry = library.read(_ELF_PROGRAM_HEADER.size)
            if len(entry) < _ELF_PROGRAM_HEADER.size:
                return None
            kind, _, offset, _, _, size, _, alignment = (
                _ELF_PROGRAM_HEADER.unpack(entry)
            )
            if kind != _ELF_NOTE_SEGMENT or size > _ELF_NOTES_LIMIT:
                continue
            library.seek(offset)
            build_id = _build_id_note(library.read(size), alignment)
            if build_id:
                return build_id.hex()
    return None


def _build_id_note(notes, alignment):
    """Return the descriptor of the GNU build id note in a note segment."""
    padding = 8 if alignment == 8 else 4
    position = 0
    while position + _ELF_NOTE_HEADER.size <= len(notes):
        name_size, descriptor_size, kind = _ELF_NOTE_HEADER.unpack_from(
            notes, position
        )
        name_start = position + _ELF_NOTE_HEADER.size
        descriptor_start = name_start + -(-name_size // padding) * padding
        descriptor_end = descriptor_start + descriptor_size
        if descriptor_end > len(notes):
            return None
        name = notes[name_start : name_start + name_size]
        if name == b"GNU\0" and kind == _ELF_BUILD_ID_NOTE:
            return notes[descriptor_start:descriptor_end]
        position = descriptor_start + -(-descriptor_size // padding) * padding
    return None


def _file_digest(path):
    """Return the SHA-256 hex digest of the contents of `path`."""
    digest = hashlib.sha256()
    with open(path, "rb") as library:
        for block in iter(lambda: library.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# The content identity of each native library this process has read, keyed
# by the path and the file facts that change when the file does. A library
# without a build id is hashed in full, tens of megabytes, so it is hashed
# once per process and again only after the file changes.
#
# Only a file that has not changed since the process started is remembered.
# A file that changed later can change again within one tick of the file
# system clock and keep every recorded fact, so it is read on each call.
_content_identities = {}


def _content_identity(path):
    """Identify a native library by its contents.

    Returns:
        `build-id:<hex>` for an ELF file that carries a GNU build id,
        otherwise `sha256:<hex>` over the whole file.

    Raises:
        OSError: The file cannot be read.
    """
    details = path.stat()
    signature = (
        str(path),
        details.st_ino,
        details.st_size,
        details.st_mtime_ns,
        details.st_ctime_ns,
    )
    identity = _content_identities.get(signature)
    if identity is not None:
        return identity
    build_id = _elf_build_id(path)
    if build_id is not None:
        identity = f"build-id:{build_id}"
    else:
        identity = f"sha256:{_file_digest(path)}"
    started = _PROCESS_START_NS
    if started is not None and _changed_ns(path) < started:
        _content_identities[signature] = identity
    return identity


def _native_identity():
    """Describe the native compiler libraries by their contents.

    Neither the modification time nor the size is part of the identity, so
    the same bytes give the same identity after a copy, an archive round
    trip, or an install that normalizes file times, and other bytes give
    another identity whatever their size and times are.

    Returns:
        A sorted list of `[file name, content identity]`, one per file of
        the nanobind extension and the C API library, or None when the
        bindings are not importable. A symlink counts as the file it leads
        to, under that file's name.
    """
    targets = set()
    for path in _native_libraries():
        try:
            targets.add(path.resolve(strict=True))
        except OSError:
            continue
    libraries = []
    for target in targets:
        try:
            libraries.append([target.name, _content_identity(target)])
        except OSError:
            continue
    if not any(name.startswith(_NATIVE_EXTENSION) for name, _ in libraries):
        return None
    return sorted(libraries)


def _process_start_ns():
    """Return when this process started, in nanoseconds since the epoch.

    The kernel reports the start in clock ticks since boot, so the result
    is at most one tick early and never late. None when the kernel does not
    report it, which is the case off Linux.

    This runs while `swage` is imported, so it never raises and never
    warns; a missing start time is reported at the first use of the cache.
    """
    try:
        with open("/proc/self/stat", encoding="ascii") as status:
            # The command name may hold spaces; fields resume after it.
            fields = status.read().rpartition(")")[2].split()
        ticks_per_second = os.sysconf("SC_CLK_TCK")
        if ticks_per_second <= 0:
            return None
        since_boot = int(fields[19]) * 1_000_000_000 // ticks_per_second
        age = time.clock_gettime_ns(time.CLOCK_BOOTTIME) - since_boot
        return time.time_ns() - age
    except Exception:
        return None


# Taken at import, and `swage/__init__.py` imports this module, so the time
# belongs to the process that loaded the package. A forked child inherits
# it instead of reading its own later start, the time of the fork.
_PROCESS_START_NS = _process_start_ns()


def _stale_identity(identity):
    """Return why `identity` may not describe the code this process runs.

    The identity is read from files on disk, but a process runs the code it
    loaded, and nothing records which bytes that was. The two are known to
    agree only when no frontend source and no native library has changed
    since the process started. A file as new as the process may have been
    loaded before or after it changed, so it is not trusted.

    Returns:
        None when every identified file is older than the process and still
        matches `identity`, otherwise a description of the first mismatch.
    """
    started = _PROCESS_START_NS
    if started is None:
        return (
            "the process start time is unavailable, so the loaded code "
            "cannot be compared with the files on disk"
        )
    frontend, problem = _hash_frontend(_package_dir(), started)
    if problem is not None:
        return problem
    if frontend != identity["frontend"]:
        return "the frontend sources changed after the cache key was derived"
    for path in _native_libraries():
        try:
            changed = _changed_ns(path)
        except OSError:
            continue
        if changed >= started:
            return _too_new(path)
    if _native_identity() != identity["native"]:
        return "the native libraries changed after the cache key was derived"
    return None


def _git_identity(root):
    """Return the HEAD revision of a checkout and whether it is clean.

    Args:
        root: Directory expected to hold `.git`.

    Returns:
        `(revision, clean)`, or `(None, False)` when `root` is not a git
        checkout or git cannot describe it.
    """
    if not (root / ".git").exists():
        return None, False
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None, False
    return revision, not dirty


def _is_swage_checkout(root, package):
    """Return whether `root` is a Swage source tree that owns `package`.

    The package two directories below a git root is Swage's own copy only
    when it is `python/swage` and the root also holds the LLVM pin. A copy
    vendored into another repository, for example at `src/swage`, sits
    under a root whose HEAD is a commit of that repository, not of Swage.
    """
    return (
        package == root / "python" / "swage"
        and (root / "cmake" / "llvm-version.txt").is_file()
    )


def _compiler_identity():
    """Identify the compiler files this process finds on disk.

    `frontend` and `native` identify the code that produces PTX and are the
    compiler fields of the cache key. They describe the files as they are
    now; `_stale_identity` decides whether that is the code this process
    loaded. `revision` and `clean` describe the sources, for diagnostics;
    they do not gate the cache. A package in a Swage git checkout takes
    them from the checkout. Any other package takes them, with `llvm`, from
    the `_build_info.json` the native build packaged, and an invalid record
    gives no revision instead of a guess. A package inside another
    repository with no build record has none.

    Returns:
        A dict with the keys `revision`, `clean`, `llvm`, `frontend`, and
        `native`. Each value is None (False for `clean`) when unavailable.
    """
    package = _package_dir()
    root = package.parents[1]
    pin = root / "cmake" / "llvm-version.txt"
    revision, clean = None, False
    llvm = pin.read_text().strip() if pin.is_file() else None
    if _is_swage_checkout(root, package):
        revision, clean = _git_identity(root)
    else:
        try:
            info = _native.build_info()
        except ValueError:
            info = None
        if info is not None:
            revision = info["source_revision"]
            clean = info["source_clean"]
            llvm = info["llvm_version"]
    return {
        "revision": revision,
        "clean": clean,
        "llvm": llvm,
        "frontend": _frontend_digest(package),
        "native": _native_identity(),
    }


def _cached_identity():
    """Return `_compiler_identity()` computed once per process.

    The identity spawns two git subprocesses and hashes the frontend, which
    would dominate launch cost if derived on every call. The cache is keyed
    on the identity function itself so a monkeypatched `_compiler_identity`
    is always honored.
    """
    global _identity_cache
    if _identity_cache is None or _identity_cache[0] is not _compiler_identity:
        _identity_cache = (_compiler_identity, _compiler_identity())
    return _identity_cache[1]


class _BindingsMismatch(SwageError):
    """The `mlir_swage` bindings were not built for the loaded `swage`."""


# The paths of a checkout that the native bindings are built from.
_NATIVE_SOURCES = (
    "CMakeLists.txt",
    "cmake",
    "include",
    "lib",
    "python/CMakeLists.txt",
    "python/SwageExtensionNanobind.cpp",
    "python/mlir_swage",
)
# The bindings module that passed `_verify_bindings`, so the check and its
# warning run once per process.
_verified_bindings = None


def _stale_native_sources(built_from):
    """Return why bindings built from `built_from` may not fit a checkout.

    Only a Swage git checkout has native sources to compare with. Its
    frontend is edited and committed without a native rebuild, and several
    working trees may share one build, so another revision alone is not a
    problem there: the native sources decide.

    Args:
        built_from: The source revision the bindings recorded.

    Returns:
        A description when the native sources of the checkout differ from
        that revision or cannot be compared with it. None when they are the
        same, when the bindings recorded no clean revision to compare with,
        and when `swage` does not run from a checkout.
    """
    identity = _cached_identity()
    revision = identity["revision"]
    if (
        revision is None
        or built_from == "unknown"
        or built_from.endswith("-dirty")
        or (revision == built_from and identity["clean"])
    ):
        return None
    root = _package_dir().parents[1]
    try:
        compared = subprocess.run(
            ["git", "diff", "--quiet", built_from, "--", *_NATIVE_SOURCES],
            cwd=root,
            check=False,
            capture_output=True,
        ).returncode
    except OSError:
        return None
    if compared == 0:
        return None
    built = f"the mlir_swage bindings were built from revision {built_from}"
    if compared == 1:
        return (
            f"{built}, and the native sources of the checkout at {root} "
            "differ from that revision; rebuild the bindings"
        )
    return (
        f"{built}, which the checkout at {root} does not have, so its "
        "native sources cannot be compared with the bindings"
    )


def _verify_bindings(native):
    """Check once that `native` was built for the `swage` that is loaded.

    The extension calls this while it is imported, when `swage` is already
    loaded. `_native_bindings` calls it for bindings that were imported
    before `swage` and for bindings from before the extension made the call.

    Bindings built for another `swage` version, and bindings that record no
    version, are refused. Another source revision is not refused: a wheel
    of the pure Python package records no revision to compare, and a
    checkout moves to a new revision with every commit. In a checkout the
    native sources are compared instead, and a difference warns once.

    Args:
        native: The `swage` submodule of the nanobind extension.

    Raises:
        _BindingsMismatch: The bindings were built for another version of
            `swage`, or carry no build identity.
    """
    global _verified_bindings
    if native is _verified_bindings:
        return
    from . import __version__

    libraries = _native_libraries()
    location = libraries[0].parent if libraries else "that are loaded"
    built_for = getattr(native, "__version__", None)
    if built_for is None:
        raise _BindingsMismatch(
            f"the mlir_swage bindings in {location} record no swage "
            f"version, so they cannot be checked against swage "
            f"{__version__} from {_package_dir()}; they were built before "
            "the bindings identified themselves. Rebuild them from the "
            "sources of this swage"
        )
    built_from = getattr(native, "__source_revision__", "unknown")
    if built_for != __version__:
        raise _BindingsMismatch(
            f"the mlir_swage bindings in {location} were built for swage "
            f"{built_for} (source revision {built_from}), but swage "
            f"{__version__} is loaded from {_package_dir()}. Install the "
            "bindings built for this version, or rebuild them from the "
            "sources of this swage"
        )
    problem = _stale_native_sources(built_from)
    if problem is not None:
        warnings.warn(problem, RuntimeWarning, stacklevel=2)
    _verified_bindings = native


def _native_bindings():
    """Import the native `swage` bindings and check them against `swage`.

    Returns:
        The `swage` submodule of the nanobind extension.

    Raises:
        ImportError: The bindings are not importable.
        _BindingsMismatch: The bindings were built for another version of
            `swage`, or carry no build identity.
    """
    try:
        from mlir_swage._mlir_libs._swageDialectsNanobind import (
            swage as native,
        )
    except ImportError as error:
        # The extension reports a failed check as an import error whose
        # cause is the error `_verify_bindings` raised.
        if isinstance(error.__cause__, _BindingsMismatch):
            raise error.__cause__ from None
        raise
    _verify_bindings(native)
    return native


def _cache_key(specialization):
    encoded = json.dumps(
        specialization, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


# The uses of the disk cache this process has given up, each with the reason
# it warned about once: "identity" and "read" end reading and publishing,
# "write" ends publishing.
_cache_off = {}
# The most entries the cache root holds after a publish, unless
# SWAGE_CACHE_MAX_ENTRIES says otherwise.
_DEFAULT_MAX_ENTRIES = 1024


def _switch_on(name):
    """Return whether the environment switch `name` is set to `1`.

    Unset, empty, and `0` are off. Any other value raises: a mistyped switch
    that stayed off would let the process write or compile after its
    operator asked it not to.
    """
    value = os.environ.get(name, "")
    if value in ("", "0"):
        return False
    if value == "1":
        return True
    raise ValueError(f"{name} must be 0 or 1; found {value!r}")


def _max_entries():
    """Return the most entries the cache root may hold after a publish."""
    value = os.environ.get("SWAGE_CACHE_MAX_ENTRIES", "")
    if not value:
        return _DEFAULT_MAX_ENTRIES
    if not (value.isascii() and value.isdigit()) or int(value) == 0:
        raise ValueError(
            "SWAGE_CACHE_MAX_ENTRIES must be a positive integer; "
            f"found {value!r}"
        )
    return int(value)


def _cache_settings():
    """Read the three cache variables, raising on a mistyped value.

    Returns:
        `(read_only, no_compile, max_entries)`.
    """
    return (
        _switch_on("SWAGE_CACHE_READ_ONLY"),
        _switch_on("SWAGE_NO_COMPILE"),
        _max_entries(),
    )


def _cache_writable():
    """Return whether this process may still change the cache root."""
    return "write" not in _cache_off and not _switch_on("SWAGE_CACHE_READ_ONLY")


def _warn_cache_off(use, reason):
    """Give up one use of the disk cache, warning the first time."""
    if use in _cache_off:
        return
    _cache_off[use] = reason
    if use == "write":
        effect = "is not written by this process"
        reuse = (
            "New kernels are reused in this process only; published "
            "entries are still read."
        )
    else:
        effect = "is off for this process"
        reuse = "Compiled kernels are reused in this process only."
    warnings.warn(
        f"Swage persistent cache {effect}: {reason}. {reuse}",
        RuntimeWarning,
        stacklevel=3,
    )


def _cache_problem(identity):
    """Return why `identity` rules out disk entries, or None when it does not.

    Reading and publishing need a native compiler, an identified frontend,
    and an identity known to describe the loaded code. Nothing is warned
    about and nothing is recorded here.
    """
    if identity["native"] is None:
        return "the native compiler libraries are not found"
    if identity["frontend"] is None:
        return (
            _hash_frontend(_package_dir())[1]
            or "the frontend sources are not identified"
        )
    return _stale_identity(identity)


def _cache_off_reason(identity):
    """Return why this process reads no disk entry, or None when it does."""
    return (
        _cache_off.get("identity")
        or _cache_off.get("read")
        or _cache_problem(identity)
    )


def _cache_usable(identity):
    """Decide whether this process may read and publish disk entries.

    A process that can compile but whose identity rules the cache out is
    warned once and keeps to process-local reuse.
    """
    if identity["native"] is None or _cache_off.keys() & {"identity", "read"}:
        return False
    problem = _cache_problem(identity)
    if problem is None:
        return True
    _warn_cache_off("identity", problem)
    return False


class _CacheStatus(NamedTuple):
    """How this process would use the disk cache right now."""

    directory: pathlib.Path
    # Why no disk entry is read, or None when lookups read the cache.
    problem: str | None
    # Whether `problem` makes every lookup raise instead of compiling.
    rejected: bool
    writes: bool
    entries: int
    max_entries: int
    compiles: bool


def _cache_status():
    """Describe how this process would use the disk cache, changing nothing.

    Unlike a lookup, this does not warn, does not record a cache it finds
    unusable, and does not create or clean the cache root. A mistyped cache
    variable raises `ValueError`, as it does for a launch.
    """
    read_only, no_compile, max_entries = _cache_settings()
    root = _cache_dir()
    problem = _cache_off_reason(_cached_identity())
    rejected = False
    entries = 0
    if problem is None:
        try:
            _check_safe(root)
            with os.scandir(root) as listing:
                entries = sum(
                    1
                    for item in listing
                    if _ENTRY_NAME.fullmatch(item.name)
                    and item.is_dir(follow_symlinks=False)
                )
        except FileNotFoundError:
            pass  # The first publish creates the root.
        except OSError as error:
            problem = f"cannot use {root}: {error}"
        except RuntimeError as error:
            problem, rejected = str(error), True
    return _CacheStatus(
        directory=root,
        problem=problem,
        rejected=rejected,
        writes=problem is None and not read_only and "write" not in _cache_off,
        entries=entries,
        max_entries=max_entries,
        compiles=not no_compile,
    )


def _refuse_compile(kernel_name, key, looked_up):
    """Raise for a kernel that is not cached while compiling is switched off.

    Args:
        kernel_name: Name of the kernel the caller asked for.
        key: Cache key of the specialization.
        looked_up: Whether the disk cache was read and held no entry. When
            it was not read, the error says why.
    """
    if looked_up:
        reason = f"no entry {key} in {_cache_dir()}"
    else:
        reason = (
            "the persistent cache is off for this process: "
            f"{_cache_off_reason(_cached_identity())}"
        )
    raise _compile_refusal(kernel_name, reason)


def _compile_refusal(kernel_name, reason):
    """Return the error for a kernel that SWAGE_NO_COMPILE=1 keeps uncompiled.

    The public launch and the private segmented runner both raise it, so
    the two refusals read alike and differ only in `reason`.
    """
    return SwageError(
        f"SWAGE_NO_COMPILE=1 refuses to compile kernel '{kernel_name}': "
        f"{reason}"
    )


def _memory_cache_limit():
    """Lazily parse the strict positive process-cache capacity."""
    global _memory_cache_entries
    limit = _memory_cache_entries
    if limit is not None:
        return limit
    with _cache_lock:
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


def _artifact_identity(key, backend, artifact_format, target, contract_json):
    material = json.dumps(
        {
            "key": key,
            "backend": backend,
            "format": artifact_format,
            "target": target,
            "contract": hashlib.sha256(contract_json.encode()).hexdigest(),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(material.encode()).hexdigest()


def _make_artifact(
    key,
    adapter,
    target,
    lowered,
    image,
    contract_json,
    contract,
):
    return _Artifact(
        key,
        adapter.name,
        adapter.artifact_format,
        target,
        lowered,
        image,
        contract_json,
        contract,
        tuple(argument.kind for argument in contract.arguments),
        _artifact_identity(
            key,
            adapter.name,
            adapter.artifact_format,
            target,
            contract_json,
        ),
    )


def _insert_artifact_locked(key, artifact, limit):
    _artifact_cache[key] = artifact
    _artifact_cache.move_to_end(key)
    while len(_artifact_cache) > limit:
        _artifact_cache.popitem(last=False)


def _cached_artifact(key, adapter, target):
    """Return the process artifact of `key`, or None, without waiting."""
    cached = _artifact_cache.get(key)
    if cached is None:
        return None
    if (
        cached.backend != adapter.name
        or cached.artifact_format != adapter.artifact_format
        or cached.target != target
    ):
        raise RuntimeError("process artifact cache identity mismatch")
    # Recency is refreshed only when nobody else holds the cache lock, so a
    # hit never waits.
    if len(_artifact_cache) > 1 and _cache_lock.acquire(blocking=False):
        try:
            if _artifact_cache.get(key) is cached:
                _artifact_cache.move_to_end(key)
        finally:
            _cache_lock.release()
    return cached


def _compile_cached(
    adapter,
    specialization,
    kernel_name,
    block_size,
    emit,
    *,
    key=None,
    lowering_kind="fixed",
    lowering_options=None,
):
    """Return the artifact for one specialization, emitting only on a miss.

    `emit` is a zero-argument callable producing the semantic module; it is
    deferred so a warm launch never pays for AST-to-MLIR emission. A warm
    launch stops at the process cache and never waits for a compile in
    progress. A miss holds the cold-path lock, so compiles are serialized,
    a second caller of the same key finds the first caller's artifact, and
    interpreter exit and `fork` wait for the compile in flight.

    A cache directory that cannot be read or written never fails the call:
    the artifact is kept for the process and one warning names the cause.
    An unsafe or corrupt entry is tamper evidence and still raises. Only
    backends with a persistent cache read and publish disk entries.

    SWAGE_CACHE_READ_ONLY=1 keeps a new artifact in the process without
    publishing it. SWAGE_NO_COMPILE=1 raises `RuntimeError` on a miss instead
    of compiling. A mistyped cache variable raises `ValueError`.
    """
    if not isinstance(lowering_kind, str) or not lowering_kind:
        raise ValueError("lowering kind must be a nonempty string")
    if lowering_options is None:
        lowering_options = {}
    if not isinstance(lowering_options, Mapping):
        raise TypeError("lowering options must be a mapping")
    lowering_options = dict(lowering_options)
    expected_options = [
        [name, lowering_options[name]] for name in sorted(lowering_options)
    ]
    codegen = specialization.get("codegen")
    target = specialization.get("target")
    if (
        specialization.get("backend") != adapter.name
        or specialization.get("format") != adapter.artifact_format
        or not isinstance(target, str)
        or not target
        or not isinstance(codegen, dict)
        or codegen.get("lowering", "fixed") != lowering_kind
        or codegen.get("options", []) != expected_options
    ):
        raise ValueError(
            "backend, format, target, lowering kind, or options do not match "
            "specialization metadata"
        )
    if key is None:
        key = _cache_key(specialization)
    limit = _memory_cache_limit()
    cached = _cached_artifact(key, adapter, target)
    if cached is not None:
        if _logger.isEnabledFor(logging.DEBUG):
            _logger.debug(
                "memory-hit backend=%s target=%s kernel=%s key=%s",
                adapter.name,
                target,
                kernel_name,
                key[:12],
            )
        return cached
    with _compile_lock:
        cached = _cached_artifact(key, adapter, target)
        if cached is not None:
            return cached
        # Read before any cache or compiler work, so a mistyped value fails
        # the call instead of being ignored.
        _, no_compile, _ = _cache_settings()
        identity = _cached_identity()
        persistent = adapter.persistent_cache and _cache_usable(identity)
        artifact = None
        if persistent:
            try:
                artifact = _read_cache_entry(
                    key,
                    specialization,
                    kernel_name,
                    block_size,
                    lowering_kind,
                    adapter,
                    target,
                )
            except OSError as error:
                _warn_cache_off("read", f"cannot use {_cache_dir()}: {error}")
                persistent = False
        if artifact is not None:
            if _logger.isEnabledFor(logging.DEBUG):
                _logger.debug(
                    "persistent-hit backend=%s target=%s kernel=%s key=%s",
                    adapter.name,
                    target,
                    kernel_name,
                    key[:12],
                )
        else:
            if no_compile:
                _refuse_compile(kernel_name, key, looked_up=persistent)
            if _logger.isEnabledFor(logging.DEBUG):
                _logger.debug(
                    "compile backend=%s target=%s kernel=%s key=%s",
                    adapter.name,
                    target,
                    kernel_name,
                    key[:12],
                )
            result = adapter.compile(
                emit(),
                kernel_name,
                block_size,
                target,
                lowering_kind,
                lowering_options,
            )
            if not isinstance(result, tuple) or len(result) != 3:
                raise RuntimeError(
                    "backend compiler must return lowered MLIR, image, and "
                    "contract"
                )
            lowered, image, contract_json = result
            if not isinstance(lowered, str) or not isinstance(
                contract_json, str
            ):
                raise RuntimeError(
                    "backend compiler returned invalid artifact data"
                )
            if adapter.artifact_format == "ptx" and not isinstance(image, str):
                raise RuntimeError("CUDA compiler returned non-text PTX")
            contract = _parse_compiler_contract(contract_json)
            _validate_contract_specialization(
                contract,
                specialization,
                kernel_name,
                block_size,
                lowering_kind,
                adapter,
            )
            artifact = _make_artifact(
                key,
                adapter,
                target,
                lowered,
                image,
                contract_json,
                contract,
            )
            # Retained before any disk write, so no failure below costs it.
            with _cache_lock:
                _insert_artifact_locked(key, artifact, limit)
            # The identity is checked again: the compile may have loaded
            # code that changed on disk since the lookup above.
            if persistent and _cache_writable() and _cache_usable(identity):
                try:
                    artifact = _write_cache_entry(
                        artifact,
                        specialization,
                        kernel_name,
                        block_size,
                        lowering_kind,
                        adapter,
                    )
                except OSError as error:
                    _warn_cache_off(
                        "write", f"cannot use {_cache_dir()}: {error}"
                    )
        with _cache_lock:
            _insert_artifact_locked(key, artifact, limit)
        _write_dumps(artifact)
        return artifact


def _native_compiler(kernel_name, backend):
    """Return the native `swage` bindings for a compile of `kernel_name`.

    Raises:
        BackendUnavailableError: The bindings cannot be imported.
        _BindingsMismatch: The bindings were built for another `swage`.
    """
    try:
        return _native_bindings()
    except _BindingsMismatch:
        raise
    except Exception as error:
        from ._frontend import _INSTALLATION

        raise BackendUnavailableError(
            "Swage launch requires the build-tree mlir_swage bindings, "
            "which the swage-compiler wheel does not include; kernel "
            f"'{kernel_name}' was not compiled",
            code="native-unavailable",
            backend=backend,
            remediation=f"see {_INSTALLATION} for the native build",
        ) from error


def _cache_dir():
    configured = os.environ.get("SWAGE_CACHE_DIR")
    if configured:
        return pathlib.Path(configured)
    base = pathlib.Path(
        os.environ.get("XDG_CACHE_HOME", pathlib.Path.home() / ".cache")
    )
    return base / "swage"


def _check_safe(path):
    details = path.lstat()
    if stat.S_ISLNK(details.st_mode):
        raise RuntimeError(f"cache entry is a symlink: {path}")
    if details.st_uid != os.geteuid():
        raise RuntimeError(
            f"cache entry is not owned by the current user: {path}"
        )
    if details.st_mode & stat.S_IWOTH:
        raise RuntimeError(f"cache entry is world-writable: {path}")


def _read_cache_entry(
    key,
    specialization,
    kernel_name,
    block_size,
    lowering_kind,
    adapter,
    target,
):
    """Return the verified entry for `key`, or None on a cache miss.

    A missing entry is a miss. An incomplete entry is also a miss: it is
    debris from a writer that died before entries were published atomically,
    and it is removed so the key can be published again. When the cache
    directory cannot be written, the debris stays, publishing is given up
    with a warning, and the miss still concerns this key only. A read-only
    process leaves the debris and does not warn. Unsafe, unreadable,
    mismatched, and corrupt entries raise `RuntimeError`, and so does a
    stored contract that does not describe the specialization. A cache
    directory that cannot be inspected raises `OSError`.
    """
    if (adapter.name, adapter.artifact_format) != ("cuda", "ptx"):
        raise RuntimeError("persistent cache is only valid for CUDA PTX")
    root = _cache_dir()
    entry = root / key
    paths = {
        "metadata": entry / "metadata.json",
        "lowered": entry / "lowered.mlir",
        "ptx": entry / "kernel.ptx",
    }
    try:
        _check_safe(root)
        _check_safe(entry)
        present = [path for path in paths.values() if os.path.lexists(path)]
        for path in present:
            _check_safe(path)
    except FileNotFoundError:
        return None
    if len(present) != len(paths):
        if _cache_writable():
            try:
                _remove_incomplete_entry(entry)
            except OSError as error:
                # The debris stays, so this key stays a miss and is not
                # published. Entries of other keys are still read.
                _warn_cache_off("write", f"cannot use {root}: {error}")
        return None
    try:
        metadata = json.loads(paths["metadata"].read_text())
        lowered = paths["lowered"].read_text()
        ptx = paths["ptx"].read_text()
    except FileNotFoundError:
        # Another process is removing this entry as incomplete.
        return None
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cache entry is unreadable: {entry}") from error
    if (
        not isinstance(metadata, dict)
        or set(metadata) != _METADATA_FIELDS
        or metadata.get("version") != _CACHE_VERSION
        or metadata.get("key") != key
        or metadata.get("backend") != adapter.name
        or metadata.get("format") != adapter.artifact_format
        or metadata.get("target") != target
    ):
        raise RuntimeError(f"cache metadata mismatch: {entry}")
    if metadata.get("specialization") != specialization:
        raise RuntimeError(f"cache specialization mismatch: {entry}")
    contract_json = metadata.get("contract")
    digests = metadata.get("digests")
    if (
        not isinstance(contract_json, str)
        or not isinstance(digests, dict)
        or set(digests) != {"lowered", "ptx", "contract"}
    ):
        raise RuntimeError(f"cache metadata mismatch: {entry}")
    if digests["contract"] != _sha256(contract_json):
        raise RuntimeError(f"cache contract digest mismatch: {entry}")
    if digests["lowered"] != _sha256(lowered):
        raise RuntimeError(f"cache lowered MLIR digest mismatch: {entry}")
    if digests["ptx"] != _sha256(ptx):
        raise RuntimeError(f"cache PTX digest mismatch: {entry}")
    contract = _parse_compiler_contract(contract_json)
    _validate_contract_specialization(
        contract,
        specialization,
        kernel_name,
        block_size,
        lowering_kind,
        adapter,
    )
    return _make_artifact(
        key, adapter, target, lowered, ptx, contract_json, contract
    )


# The fields of the metadata.json of a cache entry.
_METADATA_FIELDS = frozenset(
    {
        "version",
        "key",
        "backend",
        "format",
        "target",
        "specialization",
        "contract",
        "digests",
    }
)


def _sha256(text):
    """Return the SHA-256 hex digest of the UTF-8 encoding of `text`."""
    return hashlib.sha256(text.encode()).hexdigest()


_ENTRY_FILES = ("metadata.json", "lowered.mlir", "kernel.ptx")
# Entry names are SHA-256 hex digests, so a staging name never collides.
_ENTRY_NAME = re.compile(r"[0-9a-f]{64}")
_STAGING_PREFIX = ".staging-"
# A writer needs milliseconds between creating its staging directory and
# renaming it, so a staging directory this old belongs to a dead writer. A
# writer that was only stalled for longer finds its directory gone, fails
# to publish, and keeps its kernel for the process with one warning.
_STAGING_MAX_AGE_NS = 3600 * 1_000_000_000


def _remove_incomplete_entry(entry):
    """Remove the debris of a dead writer without losing a publish.

    The entry is renamed to a private name before it is judged, so the check
    and the removal act on the same directory. An entry that another process
    completed or published since the caller looked is renamed back.
    """
    aside = pathlib.Path(
        tempfile.mkdtemp(dir=entry.parent, prefix=_STAGING_PREFIX)
    )
    try:
        try:
            os.rename(entry, aside)
        except FileNotFoundError:
            return  # Another process removed it first.
        except (IsADirectoryError, NotADirectoryError) as error:
            raise RuntimeError(
                f"cache entry is not a directory: {entry}"
            ) from error
        if all(os.path.lexists(aside / name) for name in _ENTRY_FILES):
            try:
                os.rename(aside, entry)
            except OSError as error:
                # Another writer published the key in the meantime.
                if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                    raise
    finally:
        shutil.rmtree(aside, ignore_errors=True)


def _discard_entry(entry):
    """Remove one published entry without showing a reader part of it.

    The entry is renamed to a private name first, so a concurrent lookup
    finds the whole entry or a plain miss.
    """
    aside = pathlib.Path(
        tempfile.mkdtemp(dir=entry.parent, prefix=_STAGING_PREFIX)
    )
    try:
        os.rename(entry, aside)
    except FileNotFoundError:
        pass  # Another process evicted it first.
    finally:
        shutil.rmtree(aside, ignore_errors=True)


def _trim_cache(root, keep):
    """Remove dead staging directories and the oldest entries over the bound.

    Only a directory that the current user owns and that is named like an
    entry or like a staging directory is ever removed; anything else under
    the cache root is left alone. Entries go in the order they were
    published, oldest first, until the root holds at most `_max_entries()`.
    Reading an entry does not renew it.

    Args:
        root: The cache root.
        keep: Name of the entry just published, which always stays.
    """
    limit = _max_entries()
    dead_before = time.time_ns() - _STAGING_MAX_AGE_NS
    entries = []
    for name in os.listdir(root):
        staging = name.startswith(_STAGING_PREFIX)
        if not staging and (name == keep or not _ENTRY_NAME.fullmatch(name)):
            continue
        path = root / name
        try:
            details = path.lstat()
        except FileNotFoundError:
            continue  # Another process removed it first.
        if not stat.S_ISDIR(details.st_mode) or details.st_uid != os.geteuid():
            continue
        if not staging:
            entries.append((details.st_mtime_ns, name))
        elif details.st_mtime_ns < dead_before:
            shutil.rmtree(path, ignore_errors=True)
    # `keep` is not in `entries` and takes one place under the bound.
    excess = max(len(entries) + 1 - limit, 0)
    for _, name in sorted(entries)[:excess]:
        _discard_entry(root / name)


def _write_cache_entry(
    artifact, specialization, kernel_name, block_size, lowering_kind, adapter
):
    """Publish one cache entry atomically, then keep the root in its bound.

    The three files are written into a staging directory inside the cache
    root, which is then renamed to the entry name. A reader therefore sees
    no entry or a complete one, and a writer killed at any point leaves only
    a staging directory that no lookup reads.

    Returns:
        The artifact to use: `artifact` when this call published it, or the
        verified entry of the writer that published the same key first.
    """
    if (artifact.backend, artifact.artifact_format) != ("cuda", "ptx"):
        raise RuntimeError("persistent cache is only valid for CUDA PTX")
    root = _cache_dir()
    try:
        root.mkdir(parents=True, mode=0o700)
    except FileExistsError:
        pass
    _check_safe(root)
    metadata = {
        "version": _CACHE_VERSION,
        "key": artifact.key,
        "backend": artifact.backend,
        "format": artifact.artifact_format,
        "target": artifact.target,
        "specialization": specialization,
        "contract": artifact.contract_json,
        "digests": {
            "lowered": _sha256(artifact.lowered),
            "ptx": _sha256(artifact.image),
            "contract": _sha256(artifact.contract_json),
        },
    }
    staging = pathlib.Path(tempfile.mkdtemp(dir=root, prefix=_STAGING_PREFIX))
    try:
        _atomic_write(staging / "lowered.mlir", artifact.lowered)
        _atomic_write(staging / "kernel.ptx", artifact.image)
        _atomic_write(
            staging / "metadata.json",
            json.dumps(metadata, sort_keys=True, separators=(",", ":")),
        )
        try:
            # Renaming onto an empty directory replaces it; a published
            # entry is never empty, so it is never replaced.
            os.rename(staging, root / artifact.key)
        except OSError as error:
            if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
            published = _read_cache_entry(
                artifact.key,
                specialization,
                kernel_name,
                block_size,
                lowering_kind,
                adapter,
                artifact.target,
            )
            return artifact if published is None else published
        _trim_cache(root, artifact.key)
        return artifact
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _atomic_write(path, contents):
    descriptor, temporary = tempfile.mkstemp(dir=path.parent)
    try:
        try:
            output = os.fdopen(descriptor, "w")
        except BaseException:
            # No file object owns the descriptor, so it is closed here.
            os.close(descriptor)
            raise
        # From here the file object closes the descriptor, exactly once. A
        # second close could hit a file that another thread opened under
        # the same number in the meantime.
        with output:
            os.fchmod(output.fileno(), 0o600)
            output.write(contents)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def _write_dumps(artifact):
    dump_mlir = os.environ.get("SWAGE_DUMP_MLIR") == "1"
    dump_ptx = (
        os.environ.get("SWAGE_DUMP_PTX") == "1"
        and artifact.artifact_format == "ptx"
    )
    if not dump_mlir and not dump_ptx:
        return
    root = pathlib.Path(
        os.environ.get("SWAGE_DUMP_DIR", pathlib.Path.cwd() / "swage-dumps")
    )
    if os.path.lexists(root):
        _check_safe(root)
    else:
        root.mkdir(parents=True, mode=0o700)
    if dump_mlir:
        _atomic_write(root / f"{artifact.key}.mlir", artifact.lowered)
    if dump_ptx:
        _atomic_write(root / f"{artifact.key}.ptx", artifact.image)


def _bind_fixed_artifact(artifact, tensors, count):
    """Bind the validated dense fixed contract without rebuilding its plan."""
    return (
        artifact.argument_kinds,
        (
            tensors[0].data_ptr(),
            tensors[1].data_ptr(),
            tensors[2].data_ptr(),
            count,
        ),
    )


def bind_artifact(artifact, *, user, derived=None, plan=None, scratch=None):
    """Bind and materialize one artifact solely from its compiler contract."""
    bound = _abi.bind_kernel_contract(
        artifact.contract,
        user=user,
        derived=derived,
        plan=plan,
        scratch=scratch,
    )
    return _abi.materialize_launch_arguments(bound)


def segmented_specialization(
    semantic,
    *,
    kernel_name,
    target,
    block_size,
    lowering_kind,
    adapter,
    lowering_options=None,
    schedule=None,
):
    """Build complete deterministic cache identity for a segmented entry."""
    identity = _cached_identity()
    options = dict(lowering_options or {})
    return {
        "source": hashlib.sha256(semantic.encode()).hexdigest(),
        "kernel": kernel_name,
        "backend": adapter.name,
        "format": adapter.artifact_format,
        "target": target,
        "codegen": {
            "lowering": lowering_kind,
            "block_size": block_size,
            "options": [[key, options[key]] for key in sorted(options)],
            "schedule": dict(schedule or {}),
            "index_bits": 64,
        },
        "frontend": identity["frontend"],
        "native": identity["native"],
        "dialect_version": _DIALECT_VERSION,
    }
