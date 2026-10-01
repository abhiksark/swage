# python/swage/_runtime.py
"""Minimal CUDA Driver runtime for the canonical fixed vector-add subset."""

import ast
import collections
import ctypes
import errno
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
import warnings
import weakref
from collections.abc import Mapping
from typing import NamedTuple

from . import language

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


# The cold-path lock: every native compile and every first load of a kernel
# in this process holds it, here and in the private segmented runner.
# Compiles are serialized on purpose. The native compiler states no contract
# for two compiles running at once, it holds the GIL for a whole compile, and
# an LLVM fatal error ends the process, so a lock per key would add risk and
# no concurrency. A warm launch never takes this lock: it reads the caches
# below with one `get` each.
_compile_lock = threading.Lock()
_ptx_cache = _BoundedCache(_CACHE_LIMIT)
# Values are `(module, function)` as `_CudaDriver.load` returns them. A
# kernel that leaves this cache is unloaded once nothing else holds its
# function handle; see `_Function`.
_loaded_functions = _BoundedCache(_CACHE_LIMIT)
_driver = None
_identity_cache = None
# Device facts cannot change within a process, but tests inject fresh fake
# torch modules, so the cache is scoped to the torch module object.
_device_facts_cache = weakref.WeakKeyDictionary()
_stream_objects = {}


class _Artifact(NamedTuple):
    """A verified specialization artifact."""

    key: str
    lowered: str
    ptx: str


class _LaunchSpec(NamedTuple):
    """Validated values needed after the Python trust boundary."""

    tensors: tuple
    n: int
    block: int
    grid: tuple
    target: str
    stream: object
    descriptors: tuple


def launch(kernel, *, arguments, constexprs, grid):
    """Compile and asynchronously launch one fixed vector-add kernel.

    A `TypeError` or `ValueError` raised on the way names the kernel and
    where it is defined, after the message of the check that failed.
    """
    torch = _import_torch()
    try:
        spec = _validate_launch(kernel, arguments, constexprs, grid, torch)
        if spec.n == 0:
            return None

        memo = kernel.__dict__.setdefault("_specialization_memo", {})
        identity = _cached_identity()
        entry = memo.get((spec.block, spec.target))
        if entry is None or entry[2] is not identity:
            specialization = _specialization_data(
                kernel,
                descriptors=spec.descriptors,
                constexprs=constexprs,
                target=spec.target,
            )
            entry = (specialization, _cache_key(specialization), identity)
            memo[(spec.block, spec.target)] = entry
        specialization, key, _ = entry
        artifact = _compile_cached(
            specialization,
            kernel.__name__,
            spec.block,
            lambda: kernel.emit_mlir(
                arguments=arguments, constexprs=constexprs
            ),
            key=key,
        )
        _write_dumps(artifact)
        driver = _get_driver()
        context = driver.current_context()
        loaded_key = (artifact.key, context)
        loaded = _loaded_functions.get(loaded_key)
        if loaded is None:
            loaded = _load_cold(
                _loaded_functions,
                loaded_key,
                driver,
                artifact.ptx,
                kernel.__name__,
            )
        _, function = loaded
        abi_arguments = tuple(
            tensor.data_ptr() for tensor in spec.tensors
        ) + (spec.n,)
        driver.launch(
            function,
            spec.grid,
            spec.block,
            spec.stream.cuda_stream,
            abi_arguments,
        )
        for tensor in spec.tensors:
            tensor.record_stream(spec.stream)
        return None
    except (TypeError, ValueError) as error:
        # A subclass may take more than a message, so it is left as it is.
        if type(error) not in (TypeError, ValueError):
            raise
        raise type(error)(
            f"{error}{_launch_location(kernel)}"
        ).with_traceback(error.__traceback__) from None


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


def _import_torch():
    """Return PyTorch, or raise when it is missing or unsupported."""
    global _supported_torch
    try:
        import torch
    except Exception as error:
        raise RuntimeError(
            "Swage launch requires PyTorch; install "
            "'swage-compiler[pytorch]'"
        ) from error
    if torch is not _supported_torch:
        _require_supported_torch(torch)
        _supported_torch = torch
    return torch


def _require_supported_torch(torch):
    """Reject a PyTorch that a launch cannot use, before any work starts.

    A launch enqueues the kernel and then retains each tensor on the stream
    through `Tensor.record_stream`. Finding that method missing after the
    enqueue would leave a kernel running on storage PyTorch may reuse, so
    the release and the method are both checked here.
    """
    found = getattr(torch, "__version__", None)
    release = re.match(r"(\d+)\.(\d+)", str(found))
    required = ".".join(str(part) for part in _MIN_TORCH)
    if release is None or (int(release[1]), int(release[2])) < _MIN_TORCH:
        raise RuntimeError(
            f"Swage launch requires PyTorch {required} or newer; "
            f"found PyTorch {found}"
        )
    if not callable(getattr(torch.Tensor, "record_stream", None)):
        raise RuntimeError(
            "Swage launch requires torch.Tensor.record_stream to retain "
            f"submitted tensors; found PyTorch {found} without it"
        )


def _validate_launch_call(kernel, arguments, constexprs, grid):
    """Validate launch mappings, static block size, and parameter names."""
    if not isinstance(arguments, Mapping):
        raise TypeError("arguments must be a mapping")
    if not isinstance(constexprs, Mapping):
        raise TypeError("constexprs must be a mapping")
    block = constexprs.get("BLOCK")
    if type(block) is not int or block <= 0:
        raise ValueError("constexpr BLOCK must be a positive integer")
    if (
        not isinstance(grid, tuple)
        or len(grid) != 1
        or type(grid[0]) is not int
    ):
        raise TypeError("grid must be a one-element tuple of integers")

    parameter_names = [argument.arg for argument in kernel.function.args.args]
    if parameter_names != ["x_ptr", "y_ptr", "output_ptr", "n", "BLOCK"]:
        raise ValueError(
            "launch requires x_ptr, y_ptr, output_ptr, n, and BLOCK parameters"
        )
    runtime_names = parameter_names[:4]
    if set(arguments) != set(runtime_names):
        raise ValueError(
            "arguments must contain exactly x_ptr, y_ptr, output_ptr, and n"
        )
    if set(constexprs) != {"BLOCK"}:
        raise ValueError("constexprs must contain exactly BLOCK")
    return runtime_names, block


def _validate_launch_tensor(name, tensor, torch):
    """Validate one tensor before its raw pointer crosses the ABI."""
    if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cuda":
        raise TypeError(f"argument '{name}' must be a CUDA tensor")
    if tensor.dtype != torch.float32:
        raise TypeError(f"argument '{name}' must have dtype torch.float32")
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


def _validate_output_disjoint(names, tensors):
    """Reject an output that shares memory with a buffer the kernel reads.

    One thread would store through the output while another loads the same
    bytes through the input. Every tensor is known contiguous and rank one
    by this point, so each byte extent is exact and the half-open
    intersection is exact. It cannot see two virtual mappings of one
    physical allocation, nor aliasing created after this returns.

    Args:
        names: Argument names, the inputs first and the output last.
        tensors: The tensors in the same order.
    """
    *inputs, output = tensors
    output_start = output.data_ptr()
    output_end = output_start + output.numel() * output.element_size()
    for name, buffer in zip(names, inputs):
        start = buffer.data_ptr()
        end = start + buffer.numel() * buffer.element_size()
        if start < output_end and output_start < end:
            raise ValueError(
                f"argument '{names[-1]}' must not overlap argument '{name}'"
            )


def _validate_runtime_arguments(arguments, runtime_names, torch):
    """Validate tensor metadata, scalar bounds, lengths, and disjointness."""
    tensors = tuple(arguments[name] for name in runtime_names[:3])
    for name, tensor in zip(runtime_names, tensors):
        _validate_launch_tensor(name, tensor, torch)

    n = arguments["n"]
    if type(n) is not int or not 0 <= n < (1 << 31):
        raise ValueError("n must be a nonnegative i32")
    for name, tensor in zip(runtime_names, tensors):
        if n > tensor.numel():
            raise ValueError(f"n exceeds tensor length for argument '{name}'")
    # The two inputs may share memory; only the output is written.
    _validate_output_disjoint(runtime_names[:3], tensors)
    return tensors, n


def _validate_cuda_device(tensors, runtime_names, torch):
    """Require CUDA availability and tensors on the active device."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in PyTorch")
    current_device = torch.cuda.current_device()
    for name, tensor in zip(runtime_names, tensors):
        if tensor.device.index != current_device:
            raise ValueError(
                f"argument '{name}' must be on the current CUDA device"
            )
    return current_device


def _launch_descriptors(kernel, arguments, constexprs, runtime_names):
    """Return and verify the fixed vector-add ABI descriptors."""
    # Tensor and scalar validation pins the canonical ABI. Re-derive metadata
    # only when the annotations are not the canonical constexpr shape so its
    # diagnostics remain authoritative.
    kernel._require_plain_parameters()
    if kernel.constexpr_names == {"BLOCK"}:
        return ("ptr<f32>", "ptr<f32>", "ptr<f32>", "i32")

    runtime_types, _ = kernel._validate_inputs(None, arguments, constexprs)
    descriptors = tuple(
        "i32" if runtime_types[name] is language.int32 else "ptr<f32>"
        for name in runtime_names
    )
    if descriptors != ("ptr<f32>", "ptr<f32>", "ptr<f32>", "i32"):
        raise TypeError("launch requires three f32 pointers and one i32")
    return descriptors


def _validate_launch_geometry(block, n, grid, torch, current_device):
    """Validate the requested block and grid against the active device."""
    max_threads, target = _device_facts(torch, current_device)
    if block > max_threads:
        raise ValueError(f"BLOCK {block} exceeds device limit {max_threads}")
    expected_grid = ((n + block - 1) // block,)
    if grid != expected_grid:
        raise ValueError(f"grid must equal {expected_grid} for n and BLOCK")
    return target, _current_stream(torch, current_device)


def _validate_launch(kernel, arguments, constexprs, grid, torch):
    runtime_names, block = _validate_launch_call(
        kernel, arguments, constexprs, grid
    )
    tensors, n = _validate_runtime_arguments(arguments, runtime_names, torch)
    current_device = _validate_cuda_device(tensors, runtime_names, torch)
    descriptors = _launch_descriptors(
        kernel, arguments, constexprs, runtime_names
    )
    target, stream = _validate_launch_geometry(
        block, n, grid, torch, current_device
    )
    return _LaunchSpec(
        tensors,
        n,
        block,
        grid,
        target,
        stream,
        descriptors,
    )


def _device_facts(torch, index):
    """Cached (max threads, sm target) per device for this torch module."""
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


def _current_stream(torch, index):
    """The current stream, without rebuilding the object when unchanged."""
    raw_stream = getattr(
        getattr(torch, "_C", None), "_cuda_getCurrentRawStream", None
    )
    if raw_stream is None:
        return torch.cuda.current_stream()
    handle = raw_stream(index)
    cached = _stream_objects.get((index, handle))
    if cached is None:
        cached = torch.cuda.current_stream(index)
        _stream_objects[(index, handle)] = cached
    return cached


def _specialization_data(kernel, *, descriptors, constexprs, target):
    identity = _cached_identity()
    source_digest = getattr(kernel, "source_digest", None)
    if source_digest is None:
        normalized_source = ast.dump(kernel.function, include_attributes=False)
        source_digest = hashlib.sha256(normalized_source.encode()).hexdigest()
    block = constexprs["BLOCK"]
    return {
        "source": source_digest,
        "kernel": kernel.__name__,
        "descriptors": list(descriptors),
        "constexprs": [[key, constexprs[key]] for key in sorted(constexprs)],
        "compute_capability": target,
        "codegen": {"block_size": block, "index_bits": 64},
        "frontend": identity["frontend"],
        "native": identity["native"],
        "dialect_version": _DIALECT_VERSION,
        "llvm_version": identity["llvm"],
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
        `(digest, None)`, where the SHA-256 hex digest covers each file's
        relative name, length, and bytes, or `(None, problem)` when a file
        cannot be read, is too new, or the package holds no source.
    """
    try:
        sources = _frontend_sources(package)
        digest = hashlib.sha256()
        for name, path in sources:
            contents = path.read_bytes()
            if started is not None and _changed_ns(path) >= started:
                return None, _too_new(path)
            digest.update(f"{name}\0{len(contents)}\0".encode())
            digest.update(contents)
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


def _native_identity():
    """Describe the native compiler libraries by their file metadata.

    Returns:
        A sorted list of `[file name, size, st_mtime_ns]`, one per file of
        the nanobind extension and the C API library, or None when the
        bindings are not importable.
    """
    libraries = []
    for path in _native_libraries():
        try:
            details = path.stat()
        except OSError:
            continue
        libraries.append([path.name, details.st_size, details.st_mtime_ns])
    if not any(name.startswith(_NATIVE_EXTENSION) for name, *_ in libraries):
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


def _compiler_identity():
    """Identify the compiler files this process finds on disk.

    `frontend` and `native` identify the code that produces PTX and are the
    compiler fields of the cache key. They describe the files as they are
    now; `_stale_identity` decides whether that is the code this process
    loaded. `revision` and `clean` describe the surrounding git checkout
    for diagnostics; they do not gate the cache.

    Returns:
        A dict with the keys `revision`, `clean`, `llvm`, `frontend`, and
        `native`. Each value is None (False for `clean`) when unavailable.
    """
    package = _package_dir()
    root = package.parents[1]
    pin = root / "cmake" / "llvm-version.txt"
    revision, clean = _git_identity(root)
    return {
        "revision": revision,
        "clean": clean,
        "llvm": pin.read_text().strip() if pin.is_file() else None,
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
    return "write" not in _cache_off and not _switch_on(
        "SWAGE_CACHE_READ_ONLY"
    )


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
    return RuntimeError(
        f"SWAGE_NO_COMPILE=1 refuses to compile kernel '{kernel_name}': "
        f"{reason}"
    )


def _compile_cached(specialization, kernel_name, block_size, emit, *,
                    key=None):
    """Return the artifact for one specialization, emitting only on a miss.

    `emit` is a zero-argument callable producing the semantic module; it is
    deferred so a warm launch never pays for AST-to-MLIR emission.

    A cache directory that cannot be read or written never fails the call:
    the artifact is kept for the process and one warning names the cause.
    An unsafe or corrupt entry is tamper evidence and still raises.

    SWAGE_CACHE_READ_ONLY=1 keeps a new artifact in the process without
    publishing it. SWAGE_NO_COMPILE=1 raises `RuntimeError` on a miss instead
    of compiling. A mistyped cache variable raises `ValueError`.
    """
    if key is None:
        key = _cache_key(specialization)
    # A warm launch stops here and never waits for a compile in progress.
    cached = _ptx_cache.get(key)
    if cached is not None:
        return cached
    with _compile_lock:
        cached = _ptx_cache.get(key)
        if cached is not None:
            return cached
        # Read before any cache or compiler work, so a mistyped value fails
        # the call instead of being ignored.
        _, no_compile, _ = _cache_settings()
        identity = _cached_identity()
        persistent = _cache_usable(identity)
        if persistent:
            try:
                cached = _read_cache_entry(key, specialization)
            except OSError as error:
                _warn_cache_off("read", f"cannot use {_cache_dir()}: {error}")
                persistent = False
            if cached is not None:
                _ptx_cache[key] = cached
                return cached
        if no_compile:
            _refuse_compile(kernel_name, key, looked_up=persistent)
        target = specialization.get(
            "compute_capability", specialization.get("target")
        )
        lowered, ptx = _compile_native(
            emit(), kernel_name, block_size, target
        )
        artifact = _Artifact(key, lowered, ptx)
        # Retained before any disk write, so no failure below costs it.
        _ptx_cache[key] = artifact
        # The identity is checked again: the compile may have loaded code
        # that changed on disk since the lookup above.
        if persistent and _cache_writable() and _cache_usable(identity):
            try:
                artifact = _write_cache_entry(artifact, specialization)
            except OSError as error:
                _warn_cache_off("write", f"cannot use {_cache_dir()}: {error}")
            _ptx_cache[key] = artifact
        return artifact


def _compile_native(module, kernel_name, block_size, target):
    try:
        from mlir_swage._mlir_libs._swageDialectsNanobind import (
            swage as native_swage,
        )
    except Exception as error:
        from ._frontend import _INSTALLATION

        raise RuntimeError(
            "Swage launch requires the build-tree mlir_swage bindings, "
            "which the swage-compiler wheel does not include; kernel "
            f"'{kernel_name}' was not compiled. See {_INSTALLATION} for "
            "the native build"
        ) from error
    return native_swage._compile_ptx(
        module,
        kernel_name=kernel_name,
        block_size=block_size,
        target=target,
    )


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


def _read_cache_entry(key, specialization):
    """Return the verified entry for `key`, or None on a cache miss.

    A missing entry is a miss. An incomplete entry is also a miss: it is
    debris from a writer that died before entries were published atomically,
    and it is removed so the key can be published again. When the cache
    directory cannot be written, the debris stays, publishing is given up
    with a warning, and the miss still concerns this key only. A read-only
    process leaves the debris and does not warn. Unsafe, unreadable,
    mismatched, and corrupt entries raise `RuntimeError`. A cache directory
    that cannot be inspected raises `OSError`.
    """
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
    if metadata.get("version") != 1 or metadata.get("key") != key:
        raise RuntimeError(f"cache metadata mismatch: {entry}")
    if metadata.get("specialization") != specialization:
        raise RuntimeError(f"cache specialization mismatch: {entry}")
    digests = metadata.get("digests", {})
    if digests.get("lowered") != hashlib.sha256(lowered.encode()).hexdigest():
        raise RuntimeError(f"cache lowered MLIR digest mismatch: {entry}")
    if digests.get("ptx") != hashlib.sha256(ptx.encode()).hexdigest():
        raise RuntimeError(f"cache PTX digest mismatch: {entry}")
    return _Artifact(key, lowered, ptx)


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


def _write_cache_entry(artifact, specialization):
    """Publish one cache entry atomically, then keep the root in its bound.

    The three files are written into a staging directory inside the cache
    root, which is then renamed to the entry name. A reader therefore sees
    no entry or a complete one, and a writer killed at any point leaves only
    a staging directory that no lookup reads.

    Returns:
        The artifact to use: `artifact` when this call published it, or the
        verified entry of the writer that published the same key first.
    """
    root = _cache_dir()
    try:
        root.mkdir(parents=True, mode=0o700)
    except FileExistsError:
        pass
    _check_safe(root)
    metadata = {
        "version": 1,
        "key": artifact.key,
        "specialization": specialization,
        "digests": {
            "lowered": hashlib.sha256(artifact.lowered.encode()).hexdigest(),
            "ptx": hashlib.sha256(artifact.ptx.encode()).hexdigest(),
        },
    }
    staging = pathlib.Path(tempfile.mkdtemp(dir=root, prefix=_STAGING_PREFIX))
    try:
        _atomic_write(staging / "lowered.mlir", artifact.lowered)
        _atomic_write(staging / "kernel.ptx", artifact.ptx)
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
            published = _read_cache_entry(artifact.key, specialization)
            return artifact if published is None else published
        _trim_cache(root, artifact.key)
        return artifact
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _atomic_write(path, contents):
    descriptor, temporary = tempfile.mkstemp(dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w") as output:
            output.write(contents)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def _write_dumps(artifact):
    dump_mlir = os.environ.get("SWAGE_DUMP_MLIR") == "1"
    dump_ptx = os.environ.get("SWAGE_DUMP_PTX") == "1"
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
        _atomic_write(root / f"{artifact.key}.ptx", artifact.ptx)


def _load_cold(cache, key, driver, ptx, kernel_name):
    """Load one kernel into `cache` unless another thread just did.

    This is the miss path of a loaded-kernel cache. It takes the cold-path
    lock, which a cache hit never takes.

    Returns:
        The `(module, function)` entry of `key`.
    """
    with _compile_lock:
        loaded = cache.get(key)
        if loaded is None:
            loaded = driver.load(ptx, kernel_name)
            cache[key] = loaded
    return loaded


def _capturing():
    """Return whether this thread is capturing a CUDA graph in PyTorch."""
    torch = sys.modules.get("torch")
    return torch is not None and torch.cuda.is_current_stream_capturing()


class _Function(int):
    """The handle of a loaded kernel, which keeps its CUDA module loaded.

    The value is the function handle that `cuLaunchKernel` takes. Its module
    is queued for unloading when this object is collected, so the module
    stays loaded for whoever still holds the handle: a kernel cache, a
    prepared launch, or a launch in progress. A plain copy of the value,
    such as `int(function)`, keeps nothing loaded.
    """

    def __new__(cls, handle, module, retired):
        """Wrap a function handle resolved from a loaded module.

        Args:
            handle: Function handle returned by the driver.
            module: `(context, module handle)` to queue when this object
                is collected.
            retired: The driver's queue of modules that nothing references.
        """
        function = super().__new__(cls, handle)
        function._module = module
        function._retired = retired
        return function

    def __del__(self):
        """Queue the module. The driver unloads it before its next load."""
        self._retired.append(self._module)


class _CudaDriver:
    """Small lazy wrapper around the Linux CUDA Driver API."""

    _native_launch = None

    def __init__(self):
        try:
            self.library = ctypes.CDLL("libcuda.so.1")
        except OSError as error:
            raise RuntimeError(
                "CUDA Driver library libcuda.so.1 is unavailable"
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
        self.library.cuCtxSynchronize.argtypes = []
        self.library.cuCtxSynchronize.restype = ctypes.c_int
        self.library.cuStreamIsCapturing.argtypes = [
            pointer,
            ctypes.POINTER(ctypes.c_int),
        ]
        self.library.cuStreamIsCapturing.restype = ctypes.c_int
        # Context ids exist from CUDA 12. An older driver has only handles.
        self._context_id = getattr(self.library, "cuCtxGetId", None)
        if self._context_id is not None:
            self._context_id.argtypes = [
                pointer,
                ctypes.POINTER(ctypes.c_ulonglong),
            ]
            self._context_id.restype = ctypes.c_int
        # `(context, module handle)` of each module whose `_Function` was
        # collected, waiting for `unload_retired`.
        self._retired = collections.deque()
        # Functions that a CUDA graph may replay; their modules stay loaded.
        self._pinned = set()
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
        # The compiled launcher skips per-launch ctypes marshalling; the
        # ctypes path stays as the fallback when the build-tree bindings
        # cannot be imported. A wheel-only install cannot launch from a
        # warm cache: with no native libraries to identify, it never reads
        # the persistent cache.
        try:
            from mlir_swage._mlir_libs._swageDialectsNanobind import (
                swage as native_swage,
            )

            self._native_launch = native_swage._launch_kernel
        except ImportError:
            self._native_launch = None

    def _call(self, name, *arguments):
        result = getattr(self.library, name)(*arguments)
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
            kernel loaded in the destroyed context would pass for a kernel
            of the new one. A driver without context ids returns the
            handle.

        Raises:
            RuntimeError: No CUDA context is current on this thread.
        """
        if self._context_id is not None:
            identifier = ctypes.c_ulonglong()
            # A null context asks for the id of the current context.
            if not self._context_id(None, ctypes.byref(identifier)):
                return identifier.value
        context = ctypes.c_void_p()
        self._call("cuCtxGetCurrent", ctypes.byref(context))
        if not context.value:
            raise RuntimeError("PyTorch has no current CUDA context")
        return context.value

    def load(self, ptx, kernel_name):
        """Load one module and resolve one kernel in it.

        Modules that nothing references any more are unloaded first. A
        module whose kernel cannot be resolved is not left loaded.

        Returns:
            `(module, function)`. `function` is a `_Function`: the module
            stays loaded for as long as that object is referenced, and
            `module` is only its handle.
        """
        self.unload_retired()
        owner = self.current_context()
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
            self._retired.append((owner, module.value))
            self.unload_retired()
            raise
        return module.value, _Function(
            function.value, (owner, module.value), self._retired
        )

    def unload_retired(self):
        """Unload the modules of the current context that nothing holds.

        A module is queued only after its `_Function` was collected, so no
        thread can launch it again. Launches of it that are still queued on
        a stream are waited for: the context is synchronized once before
        the first unload.

        CUDA forbids both calls while a stream of the context captures a
        graph, and the attempt invalidates the capture. While this thread
        captures, nothing is unloaded and the modules wait for a later
        call. A capture open on another thread cannot be seen from here: it
        is invalidated, the synchronize fails, and the modules stay queued.

        A module of another context stays queued until that context is
        current. A driver error is reported as a `RuntimeWarning` and is
        never raised, because the caller is loading an unrelated kernel.
        """
        if not self._retired or _capturing():
            return
        context = self.current_context()
        idle = []
        for _ in range(len(self._retired)):
            try:
                owner, module = self._retired.popleft()
            except IndexError:
                break
            if owner == context:
                idle.append(module)
            else:
                self._retired.append((owner, module))
        if not idle:
            return
        try:
            self._call("cuCtxSynchronize")
        except RuntimeError as error:
            self._retired.extend((context, module) for module in idle)
            warnings.warn(
                f"Swage left {len(idle)} unused CUDA modules loaded: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        for module in idle:
            try:
                self._call("cuModuleUnload", ctypes.c_void_p(module))
            except RuntimeError as error:
                warnings.warn(
                    f"Swage left an unused CUDA module loaded: {error}",
                    RuntimeWarning,
                    stacklevel=2,
                )

    def _pin_if_capturing(self, function, stream):
        """Keep a kernel loaded for good when a CUDA graph captures it.

        A graph replays a captured launch for as long as the graph lives,
        and nothing ties the graph to the `_Function`.
        """
        status = ctypes.c_int()
        self._call(
            "cuStreamIsCapturing",
            ctypes.c_void_p(stream),
            ctypes.byref(status),
        )
        if status.value:
            self._pinned.add(function)

    def launch(self, function, grid, block, stream, arguments):
        # The NULL stream cannot capture, which keeps the capture query off
        # the default-stream path.
        if stream and type(function) is _Function:
            self._pin_if_capturing(function, stream)
        if self._native_launch is not None:
            self._native_launch(
                function, grid[0], block, stream,
                arguments[:3], (arguments[3],),
            )
            return
        values = [ctypes.c_void_p(value) for value in arguments[:3]]
        values.append(ctypes.c_int32(arguments[3]))
        self._launch(function, grid, block, stream, values)

    def launch_segmented(self, function, grid, block, stream, arguments):
        """Launch a private three-pointer segmented ABI.

        The pointers are followed by two i32 counts, or by three for the
        split merge kernel, whose last count is the segment count.
        """
        if self._native_launch is not None:
            self._native_launch(
                function, grid[0], block, stream,
                arguments[:3], arguments[3:],
            )
            return
        values = [ctypes.c_void_p(value) for value in arguments[:3]]
        values.extend(ctypes.c_int32(value) for value in arguments[3:])
        self._launch(function, grid, block, stream, values)

    def launch_segmented_tasks(
        self, function, grid, block, stream, arguments
    ):
        """Launch the private four-pointer, three-count task-ID ABI."""
        if self._native_launch is not None:
            self._native_launch(
                function, grid[0], block, stream,
                arguments[:4], arguments[4:],
            )
            return
        values = [ctypes.c_void_p(value) for value in arguments[:4]]
        values.extend(ctypes.c_int32(value) for value in arguments[4:])
        self._launch(function, grid, block, stream, values)

    def launch_segmented_mixed(
        self, function, grid, block, stream, arguments
    ):
        """Launch the private four-pointer, four-count fused ABI."""
        self.launch_segmented_tasks(function, grid, block, stream, arguments)

    def launch_persistent(self, function, grid, block, stream, arguments):
        """Launch the private ten-pointer, six-count persistent ABI."""
        if self._native_launch is not None:
            self._native_launch(
                function, grid[0], block, stream,
                arguments[:10], arguments[10:],
            )
            return
        values = [ctypes.c_void_p(value) for value in arguments[:10]]
        values.extend(ctypes.c_int32(value) for value in arguments[10:])
        self._launch(function, grid, block, stream, values)

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
            1,
            1,
            block,
            1,
            1,
            0,
            ctypes.c_void_p(stream),
            parameter_pointers,
            None,
        )


def _get_driver():
    """Return the process-wide driver, creating it on first use."""
    global _driver
    driver = _driver
    if driver is None:
        with _compile_lock:
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
