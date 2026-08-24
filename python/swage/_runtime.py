# python/swage/_runtime.py
"""Minimal CUDA Driver runtime for the M3 fixed vector-add subset."""

import ast
import ctypes
import hashlib
import json
import os
import pathlib
import stat
import subprocess
import tempfile
import threading
from collections.abc import Mapping
from typing import NamedTuple

from . import language

_DIALECT_VERSION = 1
_compile_lock = threading.Lock()
# ponytail: one global lock; use per-key locks only if compilation contends.
_ptx_cache = {}
_loaded_functions = {}
_driver = None


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
    """Compile and asynchronously launch one fixed vector-add kernel."""
    torch = _import_torch()
    spec = _validate_launch(kernel, arguments, constexprs, grid, torch)
    if spec.n == 0:
        return None

    module = kernel.emit_mlir(arguments=arguments, constexprs=constexprs)
    specialization = _specialization_data(
        kernel,
        descriptors=spec.descriptors,
        constexprs=constexprs,
        target=spec.target,
    )
    artifact = _compile_cached(
        module, specialization, kernel.__name__, spec.block
    )
    _write_dumps(artifact)
    driver = _get_driver()
    context = driver.current_context()
    loaded_key = (artifact.key, context)
    with _compile_lock:
        loaded = _loaded_functions.get(loaded_key)
        if loaded is None:
            loaded = driver.load(artifact.ptx, kernel.__name__)
            _loaded_functions[loaded_key] = loaded
    _, function = loaded
    abi_arguments = tuple(tensor.data_ptr() for tensor in spec.tensors) + (
        spec.n,
    )
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


def _import_torch():
    try:
        import torch
    except Exception as error:
        raise RuntimeError(
            "Swage launch requires PyTorch; install "
            "'swage-compiler[pytorch]'"
        ) from error
    return torch


def _validate_launch(kernel, arguments, constexprs, grid, torch):
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
            "M3 launch requires x_ptr, y_ptr, output_ptr, n, and BLOCK "
            "parameters"
        )
    runtime_names = parameter_names[:4]
    if set(arguments) != set(runtime_names):
        raise ValueError("arguments must contain exactly the M3 parameters")
    if set(constexprs) != {"BLOCK"}:
        raise ValueError("constexprs must contain exactly BLOCK")

    tensors = tuple(arguments[name] for name in runtime_names[:3])
    for name, tensor in zip(runtime_names, tensors):
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cuda":
            raise TypeError(f"argument '{name}' must be a CUDA tensor")
        if tensor.dtype != torch.float32:
            raise TypeError(f"argument '{name}' must have dtype torch.float32")
        if tensor.dim() != 1:
            raise TypeError(f"argument '{name}' must have rank one")
        if not tensor.is_contiguous():
            raise ValueError(f"argument '{name}' must be contiguous")

    n = arguments["n"]
    if type(n) is not int or not 0 <= n < (1 << 31):
        raise ValueError("n must be a nonnegative i32")
    for name, tensor in zip(runtime_names, tensors):
        if n > tensor.numel():
            raise ValueError(f"n exceeds tensor length for argument '{name}'")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in PyTorch")

    current_device = torch.cuda.current_device()
    for name, tensor in zip(runtime_names, tensors):
        if tensor.device.index != current_device:
            raise ValueError(
                f"argument '{name}' must be on the current CUDA device"
            )

    runtime_types, _ = kernel._validate_inputs(None, arguments, constexprs)
    descriptors = tuple(
        "i32" if runtime_types[name] is language.int32 else "ptr<f32>"
        for name in runtime_names
    )
    if descriptors != ("ptr<f32>", "ptr<f32>", "ptr<f32>", "i32"):
        raise TypeError("M3 launch requires three f32 pointers and one i32")

    properties = torch.cuda.get_device_properties(current_device)
    if block > properties.max_threads_per_block:
        raise ValueError(
            f"BLOCK {block} exceeds device limit "
            f"{properties.max_threads_per_block}"
        )
    expected_grid = ((n + block - 1) // block,)
    if grid != expected_grid:
        raise ValueError(f"grid must equal {expected_grid} for n and BLOCK")
    major, minor = torch.cuda.get_device_capability(current_device)
    stream = torch.cuda.current_stream()
    return _LaunchSpec(
        tensors,
        n,
        block,
        grid,
        f"sm_{major}{minor}",
        stream,
        descriptors,
    )


def _specialization_data(kernel, *, descriptors, constexprs, target):
    identity = _compiler_identity()
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
        "swage_revision": identity["revision"],
        "dialect_version": _DIALECT_VERSION,
        "llvm_version": identity["llvm"],
    }


def _compiler_identity():
    root = pathlib.Path(__file__).resolve().parents[2]
    pin = root / "cmake" / "llvm-version.txt"
    llvm = pin.read_text().strip() if pin.is_file() else None
    if not (root / ".git").exists():
        return {"revision": None, "clean": False, "llvm": llvm}
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
        return {"revision": None, "clean": False, "llvm": llvm}
    return {"revision": revision, "clean": not dirty, "llvm": llvm}


def _cache_key(specialization):
    encoded = json.dumps(
        specialization, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _compile_cached(module, specialization, kernel_name, block_size):
    key = _cache_key(specialization)
    with _compile_lock:
        cached = _ptx_cache.get(key)
        if cached is not None:
            return cached
        identity = _compiler_identity()
        persistent = bool(
            identity["revision"] and identity["clean"] and identity["llvm"]
        )
        if persistent:
            cached = _read_cache_entry(key, specialization)
            if cached is not None:
                _ptx_cache[key] = cached
                return cached
        target = specialization.get(
            "compute_capability", specialization.get("target")
        )
        lowered, ptx = _compile_native(
            module, kernel_name, block_size, target
        )
        artifact = _Artifact(key, lowered, ptx)
        if persistent:
            _write_cache_entry(artifact, specialization)
        _ptx_cache[key] = artifact
        return artifact


def _compile_native(module, kernel_name, block_size, target):
    try:
        from mlir_swage._mlir_libs._swageDialectsNanobind import (
            swage as native_swage,
        )
    except Exception as error:
        raise RuntimeError(
            "Swage launch requires the build-tree mlir_swage bindings"
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
    if details.st_mode & stat.S_IWOTH:
        raise RuntimeError(f"cache entry is world-writable: {path}")


def _read_cache_entry(key, specialization):
    root = _cache_dir()
    if not os.path.lexists(root):
        return None
    _check_safe(root)
    entry = root / key
    if not os.path.lexists(entry):
        return None
    _check_safe(entry)
    paths = {
        "metadata": entry / "metadata.json",
        "lowered": entry / "lowered.mlir",
        "ptx": entry / "kernel.ptx",
    }
    for path in paths.values():
        if os.path.lexists(path):
            _check_safe(path)
    if not all(os.path.lexists(path) for path in paths.values()):
        raise RuntimeError(f"cache entry is incomplete: {entry}")
    try:
        metadata = json.loads(paths["metadata"].read_text())
        lowered = paths["lowered"].read_text()
        ptx = paths["ptx"].read_text()
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


def _write_cache_entry(artifact, specialization):
    root = _cache_dir()
    if os.path.lexists(root):
        _check_safe(root)
    else:
        root.mkdir(parents=True, mode=0o700)
    entry = root / artifact.key
    if os.path.lexists(entry):
        _check_safe(entry)
    else:
        entry.mkdir(mode=0o700)
    metadata = {
        "version": 1,
        "key": artifact.key,
        "specialization": specialization,
        "digests": {
            "lowered": hashlib.sha256(artifact.lowered.encode()).hexdigest(),
            "ptx": hashlib.sha256(artifact.ptx.encode()).hexdigest(),
        },
    }
    _atomic_write(entry / "lowered.mlir", artifact.lowered)
    _atomic_write(entry / "kernel.ptx", artifact.ptx)
    _atomic_write(
        entry / "metadata.json",
        json.dumps(metadata, sort_keys=True, separators=(",", ":")),
    )


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


class _CudaDriver:
    """Small lazy wrapper around the Linux CUDA Driver API."""

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
        context = ctypes.c_void_p()
        self._call("cuCtxGetCurrent", ctypes.byref(context))
        if not context.value:
            raise RuntimeError("PyTorch has no current CUDA context")
        return context.value

    def load(self, ptx, kernel_name):
        module = ctypes.c_void_p()
        image = ctypes.create_string_buffer(ptx.encode())
        self._call(
            "cuModuleLoadData",
            ctypes.byref(module),
            ctypes.cast(image, ctypes.c_void_p),
        )
        function = ctypes.c_void_p()
        self._call(
            "cuModuleGetFunction",
            ctypes.byref(function),
            module,
            kernel_name.encode(),
        )
        return module.value, function.value

    def launch(self, function, grid, block, stream, arguments):
        values = [ctypes.c_void_p(value) for value in arguments[:3]]
        values.append(ctypes.c_int32(arguments[3]))
        self._launch(function, grid, block, stream, values)

    def launch_segmented(self, function, grid, block, stream, arguments):
        """Launch the internal three-pointer, two-count M4 ABI."""
        values = [ctypes.c_void_p(value) for value in arguments[:3]]
        values.extend(ctypes.c_int32(value) for value in arguments[3:])
        self._launch(function, grid, block, stream, values)

    def launch_segmented_tasks(
        self, function, grid, block, stream, arguments
    ):
        """Launch the internal four-pointer, two-count M7 task ABI."""
        values = [ctypes.c_void_p(value) for value in arguments[:4]]
        values.extend(ctypes.c_int32(value) for value in arguments[4:])
        self._launch(function, grid, block, stream, values)

    def launch_segmented_mixed(
        self, function, grid, block, stream, arguments
    ):
        """Launch the internal four-pointer, three-count M7 fused ABI."""
        self.launch_segmented_tasks(function, grid, block, stream, arguments)

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
    global _driver
    with _compile_lock:
        if _driver is None:
            _driver = _CudaDriver()
        return _driver


def driver_version():
    """Return the actual CUDA driver version, or ``None`` when unavailable."""
    try:
        return _get_driver().driver_version()
    except RuntimeError:
        return None
