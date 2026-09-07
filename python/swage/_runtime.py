"""Backend-neutral specialization, artifact, and launch orchestration."""

import ast
import hashlib
import json
import logging
import os
import pathlib
import stat
import subprocess
import tempfile
import threading
from collections import OrderedDict
from collections.abc import Mapping
from concurrent.futures import Future
from typing import NamedTuple

from . import _abi, _native, language
from ._backends import get_backend
from ._errors import BackendUnavailableError

_DIALECT_VERSION = 1
_CACHE_VERSION = 3
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
_cache_lock = threading.Lock()
_identity_lock = threading.Lock()
_artifact_cache = OrderedDict()
_compilations = {}
_identity_cache = None
_memory_cache_entries = None


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
    """Compile and launch one canonical fixed vector elementwise kernel."""
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
        lambda: kernel.emit_mlir(arguments=arguments, constexprs=constexprs),
        key=key,
        lowering_kind="fixed",
    )
    launch_arguments = _bind_fixed_artifact(artifact, spec.tensors, spec.n)
    if adapter.name == "cuda":
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
    if adapter.name == "cuda" and not capturing:
        from . import _cuda_backend

        _cuda_backend._prepare_fixed_launch(
            kernel, artifact, lease, spec.stream, torch, spec.tensors[0].dtype
        )
    if _logger.isEnabledFor(logging.DEBUG):
        _logger.debug(
            "%s backend=%s target=%s kernel=%s grid=%s key=%s",
            "launch complete" if adapter.name == "cpu" else "launch enqueued",
            adapter.name,
            spec.target,
            kernel.__name__,
            spec.grid,
            key[:12],
        )
    return None


def _import_torch(backend="cuda"):
    try:
        import torch
    except (ImportError, OSError) as error:
        raise BackendUnavailableError(
            "Swage launch requires PyTorch; install 'swage-compiler[pytorch]'",
            code="pytorch-unavailable",
            backend=backend,
            remediation="Install 'swage-compiler[pytorch]'.",
        ) from error
    return torch


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
    """Validate one tensor before its raw pointer crosses the ABI."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"argument '{name}' must be a {backend.upper()} tensor")
    if tensor.device.type != backend:
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
    return element_type


def _validate_runtime_arguments(
    arguments,
    runtime_names,
    torch,
    backend,
):
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
        "swage_revision": identity["revision"],
        "dialect_version": _DIALECT_VERSION,
        "llvm_version": identity["llvm"],
    }


def _compiler_identity():
    try:
        info = _native.build_info()
    except ValueError:
        # Invalid packaged provenance must never fall back to checkout data.
        return {"revision": None, "clean": False, "llvm": None}
    if info is not None:
        return {
            "revision": info["source_revision"],
            "clean": info["source_clean"],
            "llvm": info["llvm_version"],
        }
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


def _cached_identity():
    """Return `_compiler_identity()` computed once per process."""
    global _identity_cache
    with _identity_lock:
        if (
            _identity_cache is None
            or _identity_cache[0] is not _compiler_identity
        ):
            _identity_cache = (_compiler_identity, _compiler_identity())
        return _identity_cache[1]


def _cache_key(specialization):
    encoded = json.dumps(
        specialization, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _memory_cache_limit():
    """Lazily parse the strict positive process-cache capacity."""
    global _memory_cache_entries
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
    """Return one verified artifact with per-specialization coalescing."""
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
    with _cache_lock:
        cached = _artifact_cache.get(key)
        if cached is not None:
            if (
                cached.backend != adapter.name
                or cached.artifact_format != adapter.artifact_format
                or cached.target != target
            ):
                raise RuntimeError("process artifact cache identity mismatch")
            _artifact_cache.move_to_end(key)
        else:
            future = _compilations.get(key)
            owner = future is None
            if owner:
                future = Future()
                _compilations[key] = future
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
    if not owner:
        return future.result()

    try:
        identity = _cached_identity()
        persistent = bool(
            adapter.persistent_cache
            and identity["revision"]
            and identity["clean"]
            and identity["llvm"]
        )
        artifact = None
        if persistent:
            artifact = _read_cache_entry(
                key,
                specialization,
                kernel_name,
                block_size,
                lowering_kind,
                adapter,
                target,
            )
        if artifact is None:
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
            if persistent:
                _write_cache_entry(artifact, specialization)
        else:
            if _logger.isEnabledFor(logging.DEBUG):
                _logger.debug(
                    "persistent-hit backend=%s target=%s kernel=%s key=%s",
                    adapter.name,
                    target,
                    kernel_name,
                    key[:12],
                )
        _write_dumps(artifact)
    except BaseException as error:
        future.set_exception(error)
        with _cache_lock:
            if _compilations.get(key) is future:
                _compilations.pop(key)
        raise

    with _cache_lock:
        _insert_artifact_locked(key, artifact, limit)
    future.set_result(artifact)
    with _cache_lock:
        if _compilations.get(key) is future:
            _compilations.pop(key)
    return artifact


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


def _read_cache_entry(
    key,
    specialization,
    kernel_name,
    block_size,
    lowering_kind,
    adapter,
    target,
):
    if (adapter.name, adapter.artifact_format) != ("cuda", "ptx"):
        raise RuntimeError("persistent cache is only valid for CUDA PTX")
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
        "image": entry / "kernel.ptx",
    }
    for path in paths.values():
        if os.path.lexists(path):
            _check_safe(path)
    if not all(os.path.lexists(path) for path in paths.values()):
        raise RuntimeError(f"cache entry is incomplete: {entry}")
    try:
        metadata = json.loads(paths["metadata"].read_text())
        lowered = paths["lowered"].read_text()
        image = paths["image"].read_text()
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cache entry is unreadable: {entry}") from error
    if (
        not isinstance(metadata, dict)
        or set(metadata)
        != {
            "version",
            "key",
            "backend",
            "format",
            "target",
            "specialization",
            "contract",
            "digests",
        }
        or type(metadata.get("version")) is not int
        or metadata["version"] != _CACHE_VERSION
        or metadata.get("key") != key
        or metadata.get("backend") != adapter.name
        or metadata.get("format") != adapter.artifact_format
        or metadata.get("target") != target
    ):
        raise RuntimeError(f"cache metadata mismatch: {entry}")
    if metadata.get("specialization") != specialization:
        raise RuntimeError(f"cache specialization mismatch: {entry}")
    contract_json = metadata.get("contract")
    if not isinstance(contract_json, str):
        raise RuntimeError(f"cache metadata mismatch: {entry}")
    digests = metadata.get("digests")
    if not isinstance(digests, dict) or set(digests) != {
        "lowered",
        "ptx",
        "contract",
    }:
        raise RuntimeError(f"cache metadata mismatch: {entry}")
    if (
        digests.get("contract")
        != hashlib.sha256(contract_json.encode()).hexdigest()
    ):
        raise RuntimeError(f"cache contract digest mismatch: {entry}")
    contract = _parse_compiler_contract(contract_json)
    _validate_contract_specialization(
        contract,
        specialization,
        kernel_name,
        block_size,
        lowering_kind,
        adapter,
    )
    if digests.get("lowered") != hashlib.sha256(lowered.encode()).hexdigest():
        raise RuntimeError(f"cache lowered MLIR digest mismatch: {entry}")
    if digests.get("ptx") != hashlib.sha256(image.encode()).hexdigest():
        raise RuntimeError(f"cache PTX digest mismatch: {entry}")
    return _make_artifact(
        key,
        adapter,
        target,
        lowered,
        image,
        contract_json,
        contract,
    )


def _write_cache_entry(artifact, specialization):
    if (artifact.backend, artifact.artifact_format) != ("cuda", "ptx"):
        raise RuntimeError("persistent cache is only valid for CUDA PTX")
    root = _cache_dir()
    root.mkdir(parents=True, mode=0o700, exist_ok=True)
    _check_safe(root)
    entry = root / artifact.key
    entry.mkdir(mode=0o700, exist_ok=True)
    _check_safe(entry)
    metadata = {
        "version": _CACHE_VERSION,
        "key": artifact.key,
        "backend": artifact.backend,
        "format": artifact.artifact_format,
        "target": artifact.target,
        "specialization": specialization,
        "contract": artifact.contract_json,
        "digests": {
            "lowered": hashlib.sha256(artifact.lowered.encode()).hexdigest(),
            "ptx": hashlib.sha256(artifact.image.encode()).hexdigest(),
            "contract": hashlib.sha256(
                artifact.contract_json.encode()
            ).hexdigest(),
        },
    }
    _atomic_write(entry / "lowered.mlir", artifact.lowered)
    _atomic_write(entry / "kernel.ptx", artifact.image)
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
    dump_ptx = (
        os.environ.get("SWAGE_DUMP_PTX") == "1"
        and artifact.artifact_format == "ptx"
    )
    if not dump_mlir and not dump_ptx:
        return
    root = pathlib.Path(
        os.environ.get("SWAGE_DUMP_DIR", pathlib.Path.cwd() / "swage-dumps")
    )
    root.mkdir(parents=True, mode=0o700, exist_ok=True)
    _check_safe(root)
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
        "swage_revision": identity["revision"],
        "dialect_version": _DIALECT_VERSION,
        "llvm_version": identity["llvm"],
    }
