"""Closed execution-backend registry for Swage runtimes."""

from typing import Protocol


class BackendAdapter(Protocol):
    """Internal compile, residency, and launch boundary."""

    name: str
    artifact_format: str
    persistent_cache: bool

    def compile(
        self,
        module,
        kernel_name,
        block_size,
        target,
        lowering_kind,
        lowering_options,
    ): ...

    def lease(self, artifact, *, capturing=False): ...

    def launch(
        self,
        lease,
        contract,
        bindings,
        *,
        grid,
        stream,
        capturing,
    ): ...

    def release(self, lease): ...


def validate_backend(backend):
    """Return one supported backend name without implicit coercion."""
    if not isinstance(backend, str):
        raise TypeError("backend must be a string")
    if backend not in _FACTORIES:
        raise ValueError(
            f"unknown execution backend {backend!r}; expected 'cpu' or 'cuda'"
        )
    return backend


def _make_cuda_backend():
    from ._cuda_backend import CUDA_BACKEND

    return CUDA_BACKEND


def _make_cpu_backend():
    from ._cpu_backend import CPU_BACKEND

    return CPU_BACKEND


_FACTORIES = {
    "cuda": _make_cuda_backend,
    "cpu": _make_cpu_backend,
}
_ADAPTERS = {}


def get_backend(backend):
    """Return the process singleton for one validated backend name."""
    backend = validate_backend(backend)
    adapter = _ADAPTERS.get(backend)
    if adapter is None:
        adapter = _FACTORIES[backend]()
        _ADAPTERS[backend] = adapter
    return adapter
