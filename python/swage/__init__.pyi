# python/swage/__init__.pyi
"""Types for the public fixed vector elementwise contract."""

from collections.abc import Callable, Mapping
from typing import Any, Literal

__version__: str

class SwageError(RuntimeError): ...
class CompilationError(SwageError): ...

class BackendUnavailableError(SwageError):
    code: str
    backend: str
    remediation: str
    def __init__(
        self, message: str, *, code: str, backend: str, remediation: str
    ) -> None: ...

class _Kernel:
    def emit_mlir(
        self,
        *,
        signature: Mapping[str, Any] | None = ...,
        arguments: Mapping[str, Any] | None = ...,
        constexprs: Mapping[str, int],
    ) -> Any: ...
    def launch(
        self,
        *,
        arguments: Mapping[str, Any],
        constexprs: Mapping[str, int],
        grid: tuple[int],
        backend: Literal["cpu", "cuda"] = ...,
    ) -> None: ...

def jit(function: Callable[..., Any]) -> _Kernel: ...
