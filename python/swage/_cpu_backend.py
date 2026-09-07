# python/swage/_cpu_backend.py
"""Synchronous native CPU backend adapter."""

from . import _abi, _native


class _HostLease:
    """Short-lived claim that keeps an artifact-owned JIT alive."""

    __slots__ = ("entry", "released")

    def __init__(self, executable):
        self.entry = executable
        self.released = False

    def release(self):
        if self.released:
            return
        self.released = True
        self.entry = None

    def __del__(self):
        self.release()


class _CPUBackend:
    name = "cpu"
    artifact_format = "llvm-jit"
    persistent_cache = False

    def compile(
        self,
        module,
        kernel_name,
        block_size,
        target,
        lowering_kind,
        lowering_options,
    ):
        if target != "native":
            raise ValueError("CPU target must be 'native'")
        if lowering_kind != "fixed" or lowering_options:
            raise ValueError("CPU backend supports only fixed lowering")
        native_swage = _native.load_extension(backend="cpu")
        return native_swage._compile_fixed_host(
            module,
            kernel_name=kernel_name,
            block_size=block_size,
        )

    def lease(self, artifact, *, capturing=False):
        return _HostLease(artifact.image)

    def launch(
        self,
        lease,
        contract,
        bindings,
        *,
        grid,
        stream,
        capturing,
    ):
        if not isinstance(contract, _abi.KernelContract):
            raise TypeError("contract must be a KernelContract")
        if contract.backend != "cpu" or contract.launch.model != "host-call":
            raise ValueError("CPU launch requires a CPU host-call contract")
        if contract.launch.block is not None:
            raise ValueError("CPU host-call contract must not define a block")
        if grid is not None or stream is not None or capturing:
            raise ValueError(
                "CPU host-call launch has no grid, stream, or capture"
            )
        if lease.released:
            raise RuntimeError("CPU executable lease has been released")
        kinds, arguments = bindings
        expected = tuple(argument.kind for argument in contract.arguments)
        if tuple(kinds) != expected:
            raise ValueError("launch argument kinds do not match contract")
        lease.entry.invoke(tuple(kinds), tuple(arguments))

    @staticmethod
    def release(lease):
        lease.release()


CPU_BACKEND = _CPUBackend()
