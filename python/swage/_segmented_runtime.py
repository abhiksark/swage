# python/swage/_segmented_runtime.py
"""Artifact binding and owned execution for segmented programs."""

from . import _native, _runtime
from ._backends import get_backend

_CUDA_BACKEND = get_backend("cuda")


def _contract_bindings(contract, origin, available):
    keys = {
        argument.key
        for argument in contract.arguments
        if argument.origin == origin
    }
    return {key: available[key] for key in keys if key in available}


def _bind_artifact(artifact, user, derived, plan, scratch, *, materialize=True):
    contract = getattr(artifact, "contract", artifact)
    arguments = _runtime._abi.bind_kernel_contract(
        contract,
        user=user,
        derived=_contract_bindings(contract, "derived", derived),
        plan=_contract_bindings(contract, "plan", plan),
        scratch=_contract_bindings(contract, "scratch", scratch),
    )
    if not materialize:
        return arguments
    return _runtime._abi.materialize_launch_arguments(arguments)


def _emit_semantic_module(semantic):
    bindings = _native.load_ir(backend="cuda")

    with bindings.ir.Context() as context:
        bindings.swage.register_dialects(context)
        return bindings.ir.Module.parse(semantic)


def _compile_artifact(
    semantic,
    *,
    kernel_name,
    target,
    block_size,
    lowering_kind,
    lowering_options=None,
    schedule=None,
):
    specialization = _runtime.segmented_specialization(
        semantic,
        kernel_name=kernel_name,
        target=target,
        block_size=block_size,
        lowering_kind=lowering_kind,
        lowering_options=lowering_options,
        schedule=schedule,
        adapter=_CUDA_BACKEND,
    )
    return _runtime._compile_cached(
        _CUDA_BACKEND,
        specialization,
        kernel_name,
        block_size,
        lambda: _emit_semantic_module(semantic),
        lowering_kind=lowering_kind,
        lowering_options=lowering_options,
    )


def _load_entries(torch, artifacts):
    capturing = _runtime._is_current_stream_capturing(torch)
    return {
        name: _CUDA_BACKEND.lease(artifact, capturing=capturing)
        for name, artifact in artifacts.items()
    }


def _launch_immediate(torch, artifact, bindings, grid, tensors):
    stream = torch.cuda.current_stream()
    capturing = _runtime._is_current_stream_capturing(torch)
    lease = _CUDA_BACKEND.lease(artifact, capturing=capturing)
    try:
        _CUDA_BACKEND.launch(
            lease,
            artifact.contract,
            bindings,
            grid=grid,
            stream=stream.cuda_stream,
            capturing=capturing,
        )
    finally:
        _CUDA_BACKEND.release(lease)
        for tensor in tensors:
            tensor.record_stream(stream)


class _PreparedSegmentedExecution:
    """Own one explicit segmented plan and all asynchronous state."""

    __slots__ = (
        "_torch",
        "_values",
        "_offsets",
        "_output",
        "_device_index",
        "_leases",
        "_bound_arguments",
        "_referenced_tensors",
        "_prepared_plan",
        "_persistent_done",
        "_persistent_stream",
        "contracts",
        "entries",
        "plan_buffers",
        "scratch_buffers",
        "_counts",
        "resident_blocks",
        "warp_tasks",
        "cta_tasks",
        "partial_tasks",
        "merge_tasks",
    )

    def __init__(
        self,
        *,
        torch,
        values=None,
        offsets=None,
        output=None,
        artifacts=None,
        leases=None,
        prepared_plan=None,
        resident_blocks=0,
    ):
        self._torch = torch
        self._values = values
        self._offsets = offsets
        self._output = output
        self._device_index = None if offsets is None else offsets.device.index
        artifacts = artifacts or {}
        self.contracts = {
            name: artifact.contract for name, artifact in artifacts.items()
        }
        self._leases = dict(leases or {})
        self.entries = {
            name: lease.entry.function for name, lease in self._leases.items()
        }
        self._prepared_plan = prepared_plan
        self.plan_buffers = (
            {} if prepared_plan is None else prepared_plan.buffers
        )
        self.scratch_buffers = (
            {} if prepared_plan is None else prepared_plan.scratch
        )
        self._counts = {} if prepared_plan is None else prepared_plan.counts
        self._bound_arguments = {}
        self._referenced_tensors = {}
        # A prepared plan fixes argument origins and tensor identities. Keep
        # those bindings, but rematerialize pointers on every launch so a
        # storage change cannot leave a stale address in the launch ABI.
        users = (self._values, self._offsets, self._output)
        for name, contract in self.contracts.items():
            plan = self.plan_buffers
            if name == "mixed":
                plan = {
                    **plan,
                    "task_ids": plan["mixed_task_ids"],
                }
            arguments = _bind_artifact(
                contract,
                users,
                self._counts,
                plan,
                self.scratch_buffers,
                materialize=False,
            )
            self._bound_arguments[name] = arguments
            referenced = []
            for descriptor, argument in zip(contract.arguments, arguments):
                if descriptor.origin not in {"user", "plan", "scratch"}:
                    continue
                tensor = argument.value
                if not any(tensor is retained for retained in referenced):
                    referenced.append(tensor)
            self._referenced_tensors[name] = tuple(referenced)
        self._persistent_done = None
        self._persistent_stream = None
        self.resident_blocks = resident_blocks
        self.warp_tasks = self._counts.get("warp_task_count", 0)
        self.cta_tasks = self._counts.get("cta_task_count", 0)
        self.partial_tasks = self._counts.get("partial_task_count", 0)
        self.merge_tasks = self._counts.get("merge_task_count", 0)

    def __del__(self):
        for lease in self._leases.values():
            _CUDA_BACKEND.release(lease)

    def _current_stream(self, persistent=False):
        if self._device_index is None:
            return None
        if self._torch.cuda.current_device() != self._device_index:
            label = "persistent sum" if persistent else "sum"
            raise ValueError(
                f"prepared {label} must launch on its prepared device"
            )
        return self._torch.cuda.current_stream()

    def _wait_for_tasks(self, stream, persistent=False):
        self._prepared_plan.wait(self._torch, stream, persistent=persistent)

    def _launch(self, name, grid, stream):
        contract = self.contracts[name]
        bindings = _runtime._abi.materialize_launch_arguments(
            self._bound_arguments[name]
        )
        referenced = self._referenced_tensors[name]
        lease = self._leases[name]
        try:
            _CUDA_BACKEND.launch(
                lease,
                contract,
                bindings,
                grid=grid,
                stream=stream.cuda_stream,
                capturing=_runtime._is_current_stream_capturing(self._torch),
            )
        finally:
            for tensor in referenced:
                tensor.record_stream(stream)

    def launch_warp(self):
        if self._device_index is None:
            return None
        stream = self._current_stream()
        self._wait_for_tasks(stream)
        task_count = self._counts["segment_count"]
        if task_count:
            self._launch("warp", (task_count, 1, 1), stream)
        return None

    def launch_cta(self):
        if self._device_index is None:
            return None
        stream = self._current_stream()
        self._wait_for_tasks(stream)
        task_count = self._counts["segment_count"]
        if task_count:
            self._launch("cta", (task_count, 1, 1), stream)
        return None

    def launch_mixed(self):
        if self._device_index is None:
            return None
        stream = self._current_stream()
        self._wait_for_tasks(stream)
        direct_count = self.warp_tasks + self.cta_tasks
        if direct_count:
            grid = ((self.warp_tasks + 3) // 4 + self.cta_tasks, 1, 1)
            self._launch("mixed", grid, stream)
        if self.partial_tasks:
            self._launch("partial", (self.partial_tasks, 1, 1), stream)
            self._launch("merge", (self.merge_tasks, 1, 1), stream)
        return None

    def launch_persistent(self):
        if self._device_index is None:
            return None
        stream = self._current_stream(persistent=True)
        self._wait_for_tasks(stream, persistent=True)
        stream_handle = stream.cuda_stream
        capturing = self._torch.cuda.is_current_stream_capturing()
        if (
            self._persistent_stream is not None
            and self._persistent_stream != stream_handle
            and not capturing
            and not self._persistent_done.query()
        ):
            raise RuntimeError(
                "one prepared persistent execution cannot run concurrently "
                "on different streams"
            )
        self.scratch_buffers["counters"].zero_()
        self._launch("persistent", (self.resident_blocks, 1, 1), stream)
        if self._persistent_done is None:
            self._persistent_done = self._torch.cuda.Event()
        self._persistent_done.record(stream)
        self._persistent_stream = stream_handle
        return None
