# python/swage/_segmented_qualification.py
"""Private coordinator for native segmented qualification."""

from . import _native, _runtime
from . import _segmented_plan as _plan
from . import _segmented_programs as _programs
from . import _segmented_runtime as _execution
from . import _segmented_validation as _validation

_WARP_BLOCK = 32
_CTA_BLOCK = 128
_SPLIT_BLOCK = 512
_PERSISTENT_BLOCK = 512
_CTA_CHUNK_ELEMENTS = 4096


def launch_gpu(values, offsets, output, kind, block_size=128):
    """Compile and launch one internally qualified segmented reduction."""
    torch = _runtime._import_torch()
    value_count, segment_count = _validation._validate_tensors(
        values, offsets, output
    )
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("block size must be a positive integer")
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    if block_size > properties.max_threads_per_block:
        raise ValueError(
            f"block size {block_size} exceeds device limit "
            f"{properties.max_threads_per_block}"
        )
    semantic = _programs._semantic_module(kind)
    if segment_count == 0:
        return None

    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    target = f"sm_{major}{minor}"
    kernel_name = f"segmented_{kind}"
    artifact = _execution._compile_artifact(
        semantic,
        kernel_name=kernel_name,
        target=target,
        block_size=block_size,
        lowering_kind="segmented",
    )
    bindings = _execution._bind_artifact(
        artifact,
        (values, offsets, output),
        {"value_count": value_count, "segment_count": segment_count},
        {},
        {},
    )
    _execution._launch_immediate(
        torch,
        artifact,
        bindings,
        (segment_count, 1, 1),
        (values, offsets, output),
    )
    return None


def _launch_segmented_sum_tasks(
    values, offsets, output, task_ids, *, block_size
):
    """Launch one internal identity-sum task list with a warp or CTA block."""
    torch = _runtime._import_torch()
    value_count, segment_count = _validation._validate_tensors(
        values, offsets, output
    )
    if not isinstance(task_ids, torch.Tensor):
        raise TypeError("task_ids must be a torch.Tensor")
    if task_ids.dtype != torch.int32:
        raise TypeError("task_ids must have dtype torch.int32")
    if task_ids.dim() != 1:
        raise TypeError("task_ids must have rank one")
    if not task_ids.is_contiguous():
        raise ValueError("task_ids must be contiguous")
    if task_ids.device.type != "cuda":
        raise TypeError("task_ids must be a CUDA tensor")
    if task_ids.device.index != torch.cuda.current_device():
        raise ValueError("task_ids must be on the current CUDA device")
    host_task_ids = task_ids.detach().cpu().tolist()
    task_count = len(host_task_ids)
    _validation._validate_counts(value_count, task_count)
    if any(
        type(task_id) is not int or not 0 <= task_id < segment_count
        for task_id in host_task_ids
    ):
        raise ValueError("task_ids must contain valid segment IDs")
    if block_size not in {_WARP_BLOCK, _CTA_BLOCK}:
        raise ValueError("task block size must be 32 or 128")
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    if block_size > properties.max_threads_per_block:
        raise ValueError(
            f"block size {block_size} exceeds device limit "
            f"{properties.max_threads_per_block}"
        )
    if task_count == 0:
        return None

    semantic = _programs._semantic_module("sum")
    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    target = f"sm_{major}{minor}"
    kernel_name = "segmented_sum"
    artifact = _execution._compile_artifact(
        semantic,
        kernel_name=kernel_name,
        target=target,
        block_size=block_size,
        lowering_kind="segmented",
        lowering_options={"use_task_ids": True},
    )
    bindings = _execution._bind_artifact(
        artifact,
        (values, offsets, output),
        {"value_count": value_count, "task_count": task_count},
        {"task_ids": task_ids},
        {},
    )
    _execution._launch_immediate(
        torch,
        artifact,
        bindings,
        (task_count, 1, 1),
        (values, offsets, output, task_ids),
    )
    return None


def _materialize_host_plan(
    native_swage,
    module,
    host_offsets,
    value_count,
    segment_count,
    warp_max_elements,
    cta_chunk_elements,
    *,
    persistent=False,
):
    materialized = native_swage._materialize_segmented_plan(
        module,
        offsets=host_offsets,
        value_count=value_count,
        segment_count=segment_count,
        warp_max_elements=warp_max_elements,
        cta_chunk_elements=cta_chunk_elements,
    )
    return _plan._validate_materialized_plan(
        materialized,
        host_offsets,
        warp_max_elements=warp_max_elements,
        cta_chunk_elements=cta_chunk_elements,
        persistent=persistent,
    )


def _prepare_planned_sum(
    values,
    offsets,
    output,
    *,
    warp_max_elements=32,
    cta_chunk_elements=_CTA_CHUNK_ELEMENTS,
):
    """Validate, compile, bind, and own one reusable segmented sum plan."""
    torch = _runtime._import_torch()
    value_count, segment_count, host_offsets = _validation._validate_shapes(
        values, offsets, output, _validation._validate_offsets
    )
    _plan._validate_planning_limits(warp_max_elements, cta_chunk_elements)
    if segment_count == 0:
        return _execution._PreparedSegmentedExecution(torch=torch)

    native_swage = _native.load_extension(backend="cuda")

    semantic = _programs._semantic_module("sum")
    module = _execution._emit_semantic_module(semantic)
    host_plan = _materialize_host_plan(
        native_swage,
        module,
        host_offsets,
        value_count,
        segment_count,
        warp_max_elements,
        cta_chunk_elements,
    )
    direct_count = host_plan.warp_count + host_plan.cta_count
    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    target = f"sm_{major}{minor}"
    kernel_name = "segmented_sum"
    schedule = {
        "warp_max_elements": warp_max_elements,
        "cta_chunk_elements": cta_chunk_elements,
    }
    artifacts = {
        "warp": _execution._compile_artifact(
            semantic,
            kernel_name=kernel_name,
            target=target,
            block_size=_WARP_BLOCK,
            lowering_kind="segmented",
            lowering_options={"use_task_ids": True},
            schedule=schedule,
        ),
        "cta": _execution._compile_artifact(
            semantic,
            kernel_name=kernel_name,
            target=target,
            block_size=_CTA_BLOCK,
            lowering_kind="segmented",
            lowering_options={"use_task_ids": True},
            schedule=schedule,
        ),
    }
    if direct_count:
        artifacts["mixed"] = _execution._compile_artifact(
            semantic,
            kernel_name=kernel_name,
            target=target,
            block_size=_CTA_BLOCK,
            lowering_kind="segmented_fused",
            schedule=schedule,
        )
    if host_plan.partial_count:
        artifacts["partial"] = _execution._compile_artifact(
            semantic,
            kernel_name=f"{kernel_name}__partial",
            target=target,
            block_size=_SPLIT_BLOCK,
            lowering_kind="segmented_split_partial",
            schedule=schedule,
        )
        artifacts["merge"] = _execution._compile_artifact(
            semantic,
            kernel_name=f"{kernel_name}__merge",
            target=target,
            block_size=_SPLIT_BLOCK,
            lowering_kind="segmented_split_merge",
            schedule=schedule,
        )

    counts = {
        "value_count": value_count,
        "segment_count": segment_count,
        "task_count": segment_count,
        "warp_task_count": host_plan.warp_count,
        "cta_task_count": host_plan.cta_count,
        "partial_task_count": host_plan.partial_count,
        "merge_task_count": host_plan.merge_count,
    }
    placeholders = {
        "task_ids": object(),
        "partial_ranges": object(),
        "merge_ranges": object(),
    }
    scratch_placeholders = {"partials": object()}
    for artifact in artifacts.values():
        _execution._bind_artifact(
            artifact,
            (values, offsets, output),
            counts,
            placeholders,
            scratch_placeholders,
            materialize=False,
        )

    prepared_plan = _plan._prepare_planned_device_state(
        torch,
        offsets.device,
        host_plan,
        value_count=value_count,
        segment_count=segment_count,
    )
    for name, artifact in artifacts.items():
        actual_plan = prepared_plan.buffers
        if name == "mixed":
            actual_plan = {
                **prepared_plan.buffers,
                "task_ids": prepared_plan.buffers["mixed_task_ids"],
            }
        _execution._bind_artifact(
            artifact,
            (values, offsets, output),
            prepared_plan.counts,
            actual_plan,
            prepared_plan.scratch,
        )

    leases = _execution._load_entries(torch, artifacts)
    return _execution._PreparedSegmentedExecution(
        torch=torch,
        values=values,
        offsets=offsets,
        output=output,
        artifacts=artifacts,
        leases=leases,
        prepared_plan=prepared_plan,
    )


def _prepare_persistent_sum(
    values,
    offsets,
    output,
    *,
    warp_max_elements=32,
    cta_chunk_elements=_CTA_CHUNK_ELEMENTS,
    resident_blocks=None,
):
    """Prepare one reusable resident execution with explicit owned state."""
    if resident_blocks is not None and (
        type(resident_blocks) is not int
        or resident_blocks <= 0
        or resident_blocks > (1 << 32) - 1
    ):
        raise ValueError("resident_blocks must be a positive u32")
    torch = _runtime._import_torch()
    value_count, segment_count, host_offsets = _validation._validate_shapes(
        values, offsets, output, _validation._validate_offsets
    )
    _plan._validate_planning_limits(warp_max_elements, cta_chunk_elements)
    if segment_count == 0:
        return _execution._PreparedSegmentedExecution(torch=torch)

    native_swage = _native.load_extension(backend="cuda")

    semantic = _programs._semantic_module("sum")
    module = _execution._emit_semantic_module(semantic)
    host_plan = _materialize_host_plan(
        native_swage,
        module,
        host_offsets,
        value_count,
        segment_count,
        warp_max_elements,
        cta_chunk_elements,
        persistent=True,
    )
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    active_blocks = _plan._persistent_active_blocks(
        host_plan, properties, resident_blocks
    )
    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    target = f"sm_{major}{minor}"
    kernel_name = "segmented_sum"
    schedule = {
        "warp_max_elements": warp_max_elements,
        "cta_chunk_elements": cta_chunk_elements,
    }
    artifact = _execution._compile_artifact(
        semantic,
        kernel_name=kernel_name,
        target=target,
        block_size=_PERSISTENT_BLOCK,
        lowering_kind="segmented_persistent",
        schedule=schedule,
    )
    counts = {
        "value_count": value_count,
        "warp_task_count": host_plan.warp_count,
        "cta_task_count": host_plan.cta_count,
        "partial_task_count": host_plan.partial_count,
        "merge_task_count": host_plan.merge_count,
    }
    plan_placeholders = {
        "warp_task_ids": object(),
        "cta_task_ids": object(),
        "partial_ranges": object(),
        "partial_merge_ids": object(),
        "merge_ranges": object(),
    }
    scratch_placeholders = {"partials": object(), "counters": object()}
    _execution._bind_artifact(
        artifact,
        (values, offsets, output),
        counts,
        plan_placeholders,
        scratch_placeholders,
        materialize=False,
    )

    prepared_plan = _plan._prepare_persistent_device_state(
        torch, offsets.device, host_plan, value_count=value_count
    )
    _execution._bind_artifact(
        artifact,
        (values, offsets, output),
        prepared_plan.counts,
        prepared_plan.buffers,
        prepared_plan.scratch,
    )
    artifacts = {"persistent": artifact}
    leases = _execution._load_entries(torch, artifacts)
    return _execution._PreparedSegmentedExecution(
        torch=torch,
        values=values,
        offsets=offsets,
        output=output,
        artifacts=artifacts,
        leases=leases,
        prepared_plan=prepared_plan,
        resident_blocks=active_blocks,
    )


def launch_softmax_gpu(values, offsets, output, block_size=128):
    """Compile and launch the internally qualified ragged softmax."""
    torch = _runtime._import_torch()
    value_count, segment_count = _validation._validate_softmax_tensors(
        values, offsets, output
    )
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("block size must be a positive integer")
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    if block_size > properties.max_threads_per_block:
        raise ValueError(
            f"block size {block_size} exceeds device limit "
            f"{properties.max_threads_per_block}"
        )
    if segment_count == 0:
        return None

    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    target = f"sm_{major}{minor}"
    artifact = _execution._compile_artifact(
        _programs._SOFTMAX_MODULE,
        kernel_name="ragged_softmax",
        target=target,
        block_size=block_size,
        lowering_kind="segmented",
    )
    bindings = _execution._bind_artifact(
        artifact,
        (values, offsets, output),
        {"value_count": value_count, "segment_count": segment_count},
        {},
        {},
    )
    _execution._launch_immediate(
        torch,
        artifact,
        bindings,
        (segment_count, 1, 1),
        (values, offsets, output),
    )
    return None
