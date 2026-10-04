# python/swage/_segmented_plan.py
"""Validated host plans and owned device plan state."""

from dataclasses import dataclass
from itertools import pairwise

from ._segmented_validation import _I32_LIMIT

_WARP_BLOCK = 32
_PERSISTENT_BLOCK = 512


@dataclass(frozen=True)
class _SegmentedPlan:
    warp_ids: tuple
    cta_ids: tuple
    partial_records: tuple
    merge_records: tuple
    partial_merge_ids: tuple

    @property
    def warp_count(self):
        return len(self.warp_ids)

    @property
    def cta_count(self):
        return len(self.cta_ids)

    @property
    def partial_count(self):
        return len(self.partial_records) // 2

    @property
    def merge_count(self):
        return len(self.merge_records) // 3


@dataclass
class _PreparedPlan:
    buffers: dict
    scratch: dict
    counts: dict
    tasks_ready: object
    tasks_ready_complete: bool = False

    def wait(self, torch, stream, *, persistent=False):
        if self.tasks_ready_complete:
            return
        if torch.cuda.is_current_stream_capturing():
            label = "persistent sum" if persistent else "reduction"
            raise RuntimeError(
                f"prepared {label} must launch once after task "
                "initialization before CUDA graph capture"
            )
        if self.tasks_ready.query():
            self.tasks_ready_complete = True
        else:
            stream.wait_event(self.tasks_ready)


def _validate_planning_limits(warp_max_elements, cta_chunk_elements):
    if (
        type(warp_max_elements) is not int
        or type(cta_chunk_elements) is not int
        or not 0 < warp_max_elements <= cta_chunk_elements < _I32_LIMIT
    ):
        raise ValueError(
            "planning limits must satisfy 0 < warp-max-elements <= "
            "cta-chunk-elements <= INT32_MAX"
        )


def _expected_plan(host_offsets, warp_max_elements, cta_chunk_elements):
    warp_ids = []
    cta_ids = []
    partial_records = []
    merge_records = []
    for segment_id, (begin, end) in enumerate(pairwise(host_offsets)):
        length = end - begin
        if length <= warp_max_elements:
            warp_ids.append(segment_id)
        elif length <= cta_chunk_elements:
            cta_ids.append(segment_id)
        else:
            partial_begin = len(partial_records) // 2
            for chunk_begin in range(begin, end, cta_chunk_elements):
                partial_records.extend(
                    [chunk_begin, min(end, chunk_begin + cta_chunk_elements)]
                )
            merge_records.extend(
                [segment_id, partial_begin, len(partial_records) // 2]
            )
    return warp_ids, cta_ids, partial_records, merge_records


def _validate_materialized_plan(
    materialized,
    host_offsets,
    *,
    warp_max_elements,
    cta_chunk_elements,
    persistent=False,
):
    """Validate native plan records and freeze their host metadata."""
    expected = _expected_plan(
        host_offsets, warp_max_elements, cta_chunk_elements
    )
    if materialized != expected:
        raise RuntimeError(
            "materialized plan does not match classified metadata"
        )
    warp_ids, cta_ids, partial_records, merge_records = materialized
    partial_merge_ids = []
    if persistent:
        partial_merge_ids = [-1] * (len(partial_records) // 2)
        for merge_id in range(len(merge_records) // 3):
            begin = merge_records[merge_id * 3 + 1]
            end = merge_records[merge_id * 3 + 2]
            for partial_id in range(begin, end):
                if partial_merge_ids[partial_id] != -1:
                    raise RuntimeError(
                        "partial task belongs to multiple merges"
                    )
                partial_merge_ids[partial_id] = merge_id
        if any(merge_id < 0 for merge_id in partial_merge_ids):
            raise RuntimeError("partial task has no merge dependency")
    return _SegmentedPlan(
        tuple(warp_ids),
        tuple(cta_ids),
        tuple(partial_records),
        tuple(merge_records),
        tuple(partial_merge_ids),
    )


def _prepare_planned_device_state(
    torch, device, host_plan, *, value_count, segment_count, split=True
):
    # A preparation that runs every segment as direct CTA work never launches
    # the split kernels, so it allocates neither their records nor scratch.
    direct_count = host_plan.warp_count + host_plan.cta_count
    buffers = {
        "task_ids": torch.arange(
            segment_count, dtype=torch.int32, device=device
        )
    }
    if direct_count:
        buffers["mixed_task_ids"] = torch.tensor(
            [*host_plan.warp_ids, *host_plan.cta_ids],
            dtype=torch.int32,
            device=device,
        )
    scratch = {}
    if host_plan.partial_count and split:
        buffers["partial_ranges"] = torch.tensor(
            host_plan.partial_records, dtype=torch.int32, device=device
        )
        buffers["merge_ranges"] = torch.tensor(
            host_plan.merge_records, dtype=torch.int32, device=device
        )
        scratch["partials"] = torch.empty(
            host_plan.partial_count, dtype=torch.float32, device=device
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
    ready = torch.cuda.Event()
    ready.record(torch.cuda.current_stream())
    return _PreparedPlan(buffers, scratch, counts, ready)


def _prepare_persistent_device_state(torch, device, host_plan, *, value_count):
    buffers = {
        "warp_task_ids": torch.tensor(
            host_plan.warp_ids, dtype=torch.int32, device=device
        ),
        "cta_task_ids": torch.tensor(
            host_plan.cta_ids, dtype=torch.int32, device=device
        ),
        "partial_ranges": torch.tensor(
            host_plan.partial_records, dtype=torch.int32, device=device
        ),
        "partial_merge_ids": torch.tensor(
            host_plan.partial_merge_ids, dtype=torch.int32, device=device
        ),
        "merge_ranges": torch.tensor(
            host_plan.merge_records, dtype=torch.int32, device=device
        ),
    }
    scratch = {
        "partials": torch.empty(
            host_plan.partial_count, dtype=torch.float32, device=device
        ),
        "counters": torch.zeros(
            3 + host_plan.merge_count, dtype=torch.int32, device=device
        ),
    }
    counts = {
        "value_count": value_count,
        "warp_task_count": host_plan.warp_count,
        "cta_task_count": host_plan.cta_count,
        "partial_task_count": host_plan.partial_count,
        "merge_task_count": host_plan.merge_count,
    }
    ready = torch.cuda.Event()
    ready.record(torch.cuda.current_stream())
    return _PreparedPlan(buffers, scratch, counts, ready)


def _persistent_active_blocks(host_plan, properties, resident_blocks):
    if _PERSISTENT_BLOCK > properties.max_threads_per_block:
        raise ValueError(
            f"persistent block size {_PERSISTENT_BLOCK} exceeds device limit "
            f"{properties.max_threads_per_block}"
        )
    warp_slots = _PERSISTENT_BLOCK // _WARP_BLOCK
    work_groups = (
        host_plan.cta_count
        + host_plan.partial_count
        + (host_plan.warp_count + warp_slots - 1) // warp_slots
    )
    if resident_blocks is None:
        resident_blocks = properties.multi_processor_count * 2
    return min(resident_blocks, work_groups)
