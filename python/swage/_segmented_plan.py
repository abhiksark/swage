# python/swage/_segmented_plan.py
"""Planning limits, admission, classification, and schedule selection.

A planned launch admits its program once per pair of planning limits,
classifies its offsets on the host into warp, CTA, partial, and merge
records, and uploads the records in one buffer. Nothing here compiles or
launches a kernel.
"""

from . import _artifact, _runtime
from . import _segmented_runtime as _execution
from . import _segmented_validation as _validation


def _planning_limits(warp_max_elements, cta_chunk_elements):
    """Replace each omitted planning limit with the default of the target."""
    target = _execution._target_description()
    if warp_max_elements is None:
        warp_max_elements = target.default_warp_max_elements
    if cta_chunk_elements is None:
        cta_chunk_elements = target.default_cta_chunk_elements
    return warp_max_elements, cta_chunk_elements


# The segment ids 0, 1, 2, ... of each CUDA device index, which a launch of
# one task per segment reads as its task list. They depend on the segment
# count only, so one tensor per device serves every preparation: it is
# uploaded once, complete when the upload returns, and only read afterwards.
# It holds 4 MiB of device memory per device for as long as the process runs.
_IDENTITY_LIMIT = 1 << 20
_identity_memo = {}


def _identity_ids(torch, device, count):
    """Return a tensor that starts with the segment ids 0 to `count - 1`.

    Args:
        torch: The PyTorch module.
        device: CUDA device of the prepared tensors.
        count: Number of ids a launch reads, the segment count of the
            preparation.

    Returns:
        An int32 tensor on the device with at least `count` ascending ids.
        Up to `_IDENTITY_LIMIT` ids it is the shared tensor of the device,
        which needs no kernel and no wait. A longer list is filled on the
        device for this preparation alone, asynchronously, on the current
        stream.
    """
    if count > _IDENTITY_LIMIT:
        return torch.arange(count, dtype=torch.int32, device=device)
    ids = _identity_memo.get(device.index)
    if ids is None:
        import numpy

        ids = _identity_memo[device.index] = torch.tensor(
            numpy.arange(_IDENTITY_LIMIT, dtype=numpy.int32),
            dtype=torch.int32,
            device=device,
        )
    return ids


# The element programs of a batch that selection may send to the pure CTA
# kernel cost at most this many relative work units, as the native estimate
# counts them.
_ELEMENT_WORK_BUDGET = 32


def _has_small_element_program(module):
    """Conservatively bound work in already-admitted element regions.

    The native estimate holds the weight of each operation. It gives no
    estimate for a region that holds an operation without a weight, and
    such a program is not small.
    """
    from mlir_swage._mlir_libs._swageDialectsNanobind import (
        swage as native_swage,
    )

    work = native_swage._element_work(module)
    return work is not None and work <= _ELEMENT_WORK_BUDGET


# What a preparation needs to know about a program, none of which depends on
# the offsets. `_module_memo` keeps one parsed module per semantic module
# text with the result of inspecting its element program. `_admitted` keeps
# that result per program text and pair of planning limits that planning
# admission has accepted, so admission runs once per program and limits and
# a later preparation only classifies its offsets.
#
# Threads share a parsed module, which is safe because planning admission
# holds the GIL and leaves the module unchanged. Compiles do not use these
# modules: on a miss `_compile_once` parses the text in a context of its
# own, so a compile never runs on a context shared here.
#
# Both memos are bounded like the kernel memos: each keeps
# `_runtime._CACHE_LIMIT` entries and then forgets its oldest, which is
# parsed or admitted again on its next use. A hit takes no lock.
_module_memo = _runtime._BoundedCache(_runtime._CACHE_LIMIT)
_admitted = _runtime._BoundedCache(_runtime._CACHE_LIMIT)


def _parsed_module(module_text):
    """Parse and inspect one semantic module at most once per process.

    Args:
        module_text: Semantic module text that identifies the program.

    Returns:
        The module, parsed in a context that it keeps alive, and whether
        `_has_small_element_program` holds for it. A text that does not
        parse is not kept.
    """
    entry = _module_memo.get(module_text)
    if entry is None:
        from mlir_swage import ir
        from mlir_swage.dialects import swage

        context = ir.Context()
        swage.register_dialects(context)
        module = ir.Module.parse(module_text, context=context)
        entry = (module, _has_small_element_program(module))
        with _execution._memo_lock:
            _module_memo[module_text] = entry
    return entry


def _admit_program(
    module_text, kernel_name, warp_max_elements, cta_chunk_elements
):
    """Admit one program for planning under one pair of limits, once.

    Planning admission decides whether a program can be classified and
    whether the limits are valid. Neither depends on the offsets, so
    admission runs at the first preparation of a program with a pair of
    limits, on a layout without segments. Later preparations classify their
    offsets without the module.

    Args:
        module_text: Semantic module text that identifies the program.
        kernel_name: Name of the segment function in the module to admit.
        warp_max_elements: Largest segment assigned to direct warp work.
        cta_chunk_elements: Largest input range assigned to one CTA task.

    Returns:
        Whether `_has_small_element_program` holds for the program.

    Raises:
        ValueError: Planning admission rejects the program or the limits. A
            rejection is not kept, so every preparation raises it again.
        RuntimeError: The selected artifact does not hold the program
            under these limits.
    """
    artifact = _artifact.selected()
    if artifact is not None:
        # The build host ran planning admission and recorded its answer.
        return artifact.admit(
            module_text, warp_max_elements, cta_chunk_elements
        )
    key = (module_text, kernel_name, warp_max_elements, cta_chunk_elements)
    small_element_program = _admitted.get(key)
    if small_element_program is None:
        import numpy
        from mlir_swage._mlir_libs._swageDialectsNanobind import (
            swage as native_swage,
        )

        module, small_element_program = _parsed_module(module_text)
        native_swage._materialize_segmented_plan(
            module,
            kernel_name,
            offsets=numpy.zeros(1, dtype=numpy.int32),
            value_count=0,
            segment_count=0,
            warp_max_elements=warp_max_elements,
            cta_chunk_elements=cta_chunk_elements,
        )
        with _execution._memo_lock:
            _admitted[key] = small_element_program
    return small_element_program


def _classifying_validator(warp_max_elements, cta_chunk_elements):
    """Return an offsets validator that classifies the offsets it admits.

    The native classifier checks everything `_validate_offsets` checks about
    a host int32 array, so a preparation walks valid offsets once, there,
    instead of once to validate and once to classify. The classifier runs at
    the point where `_validate_shapes` validates the offsets, which keeps
    the order of every error.

    When the classifier refuses, `_validate_offsets` runs and raises its own
    message for offsets it refuses too. Offsets it admits were refused for
    the planning limits or for the size of the plan; the classification is
    then left out, and the caller classifies again after it has admitted
    the program, which reports the limits first.

    Args:
        warp_max_elements: Largest segment assigned to direct warp work.
        cta_chunk_elements: Largest input range assigned to one CTA task.

    Returns:
        The validator, with the signature of `_validate_offsets`, and a list
        that holds the result of `_classify_segments` once the validator
        has admitted and classified the offsets.
    """
    classification = []

    def validate(host_offsets, value_count, output_count):
        native_swage = _execution._native_swage()
        segment_count = len(host_offsets) - 1
        try:
            classification.append(
                native_swage._classify_segments(
                    host_offsets,
                    value_count=value_count,
                    segment_count=segment_count,
                    warp_max_elements=warp_max_elements,
                    cta_chunk_elements=cta_chunk_elements,
                )
            )
        except (TypeError, ValueError):
            return _validation._validate_offsets(
                host_offsets, value_count, output_count
            )
        if type(output_count) is not int or output_count < segment_count:
            classification.clear()
            return _validation._validate_offsets(
                host_offsets, value_count, output_count
            )
        return segment_count

    return validate, classification


def _selects_direct_cta(
    torch,
    device,
    host_offsets,
    segment_count,
    merge_count,
    cta_chunk_elements,
    small_element_program,
):
    """Return whether a batch runs the pure CTA kernel instead of splitting.

    The prepared path and the one-shot path both ask here, so a batch gets
    the same schedule, and a sum the same bits, on either.

    Args:
        torch: The PyTorch module.
        device: CUDA device of the batch.
        host_offsets: The validated offsets on the host.
        segment_count: Number of segments of the batch.
        merge_count: Number of merge tasks of its classification.
        cta_chunk_elements: Largest input range assigned to one CTA task.
        small_element_program: What `_admit_program` returned.
    """
    # ponytail: a measured two-chunk rule, not a general cost model.
    # Retain splitting for sparse batches, larger tails, or mixed lengths.
    # Every segment is split when the merge count equals the segment count,
    # so the longest segment is read only then.
    return bool(
        segment_count > 0
        and cta_chunk_elements
        == _execution._target_description().default_cta_chunk_elements
        and merge_count == segment_count
        and int((host_offsets[1:] - host_offsets[:-1]).max())
        <= 2 * cta_chunk_elements
        and segment_count
        >= torch.cuda.get_device_properties(device).multi_processor_count
        and small_element_program
    )


def _classification(
    found,
    host_offsets,
    *,
    value_count,
    segment_count,
    warp_max_elements,
    cta_chunk_elements,
):
    """Return the classification of validated offsets.

    Args:
        found: The list `_classifying_validator` returned, which holds the
            classification when the validator made one.
        host_offsets: The validated host offsets.
        value_count: Number of values the offsets index.
        segment_count: Number of segments.
        warp_max_elements: Largest segment assigned to direct warp work.
        cta_chunk_elements: Largest input range assigned to one CTA task.

    Returns:
        The records in one int32 array, then the warp, CTA, partial, and
        merge counts.

    Raises:
        ValueError: The classifier refused offsets that are valid. The
            program and the limits were admitted before, so this is the
            reason, for example the size of the plan.
    """
    if not found:
        found.append(
            _execution._native_swage()._classify_segments(
                host_offsets,
                value_count=value_count,
                segment_count=segment_count,
                warp_max_elements=warp_max_elements,
                cta_chunk_elements=cta_chunk_elements,
            )
        )
    return found[0]
