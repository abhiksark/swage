# tests/python/test_segment_launch_cleanup.py
"""LLVM-free fault injection for segmented launch cleanup."""

import types

import pytest
from swage import _abi, _cuda_backend, _runtime
from swage import _segmented_plan as _plan
from swage import _segmented_programs as _programs
from swage import _segmented_qualification as _qualification
from swage import _segmented_runtime as _execution
from swage import _segmented_validation as _validation


class _Tensor:
    """Observe allocator retention and the output version without a GPU."""

    def __init__(self, state, name, pointer, dtype="float32"):
        self.state = state
        self.name = name
        self.pointer = pointer
        self.dtype = dtype
        self.device = types.SimpleNamespace(index=0)
        self.version = 0

    def data_ptr(self):
        return self.pointer

    @property
    def _version(self):
        return self.version

    def is_inference(self):
        return False

    def numel(self):
        return {"values": 260, "offsets": 3}.get(self.name, 2)

    def zero_(self):
        self.state.zeroed.append(self.state.stream.cuda_stream)
        if self.state.zero_error is not None:
            raise self.state.zero_error
        return self

    def record_stream(self, stream):
        assert stream is self.state.stream
        self.state.retained.append(self.name)
        self.state.trace.append(("record", self.name))
        error = self.state.record_errors.get(self.name)
        if error is not None:
            raise error


class _Lease:
    """Observe explicit releases while the real enqueue uses its entry."""

    def __init__(self, state, entry):
        self.state = state
        self.entry = entry

    def release(self):
        name = self.entry.function
        self.state.released.append(name)
        error = self.state.release_errors.get(name)
        if error is not None:
            raise error


class _Driver:
    """Inject failures below the real CUDA enqueue and fence helpers."""

    def __init__(self, state):
        self.state = state
        self.last_entry = None

    def event_create(self):
        if self.state.event_create_error is not None:
            raise self.state.event_create_error
        return object()

    def launch_entry(self, function, contract, arguments, grid, stream):
        assert contract.entry == function
        assert arguments[0]
        assert grid[0] > 0
        assert stream == self.state.stream.cuda_stream
        self.last_entry = function
        self.state.launched.append(function)
        self.state.trace.append(("launch", function))
        error = self.state.driver_errors.get(("launch", function))
        if error is not None:
            raise error

    def event_record(self, event, stream):
        assert stream == self.state.stream.cuda_stream
        self.state.fenced.append(self.last_entry)
        error = self.state.driver_errors.get(("record", self.last_entry))
        if error is not None:
            raise error


class _Event:
    """Model task readiness and persistent completion without a GPU."""

    def __init__(self, state):
        self.state = state
        self.role = "tasks" if not state.events else "completion"
        state.events.append(self)

    def record(self, stream):
        self.state.event_records.append((self.role, stream.cuda_stream))
        if self.role == "completion":
            if self.state.completion_record_error is not None:
                raise self.state.completion_record_error

    def query(self):
        if self.role == "completion":
            return self.state.completion_ready
        return self.state.tasks_ready


def _kernel(name):
    """Build the contracts the real binder consumes for these phases."""
    def user(kind, index, access=None):
        return _abi.KernelArgument(
            kind, "user", source_index=index, access=access
        )

    def keyed(kind, origin, key, access=None):
        return _abi.KernelArgument(kind, origin, key=key, access=access)

    if name == "partial":
        arguments = (
            user("ptr", 0, "read"),
            keyed("ptr", "plan", "partial_ranges", "read"),
            keyed("ptr", "scratch", "scratch", "write"),
            user("i32", 3),
            keyed("i32", "derived", "partial_count"),
        )
    elif name == "merge":
        arguments = (
            keyed("ptr", "scratch", "scratch", "read"),
            user("ptr", 2, "write"),
            keyed("ptr", "plan", "merge_records", "read"),
            keyed("i32", "derived", "partial_count"),
            keyed("i32", "derived", "merge_count"),
            user("i32", 4),
        )
    else:
        arguments = (
            user("ptr", 0, "read"),
            user("ptr", 1, "read"),
            user("ptr", 2, "write"),
            user("i32", 3),
            user("i32", 4),
        )
        if name in {"warp", "cta"}:
            arguments += (
                keyed("ptr", "plan", "task_ids", "read"),
                keyed("i32", "derived", "task_count"),
            )
        elif name == "persistent":
            arguments += (
                *(
                    keyed("ptr", "plan", key, "read")
                    for key in (
                        "warp_ids", "cta_ids", "partial_ranges",
                        "partial_merge_ids", "merge_records",
                    )
                ),
                keyed("ptr", "scratch", "scratch", "readwrite"),
                keyed("ptr", "scratch", "counters", "readwrite"),
                *(
                    keyed("i32", "derived", key)
                    for key in (
                        "warp_task_count", "cta_task_count", "partial_count",
                        "merge_count",
                    )
                ),
            )
    contract = _abi.KernelContract(
        2,
        "cuda",
        name,
        _abi.KernelLaunch("spmd-grid", (128, 1, 1)),
        arguments,
    )
    return _execution._Kernel(
        name, _abi.serialize_kernel_contract(contract), contract
    )


@pytest.fixture
def launch_state(monkeypatch):
    """Replace GPU allocation and native compilation, keeping launch real."""
    state = types.SimpleNamespace(
        stream=types.SimpleNamespace(cuda_stream=17),
        retained=[],
        released=[],
        leased=[],
        launched=[],
        fenced=[],
        trace=[],
        record_errors={},
        release_errors={},
        driver_errors={},
        event_create_error=None,
        allocation_error=None,
        lease_error=None,
        direct_count=1,
        partial_count=2,
        merge_count=1,
        events=[],
        event_records=[],
        completion_record_error=None,
        completion_ready=True,
        tasks_ready=True,
        capturing=False,
        shared_versions=False,
        zeroed=[],
        zero_error=None,
    )
    state.values = _Tensor(state, "values", 0x1000)
    state.offsets = _Tensor(state, "offsets", 0x2000, "int32")
    state.output = _Tensor(state, "output", 0x3000)
    state.records = _Tensor(state, "records", 0x4000, "int32")
    state.scratch = _Tensor(state, "scratch", 0x5000)
    state.tasks = _Tensor(state, "tasks", 0x6000, "int32")
    state.counters = _Tensor(state, "counters", 0x7000, "int32")
    driver = _Driver(state)

    def allocate(name):
        if state.allocation_error is not None:
            failed_name, error = state.allocation_error
            if failed_name == name:
                raise error
        return getattr(state, name)

    def advance_version(tensor):
        tensor.version += 1
        state.trace.append(("version", tensor.name))
        if state.shared_versions and tensor is state.output:
            state.offsets.version += 1

    def empty(*args, dtype, **kwargs):
        return allocate("counters" if dtype == "int32" else "scratch")

    torch = types.SimpleNamespace(
        int32="int32",
        float32="float32",
        cuda=types.SimpleNamespace(
            current_device=lambda: 0,
            current_stream=lambda *args: state.stream,
            is_current_stream_capturing=lambda: state.capturing,
            Event=lambda: _Event(state),
            get_device_properties=lambda *args: types.SimpleNamespace(
                max_threads_per_block=1024,
                multi_processor_count=1,
            ),
        ),
        tensor=lambda *args, **kwargs: allocate("records"),
        empty=empty,
        autograd=types.SimpleNamespace(
            graph=types.SimpleNamespace(increment_version=advance_version)
        ),
    )
    state.torch = torch

    def lease(_torch, kernel):
        if state.lease_error is not None and state.leased:
            raise state.lease_error
        entry = _cuda_backend._LoadedModule(
            kernel.identity, 1, object(), kernel.contract.entry, driver
        )
        state.leased.append(kernel.contract.entry)
        return _Lease(state, entry)

    names = {
        "_compile_fused_segmented_reduction_ptx": "direct",
        "_compile_split_partial_reduction_ptx": "partial",
        "_compile_split_merge_reduction_ptx": "merge",
        "_compile_segmented_reduction_ptx": "task",
        "_compile_persistent_segmented_reduction_ptx": "persistent",
    }
    native = types.SimpleNamespace()
    for name, phase in names.items():
        setattr(native, name, types.SimpleNamespace(phase=phase))
    monkeypatch.setattr(_runtime, "_import_torch", lambda: torch)
    monkeypatch.setattr(_execution, "_native_swage", lambda: native)
    monkeypatch.setattr(_qualification, "_native_bindings", lambda: native)

    def compile_kernel(compiler, *args, **kwargs):
        if compiler.phase == "task":
            if getattr(state, "rows", False):
                phase = "direct"
            else:
                phase = "warp" if kwargs["block_size"] == 32 else "cta"
        else:
            phase = compiler.phase
        return _kernel(phase)

    monkeypatch.setattr(
        _execution,
        "_compile_once",
        compile_kernel,
    )
    monkeypatch.setattr(_execution, "_lease", lease)
    monkeypatch.setattr(_execution, "_target", lambda *args: "sm_86")
    monkeypatch.setattr(
        _execution,
        "_target_description",
        lambda: types.SimpleNamespace(
            subgroup_width=32,
            cta_block_threads=128,
            persistent_block_threads=128,
            default_warp_max_elements=32,
            default_cta_chunk_elements=128,
        ),
    )
    monkeypatch.setattr(_cuda_backend, "ensure_context", lambda *args: 1)
    monkeypatch.setattr(
        _cuda_backend, "is_current_stream_capturing",
        lambda _torch: state.capturing,
    )
    monkeypatch.setattr(_cuda_backend, "_poll_deferred", lambda *a, **k: None)
    monkeypatch.setattr(
        _validation, "_validate_shapes", lambda *a, **k: (260, 2, [0, 4, 260])
    )
    monkeypatch.setattr(_plan, "_admit_program", lambda *a, **k: True)
    monkeypatch.setattr(_plan, "_selects_direct_cta", lambda *a, **k: False)
    monkeypatch.setattr(_plan, "_identity_ids", lambda *a, **k: state.tasks)
    monkeypatch.setattr(
        _plan,
        "_classification",
        lambda *a, **k: (
            [0, 4, 128, 128, 260, 0, 2], state.direct_count, 0,
            state.partial_count, state.merge_count
        ),
    )
    return state


def _launch(state, helper):
    """Exercise the real one-shot launch helper under test."""
    module = _programs._semantic_module("sum")
    if helper == "one":
        return _qualification._launch_one(
            state.torch,
            _kernel("direct"),
            module,
            (state.values, state.offsets, state.output, state.records),
            {"value_count": 260, "segment_count": 2},
            2,
            {},
        )
    if helper == "rows":
        state.rows = True
        state.torch.Tensor = _Tensor
        state.values.dim = lambda: 2
        state.values.shape = (260, 4)
        return _qualification._launch_planned_rows(
            state.values,
            state.offsets,
            state.output,
            module_text=_programs._semantic_module("sum", rank=2),
            kernel_name="segmented_sum",
        )
    return _qualification._launch_planned_reduction(
        state.values,
        state.offsets,
        state.output,
        module_text=module,
        kernel_name="segmented_sum",
    )


@pytest.mark.parametrize("helper", ["one", "mixed"])
def test_fence_error_retains_tensors_and_marks_attempted_output(
    launch_state, helper
):
    """A submitted launch remains live when its completion fence fails."""
    state = launch_state
    primary = RuntimeError("cuEventRecord failed after launch")
    state.driver_errors["record", "direct"] = primary

    with pytest.raises(RuntimeError) as caught:
        _launch(state, helper)

    assert caught.value is primary
    assert state.launched == ["direct"]
    assert state.retained == ["values", "offsets", "output", "records"] + (
        ["scratch"] if helper == "mixed" else []
    )
    assert state.output.version == 1
    assert state.released == (
        ["direct", "partial", "merge"] if helper == "mixed" else ["direct"]
    )


def test_later_phase_error_marks_the_completed_direct_output(launch_state):
    """A failing partial phase cannot hide an earlier direct output write."""
    state = launch_state
    primary = RuntimeError("partial launch failed")
    state.driver_errors["launch", "partial"] = primary

    with pytest.raises(RuntimeError) as caught:
        _launch(state, "mixed")

    assert caught.value is primary
    assert state.launched == ["direct", "partial"]
    assert state.retained == [
        "values", "offsets", "output", "records", "scratch"
    ]
    assert state.output.version == 1
    assert state.released == ["direct", "partial", "merge"]


@pytest.mark.parametrize("helper", ["one", "mixed"])
@pytest.mark.parametrize("primary_failure", [False, True])
@pytest.mark.parametrize("release_failure", [False, True])
def test_retention_failure_attempts_every_cleanup_action(
    launch_state, helper, primary_failure, release_failure
):
    """One failed allocator recording cannot skip later recording or version."""
    state = launch_state
    first_cleanup = RuntimeError("values record_stream failed")
    state.record_errors["values"] = first_cleanup
    state.record_errors["offsets"] = RuntimeError(
        "offsets record_stream failed"
    )
    if release_failure:
        state.release_errors["direct"] = RuntimeError("lease release failed")
    primary = RuntimeError("cuEventRecord failed after launch")
    if primary_failure:
        state.driver_errors["record", "direct"] = primary

    with pytest.raises(RuntimeError) as caught:
        _launch(state, helper)

    assert caught.value is (primary if primary_failure else first_cleanup)
    assert state.retained == ["values", "offsets", "output", "records"] + (
        ["scratch"] if helper == "mixed" else []
    )
    assert state.output.version == 1
    assert state.released == (
        ["direct", "partial", "merge"] if helper == "mixed" else ["direct"]
    )
    assert state.launched == (
        ["direct", "partial", "merge"]
        if helper == "mixed" and not primary_failure
        else ["direct"]
    )


@pytest.mark.parametrize("helper", ["one", "mixed"])
def test_release_failure_cannot_skip_retention_or_other_releases(
    launch_state, helper
):
    """Release all leases and preserve tensor metadata after a release error."""
    state = launch_state
    primary = RuntimeError("lease release failed")
    state.release_errors["direct"] = primary

    with pytest.raises(RuntimeError) as caught:
        _launch(state, helper)

    assert caught.value is primary
    assert state.retained == ["values", "offsets", "output", "records"] + (
        ["scratch"] if helper == "mixed" else []
    )
    assert state.output.version == 1
    assert state.released == (
        ["direct", "partial", "merge"] if helper == "mixed" else ["direct"]
    )


@pytest.mark.parametrize(
    "point", ["validation", "records", "scratch", "binding", "lease"]
)
@pytest.mark.parametrize("helper", ["mixed", "rows"])
def test_planned_preenqueue_error_only_releases_acquired_leases(
    launch_state, monkeypatch, point, helper
):
    """Rejected setup cannot imply output mutation or allocator retention."""
    state = launch_state
    primary = RuntimeError(f"{point} failed before enqueue")

    def fail(*args, **kwargs):
        raise primary

    if point == "validation":
        monkeypatch.setattr(_validation, "_validate_shapes", fail)
    elif point in {"records", "scratch"}:
        state.allocation_error = point, primary
    elif point == "binding":
        monkeypatch.setattr(_execution, "_bind", fail)
    else:
        state.lease_error = primary

    with pytest.raises(RuntimeError) as caught:
        _launch(state, helper)

    assert caught.value is primary
    assert state.launched == []
    assert state.retained == []
    assert state.output.version == 0
    assert state.released == (
        [] if point == "validation"
        else ["direct"] if point == "lease"
        else ["direct", "partial", "merge"]
    )


def test_one_binding_error_leaves_output_unmarked(launch_state, monkeypatch):
    """Binding fails before the launch acquires a lease or tries to enqueue."""
    state = launch_state
    primary = ValueError("bad launch binding")

    def fail(*args, **kwargs):
        raise primary

    monkeypatch.setattr(_execution, "_bind", fail)
    with pytest.raises(ValueError) as caught:
        _launch(state, "one")

    assert caught.value is primary
    assert state.launched == state.retained == state.released == []
    assert state.output.version == 0


def test_one_stream_error_leaves_no_lease_to_release(launch_state, monkeypatch):
    """Read the current stream before taking ownership of a kernel lease."""
    state = launch_state
    primary = RuntimeError("current stream unavailable")

    def fail():
        raise primary

    monkeypatch.setattr(state.torch.cuda, "current_stream", fail)
    with pytest.raises(RuntimeError) as caught:
        _launch(state, "one")

    assert caught.value is primary
    assert state.launched == state.retained == state.leased == []
    assert state.output.version == 0


@pytest.mark.parametrize("helper", ["mixed", "rows"])
def test_scratch_only_failure_does_not_mark_user_output(launch_state, helper):
    """A partial phase can write scratch before any output phase is tried."""
    state = launch_state
    state.direct_count = 0
    primary = RuntimeError("partial launch failed")
    state.driver_errors["launch", "partial"] = primary

    with pytest.raises(RuntimeError) as caught:
        _launch(state, helper)

    assert caught.value is primary
    assert state.launched == ["partial"]
    assert state.retained == ["values", "offsets", "records", "scratch"]
    assert state.output.version == 0
    assert state.released == ["partial", "merge"]


def test_merge_fence_error_marks_output_after_scratch_only_work(launch_state):
    """The merge begins output ownership for a batch without direct work."""
    state = launch_state
    state.direct_count = 0
    primary = RuntimeError("merge cuEventRecord failed")
    state.driver_errors["record", "merge"] = primary

    with pytest.raises(RuntimeError) as caught:
        _launch(state, "mixed")

    assert caught.value is primary
    assert state.launched == ["partial", "merge"]
    assert state.retained == [
        "values", "offsets", "output", "records", "scratch"
    ]
    assert state.output.version == 1
    assert state.released == ["partial", "merge"]


@pytest.mark.parametrize("point", ["capture", "event_allocation"])
def test_opaque_enqueue_error_conservatively_marks_attempted_output(
    launch_state, monkeypatch, point
):
    """Enqueue cannot tell its caller whether the driver accepted work."""
    state = launch_state
    primary = RuntimeError(f"{point} failed inside enqueue")
    if point == "event_allocation":
        state.event_create_error = primary
    else:
        def fail(_torch):
            raise primary

        monkeypatch.setattr(_cuda_backend, "is_current_stream_capturing", fail)

    with pytest.raises(RuntimeError) as caught:
        _launch(state, "one")

    assert caught.value is primary
    assert state.launched == []
    assert state.retained == ["values", "offsets", "output", "records"]
    assert state.output.version == 1
    assert state.released == ["direct"]


@pytest.mark.parametrize("narrowed", [False, True])
@pytest.mark.parametrize(
    "point,phase,expected_launches",
    [
        ("launch", "direct", ["direct"]),
        ("record", "direct", ["direct"]),
        ("launch", "partial", ["direct", "partial"]),
        ("record", "merge", ["direct", "partial", "merge"]),
    ],
)
def test_row_stripe_error_retains_storage_and_marks_attempted_output(
    launch_state, narrowed, point, phase, expected_launches
):
    """Keep all split-launch storage live after any attempted output write."""
    state = launch_state
    if narrowed:
        state.offsets.dtype = "int64"
    primary = RuntimeError(f"row {phase} {point} failed")
    state.driver_errors[point, phase] = primary

    with pytest.raises(RuntimeError) as caught:
        _launch(state, "rows")

    assert caught.value is primary
    assert state.launched == expected_launches
    assert state.retained == (
        ["values", "output", "records", "scratch"] if narrowed
        else ["values", "offsets", "output", "records", "scratch"]
    )
    assert state.output.version == 1
    assert state.released == ["direct", "partial", "merge"]


@pytest.mark.parametrize("primary_failure", [False, True])
@pytest.mark.parametrize("cleanup", ["retention", "release", "version"])
def test_row_stripe_cleanup_error_attempts_remaining_actions(
    launch_state, monkeypatch, primary_failure, cleanup
):
    """Preserve the first error while completing every cleanup action."""
    state = launch_state
    first_cleanup = RuntimeError(f"row {cleanup} failed")
    if cleanup == "retention":
        state.record_errors["values"] = first_cleanup
    elif cleanup == "release":
        state.release_errors["direct"] = first_cleanup
    else:
        def fail_version(tensor):
            tensor.version += 1
            raise first_cleanup

        monkeypatch.setattr(
            state.torch.autograd.graph, "increment_version", fail_version
        )
    primary = RuntimeError("row direct completion fence failed")
    if primary_failure:
        state.driver_errors["record", "direct"] = primary

    with pytest.raises(RuntimeError) as caught:
        _launch(state, "rows")

    assert caught.value is (primary if primary_failure else first_cleanup)
    assert state.retained == [
        "values", "offsets", "output", "records", "scratch"
    ]
    assert state.output.version == 1
    assert state.released == ["direct", "partial", "merge"]


@pytest.mark.parametrize("narrowed", [False, True])
def test_row_stripe_unsplit_error_retains_identity_tasks(
    launch_state, narrowed
):
    """The upload-free identity path still retains every borrowed tensor."""
    state = launch_state
    state.partial_count = state.merge_count = 0
    if narrowed:
        state.offsets.dtype = "int64"
    primary = RuntimeError("row identity fence failed")
    state.driver_errors["record", "direct"] = primary

    with pytest.raises(RuntimeError) as caught:
        _launch(state, "rows")

    assert caught.value is primary
    assert state.launched == ["direct"]
    assert state.retained == [
        "values", "records" if narrowed else "offsets", "output", "tasks"
    ]
    assert state.output.version == 1
    assert state.released == ["direct"]


def _prepare(state, path):
    """Prepare the real callable while native compilation stays replaced."""
    if path == "persistent":
        return _qualification._prepare_persistent_sum(
            state.values, state.offsets, state.output
        ).launch
    policies = _qualification._prepare_planned_reduction(
        state.values,
        state.offsets,
        state.output,
        module_text=_programs._semantic_module("sum"),
        kernel_name="segmented_sum",
    )
    return getattr(policies, path)


@pytest.mark.parametrize("path", ["warp", "cta", "mixed", "persistent"])
@pytest.mark.parametrize("shared_versions", [False, True])
def test_prepared_fence_error_retains_storage_and_tracks_output_version(
    launch_state, path, shared_versions
):
    """Fence errors retain prepared storage and advance aliased counters."""
    state = launch_state
    state.shared_versions = shared_versions
    launch = _prepare(state, path)
    before = state.output.version
    entry = "direct" if path == "mixed" else path
    primary = RuntimeError("cuEventRecord failed after prepared launch")
    state.driver_errors["record", entry] = primary

    with pytest.raises(RuntimeError) as caught:
        launch()

    assert caught.value is primary
    expected = {"values", "offsets", "output"}
    expected.add("tasks" if path in {"warp", "cta"} else "records")
    if path == "persistent":
        expected.update(("scratch", "counters"))
    assert set(state.retained) == expected
    assert state.output.version == before + 1
    assert state.offsets.version == (
        state.output.version if shared_versions else 0
    )
    assert state.released == []

    state.driver_errors.clear()
    launch()
    assert state.output.version == before + 2
    assert state.released == []
    previous_launches = state.launched.copy()
    state.offsets.version += 1
    with pytest.raises(RuntimeError, match="offsets changed after preparation"):
        launch()
    assert state.launched == previous_launches


@pytest.mark.parametrize("path", ["warp", "cta", "mixed", "persistent"])
@pytest.mark.parametrize("primary_failure", [False, True])
def test_prepared_retention_error_attempts_remaining_recording_and_version(
    launch_state, path, primary_failure
):
    """Allocator errors cannot mask launches or skip prepared version state."""
    state = launch_state
    state.shared_versions = True
    launch = _prepare(state, path)
    before = state.output.version
    first_cleanup = RuntimeError("values record_stream failed")
    state.record_errors["values"] = first_cleanup
    state.record_errors["offsets"] = RuntimeError("second recording failed")
    primary = RuntimeError("cuEventRecord failed after prepared launch")
    entry = "direct" if path == "mixed" else path
    if primary_failure:
        state.driver_errors["record", entry] = primary

    with pytest.raises(RuntimeError) as caught:
        launch()

    assert caught.value is (primary if primary_failure else first_cleanup)
    expected = {"values", "offsets", "output"}
    expected.add("tasks" if path in {"warp", "cta"} else "records")
    if path == "persistent":
        expected.update(("scratch", "counters"))
    assert expected <= set(state.retained)
    assert state.output.version == state.offsets.version == before + 1
    assert state.released == []
    state.driver_errors.clear()
    state.record_errors.clear()
    launch()
    assert state.output.version == state.offsets.version == before + 2


@pytest.mark.parametrize("direct_count", [0, 1])
def test_prepared_partial_error_marks_only_attempted_user_output(
    launch_state, direct_count
):
    """Scratch-only work leaves output unmarked; prior direct work marks it."""
    state = launch_state
    state.direct_count = direct_count
    state.shared_versions = True
    launch = _prepare(state, "mixed")
    before = state.output.version
    primary = RuntimeError("prepared partial launch failed")
    state.driver_errors["launch", "partial"] = primary

    with pytest.raises(RuntimeError) as caught:
        launch()

    assert caught.value is primary
    assert state.launched == (["direct"] if direct_count else []) + ["partial"]
    expected = {"values", "offsets", "records", "scratch"}
    if direct_count:
        expected.add("output")
    assert set(state.retained) == expected
    assert state.output.version == state.offsets.version
    assert state.output.version == before + direct_count
    assert state.released == []
    state.driver_errors.clear()
    launch()
    assert state.output.version == state.offsets.version
    assert state.output.version == before + direct_count + 1


@pytest.mark.parametrize("capturing", [False, True])
def test_persistent_failed_completion_cannot_reuse_a_stale_event(
    launch_state, capturing
):
    """A prior complete event cannot authorize reuse after recording fails."""
    state = launch_state
    launch = _prepare(state, "persistent")
    before = state.output.version
    launch()
    first_stream = state.stream
    primary = RuntimeError("persistent completion record failed")
    state.completion_record_error = primary
    with pytest.raises(RuntimeError) as caught:
        launch()

    assert caught.value is primary
    previous_launches = state.launched.copy()
    previous_resets = state.zeroed.copy()
    state.stream = types.SimpleNamespace(cuda_stream=29)
    state.capturing = capturing
    with pytest.raises(
        RuntimeError, match="uncertain|completion.*another stream"
    ):
        launch()
    assert state.launched == previous_launches
    assert state.zeroed == previous_resets
    assert state.output.version == before + 2

    state.stream = first_stream
    state.capturing = False
    state.completion_record_error = None
    state.completion_ready = False
    launch()
    state.stream = types.SimpleNamespace(cuda_stream=29)
    with pytest.raises(RuntimeError, match="in flight on another stream"):
        launch()
    state.completion_ready = True
    launch()
    assert state.output.version == before + 4
    assert state.released == []


@pytest.mark.parametrize("completion_fails", [False, True])
def test_persistent_enqueue_error_still_attempts_a_fresh_completion_fence(
    launch_state, completion_fails
):
    """Failed enqueues need a fresh fence before scratch changes streams."""
    state = launch_state
    launch = _prepare(state, "persistent")
    before = state.output.version
    primary = RuntimeError("persistent driver fence failed")
    state.driver_errors["record", "persistent"] = primary
    if completion_fails:
        state.completion_record_error = RuntimeError("completion fence failed")
    state.completion_ready = False

    with pytest.raises(RuntimeError) as caught:
        launch()

    assert caught.value is primary
    assert state.event_records[-1] == ("completion", 17)
    assert state.output.version == before + 1
    assert set(state.retained) == {
        "values", "offsets", "output", "records", "scratch", "counters"
    }
    state.driver_errors.clear()
    state.stream = types.SimpleNamespace(cuda_stream=29)
    with pytest.raises(RuntimeError, match="another stream"):
        launch()
    state.completion_ready = True
    if completion_fails:
        with pytest.raises(RuntimeError, match="uncertain|completion"):
            launch()
    else:
        launch()
    assert state.released == []


def test_persistent_counter_reset_error_does_not_attempt_output_write(
    launch_state,
):
    """A failed asynchronous reset retains counters without marking output."""
    state = launch_state
    launch = _prepare(state, "persistent")
    before = state.output.version
    primary = RuntimeError("counter reset failed")
    state.zero_error = primary
    state.completion_record_error = RuntimeError("completion record failed")

    with pytest.raises(RuntimeError) as caught:
        launch()

    assert caught.value is primary
    assert state.launched == []
    assert state.retained == ["counters"]
    assert state.output.version == before
    assert state.event_records[-1] == ("completion", 17)
    state.stream = types.SimpleNamespace(cuda_stream=29)
    with pytest.raises(RuntimeError, match="uncertain|completion"):
        launch()
    assert state.zeroed == [17]
    assert state.released == []


@pytest.mark.parametrize("failure", [None, "launch", "retention"])
def test_persistent_capture_restores_state_only_after_success(
    launch_state, failure
):
    """Successful capture keeps the protocol; failed capture blocks reuse."""
    state = launch_state
    launch = _prepare(state, "persistent")
    launch()
    before = state.output.version
    completion_records = state.event_records.copy()
    state.capturing = True
    primary = RuntimeError("captured launch failed")
    if failure == "launch":
        state.driver_errors["launch", "persistent"] = primary
    elif failure == "retention":
        state.record_errors["values"] = primary

    if failure is None:
        launch()
    else:
        with pytest.raises(RuntimeError) as caught:
            launch()
        assert caught.value is primary

    assert state.output.version == before + 1
    assert state.event_records == completion_records
    state.driver_errors.clear()
    state.record_errors.clear()
    state.capturing = False
    state.stream = types.SimpleNamespace(cuda_stream=29)
    previous_launches = state.launched.copy()
    if failure is None:
        launch()
        assert state.output.version == before + 2
    else:
        with pytest.raises(RuntimeError, match="uncertain"):
            launch()
        assert state.launched == previous_launches
    assert state.released == []


@pytest.mark.parametrize("path", ["warp", "cta", "mixed", "persistent"])
def test_prepared_rebound_storage_error_leaves_no_attempted_write(
    launch_state, path
):
    """The storage guard runs before recording, reset, or output mutation."""
    state = launch_state
    launch = _prepare(state, path)
    before = state.output.version
    state.values.pointer += 16

    with pytest.raises(RuntimeError, match="values is bound to other storage"):
        launch()

    assert state.launched == state.retained == state.zeroed == []
    assert state.output.version == before
    assert state.released == []


def test_prepared_mixed_records_each_phase_before_the_next_enqueue(
    launch_state,
):
    """Keep each phase's original allocator-recording order and call count."""
    state = launch_state
    launch = _prepare(state, "mixed")
    state.trace.clear()
    launch()

    assert state.trace == [
        ("launch", "direct"),
        ("record", "values"),
        ("record", "offsets"),
        ("record", "output"),
        ("record", "records"),
        ("launch", "partial"),
        ("record", "values"),
        ("record", "offsets"),
        ("record", "records"),
        ("record", "scratch"),
        ("launch", "merge"),
        ("record", "offsets"),
        ("record", "output"),
        ("record", "records"),
        ("record", "scratch"),
        ("version", "output"),
    ]
    assert state.released == []


def test_prepared_direct_recording_error_stops_later_phases(launch_state):
    """Failed direct retention marks output but stops partial and merge."""
    state = launch_state
    launch = _prepare(state, "mixed")
    state.trace.clear()
    before = state.output.version
    primary = RuntimeError("direct values recording failed")
    state.record_errors["values"] = primary

    with pytest.raises(RuntimeError) as caught:
        launch()

    assert caught.value is primary
    assert state.trace == [
        ("launch", "direct"),
        ("record", "values"),
        ("record", "offsets"),
        ("record", "output"),
        ("record", "records"),
        ("version", "output"),
    ]
    assert state.output.version == before + 1
    assert state.released == []


def test_prepared_partial_enqueue_error_keeps_each_phase_recording(
    launch_state,
):
    """A failed partial enqueue retains its storage after direct retention."""
    state = launch_state
    launch = _prepare(state, "mixed")
    state.trace.clear()
    before = state.output.version
    primary = RuntimeError("partial enqueue failed")
    state.driver_errors["launch", "partial"] = primary

    with pytest.raises(RuntimeError) as caught:
        launch()

    assert caught.value is primary
    assert state.trace == [
        ("launch", "direct"),
        ("record", "values"),
        ("record", "offsets"),
        ("record", "output"),
        ("record", "records"),
        ("launch", "partial"),
        ("record", "values"),
        ("record", "offsets"),
        ("record", "records"),
        ("record", "scratch"),
        ("version", "output"),
    ]
    assert state.output.version == before + 1
    assert state.released == []


@pytest.mark.parametrize("helper", ["one", "mixed"])
@pytest.mark.parametrize("failure", ["version", "enqueue", "allocator"])
def test_version_failure_preserves_error_and_releases_all_one_shot_leases(
    launch_state, monkeypatch, helper, failure
):
    """A failed version update cannot skip releases or mask an earlier error."""
    state = launch_state
    version_error = RuntimeError("output version update failed")
    primary = version_error
    attempts = []

    def fail_version(tensor):
        attempts.append(tensor)
        raise version_error

    monkeypatch.setattr(
        state.torch.autograd.graph, "increment_version", fail_version
    )
    if failure != "version":
        primary = RuntimeError(f"{failure} failed first")
        if failure == "enqueue":
            state.driver_errors["record", "direct"] = primary
        else:
            state.record_errors["values"] = primary
    state.release_errors["direct"] = RuntimeError("release failed later")

    with pytest.raises(RuntimeError) as caught:
        _launch(state, helper)

    assert caught.value is primary
    assert attempts == [state.output]
    assert state.retained == ["values", "offsets", "output", "records"] + (
        ["scratch"] if helper == "mixed" else []
    )
    assert state.released == (
        ["direct", "partial", "merge"] if helper == "mixed" else ["direct"]
    )


@pytest.mark.parametrize("failure", ["version", "enqueue", "allocator"])
def test_persistent_version_failure_preserves_fence_and_unlocks_launch(
    launch_state, monkeypatch, failure
):
    """Version errors leave a valid fence and a reusable launch lock."""
    state = launch_state
    launch = _prepare(state, "persistent")
    original_version = state.torch.autograd.graph.increment_version
    version_error = RuntimeError("output version update failed")
    primary = version_error
    attempts = []

    def fail_version(tensor):
        attempts.append(tensor)
        raise version_error

    monkeypatch.setattr(
        state.torch.autograd.graph, "increment_version", fail_version
    )
    if failure != "version":
        primary = RuntimeError(f"{failure} failed first")
        if failure == "enqueue":
            state.driver_errors["record", "persistent"] = primary
        else:
            state.record_errors["values"] = primary

    with pytest.raises(RuntimeError) as caught:
        launch()

    assert caught.value is primary
    assert attempts == [state.output]
    assert state.retained == [
        "values", "offsets", "output", "records", "scratch", "counters"
    ]
    assert state.event_records[-1] == ("completion", 17)
    assert state.released == []
    state.driver_errors.clear()
    state.record_errors.clear()
    monkeypatch.setattr(
        state.torch.autograd.graph, "increment_version", original_version
    )
    state.stream = types.SimpleNamespace(cuda_stream=23)
    launch()
    assert state.launched == ["persistent", "persistent"]
    assert state.event_records[-1] == ("completion", 23)
    assert state.released == []
