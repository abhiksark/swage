# python/tests/mlir/test_segmented_cache.py
"""Kernel memo and launch guards of the private segmented runner."""

import threading
import time
import weakref

import pytest
import torch
from swage import _runtime
from swage import _segmented_qualification as qualification

_requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"
)
_COMPILERS = (
    "_compile_segmented_reduction_ptx",
    "_compile_fused_segmented_reduction_ptx",
    "_compile_split_partial_reduction_ptx",
    "_compile_split_merge_reduction_ptx",
    "_compile_persistent_segmented_reduction_ptx",
)
_PLANNED_COMPILES = sorted([_COMPILERS[0], *_COMPILERS[:4]])
_STALE = "offsets changed after preparation; prepare again"
_SENTINEL = -5.0


@pytest.fixture(autouse=True)
def _fresh_memo(monkeypatch):
    """Start every test from empty memos so counts ignore test order."""
    monkeypatch.setattr(qualification, "_ptx_memo", {})
    monkeypatch.setattr(
        qualification, "_load_memo", weakref.WeakKeyDictionary()
    )


class _CountingDriver:
    """Forward to the real driver while recording every module load."""

    def __init__(self, driver):
        self._driver = driver
        self.loads = []

    def load(self, ptx, kernel_name):
        self.loads.append(kernel_name)
        return self._driver.load(ptx, kernel_name)

    def __getattr__(self, name):
        return getattr(self._driver, name)


class _Counts:
    """Native compiles and driver loads observed since the last reset."""

    def __init__(self):
        self.compiles = []
        self.driver = _CountingDriver(_runtime._get_driver())

    @property
    def loads(self):
        return self.driver.loads

    def reset(self):
        self.compiles.clear()
        self.loads.clear()


@pytest.fixture
def counts(monkeypatch):
    """Wrap the native compile functions and the driver load with counters."""
    from mlir_swage._mlir_libs._swageDialectsNanobind import swage as native

    observed = _Counts()

    def counting(name, original):
        def compile_ptx(*args, **kwargs):
            observed.compiles.append(name)
            return original(*args, **kwargs)

        return compile_ptx

    for name in _COMPILERS:
        monkeypatch.setattr(native, name, counting(name, getattr(native, name)))
    monkeypatch.setattr(_runtime, "_get_driver", lambda: observed.driver)
    return observed


def _case(lengths):
    """Build position-dependent integral values with exact f32 references."""
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    values = torch.tensor(
        [index % 7 + 1 for index in range(offsets[-1])], dtype=torch.float32
    )
    sums = []
    maxima = []
    for begin, end in zip(offsets, offsets[1:]):
        segment = values[begin:end].double()
        sums.append(float(segment.sum()))
        maxima.append(float(segment.max()) if end > begin else float("-inf"))
    return (
        values.cuda(),
        torch.tensor(offsets, device="cuda", dtype=torch.int32),
        {
            "sum": torch.tensor(sums, dtype=torch.float32),
            "max": torch.tensor(maxima, dtype=torch.float32),
        },
    )


def _softmax_reference(values, offsets):
    """Compute the segmented softmax in float64."""
    host_values = values.cpu().double()
    bounds = offsets.cpu().tolist()
    pieces = [
        torch.softmax(host_values[begin:end], 0)
        for begin, end in zip(bounds, bounds[1:])
    ]
    return torch.cat(pieces).float()


def _prepare_planned(kind, values, offsets, output):
    return qualification._prepare_planned_reduction(
        values,
        offsets,
        output,
        module_text=qualification._semantic_module(kind),
        kernel_name=f"segmented_{kind}",
    )


def _assert_policies(prepared, output, expected):
    """Launch every prepared policy and require the exact reference."""
    for launch in prepared:
        output.fill_(float("nan"))
        launch()
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


def _forbid_host_readback(patch):
    """Fail on any call that would wait for the device."""

    def fail(*_args, **_kwargs):
        pytest.fail("a prepared launch must not synchronize with the device")

    for name in ("cpu", "tolist", "item", "numpy"):
        patch.setattr(torch.Tensor, name, fail)
    patch.setattr(torch.cuda, "synchronize", fail)


class _FakeCompiler:
    """A compile function that returns a distinct PTX string per call."""

    def __init__(self):
        self.calls = []

    def __call__(self, module, **options):
        self.calls.append((module, options))
        return "lowered", f"ptx{len(self.calls)}"


class _FakeDriver:
    """A driver whose current context is set by the test."""

    def __init__(self, context=1):
        self.context = context
        self.loads = []

    def current_context(self):
        return self.context

    def load(self, ptx, kernel_name):
        self.loads.append((self.context, ptx, kernel_name))
        return 100 + len(self.loads), len(self.loads)


_OPTIONS = {
    "kernel_name": "segmented_sum",
    "block_size": 128,
    "target": "sm_86",
}


def test_compile_memo_reuses_one_compile_for_one_key():
    """Return the first PTX again without calling the compiler."""
    compiler = _FakeCompiler()
    module = object()

    first = qualification._compile_once(
        compiler, "program", module=module, **_OPTIONS
    )
    second = qualification._compile_once(
        compiler, "program", module=module, **_OPTIONS
    )

    assert first == second == "ptx1"
    assert compiler.calls == [(module, _OPTIONS)]


@pytest.mark.parametrize(
    "change",
    [
        {"target": "sm_87"},
        {"block_size": 32},
        {"kernel_name": "segmented_max"},
        {"use_task_ids": True},
    ],
    ids=["target", "block-size", "kernel-name", "option"],
)
def test_compile_memo_key_separates_targets_and_options(change):
    """Compile again when the target or any code generation option differs."""
    compiler = _FakeCompiler()
    module = object()

    first = qualification._compile_once(
        compiler, "program", module=module, **_OPTIONS
    )
    changed = qualification._compile_once(
        compiler, "program", module=module, **{**_OPTIONS, **change}
    )
    repeated = qualification._compile_once(
        compiler, "program", module=module, **_OPTIONS
    )

    assert (first, changed, repeated) == ("ptx1", "ptx2", "ptx1")
    assert len(compiler.calls) == 2


def test_compile_memo_key_separates_programs_and_compile_functions():
    """Honor a different program and a replaced compile function."""
    compiler = _FakeCompiler()
    replacement = _FakeCompiler()
    module = object()

    qualification._compile_once(compiler, "sum", module=module, **_OPTIONS)
    qualification._compile_once(compiler, "max", module=module, **_OPTIONS)
    qualification._compile_once(replacement, "sum", module=module, **_OPTIONS)
    qualification._compile_once(replacement, "sum", module=module, **_OPTIONS)

    assert len(compiler.calls) == 2
    assert len(replacement.calls) == 1


def test_compile_memo_parses_the_program_only_on_a_miss():
    """Parse the semantic text for the compiler, then never again."""
    from mlir_swage import ir

    compiler = _FakeCompiler()
    seen = []

    def compile_ptx(module, **options):
        seen.append(isinstance(module, ir.Module))
        return compiler(str(module), **options)

    text = qualification._semantic_module("sum")
    first = qualification._compile_once(compile_ptx, text, **_OPTIONS)
    second = qualification._compile_once(compile_ptx, text, **_OPTIONS)

    assert first == second == "ptx1"
    assert seen == [True]
    assert "swage.reduce" in compiler.calls[0][0]


def test_compile_failure_is_not_memoized():
    """Retry a compile that raised instead of remembering the failure."""
    compiler = _FakeCompiler()
    attempts = []

    def compile_ptx(module, **options):
        attempts.append(module)
        if len(attempts) == 1:
            raise RuntimeError("compile failed")
        return compiler(module, **options)

    with pytest.raises(RuntimeError, match="compile failed"):
        qualification._compile_once(
            compile_ptx, "program", module=object(), **_OPTIONS
        )
    ptx = qualification._compile_once(
        compile_ptx, "program", module=object(), **_OPTIONS
    )

    assert ptx == "ptx1"
    assert len(attempts) == 2


def test_compile_memo_compiles_once_across_threads():
    """Hold the module lock so concurrent first uses share one compile."""
    compiler = _FakeCompiler()
    workers = 8
    barrier = threading.Barrier(workers)
    results = []

    def compile_ptx(module, **options):
        # Give every other worker time to reach the memo before it is filled.
        time.sleep(0.05)
        return compiler(module, **options)

    def prepare():
        barrier.wait()
        results.append(
            qualification._compile_once(
                compile_ptx, "program", module=object(), **_OPTIONS
            )
        )

    threads = [threading.Thread(target=prepare) for _ in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results == ["ptx1"] * workers
    assert len(compiler.calls) == 1


def test_load_memo_reuses_one_module_within_a_context():
    """Load one module per PTX and kernel name in one context."""
    driver = _FakeDriver()

    first = qualification._load_once(driver, "ptx", "segmented_sum")
    second = qualification._load_once(driver, "ptx", "segmented_sum")
    renamed = qualification._load_once(driver, "ptx", "segmented_sum__merge")
    other = qualification._load_once(driver, "other ptx", "segmented_sum")

    assert first == second == (101, 1)
    assert renamed == (102, 2)
    assert other == (103, 3)
    assert driver.loads == [
        (1, "ptx", "segmented_sum"),
        (1, "ptx", "segmented_sum__merge"),
        (1, "other ptx", "segmented_sum"),
    ]


def test_load_memo_does_not_cross_cuda_contexts():
    """Load again in a second context and keep both handles apart."""
    driver = _FakeDriver(context=1)
    first = qualification._load_once(driver, "ptx", "segmented_sum")

    driver.context = 2
    second = qualification._load_once(driver, "ptx", "segmented_sum")
    assert qualification._load_once(driver, "ptx", "segmented_sum") == second

    driver.context = 1
    assert qualification._load_once(driver, "ptx", "segmented_sum") == first

    assert first != second
    assert [load[0] for load in driver.loads] == [1, 2]


def test_load_memo_does_not_cross_drivers():
    """Keep a handle with the driver that loaded it."""
    first_driver = _FakeDriver()
    second_driver = _FakeDriver()

    qualification._load_once(first_driver, "ptx", "segmented_sum")
    qualification._load_once(second_driver, "ptx", "segmented_sum")
    qualification._load_once(second_driver, "ptx", "segmented_sum")

    assert len(first_driver.loads) == 1
    assert len(second_driver.loads) == 1


def test_driver_without_a_context_is_never_memoized():
    """Do not reuse a handle whose context the driver cannot name."""

    class _Driver:
        def __init__(self):
            self.loads = 0

        def load(self, _ptx, _kernel_name):
            self.loads += 1
            return 1, self.loads

    driver = _Driver()

    first = qualification._load_once(driver, "ptx", "segmented_sum")
    second = qualification._load_once(driver, "ptx", "segmented_sum")

    assert (first, second) == ((1, 1), (1, 2))


@pytest.mark.parametrize("name", ["values", "output"])
@pytest.mark.parametrize(
    "validate",
    [
        qualification._validate_tensors,
        qualification._validate_softmax_tensors,
    ],
    ids=["reduction", "softmax"],
)
def test_rejects_tensors_that_require_grad(validate, name):
    """Refuse a raw-pointer write into or from an autograd leaf."""
    tensors = {
        "values": torch.ones(6),
        "offsets": torch.tensor([0, 2, 6], dtype=torch.int32),
        "output": torch.zeros(6),
    }
    assert validate(**tensors, require_cuda=False) == (6, 2)
    tensors[name].requires_grad_()

    with pytest.raises(
        ValueError,
        match=(
            f"{name} must not require grad; segmented kernels write "
            "through raw pointers"
        ),
    ):
        validate(**tensors, require_cuda=False)


def test_rejects_values_computed_from_a_tensor_that_requires_grad():
    """Refuse a non-leaf value that still belongs to an autograd graph."""
    weights = torch.ones(4, requires_grad=True)

    with pytest.raises(ValueError, match="values must not require grad"):
        qualification._validate_tensors(
            weights * 2,
            torch.tensor([0, 4], dtype=torch.int32),
            torch.zeros(1),
            require_cuda=False,
        )


def test_offsets_version_detects_only_in_place_offset_writes():
    """Compare the version counter recorded for one offsets tensor."""
    offsets = torch.tensor([0, 2, 6], dtype=torch.int32)
    version = qualification._offsets_version(offsets)

    qualification._require_unchanged_offsets(offsets, version)
    offsets[1] += 1

    with pytest.raises(RuntimeError, match=_STALE):
        qualification._require_unchanged_offsets(offsets, version)


def test_offsets_version_rejects_inference_tensors():
    """Refuse offsets whose in-place changes PyTorch does not count."""
    with torch.inference_mode():
        offsets = torch.tensor([0, 2, 6], dtype=torch.int32)

    with pytest.raises(
        ValueError, match="offsets must not be an inference tensor"
    ):
        qualification._offsets_version(offsets)


@_requires_cuda
def test_second_planned_preparation_compiles_and_loads_nothing(counts):
    """Reuse every kernel of one program for new offsets, not for a new one."""
    first_case = _case([1, 33, 4097, 0, 32])
    second_case = _case([40, 2, 8193, 5])
    outputs = [torch.empty(5, device="cuda"), torch.empty(4, device="cuda")]

    first = _prepare_planned("sum", *first_case[:2], outputs[0])
    assert sorted(counts.compiles) == _PLANNED_COMPILES
    assert sorted(counts.loads) == [
        "segmented_sum",
        "segmented_sum",
        "segmented_sum",
        "segmented_sum__merge",
        "segmented_sum__partial",
    ]
    counts.reset()

    second = _prepare_planned("sum", *second_case[:2], outputs[1])
    assert counts.compiles == []
    assert counts.loads == []
    _assert_policies(first, outputs[0], first_case[2]["sum"])
    _assert_policies(second, outputs[1], second_case[2]["sum"])

    maximum = _prepare_planned("max", *second_case[:2], outputs[1])
    assert sorted(counts.compiles) == _PLANNED_COMPILES
    assert sorted(counts.loads) == [
        "segmented_max",
        "segmented_max",
        "segmented_max",
        "segmented_max__merge",
        "segmented_max__partial",
    ]
    _assert_policies(maximum, outputs[1], second_case[2]["max"])
    counts.reset()

    again = _prepare_planned("sum", *first_case[:2], outputs[0])
    assert counts.compiles == []
    assert counts.loads == []
    _assert_policies(again, outputs[0], first_case[2]["sum"])


@_requires_cuda
def test_replaced_compile_function_is_not_served_from_the_memo(monkeypatch):
    """Call a monkeypatched compiler even after the real one is memoized."""
    from mlir_swage._mlir_libs._swageDialectsNanobind import swage as native

    values, offsets, expected = _case([1, 33, 257])
    output = torch.full((3,), float("nan"), device="cuda")
    qualification.launch_gpu(values, offsets, output, "sum")
    torch.testing.assert_close(output.cpu(), expected["sum"], rtol=0, atol=0)

    def replaced(*_args, **_kwargs):
        raise RuntimeError("replaced compiler was called")

    monkeypatch.setattr(native, "_compile_segmented_reduction_ptx", replaced)

    with pytest.raises(RuntimeError, match="replaced compiler was called"):
        qualification.launch_gpu(values, offsets, output, "sum")


@_requires_cuda
def test_second_persistent_preparation_compiles_and_loads_nothing(counts):
    """Reuse the resident kernel when only the offsets change."""
    for index, lengths in enumerate([[1, 33, 4097, 0, 32], [40, 2, 8193, 5]]):
        values, offsets, expected = _case(lengths)
        output = torch.full((len(lengths),), float("nan"), device="cuda")

        prepared = qualification._prepare_persistent_sum(
            values, offsets, output, resident_blocks=3
        )
        prepared.launch()

        torch.testing.assert_close(
            output.cpu(), expected["sum"], rtol=0, atol=0
        )
        assert counts.compiles == ([_COMPILERS[4]] if index == 0 else [])
        assert counts.loads == (["segmented_sum"] if index == 0 else [])
        counts.reset()


@_requires_cuda
@pytest.mark.parametrize("kind", ["sum", "max"])
def test_second_direct_launch_compiles_and_loads_nothing(counts, kind):
    """Reuse the one-CTA kernel across calls with different offsets."""
    for index, lengths in enumerate([[1, 33, 257, 0], [129, 2, 64]]):
        values, offsets, expected = _case(lengths)
        output = torch.full((len(lengths),), float("nan"), device="cuda")

        qualification.launch_gpu(values, offsets, output, kind)

        torch.testing.assert_close(output.cpu(), expected[kind], rtol=0, atol=0)
        assert counts.compiles == ([_COMPILERS[0]] if index == 0 else [])
        assert counts.loads == ([f"segmented_{kind}"] if index == 0 else [])
        counts.reset()


@_requires_cuda
def test_direct_launch_compiles_again_for_another_block_size(counts):
    """Treat the block size as part of the compiled kernel identity."""
    values, offsets, expected = _case([1, 33, 257, 0])
    output = torch.full((4,), float("nan"), device="cuda")

    for block_size in (128, 32, 128):
        output.fill_(float("nan"))
        qualification.launch_gpu(
            values, offsets, output, "sum", block_size=block_size
        )
        torch.testing.assert_close(
            output.cpu(), expected["sum"], rtol=0, atol=0
        )

    assert counts.compiles == [_COMPILERS[0]] * 2
    assert counts.loads == ["segmented_sum"] * 2


@_requires_cuda
def test_second_softmax_launch_compiles_and_loads_nothing(counts):
    """Reuse the softmax kernel across calls with different offsets."""
    for index, lengths in enumerate([[1, 33, 257], [129, 2, 64]]):
        values, offsets, _ = _case(lengths)
        output = torch.full((values.numel(),), float("nan"), device="cuda")

        qualification.launch_softmax_gpu(values, offsets, output)

        torch.testing.assert_close(
            output.cpu(),
            _softmax_reference(values, offsets),
            rtol=1e-4,
            atol=1e-6,
        )
        assert counts.compiles == ([_COMPILERS[0]] if index == 0 else [])
        assert counts.loads == (["ragged_softmax"] if index == 0 else [])
        counts.reset()


@_requires_cuda
def test_second_task_launch_compiles_and_loads_nothing(counts):
    """Reuse the task-list kernel across calls with different offsets."""
    for index, lengths in enumerate([[1, 33, 257, 0], [129, 2, 64]]):
        values, offsets, expected = _case(lengths)
        output = torch.full((len(lengths),), float("nan"), device="cuda")
        task_ids = torch.arange(len(lengths), device="cuda", dtype=torch.int32)

        qualification._launch_segmented_sum_tasks(
            values, offsets, output, task_ids, block_size=128
        )

        torch.testing.assert_close(
            output.cpu(), expected["sum"], rtol=0, atol=0
        )
        assert counts.compiles == ([_COMPILERS[0]] if index == 0 else [])
        assert counts.loads == (["segmented_sum"] if index == 0 else [])
        counts.reset()


def _prepared_launch(policy, values, offsets, output):
    """Prepare one named policy over a batch with direct and split work."""
    if policy == "persistent":
        return qualification._prepare_persistent_sum(
            values, offsets, output, resident_blocks=3
        ).launch
    return getattr(
        qualification._prepare_planned_sum(values, offsets, output), policy
    )


@_requires_cuda
@pytest.mark.parametrize("policy", ["warp", "cta", "mixed", "persistent"])
def test_in_place_offsets_change_stops_the_next_launch(policy, monkeypatch):
    """Raise on stale offsets without launching or waiting for the device."""
    values, offsets, expected = _case([1, 33, 4097, 2])
    output = torch.full((4,), float("nan"), device="cuda")
    launch = _prepared_launch(policy, values, offsets, output)

    # Refreshing the values in place is the supported way to reuse a plan.
    values.mul_(2)
    with monkeypatch.context() as patch:
        _forbid_host_readback(patch)
        launch()
    torch.testing.assert_close(
        output.cpu(), expected["sum"] * 2, rtol=0, atol=0
    )

    # The changed offsets are still valid; only the prepared plan is stale.
    offsets[1] += 1
    output.fill_(_SENTINEL)
    torch.cuda.synchronize()
    with monkeypatch.context() as patch:
        _forbid_host_readback(patch)
        for _ in range(2):
            with pytest.raises(RuntimeError, match=_STALE):
                launch()
    torch.cuda.synchronize()

    assert output.cpu().tolist() == [_SENTINEL] * 4


@_requires_cuda
def test_in_place_offsets_change_stops_an_empty_prepared_launch():
    """Apply the same contract when the prepared batch has no segments."""
    values = torch.empty(0, device="cuda")
    offsets = torch.zeros(1, device="cuda", dtype=torch.int32)
    output = torch.empty(0, device="cuda")
    planned = qualification._prepare_planned_sum(values, offsets, output)
    persistent = qualification._prepare_persistent_sum(values, offsets, output)
    launches = [*planned, persistent.launch]
    assert [launch() for launch in launches] == [None] * 4

    offsets.zero_()

    for launch in launches:
        with pytest.raises(RuntimeError, match=_STALE):
            launch()


@_requires_cuda
@pytest.mark.parametrize(
    "prepare",
    [
        qualification._prepare_planned_sum,
        qualification._prepare_persistent_sum,
    ],
    ids=["planned", "persistent"],
)
def test_preparation_rejects_inference_offsets(prepare, counts):
    """Refuse offsets that cannot be checked at launch, before any work."""
    values = torch.ones(33, device="cuda")
    with torch.inference_mode():
        offsets = torch.tensor([0, 33], device="cuda", dtype=torch.int32)
    output = torch.empty(1, device="cuda")

    with pytest.raises(
        ValueError, match="offsets must not be an inference tensor"
    ):
        prepare(values, offsets, output)

    assert counts.compiles == []
    assert counts.loads == []


def _launch_sum(values, offsets, output, block_size):
    qualification.launch_gpu(values, offsets, output, "sum", block_size)


def _launch_softmax(values, offsets, output, block_size):
    qualification.launch_softmax_gpu(values, offsets, output, block_size)


_LAUNCHERS = pytest.mark.parametrize(
    "launch", [_launch_sum, _launch_softmax], ids=["sum", "softmax"]
)


@_requires_cuda
@_LAUNCHERS
@pytest.mark.parametrize("block_size", [65, 96, 129, 160, 513, 992])
def test_rejects_a_block_size_without_a_power_of_two_warp_count(
    launch, block_size, counts
):
    """Refuse three, five, or more odd warps before compiling anything."""
    values, offsets, _ = _case([1, 33, 257])
    output = torch.full((values.numel(),), _SENTINEL, device="cuda")

    with pytest.raises(
        ValueError,
        match=(
            "block size must give a power-of-two warp count, "
            f"got {block_size}$"
        ),
    ):
        launch(values, offsets, output, block_size)

    assert counts.compiles == []
    assert counts.loads == []
    assert output.cpu().tolist() == [_SENTINEL] * values.numel()


@_requires_cuda
@_LAUNCHERS
@pytest.mark.parametrize(
    "block_size", [1, 31, 32, 33, 40, 64, 97, 100, 128, 256, 512, 1024]
)
def test_admits_a_block_size_with_a_power_of_two_warp_count(
    launch, block_size
):
    """Accept one, two, four, and more warps, including a partial last warp.

    An empty batch returns before compilation, so this checks the guard
    alone and leaves kernel exactness at unusual sizes to the compiler tests.
    """
    values = torch.empty(0, device="cuda")
    offsets = torch.zeros(1, device="cuda", dtype=torch.int32)
    output = torch.empty(0, device="cuda")

    launch(values, offsets, output, block_size)


@_requires_cuda
def test_block_size_128_launches_both_helpers(counts):
    """Keep the four-warp default working through both launch helpers."""
    values, offsets, expected = _case([1, 33, 257])
    reduced = torch.full((3,), float("nan"), device="cuda")
    normalized = torch.full((values.numel(),), float("nan"), device="cuda")

    _launch_sum(values, offsets, reduced, 128)
    _launch_softmax(values, offsets, normalized, 128)

    torch.testing.assert_close(reduced.cpu(), expected["sum"], rtol=0, atol=0)
    torch.testing.assert_close(
        normalized.cpu(),
        _softmax_reference(values, offsets),
        rtol=1e-4,
        atol=1e-6,
    )
    assert counts.compiles == [_COMPILERS[0]] * 2
