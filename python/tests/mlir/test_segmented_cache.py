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
    monkeypatch.setattr(
        qualification,
        "_ptx_memo",
        _runtime._BoundedCache(_runtime._CACHE_LIMIT),
    )
    monkeypatch.setattr(
        qualification, "_load_memo", weakref.WeakKeyDictionary()
    )
    # A caller's switch would make every miss in this file a refusal.
    monkeypatch.delenv("SWAGE_NO_COMPILE", raising=False)


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
        return "lowered", f"ptx{len(self.calls)}", "{}"


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


def test_load_memo_hit_does_not_read_the_ptx_text_again():
    """Look a loaded kernel up by its text without hashing the text."""

    class _Text(str):
        def encode(self, *_args, **_kwargs):
            pytest.fail("a lookup must not encode the PTX text")

    driver = _FakeDriver()
    ptx = _Text("ptx")

    first = qualification._load_once(driver, ptx, "segmented_sum")
    second = qualification._load_once(driver, ptx, "segmented_sum")
    other = qualification._load_once(driver, _Text("other"), "segmented_sum")

    assert first is second
    assert other != first
    assert [loaded[1] for loaded in driver.loads] == ["ptx", "other"]


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


class _ForbiddenLock:
    """A lock that fails the test when anything tries to take it."""

    def __enter__(self):
        pytest.fail("a memo hit took the cold-path lock")

    def __exit__(self, *_error):
        return False


def test_memo_hits_take_no_lock(monkeypatch):
    """Serve a compiled and a loaded kernel without the cold-path lock."""
    compiler = _FakeCompiler()
    driver = _FakeDriver()
    module = object()
    ptx = qualification._compile_once(
        compiler, "program", module=module, **_OPTIONS
    )
    handles = qualification._load_once(driver, ptx, "segmented_sum")
    monkeypatch.setattr(qualification, "_memo_lock", _ForbiddenLock())

    again = qualification._compile_once(
        compiler, "program", module=module, **_OPTIONS
    )

    assert again == ptx
    assert qualification._load_once(driver, ptx, "segmented_sum") == handles
    assert len(compiler.calls) == len(driver.loads) == 1


def test_memo_hits_do_not_wait_for_a_cold_compile():
    """Serve known kernels while another thread compiles a new program."""
    compiler = _FakeCompiler()
    driver = _FakeDriver()
    compiling = threading.Event()
    finish = threading.Event()

    def slow_compile(module, **options):
        compiling.set()
        # Longer than the reader below is given, so a blocked reader is
        # seen blocked before the compile ends.
        finish.wait(60)
        return compiler(module, **options)

    warm = qualification._compile_once(
        compiler, "warm", module=object(), **_OPTIONS
    )
    handles = qualification._load_once(driver, warm, "segmented_sum")
    served = []

    def serve():
        served.append(
            qualification._compile_once(
                compiler, "warm", module=object(), **_OPTIONS
            )
        )
        served.append(qualification._load_once(driver, warm, "segmented_sum"))

    cold = threading.Thread(
        target=lambda: qualification._compile_once(
            slow_compile, "cold", module=object(), **_OPTIONS
        )
    )
    reader = threading.Thread(target=serve)
    cold.start()
    try:
        started = compiling.wait(5)
        reader.start()
        reader.join(5)
        blocked = reader.is_alive()
    finally:
        finish.set()
        cold.join()
        reader.join()

    assert started
    assert not blocked
    assert served == [warm, handles]
    assert len(driver.loads) == 1


def test_compile_memo_forgets_its_oldest_program_at_the_bound(monkeypatch):
    """Keep at most the bound and compile a forgotten program again."""
    monkeypatch.setattr(qualification, "_ptx_memo", _runtime._BoundedCache(2))
    compiler = _FakeCompiler()
    module = object()

    results = [
        qualification._compile_once(compiler, text, module=module, **_OPTIONS)
        for text in ("a", "b", "a", "c", "a")
    ]

    assert results == ["ptx1", "ptx2", "ptx1", "ptx3", "ptx4"]
    assert len(qualification._ptx_memo) == 2


def test_load_memo_forgets_its_oldest_kernel_at_the_bound(monkeypatch):
    """Keep at most the bound per driver and load a forgotten kernel again."""
    monkeypatch.setattr(_runtime, "_CACHE_LIMIT", 2)
    driver = _FakeDriver()

    results = [
        qualification._load_once(driver, ptx, "segmented_sum")
        for ptx in ("a", "b", "a", "c", "a")
    ]

    assert [function for _, function in results] == [1, 2, 1, 3, 4]
    assert len(qualification._load_memo[driver]) == 2
    assert qualification._load_memo[driver].limit == 2


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
    tensors = _reduction_tensors()
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


def _reduction_tensors():
    """Return host tensors that both private validators accept."""
    return {
        "values": torch.ones(6),
        "offsets": torch.tensor([0, 2, 6], dtype=torch.int32),
        "output": torch.zeros(6),
    }


@pytest.mark.parametrize("name", ["values", "offsets", "output"])
@pytest.mark.parametrize(
    "validate",
    [
        qualification._validate_tensors,
        qualification._validate_softmax_tensors,
    ],
    ids=["reduction", "softmax"],
)
def test_rejects_lazy_negation_views(validate, name):
    """Refuse a view whose storage holds the opposite of what it shows."""
    tensors = _reduction_tensors()
    assert validate(**tensors, require_cuda=False) == (6, 2)
    tensors[name] = tensors[name]._neg_view()
    assert tensors[name].is_neg() and tensors[name].is_contiguous()

    with pytest.raises(
        ValueError,
        match=(
            f"^{name} must not be a lazy negation view; pass "
            r"tensor.resolve_neg\(\)$"
        ),
    ):
        validate(**tensors, require_cuda=False)


@pytest.mark.parametrize("name", ["values", "offsets", "output"])
def test_rejects_lazy_conjugate_views(monkeypatch, name):
    """Refuse a conjugate view, which no admitted dtype can build today."""
    tensors = _reduction_tensors()
    target = tensors[name]
    monkeypatch.setattr(
        torch.Tensor, "is_conj", lambda tensor: tensor is target
    )

    with pytest.raises(
        ValueError,
        match=(
            f"^{name} must not be a lazy conjugate view; pass "
            r"tensor.resolve_conj\(\)$"
        ),
    ):
        qualification._validate_tensors(**tensors, require_cuda=False)


def test_cpu_oracle_rejects_the_negation_view_the_review_summed():
    """Refuse the view of ones that the kernels would sum as plus four."""
    view = torch.ones(4)._neg_view()
    offsets = torch.tensor([0, 4], dtype=torch.int32)
    assert float(view.sum()) == -4.0

    with pytest.raises(ValueError, match="values must not be a lazy negation"):
        qualification._validate_tensors(
            view, offsets, torch.zeros(1), require_cuda=False
        )
    with pytest.raises(ValueError, match="values must not be a lazy negation"):
        qualification.cpu_oracle(view, offsets, "sum")
    assert qualification._validate_tensors(
        view.resolve_neg(), offsets, torch.zeros(1), require_cuda=False
    ) == (4, 1)


@_requires_cuda
def test_gpu_entry_points_reject_a_negation_view(counts):
    """Refuse the view before any compile, load, or launch on the device."""
    values, offsets, _ = _case([3, 5])
    output = torch.full((8,), _SENTINEL, device="cuda")
    view = values._neg_view()
    task_ids = torch.arange(2, device="cuda", dtype=torch.int32)
    reason = "values must not be a lazy negation view"

    with pytest.raises(ValueError, match=reason):
        qualification.launch_gpu(view, offsets, output, "sum")
    with pytest.raises(ValueError, match=reason):
        qualification.launch_softmax_gpu(view, offsets, output)
    with pytest.raises(ValueError, match=reason):
        _prepare_planned("sum", view, offsets, output)
    with pytest.raises(ValueError, match=reason):
        qualification._prepare_persistent_sum(view, offsets, output)
    with pytest.raises(
        ValueError, match="task_ids must not be a lazy negation view"
    ):
        qualification._launch_segmented_sum_tasks(
            values, offsets, output, task_ids._neg_view(), block_size=32
        )

    torch.cuda.synchronize()
    assert counts.compiles == counts.loads == []
    assert torch.all(output == _SENTINEL)


@pytest.fixture
def _no_compile(monkeypatch):
    """Switch compiling off for one test, whatever the caller exported."""
    monkeypatch.setenv("SWAGE_NO_COMPILE", "1")


def test_no_compile_mode_serves_a_held_kernel_and_refuses_another(
    monkeypatch,
):
    """Return what the process compiled and raise instead of compiling."""
    compiler = _FakeCompiler()
    held = qualification._compile_once(
        compiler, "program", module=object(), **_OPTIONS
    )
    monkeypatch.setenv("SWAGE_NO_COMPILE", "1")

    again = qualification._compile_once(
        compiler, "program", module=object(), **_OPTIONS
    )
    for _ in range(2):
        with pytest.raises(
            RuntimeError,
            match=(
                "^SWAGE_NO_COMPILE=1 refuses to compile kernel "
                "'segmented_sum': this process does not hold it for block "
                "size 32 and target sm_86, and the private segmented path "
                "has no persistent cache$"
            ),
        ):
            qualification._compile_once(
                compiler,
                "program",
                module=object(),
                **{**_OPTIONS, "block_size": 32},
            )

    assert held == again == "ptx1"
    assert len(compiler.calls) == 1
    assert len(qualification._ptx_memo) == 1


def test_no_compile_mode_does_not_parse_the_program(_no_compile):
    """Refuse before the semantic text is parsed for the compiler."""
    compiler = _FakeCompiler()

    with pytest.raises(RuntimeError, match="SWAGE_NO_COMPILE=1 refuses"):
        qualification._compile_once(compiler, "not a module", **_OPTIONS)

    assert compiler.calls == []


def test_no_compile_mode_names_a_kernel_without_a_block_size(_no_compile):
    """Describe a fused, split, or persistent kernel by its target alone."""
    with pytest.raises(
        RuntimeError,
        match=(
            "refuses to compile kernel 'segmented_sum': this process does "
            "not hold it for target sm_86, and"
        ),
    ):
        qualification._compile_once(
            _FakeCompiler(),
            "program",
            module=object(),
            kernel_name="segmented_sum",
            target="sm_86",
        )


@pytest.mark.parametrize("value", ["yes", "true", "2"])
def test_compile_memo_rejects_a_mistyped_no_compile_switch(monkeypatch, value):
    """Fail on a miss instead of compiling under an unreadable switch."""
    compiler = _FakeCompiler()
    monkeypatch.setenv("SWAGE_NO_COMPILE", value)

    with pytest.raises(
        ValueError, match=f"SWAGE_NO_COMPILE must be 0 or 1; found '{value}'"
    ):
        qualification._compile_once(
            compiler, "program", module=object(), **_OPTIONS
        )

    assert compiler.calls == []


@_requires_cuda
def test_no_compile_mode_stops_every_private_entry_point(
    counts, monkeypatch
):
    """Launch kernels the process holds and refuse the ones it does not."""
    values, offsets, expected = _case([3, 5, 40])
    output = torch.full((3,), _SENTINEL, device="cuda")
    qualification.launch_gpu(values, offsets, output, "sum")
    torch.cuda.synchronize()
    counts.reset()
    monkeypatch.setenv("SWAGE_NO_COMPILE", "1")
    refusal = "SWAGE_NO_COMPILE=1 refuses to compile kernel"

    output.fill_(_SENTINEL)
    qualification.launch_gpu(values, offsets, output, "sum")
    torch.cuda.synchronize()
    torch.testing.assert_close(output.cpu(), expected["sum"], rtol=0, atol=0)

    output.fill_(_SENTINEL)
    softmax_output = torch.full((48,), _SENTINEL, device="cuda")
    with pytest.raises(RuntimeError, match=f"{refusal} 'segmented_max'"):
        qualification.launch_gpu(values, offsets, output, "max")
    with pytest.raises(RuntimeError, match=f"{refusal} 'segmented_sum'"):
        qualification.launch_gpu(values, offsets, output, "sum", block_size=64)
    with pytest.raises(RuntimeError, match=f"{refusal} 'ragged_softmax'"):
        qualification.launch_softmax_gpu(values, offsets, softmax_output)
    with pytest.raises(RuntimeError, match=f"{refusal} 'segmented_sum'"):
        _prepare_planned("sum", values, offsets, output)
    with pytest.raises(RuntimeError, match=f"{refusal} 'segmented_sum'"):
        qualification._prepare_persistent_sum(values, offsets, output)

    torch.cuda.synchronize()
    assert counts.compiles == counts.loads == []
    assert torch.all(output == _SENTINEL)
    assert torch.all(softmax_output == _SENTINEL)


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


def _write_through_data(offsets, new_offsets):
    offsets.data.copy_(new_offsets)


def _write_through_dlpack(offsets, new_offsets):
    torch.from_dlpack(offsets.__dlpack__()).copy_(new_offsets)


@_requires_cuda
@pytest.mark.parametrize("policy", ["warp", "cta", "mixed", "persistent"])
@pytest.mark.parametrize(
    "write",
    [_write_through_data, _write_through_dlpack],
    ids=["data", "dlpack"],
)
def test_uncounted_offsets_write_is_not_detected(policy, write):
    """Pin the documented limit of the stale-offsets check.

    PyTorch does not count a write through `.data` or through a DLPack
    alias, so the version counter stays put and the launch proceeds. A
    segment that the plan runs as one task is reduced over the new offsets.
    A segment that the plan split keeps its prepared ranges, so the output
    of `mixed` and of the persistent kernel matches neither layout. The
    kernels clamp every range, which test_segmented_bounds.py covers.
    """
    old_lengths, new_lengths = [10, 5000, 20, 3], [5000, 10, 3, 20]
    values, offsets, old = _case(old_lengths)
    _, new_offsets, new = _case(new_lengths)
    output = torch.full((4,), _SENTINEL, device="cuda")
    launch = _prepared_launch(policy, values, offsets, output)
    version = offsets._version

    write(offsets, new_offsets)
    launch()

    torch.cuda.synchronize()
    assert offsets._version == version
    assert torch.equal(offsets, new_offsets)
    result = output.cpu()
    # Segment 1 has 5000 elements at preparation and is the only split one.
    split = policy in ("mixed", "persistent")
    expected = new["sum"].clone()
    if split:
        expected[1] = old["sum"][1]
        assert not torch.equal(result, new["sum"])
    assert not torch.equal(result, old["sum"])
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@_requires_cuda
@pytest.mark.parametrize("policy", ["warp", "cta", "mixed", "persistent"])
def test_write_to_a_sibling_view_stops_the_launch(policy):
    """Pin the documented false refusal of the stale-offsets check.

    Views of one tensor share one version counter. A write to another view
    therefore stops the launch although the offsets did not change, and a
    clone of the offsets, which has a counter of its own, does not.
    """
    values, host_offsets, expected = _case([1, 33, 4097, 2])
    arena = torch.zeros(10, dtype=torch.int32, device="cuda")
    arena[:5] = host_offsets
    offsets, sibling = arena[:5], arena[5:]
    output = torch.full((4,), _SENTINEL, device="cuda")
    launch = _prepared_launch(policy, values, offsets, output)
    own_output = torch.full((4,), _SENTINEL, device="cuda")
    own_launch = _prepared_launch(policy, values, offsets.clone(), own_output)

    sibling.fill_(7)

    assert torch.equal(offsets, host_offsets)
    with pytest.raises(RuntimeError, match=_STALE):
        launch()
    own_launch()
    torch.cuda.synchronize()
    assert output.cpu().tolist() == [_SENTINEL] * 4
    torch.testing.assert_close(
        own_output.cpu(), expected["sum"], rtol=0, atol=0
    )


@_requires_cuda
@pytest.mark.parametrize("policy", ["warp", "cta", "mixed", "persistent"])
def test_offsets_from_outside_inference_mode_launch_inside_it(policy):
    """Admit normal offsets in inference mode, also as a clone made outside.

    Only a tensor created under `torch.inference_mode()` lacks a version
    counter. Preparing and launching inside the context is fine.
    """
    with torch.inference_mode():
        values, inference_offsets, expected = _case([1, 33, 4097, 2])
        output = torch.full((4,), _SENTINEL, device="cuda")
    offsets = inference_offsets.clone()
    assert inference_offsets.is_inference() and not offsets.is_inference()

    with torch.inference_mode():
        launch = _prepared_launch(policy, values, offsets, output)
        launch()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            output.cpu(), expected["sum"], rtol=0, atol=0
        )


_GUARD = 8


def _rebind_into_an_arena(tensor, length):
    """Rebind `tensor` to `length` elements inside a larger tensor.

    `tensor.data = other` changes the storage, the length, and even the
    dtype of a tensor object without advancing its version counter. The new
    storage lies inside an arena that is longer than the prepared tensor
    was, with guard elements before it, so whatever a launch with the
    prepared counts would read or write stays inside this test's memory.

    Returns:
        The arena, and the storage the tensor had, which the caller keeps
        alive so that its address is not handed out again.
    """
    prepared = tensor.data
    fill = 0 if tensor.dtype == torch.int32 else -7.0
    arena = torch.full(
        (prepared.numel() + 2 * _GUARD,),
        fill,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    tensor.data = arena[_GUARD:_GUARD + length]
    return arena, prepared


@_requires_cuda
@pytest.mark.parametrize("policy", ["warp", "cta", "mixed", "persistent"])
@pytest.mark.parametrize("name", ["values", "offsets", "output"])
@pytest.mark.parametrize("length", ["one element", "the prepared length"])
def test_rebound_tensor_stops_the_next_launch(
    policy, name, length, monkeypatch
):
    """Raise before anything is enqueued when a tensor has other storage."""
    values, offsets, expected = _case([1, 33, 4097, 2])
    output = torch.full((4,), _SENTINEL, device="cuda")
    tensors = {"values": values, "offsets": offsets, "output": output}
    launch = _prepared_launch(policy, values, offsets, output)
    launch()
    torch.cuda.synchronize()
    launch()
    torch.cuda.synchronize()
    torch.testing.assert_close(output.cpu(), expected["sum"], rtol=0, atol=0)
    prepared_output = output.data
    prepared_output.fill_(_SENTINEL)
    count = tensors[name].numel()

    arena, prepared = _rebind_into_an_arena(
        tensors[name], 1 if length == "one element" else count
    )
    untouched = arena.clone()
    torch.cuda.synchronize()
    with monkeypatch.context() as patch:
        _forbid_host_readback(patch)
        for _ in range(2):
            with pytest.raises(
                RuntimeError,
                match=(
                    f"^{name} is bound to other storage than at "
                    f"preparation: found {tensors[name].numel()} .* "
                    f"prepared with {count} .*; prepare again$"
                ),
            ):
                launch()
    torch.cuda.synchronize()

    assert torch.equal(arena, untouched)
    assert prepared_output.cpu().tolist() == [_SENTINEL] * 4
    assert prepared.numel() == count


@_requires_cuda
@pytest.mark.parametrize("policy", ["warp", "cta", "mixed", "persistent"])
def test_tensor_rebound_to_another_dtype_in_place_stops_the_launch(policy):
    """Refuse the prepared address and count under another element type."""
    values, offsets, _ = _case([1, 33, 4097, 2])
    output = torch.full((4,), _SENTINEL, device="cuda")
    launch = _prepared_launch(policy, values, offsets, output)
    torch.cuda.synchronize()
    address = values.data_ptr()

    values.data = values.data.view(torch.int32)
    assert (values.data_ptr(), values.numel()) == (address, 4133)

    with pytest.raises(
        RuntimeError,
        match=(
            "values is bound to other storage than at preparation: found "
            "4133 torch.int32 .* prepared with 4133 torch.float32"
        ),
    ):
        launch()
    torch.cuda.synchronize()
    assert output.cpu().tolist() == [_SENTINEL] * 4


@_requires_cuda
@pytest.mark.parametrize("policy", ["warp", "cta", "mixed", "persistent"])
def test_in_place_writes_to_values_and_output_still_launch(policy):
    """Keep a prepared launch usable while its buffers are refilled."""
    values, offsets, expected = _case([1, 33, 4097, 2])
    output = torch.full((4,), _SENTINEL, device="cuda")
    launch = _prepared_launch(policy, values, offsets, output)

    for scale in (1, 3, 5):
        values.copy_(_case([1, 33, 4097, 2])[0] * scale)
        output.fill_(_SENTINEL)
        launch()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            output.cpu(), expected["sum"] * scale, rtol=0, atol=0
        )


def _one_shot_launch(path, values, offsets, output):
    """Return a launch of one unprepared private entry point."""
    if path == "one-cta":
        return lambda: qualification.launch_gpu(values, offsets, output, "sum")
    if path == "softmax":
        return lambda: qualification.launch_softmax_gpu(
            values, offsets, output
        )
    assert path == "tasks"
    task_ids = torch.arange(
        offsets.numel() - 1, dtype=torch.int32, device="cuda"
    )
    return lambda: qualification._launch_segmented_sum_tasks(
        values, offsets, output, task_ids, block_size=32
    )


_PREPARED = ["warp", "cta", "mixed", "persistent"]
_ONE_SHOT = ["one-cta", "softmax", "tasks"]


@_requires_cuda
@pytest.mark.parametrize("path", [*_PREPARED, *_ONE_SHOT])
def test_every_launch_advances_the_output_version_only(path):
    """Tell autograd that the output was written, on every private path.

    PyTorch cannot see a kernel store. Without the advance, a backward pass
    that saved the output would use the overwritten values and return a
    wrong gradient without an error. The counters of values and offsets
    must stay put: the offsets counter is what a prepared launch compares.
    """
    values, offsets, expected = _case([1, 33, 4097, 2])
    size = values.numel() if path == "softmax" else 4
    output = torch.zeros(size, device="cuda")
    if path in _PREPARED:
        launch = _prepared_launch(path, values, offsets, output)
        # Preparation advances it once, to see whether offsets share it.
        assert output._version == 1
    else:
        launch = _one_shot_launch(path, values, offsets, output)
        assert output._version == 0
    weights = torch.ones(size, device="cuda", requires_grad=True)
    kept = [values._version, offsets._version]

    for _ in range(3):
        loss = (weights * output).sum()
        version = output._version
        launch()
        torch.cuda.synchronize()
        assert output._version > version
        with pytest.raises(
            RuntimeError, match="modified by an inplace operation"
        ):
            loss.backward()

    assert [values._version, offsets._version] == kept
    if path != "softmax":
        torch.testing.assert_close(
            output.cpu(), expected["sum"], rtol=0, atol=0
        )


@_requires_cuda
@pytest.mark.parametrize("policy", _PREPARED)
def test_offsets_that_share_the_output_version_counter_still_launch(policy):
    """Do not read the launch's own output write as changed offsets.

    Views of one tensor share one version counter, also views of another
    dtype that share no byte. Advancing the output version then advances
    the offsets version, which must not stop the next launch, while a
    write to the offsets must still stop it.
    """
    values, host_offsets, expected = _case([1, 33, 4097, 2])
    arena = torch.zeros(9, device="cuda")
    offsets = arena[:5].view(torch.int32)
    offsets.copy_(host_offsets)
    output = arena[5:]
    launch = _prepared_launch(policy, values, offsets, output)

    for _ in range(3):
        version = offsets._version
        launch()
        torch.cuda.synchronize()
        assert offsets._version == output._version > version
        torch.testing.assert_close(
            output.cpu(), expected["sum"], rtol=0, atol=0
        )

    offsets[1] += 1
    with pytest.raises(RuntimeError, match=_STALE):
        launch()


@_requires_cuda
@pytest.mark.parametrize("path", _ONE_SHOT)
def test_one_shot_launch_accepts_inference_tensors(path):
    """Launch on tensors that have no version counter to advance."""
    with torch.inference_mode():
        values, offsets, expected = _case([1, 33, 4097, 2])
        size = values.numel() if path == "softmax" else 4
        output = torch.zeros(size, device="cuda")
        assert output.is_inference()

        _one_shot_launch(path, values, offsets, output)()

        torch.cuda.synchronize()
        if path != "softmax":
            torch.testing.assert_close(
                output.cpu(), expected["sum"], rtol=0, atol=0
            )


@_requires_cuda
def test_rebound_tensor_stops_an_empty_prepared_launch():
    """Apply the same contract when the prepared batch has no segments."""
    values = torch.empty(0, device="cuda")
    offsets = torch.zeros(1, device="cuda", dtype=torch.int32)
    output = torch.empty(0, device="cuda")
    planned = qualification._prepare_planned_sum(values, offsets, output)
    persistent = qualification._prepare_persistent_sum(values, offsets, output)
    launches = [*planned, persistent.launch]
    assert [launch() for launch in launches] == [None] * 4

    output.data = torch.zeros(3, device="cuda")

    for launch in launches:
        with pytest.raises(
            RuntimeError,
            match="output is bound to other storage than at preparation",
        ):
            launch()


@_requires_cuda
def test_task_ids_must_not_overlap_the_output(counts):
    """Refuse task IDs that a kernel would overwrite while it reads them."""
    count = 1 << 10
    values = torch.ones(count, device="cuda")
    offsets = torch.arange(count + 1, dtype=torch.int32, device="cuda")
    output = torch.full((count,), _SENTINEL, device="cuda")
    arena = torch.arange(2 * count, dtype=torch.int32, device="cuda") % count
    # The same bytes as the output, read as task IDs, and every ID valid.
    shared = output.view(torch.int32)
    shared.copy_(arena[:count].flip(0))
    kept = output.clone()

    with pytest.raises(
        ValueError, match="^output must not overlap the task_ids buffer$"
    ):
        qualification._launch_segmented_sum_tasks(
            values, offsets, output, shared, block_size=32
        )
    torch.cuda.synchronize()
    assert torch.equal(output, kept)
    assert counts.compiles == counts.loads == []

    # Task IDs that only touch the output are admitted and run.
    for task_ids in (arena[:count], arena[count:]):
        output.fill_(_SENTINEL)
        qualification._launch_segmented_sum_tasks(
            values, offsets, output, task_ids, block_size=32
        )
        torch.cuda.synchronize()
        assert torch.all(output == 1.0)


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
