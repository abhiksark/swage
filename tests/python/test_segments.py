# tests/python/test_segments.py
"""LLVM-free tests for the public segmented calls.

These run on a wheel-only install: no native bindings, no GPU, and a small
stand-in for PyTorch. They pin the public names, the order of the checks a
call makes before it needs the native build, and the errors of an install
that cannot run a segmented call.
"""

import inspect
import pathlib
import re
import subprocess
import sys
import types

import pytest
import swage

FUNCTIONS = [swage.segment_reduce, swage.segment_softmax]


@pytest.fixture(autouse=True)
def _no_native_bindings(monkeypatch):
    """Make `mlir_swage` unimportable, as it is on a wheel-only install."""
    monkeypatch.setitem(sys.modules, "mlir_swage", None)


class _Tensor:
    """The metadata of one tensor, as the checks read it.

    A tensor of rank two has `count` rows of one column, unless `columns`
    gives it another width.
    """

    def __init__(
        self,
        torch,
        count,
        *,
        dtype=None,
        pointer=0x1000,
        rank=1,
        columns=1,
        contiguous=True,
        negative=False,
        requires_grad=False,
        device="cuda:0",
    ):
        self.dtype = torch.float32 if dtype is None else dtype
        self.device = device
        self.requires_grad = requires_grad
        self.shape = (count, columns, *(1,) * (rank - 2))[:rank]
        self._count = count * (columns if rank > 1 else 1)
        self._pointer = pointer
        self._rank = rank
        self._contiguous = contiguous
        self._negative = negative

    def dim(self):
        return self._rank

    def numel(self):
        return self._count

    def is_contiguous(self):
        return self._contiguous

    def is_neg(self):
        return self._negative

    def is_conj(self):
        return False

    def element_size(self):
        return 4

    def data_ptr(self):
        return self._pointer

    def record_stream(self, stream):
        raise AssertionError("a tensor was retained without a launch")


class _Dtype:
    """A stand-in for one tensor dtype, which prints as PyTorch prints it."""

    def __init__(self, name):
        self._name = name

    def __repr__(self):
        return self._name


def _fake_torch(monkeypatch, version="2.6.0"):
    """Install a PyTorch stand-in that passes the launch requirements."""
    torch = types.ModuleType("torch")
    torch.__version__ = version
    torch.float32 = _Dtype("torch.float32")
    torch.float64 = _Dtype("torch.float64")
    torch.int32 = _Dtype("torch.int32")
    torch.Tensor = _Tensor
    torch.autograd = types.SimpleNamespace(
        graph=types.SimpleNamespace(increment_version=lambda tensor: None)
    )
    # The calls wrap their bodies against `torch.compile`; nothing compiles.
    torch.compiler = types.SimpleNamespace(disable=lambda function: function)
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


def _inputs(torch):
    """Return six values and the offsets of four segments."""
    values = _Tensor(torch, 6, pointer=0x1000)
    offsets = _Tensor(torch, 5, dtype=torch.int32, pointer=0x2000)
    return values, offsets


def _call(function, values, offsets, **keywords):
    """Call either public function with the arguments both take."""
    if function is swage.segment_reduce:
        return function(values, offsets, "sum", **keywords)
    return function(values, offsets, **keywords)


def _result_count(function):
    """Return the result size of a call on `_inputs`."""
    return 4 if function is swage.segment_reduce else 6


def test_swage_exports_the_two_segmented_calls():
    """Publish the two calls and nothing else of the segmented runner."""
    assert swage.__all__ == [
        "CompilationError",
        "jit",
        "segment_reduce",
        "segment_softmax",
    ]
    assert swage.segment_reduce.__module__ == "swage._segments"
    assert swage.segment_softmax.__module__ == "swage._segments"


def test_segmented_calls_have_the_documented_signatures():
    """Keep `out` keyword-only and without a second optional argument."""
    assert str(inspect.signature(swage.segment_reduce)) == (
        "(values, offsets, kind, *, out=None)"
    )
    assert str(inspect.signature(swage.segment_softmax)) == (
        "(values, offsets, *, out=None)"
    )


def test_importing_the_segmented_calls_stays_light():
    """Import no PyTorch, numpy, or native bindings with the package."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "import swage\n"
            "swage.segment_reduce, swage.segment_softmax\n"
            "for name in ('torch', 'numpy', 'mlir_swage'):\n"
            "    assert name not in sys.modules, name",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("function", FUNCTIONS)
def test_segmented_calls_require_pytorch(function, monkeypatch):
    """Name the extra to install when PyTorch is missing."""
    monkeypatch.setitem(sys.modules, "torch", None)

    with pytest.raises(
        RuntimeError,
        match=r"requires PyTorch; install 'swage-compiler\[pytorch\]'$",
    ):
        _call(function, object(), object())


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("version", ["2.5.1", "2.5.1+cu124", "1.13.1"])
def test_segmented_calls_reject_a_pytorch_below_the_floor(
    function, version, monkeypatch
):
    """Apply the floor of `launch()` before looking at an argument."""
    _fake_torch(monkeypatch, version=version)

    with pytest.raises(
        RuntimeError,
        match=(
            "requires PyTorch 2.6 or newer; found PyTorch "
            f"{re.escape(version)}$"
        ),
    ):
        _call(function, object(), object())


@pytest.mark.parametrize("function", FUNCTIONS)
def test_segmented_calls_name_the_installation_page_without_bindings(
    function, monkeypatch
):
    """Fail a wheel-only install with the missing-bindings message.

    The arguments pass every check that needs no native build, so the
    error is the one a correct call meets on the published wheel.
    """
    torch = _fake_torch(monkeypatch)
    values, offsets = _inputs(torch)
    out = _Tensor(torch, _result_count(function), pointer=0x3000)

    for keywords in ({}, {"out": out}):
        with pytest.raises(
            RuntimeError,
            match=(
                f"^Swage {function.__name__}\\(\\) requires the build-tree "
                "mlir_swage bindings, which the swage-compiler wheel does "
                "not include; nothing was launched. See "
                "docs/getting-started/installation.md in "
                "https://github.com/abhiksark/swage for the native build$"
            ),
        ):
            _call(function, values, offsets, **keywords)


def _importable_bindings(monkeypatch):
    """Make the import that the bindings check performs succeed."""
    native = types.ModuleType("mlir_swage._mlir_libs._swageDialectsNanobind")
    native.swage = types.SimpleNamespace()
    libraries = types.ModuleType("mlir_swage._mlir_libs")
    libraries._swageDialectsNanobind = native
    package = types.ModuleType("mlir_swage")
    package._mlir_libs = libraries
    for module in (package, libraries, native):
        monkeypatch.setitem(sys.modules, module.__name__, module)


@pytest.mark.parametrize("function", FUNCTIONS)
def test_segmented_calls_name_numpy_when_it_is_missing(function, monkeypatch):
    """Say that numpy is required, before any device work.

    The calls copy the offsets into a numpy array on the host. An install
    with PyTorch and the bindings and without numpy would otherwise fail
    inside PyTorch, with a message that names neither the call nor a
    remedy.
    """
    torch = _fake_torch(monkeypatch)
    values, offsets = _inputs(torch)
    _importable_bindings(monkeypatch)
    monkeypatch.setitem(sys.modules, "numpy", None)

    with pytest.raises(
        RuntimeError,
        match=(
            f"^Swage {function.__name__}\\(\\) requires numpy, which "
            "cannot be imported; nothing was launched. Install "
            "'swage-compiler\\[pytorch\\]', which declares it. See "
            "docs/getting-started/installation.md in "
            "https://github.com/abhiksark/swage for the requirements$"
        ),
    ):
        _call(function, values, offsets)


@pytest.mark.parametrize("function", FUNCTIONS)
def test_missing_bindings_are_reported_before_missing_numpy(
    function, monkeypatch
):
    """Name the native build first: its requirements bring numpy along."""
    torch = _fake_torch(monkeypatch)
    values, offsets = _inputs(torch)
    monkeypatch.setitem(sys.modules, "numpy", None)

    with pytest.raises(RuntimeError, match="requires the build-tree"):
        _call(function, values, offsets)


def test_the_pytorch_extra_declares_numpy():
    """Install numpy with the extra that the calls name."""
    project = pathlib.Path(__file__).parents[2] / "pyproject.toml"
    extra = re.search(r"^pytorch = \[(.*)\]$", project.read_text(), re.M)

    assert extra is not None
    assert [name.strip() for name in extra[1].split(",")] == [
        '"torch>=2.6"',
        '"numpy"',
    ]


@pytest.mark.parametrize("kind", ["median", "prod", "MIN", "", None, 1])
def test_segment_reduce_rejects_an_unsupported_kind_without_bindings(
    kind, monkeypatch
):
    """Report a wrong kind on a wheel-only install too."""
    torch = _fake_torch(monkeypatch)
    values, offsets = _inputs(torch)

    with pytest.raises(
        ValueError,
        match=f"^kind must be 'sum', 'max', 'min', or 'mean', got {kind!r}$",
    ):
        swage.segment_reduce(values, offsets, kind)


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("name", ["values", "offsets"])
def test_segmented_calls_reject_an_input_that_is_not_a_tensor(
    function, name, monkeypatch
):
    """Never convert a list."""
    torch = _fake_torch(monkeypatch)
    arguments = dict(zip(("values", "offsets"), _inputs(torch)))
    arguments[name] = [0, 1, 2]

    with pytest.raises(TypeError, match=f"^{name} must be a torch.Tensor$"):
        _call(function, arguments["values"], arguments["offsets"])


@pytest.mark.parametrize("function", FUNCTIONS)
def test_segmented_calls_reject_values_that_require_grad(
    function, monkeypatch
):
    """Refuse a gradient the call cannot record, and name the remedy."""
    torch = _fake_torch(monkeypatch)
    _, offsets = _inputs(torch)
    values = _Tensor(torch, 6, requires_grad=True)

    with pytest.raises(
        ValueError,
        match=(
            "^values must not require grad; a segmented call records no "
            r"gradient, so pass values.detach\(\)$"
        ),
    ):
        _call(function, values, offsets)


def _wrong_out(torch, case, count):
    """Build one `out` the contract refuses for a result of `count`."""
    if case == "list":
        return [0.0] * count
    if case == "dtype":
        return _Tensor(torch, count, dtype=torch.float64, pointer=0x3000)
    if case == "rank":
        return _Tensor(torch, count, rank=2, pointer=0x3000)
    if case == "longer":
        return _Tensor(torch, count + 1, pointer=0x3000)
    if case == "shorter":
        return _Tensor(torch, count - 1, pointer=0x3000)
    if case == "strided":
        return _Tensor(torch, count, contiguous=False, pointer=0x3000)
    if case == "negated":
        return _Tensor(torch, count, negative=True, pointer=0x3000)
    if case == "requires-grad":
        return _Tensor(torch, count, requires_grad=True, pointer=0x3000)
    if case == "device":
        return _Tensor(torch, count, device="cpu", pointer=0x3000)
    if case == "values":
        # The last value and the first result element share four bytes.
        return _Tensor(torch, count, pointer=0x1000 + 5 * 4)
    assert case == "offsets"
    return _Tensor(torch, count, pointer=0x2000 - (count - 1) * 4)


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize(
    ("case", "error", "message"),
    [
        ("list", TypeError, "out must be a torch.Tensor or None$"),
        (
            "dtype",
            TypeError,
            "out must have the dtype of values, torch.float32$",
        ),
        ("rank", TypeError, "out must have rank one$"),
        ("longer", ValueError, "out must have exactly [46] elements, one per"),
        ("shorter", ValueError, "out must have exactly [46] elements, one per"),
        ("strided", ValueError, "out must be contiguous$"),
        ("negated", ValueError, "out must not be a lazy negation view"),
        ("requires-grad", ValueError, "out must not require grad"),
        (
            "device",
            ValueError,
            "out must be on the device of values: found cpu, values are on "
            "cuda:0$",
        ),
        ("values", ValueError, "out must not overlap values$"),
        ("offsets", ValueError, "out must not overlap offsets$"),
    ],
)
def test_segmented_calls_reject_an_out_outside_the_contract(
    function, case, error, message, monkeypatch
):
    """Name `out` in every error about the result tensor."""
    torch = _fake_torch(monkeypatch)
    values, offsets = _inputs(torch)
    out = _wrong_out(torch, case, _result_count(function))

    with pytest.raises(error, match=f"^{message}"):
        _call(function, values, offsets, out=out)


@pytest.mark.parametrize("function", FUNCTIONS)
def test_an_out_that_ends_where_an_input_begins_does_not_overlap(
    function, monkeypatch
):
    """Compare half-open byte ranges, so adjacent buffers are admitted."""
    torch = _fake_torch(monkeypatch)
    values, offsets = _inputs(torch)
    count = _result_count(function)
    before_values = _Tensor(torch, count, pointer=0x1000 - 4 * count)
    after_values = _Tensor(torch, count, pointer=0x1000 + 4 * 6)

    for out in (before_values, after_values):
        # The call passes the `out` checks and stops at the missing build.
        with pytest.raises(RuntimeError, match="requires the build-tree"):
            _call(function, values, offsets, out=out)


def test_a_float64_reduction_takes_a_float64_out(monkeypatch):
    """Require the dtype of the values, which a reduction never casts."""
    torch = _fake_torch(monkeypatch)
    _, offsets = _inputs(torch)
    values = _Tensor(torch, 6, dtype=torch.float64, pointer=0x1000)
    narrow = _Tensor(torch, 4, pointer=0x3000)
    wide = _Tensor(torch, 4, dtype=torch.float64, pointer=0x3000)

    with pytest.raises(
        TypeError, match="^out must have the dtype of values, torch.float64$"
    ):
        swage.segment_reduce(values, offsets, "sum", out=narrow)
    # A float64 out passes the `out` checks and stops at the missing build.
    with pytest.raises(RuntimeError, match="requires the build-tree"):
        swage.segment_reduce(values, offsets, "sum", out=wide)


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("rank", [0, 3])
def test_segmented_calls_reject_values_of_another_rank_without_bindings(
    rank, function, monkeypatch
):
    """Name the two ranks of values on a wheel-only install too."""
    torch = _fake_torch(monkeypatch)
    _, offsets = _inputs(torch)
    values = _Tensor(torch, 6, rank=rank)

    with pytest.raises(TypeError, match="^values must have rank one or two$"):
        _call(function, values, offsets)


@pytest.mark.parametrize(
    ("out", "found"),
    [
        ({"count": 4, "rank": 2, "columns": 2}, r"\(4, 2\)"),
        ({"count": 5, "rank": 2, "columns": 3}, r"\(5, 3\)"),
        ({"count": 12}, r"\(12,\)"),
        ({"count": 4, "rank": 3, "columns": 3}, r"\(4, 3, 1\)"),
    ],
)
def test_out_of_rows_has_one_row_per_segment_and_one_column_per_feature(
    out, found, monkeypatch
):
    """State the shape a result of `[N, D]` values has, and the one found."""
    torch = _fake_torch(monkeypatch)
    _, offsets = _inputs(torch)
    values = _Tensor(torch, 6, rank=2, columns=3)
    count = out.pop("count")

    with pytest.raises(
        ValueError,
        match=(
            r"^out must have shape \(4, 3\), one row per segment and one "
            rf"column per feature; found {found}$"
        ),
    ):
        swage.segment_reduce(
            values, offsets, "sum", out=_Tensor(torch, count, **out)
        )


@pytest.mark.parametrize(
    ("out", "found"),
    [
        ({"count": 6, "rank": 2, "columns": 2}, r"\(6, 2\)"),
        ({"count": 4, "rank": 2, "columns": 3}, r"\(4, 3\)"),
        ({"count": 18}, r"\(18,\)"),
        ({"count": 6, "rank": 3, "columns": 3}, r"\(6, 3, 1\)"),
    ],
)
def test_a_softmax_out_of_rows_has_the_shape_of_the_values(
    out, found, monkeypatch
):
    """State the shape a softmax of `[N, D]` values has, and the one found."""
    torch = _fake_torch(monkeypatch)
    _, offsets = _inputs(torch)
    values = _Tensor(torch, 6, rank=2, columns=3)
    count = out.pop("count")

    with pytest.raises(
        ValueError,
        match=(
            r"^out must have shape \(6, 3\), the shape of values; found "
            rf"{found}$"
        ),
    ):
        swage.segment_softmax(values, offsets, out=_Tensor(torch, count, **out))


def test_rank_two_values_reach_the_missing_bindings_error(monkeypatch):
    """Check `[N, D]` values and their out before the bindings.

    A reduction takes an `[S, D]` out and a softmax an `[N, D]` one.
    """
    torch = _fake_torch(monkeypatch)
    _, offsets = _inputs(torch)
    values = _Tensor(torch, 6, rank=2, columns=3)
    out = _Tensor(torch, 4, rank=2, columns=3, pointer=0x9000)
    weights = _Tensor(torch, 6, rank=2, columns=3, pointer=0x9000)

    for keywords in ({}, {"out": out}):
        with pytest.raises(RuntimeError, match="requires the build-tree"):
            swage.segment_reduce(values, offsets, "mean", **keywords)
    for keywords in ({}, {"out": weights}):
        with pytest.raises(RuntimeError, match="requires the build-tree"):
            swage.segment_softmax(values, offsets, **keywords)


def test_out_size_names_what_one_element_belongs_to(monkeypatch):
    """Say whether the result has one element per segment or per value."""
    torch = _fake_torch(monkeypatch)
    values, offsets = _inputs(torch)
    out = _Tensor(torch, 5, pointer=0x3000)

    with pytest.raises(
        ValueError,
        match="^out must have exactly 4 elements, one per segment; found 5$",
    ):
        swage.segment_reduce(values, offsets, "max", out=out)
    with pytest.raises(
        ValueError,
        match="^out must have exactly 6 elements, one per value; found 5$",
    ):
        swage.segment_softmax(values, offsets, out=out)
