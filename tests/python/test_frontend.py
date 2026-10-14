# tests/python/test_frontend.py
"""LLVM-free tests for the public compile-only frontend API."""

import importlib.util
import inspect
import os
import pathlib
import subprocess
import sys
import types
from unittest import mock

import pytest
import swage as sw
import swage.language as lang
import swage.language as sl
from swage.language import constexpr, program_id

SIGNATURE = {"x_ptr": sl.pointer(sl.float32), "n": sl.int32}


@pytest.fixture(autouse=True)
def _no_native_bindings(monkeypatch):
    """Make `mlir_swage` unimportable, as it is on a wheel-only install.

    This tier normally runs without the native package on the path. Blocking
    the import keeps every test here honest when it is on the path too: a
    diagnostic that only the native emitter could raise would surface as the
    missing-bindings `RuntimeError` instead.
    """
    monkeypatch.setitem(sys.modules, "mlir_swage", None)


def _reason(kernel, signature=None, constexprs=None):
    """Return the diagnostic text after the location and the kernel name."""
    with pytest.raises(sw.CompilationError) as caught:
        kernel.emit_mlir(
            signature=SIGNATURE if signature is None else signature,
            constexprs={"BLOCK": 8} if constexprs is None else constexprs,
        )
    message = str(caught.value)
    assert message.startswith(f"{kernel.filename}:")
    return message.partition(f": {kernel.__name__}: ")[2]


def _assert_passes_check(kernel, signature=None, constexprs=None):
    """Assert a body is accepted, so only the native package is missing."""
    with pytest.raises(RuntimeError, match="requires the mlir_swage"):
        kernel.emit_mlir(
            signature=SIGNATURE if signature is None else signature,
            constexprs={"BLOCK": 8} if constexprs is None else constexprs,
        )


def _kernel_from_source(directory, source, module_name="generated_kernel"):
    """Import generated source from a real file and return its `kernel`."""
    path = directory / f"{module_name}.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.kernel


def test_importing_swage_does_not_import_optional_dependencies():
    """Keep the public package usable without native or PyTorch packages."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "import swage\n"
            "assert 'mlir_swage' not in sys.modules\n"
            "assert 'torch' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("failure_type", [ModuleNotFoundError, OSError])
def test_missing_native_bindings_expose_availability_error(failure_type):
    """Expose import/link failures without requiring build-tree bindings."""
    failure = failure_type("native binding unavailable")
    original_import = __import__

    def import_without_native(name, *args, **kwargs):
        if name == "mlir_swage" or name.startswith("mlir_swage."):
            raise failure
        return original_import(name, *args, **kwargs)

    @sw.jit
    def kernel():
        return

    with mock.patch("builtins.__import__", side_effect=import_without_native):
        with pytest.raises(sw.BackendUnavailableError) as caught:
            kernel.emit_mlir(signature={}, constexprs={})

    error = caught.value
    assert isinstance(error, sw.SwageError)
    assert isinstance(error, RuntimeError)
    assert error.code == "native-unavailable"
    assert error.backend == "native"
    assert isinstance(error.remediation, str)
    assert error.__cause__ is failure


@pytest.mark.parametrize(
    "symbolic_call",
    [
        lambda: sl.program_id(0),
        lambda: sl.arange(0, 1),
        lambda: sl.load(None, mask=None, other=None),
        lambda: sl.store(None, None, mask=None),
    ],
)
def test_symbolic_language_calls_fail_outside_jit(symbolic_call):
    """Reject symbolic operations instead of pretending to execute them."""
    with pytest.raises(RuntimeError, match="only available inside @swage.jit"):
        symbolic_call()


def test_symbolic_signatures_declare_the_keywords_the_grammar_requires():
    """Do not advertise a default for a keyword a kernel must pass."""
    load = inspect.signature(sl.load).parameters
    store = inspect.signature(sl.store).parameters

    for parameter in (load["mask"], load["other"], store["mask"]):
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is inspect.Parameter.empty


def test_decorating_a_kernel_does_not_execute_its_body():
    """Capture source without running arbitrary user code."""
    calls = []

    @sw.jit
    def kernel():
        calls.append("executed")

    assert calls == []
    assert kernel.__name__ == "kernel"


def test_stacked_decorator_is_rejected():
    """Reject decorator semantics that the frontend would otherwise ignore."""

    def passthrough(function):
        return function

    with pytest.raises(
        sw.CompilationError,
        match="only @swage.jit may decorate a kernel",
    ):

        @sw.jit
        @passthrough
        def kernel():
            return


@pytest.mark.parametrize(
    ("value", "reason"),
    [
        (True, "constexpr 'VALUE' must be an integer"),
        (1.5, "constexpr 'VALUE' must be an integer"),
        (1 << 63, "constexpr 'VALUE' must fit signed 64-bit"),
        (-(1 << 63) - 1, "constexpr 'VALUE' must fit signed 64-bit"),
    ],
)
def test_constexpr_values_are_validated_before_native_import(value, reason):
    """Keep unsupported constexpr values inside the diagnostic boundary."""

    @sw.jit
    def kernel(VALUE: sl.constexpr):
        return

    with pytest.raises(sw.CompilationError, match=reason):
        kernel.emit_mlir(signature={}, constexprs={"VALUE": value})


@pytest.mark.parametrize(
    ("keywords", "reason"),
    [
        ({}, "exactly one of signature or arguments is required"),
        (
            {"signature": {}, "arguments": {}},
            "exactly one of signature or arguments is required",
        ),
    ],
)
def test_emit_requires_one_runtime_input_mode(keywords, reason):
    """Reject ambiguous runtime type inputs before native imports."""

    @sw.jit
    def kernel():
        return

    with pytest.raises(sw.CompilationError, match=reason):
        kernel.emit_mlir(constexprs={}, **keywords)


def test_argument_keys_are_validated_before_importing_pytorch():
    """Report mapping mistakes without requiring the optional dependency."""

    @sw.jit
    def kernel(value):
        return

    with mock.patch("builtins.__import__", side_effect=ImportError("missing")):
        with pytest.raises(
            sw.CompilationError,
            match=(
                "arguments keys must match runtime parameters; missing: value"
            ),
        ):
            kernel.emit_mlir(arguments={}, constexprs={})


def test_missing_pytorch_has_an_installation_hint():
    """Keep optional-dependency failures inside the diagnostic boundary."""

    @sw.jit
    def kernel(value):
        return

    real_import = __import__

    def import_without_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    with mock.patch("builtins.__import__", side_effect=import_without_torch):
        with pytest.raises(sw.CompilationError) as caught:
            kernel.emit_mlir(arguments={"value": 1}, constexprs={})

    message = str(caught.value)
    assert message.startswith(f"{__file__}:")
    assert message.endswith(
        "kernel: PyTorch metadata inference requires 'swage-compiler[pytorch]'"
    )


def test_pytorch_metadata_failures_have_an_installation_hint(monkeypatch):
    """Translate dependency metadata errors into stable diagnostics."""

    class Tensor:
        @property
        def layout(self):
            raise RuntimeError("broken metadata")

    fake_torch = types.SimpleNamespace(
        Tensor=Tensor,
        float32=object(),
        strided=object(),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    @sw.jit
    def kernel(value):
        return

    with pytest.raises(sw.CompilationError) as caught:
        kernel.emit_mlir(arguments={"value": Tensor()}, constexprs={})

    assert str(caught.value).endswith(
        "kernel: could not read PyTorch metadata for parameter 'value'; "
        "install 'swage-compiler[pytorch]'"
    )


@pytest.mark.parametrize(
    "value",
    [True, -(1 << 31) - 1, 1 << 31, 1.5, object()],
)
def test_inferred_scalars_reject_unsupported_values(value, monkeypatch):
    """Accept only non-boolean Python integers in the signed i32 range."""

    class Tensor:
        pass

    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(
            Tensor=Tensor,
            float32=object(),
            strided=object(),
        ),
    )

    @sw.jit
    def kernel(runtime_value):
        return

    with pytest.raises(
        sw.CompilationError,
        match="unsupported argument for parameter 'runtime_value'",
    ):
        kernel.emit_mlir(arguments={"runtime_value": value}, constexprs={})


def test_calling_a_decorated_kernel_points_to_launch():
    """Keep direct calls unavailable after adding the explicit launch API."""

    @sw.jit
    def kernel():
        return

    with pytest.raises(RuntimeError, match=r"use kernel\.launch\(\)"):
        kernel()


def test_non_ascii_kernel_names_are_rejected_at_capture():
    """Refuse names PTX cannot represent before any native work happens."""
    with pytest.raises(
        sw.CompilationError, match="kernel name must be an ASCII identifier"
    ):

        @sw.jit
        def añadir(x_ptr):
            return


@sw.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n, BLOCK: sl.constexpr):
    """Add two vectors elementwise under a bounds mask."""
    pid = sl.program_id(0)
    offsets = pid * BLOCK + sl.arange(0, BLOCK)
    mask = offsets < n
    x = sl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = sl.load(y_ptr + offsets, mask=mask, other=0.0)
    sl.store(output_ptr + offsets, x + y, mask=mask)


ADD_SIGNATURE = {
    "x_ptr": sl.pointer(sl.float32),
    "y_ptr": sl.pointer(sl.float32),
    "output_ptr": sl.pointer(sl.float32),
    "n": sl.int32,
}


def test_body_outside_the_language_is_diagnosed_without_native_bindings():
    """Check a body with the pure package alone, as a wheel install does."""

    @sw.jit
    def bad(x_ptr, n, BLOCK: sl.constexpr):
        for i in range(3):
            x_ptr[i] = x_ptr[i] / 0
        return 42

    line = inspect.getsourcelines(bad.python_function)[1] + 2
    with pytest.raises(sw.CompilationError) as caught:
        bad.emit_mlir(signature=SIGNATURE, constexprs={"BLOCK": 8})

    assert str(caught.value) == (
        f"{__file__}:{line}:9: bad: unsupported statement 'For'"
    )


def test_accepted_body_without_native_bindings_names_the_installation_page():
    """Say the body passed and where the native build is described."""
    with pytest.raises(RuntimeError) as caught:
        add_kernel.emit_mlir(signature=ADD_SIGNATURE, constexprs={"BLOCK": 128})

    message = str(caught.value)
    assert not isinstance(caught.value, sw.CompilationError)
    assert message.startswith(
        "Swage emit_mlir() requires the mlir_swage bindings"
    )
    assert "kernel 'add_kernel' passed the language check" in message
    assert "docs/getting-started/installation.md" in message


_WHEEL_ONLY_SCRIPT = """
import importlib.util
import sys

import swage as sw
import swage.language as sl

assert importlib.util.find_spec("mlir_swage.ir") is None
SIGNATURE = {"x_ptr": sl.pointer(sl.float32), "n": sl.int32}


@sw.jit
def bad(x_ptr, n, BLOCK: sl.constexpr):
    for i in range(n):
        x_ptr[i] = x_ptr[i] * 2


@sw.jit
def good(x_ptr, n, BLOCK: sl.constexpr):
    offsets = sl.program_id(0) * BLOCK + sl.arange(0, BLOCK)
    mask = offsets < n
    x = sl.load(x_ptr + offsets, mask=mask, other=0.0)
    sl.store(x_ptr + offsets, x + x, mask=mask)


for kernel in (bad, good):
    try:
        kernel.emit_mlir(signature=SIGNATURE, constexprs={"BLOCK": 8})
    except Exception as error:
        print(f"{type(error).__name__}|{error}")
assert "mlir_swage.ir" not in sys.modules
assert "torch" not in sys.modules
"""


def test_a_process_with_only_the_pure_package_checks_kernel_bodies(tmp_path):
    """Run the check where the bindings are absent, not merely blocked.

    A checkout has the binding sources under `python/mlir_swage`, which
    import as an empty namespace package, so the script asserts that the
    native `mlir_swage.ir` module is what is missing.
    """
    script = tmp_path / "wheel_only.py"
    script.write_text(_WHEEL_ONLY_SCRIPT, encoding="utf-8")
    package_parent = pathlib.Path(sw.__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(package_parent)},
    )

    assert result.returncode == 0, result.stderr
    rejected, accepted = result.stdout.splitlines()
    assert rejected == (
        f"CompilationError|{script}:14:5: bad: unsupported statement 'For'"
    )
    assert accepted.startswith(
        "BackendUnavailableError|Swage emit_mlir() requires the mlir_swage "
        "bindings"
    )
    assert "kernel 'good' passed the language check" in accepted
    assert "docs/getting-started/installation.md" in accepted


_LOAD_OTHER_SOURCE = """
import swage as sw
import swage.language as sl


@sw.jit
def kernel(x_ptr, n, BLOCK: sl.constexpr):
    offsets = sl.program_id(0) * BLOCK + sl.arange(0, BLOCK)
    mask = offsets < n
    x = sl.load(x_ptr + offsets, mask=mask, other={other})
    sl.store(x_ptr + offsets, x, mask=mask)
"""
_HUGE_INTEGER = "1" + "0" * 400


_REJECTED_OTHERS = [
    ("1e40", "sl.load other=1e+40 is outside the float32 range"),
    ("-1e40", "sl.load other=-1e+40 is outside the float32 range"),
    (
        "3.4028236e38",
        "sl.load other=3.4028236e+38 is outside the float32 range",
    ),
    (
        _HUGE_INTEGER,
        f"sl.load other={_HUGE_INTEGER} is outside the float32 range",
    ),
    ("1e999", "sl.load other must be a finite literal"),
    ("-1e999", "sl.load other must be a finite literal"),
    ("1e-50", "sl.load other=1e-50 rounds to zero in float32"),
    ("-1e-50", "sl.load other=-1e-50 rounds to zero in float32"),
    (
        "16777217",
        "sl.load other=16777217 is an integer that float32 cannot hold exactly",
    ),
    (
        "-16777217",
        "sl.load other=-16777217 is an integer that float32 cannot "
        "hold exactly",
    ),
    ("True", "sl.load other must be a numeric literal"),
    ("+1.0", "sl.load other must be a numeric literal"),
    ("--1.0", "sl.load other must be a numeric literal"),
    ("1.0 + 1.0", "sl.load other must be a numeric literal"),
    ("n", "sl.load other must be a numeric literal"),
]


@pytest.mark.parametrize(
    ("other", "reason"),
    _REJECTED_OTHERS,
    ids=[other[:16] for other, _ in _REJECTED_OTHERS],
)
def test_load_other_rejects_what_float32_cannot_represent(
    tmp_path, other, reason
):
    """Refuse a literal whose float32 value differs in kind from the text."""
    kernel = _kernel_from_source(
        tmp_path, _LOAD_OTHER_SOURCE.format(other=other)
    )

    assert _reason(kernel) == reason


@pytest.mark.parametrize(
    "other",
    [
        "-1.0",
        "0.1",
        "-0.0",
        "3.4028235e38",
        "-3.4028235e38",
        "1e-45",
        "0",
        "-1",
        "16777216",
        "-16777216",
    ],
)
def test_load_other_accepts_what_float32_represents(tmp_path, other):
    """Accept a signed literal that rounds to a finite nonzero float32."""
    kernel = _kernel_from_source(
        tmp_path, _LOAD_OTHER_SOURCE.format(other=other)
    )

    _assert_passes_check(kernel)


def test_parameter_defaults_are_rejected():
    """Refuse a default the frontend would otherwise silently ignore."""

    @sw.jit
    def runtime_default(x_ptr, n=5):
        return

    @sw.jit
    def constexpr_default(x_ptr, n, BLOCK: sl.constexpr = 64):
        return

    suffix = (
        "has a default value; defaults are unsupported, so pass every "
        "value when the kernel is emitted or launched"
    )
    assert _reason(runtime_default, constexprs={}) == f"parameter 'n' {suffix}"
    assert _reason(constexpr_default) == f"parameter 'BLOCK' {suffix}"
    source, first = inspect.getsourcelines(constexpr_default.python_function)
    with pytest.raises(sw.CompilationError) as caught:
        constexpr_default.emit_mlir(
            signature=SIGNATURE, constexprs={"BLOCK": 128}
        )
    column = source[1].index("64") + 1
    assert str(caught.value).startswith(f"{__file__}:{first + 1}:{column}: ")


def test_foreign_parameter_annotations_are_rejected():
    """Refuse an annotation that reads as a type the kernel does not get."""

    @sw.jit
    def float_annotation(x_ptr, n: float):
        return

    @sw.jit
    def string_annotation(x_ptr, n, BLOCK: "sl.constexpr"):
        return

    @sw.jit
    def bare_marker(x_ptr, n, BLOCK: constexpr):
        return

    suffix = (
        "the only accepted annotation is constexpr through a name bound to "
        "the swage.language module, such as sl.constexpr"
    )
    assert _reason(float_annotation, constexprs={}) == (
        f"unsupported annotation 'float' on parameter 'n'; {suffix}"
    )
    assert _reason(string_annotation) == (
        "unsupported string annotation 'sl.constexpr' on parameter "
        f"'BLOCK'; {suffix}"
    )
    assert _reason(bare_marker) == (
        f"unsupported annotation 'constexpr' on parameter 'BLOCK'; {suffix}"
    )


_RETURN_ANNOTATION_SOURCE = """
import swage as sw
import swage.language as sl


@sw.jit
def kernel(x_ptr, n) -> {annotation}:
    return
"""


@pytest.mark.parametrize(
    ("annotation", "described"),
    [
        ("float", "annotation 'float'"),
        ("sl.int32", "annotation 'sl.int32'"),
        ("'None'", "string annotation 'None'"),
    ],
)
def test_return_annotations_other_than_none_are_rejected(
    tmp_path, annotation, described
):
    """Refuse a return annotation that promises a value."""
    kernel = _kernel_from_source(
        tmp_path, _RETURN_ANNOTATION_SOURCE.format(annotation=annotation)
    )

    assert _reason(kernel, constexprs={}) == (
        f"unsupported return {described}; a kernel returns nothing, so "
        "the only accepted return annotation is None"
    )


def test_the_none_return_annotation_is_accepted(tmp_path):
    """Accept the one return annotation that states what a kernel does."""
    from swage import _runtime

    kernel = _kernel_from_source(
        tmp_path, _RETURN_ANNOTATION_SOURCE.format(annotation="None")
    )

    @sw.jit
    def add_kernel(x_ptr, y_ptr, output_ptr, n, BLOCK: sl.constexpr) -> None:
        pid = sl.program_id(0)
        offsets = pid * BLOCK + sl.arange(0, BLOCK)
        mask = offsets < n
        x = sl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = sl.load(y_ptr + offsets, mask=mask, other=0.0)
        sl.store(output_ptr + offsets, x + y, mask=mask)

    _assert_passes_check(kernel, constexprs={})
    _assert_passes_check(
        add_kernel, signature=ADD_SIGNATURE, constexprs={"BLOCK": 128}
    )
    assert _runtime._launch_descriptors(add_kernel, sl.float32) == (
        "ptr<f32>",
        "ptr<f32>",
        "ptr<f32>",
        "i32",
    )


def test_index_arithmetic_on_compile_time_operands_is_range_checked():
    """Refuse a product that wraps in 64 bits where Python would not."""

    @sw.jit
    def lane_product(x_ptr, n, BLOCK: sl.constexpr):
        lanes = sl.arange(0, BLOCK)
        _ = lanes * lanes * 4611686018427387904 * 4

    @sw.jit
    def literal_product(x_ptr, n, BLOCK: sl.constexpr):
        _ = 4611686018427387904 * 4

    @sw.jit
    def literal_sum(x_ptr, n, BLOCK: sl.constexpr):
        _ = 9223372036854775807 + BLOCK

    @sw.jit
    def constexpr_product(x_ptr, n, SCALE: sl.constexpr):
        _ = SCALE * 4611686018427387904

    assert _reason(lane_product) == (
        "'*' on compile-time index operands leaves signed 64-bit; the "
        "result ranges from 0 to 225972614902942007296"
    )
    assert _reason(literal_product) == (
        "'*' on compile-time index operands leaves signed 64-bit; the "
        "result ranges from 18446744073709551616 to 18446744073709551616"
    )
    assert _reason(literal_sum) == (
        "'+' on compile-time index operands leaves signed 64-bit; the "
        "result ranges from 9223372036854775815 to 9223372036854775815"
    )
    assert _reason(constexpr_product, constexprs={"SCALE": -3}) == (
        "'*' on compile-time index operands leaves signed 64-bit; the "
        "result ranges from -13835058055282163712 to -13835058055282163712"
    )
    _assert_passes_check(constexpr_product, constexprs={"SCALE": -2})
    _assert_passes_check(constexpr_product, constexprs={"SCALE": 1})


def test_index_arithmetic_on_a_program_coordinate_is_not_range_checked():
    """Leave run-time products to the documented 64-bit wrapping."""

    @sw.jit
    def runtime_product(x_ptr, n, BLOCK: sl.constexpr):
        _ = sl.program_id(0) * 9223372036854775807 + sl.arange(0, BLOCK)

    _assert_passes_check(runtime_product)


def test_language_module_is_matched_by_object_under_any_name():
    """Accept the module under an alias for the marker and for the calls."""

    @sw.jit
    def aliased(x_ptr, n, BLOCK: lang.constexpr):
        offsets = lang.program_id(0) * BLOCK + lang.arange(0, BLOCK)
        mask = offsets < n
        x = lang.load(x_ptr + offsets, mask=mask, other=0.0)
        lang.store(x_ptr + offsets, x, mask=mask)

    @sw.jit
    def mixed(x_ptr, n, BLOCK: sl.constexpr):
        offsets = lang.program_id(0) * BLOCK + sl.arange(0, BLOCK)
        mask = offsets < n
        x = sl.load(x_ptr + offsets, mask=mask, other=0.0)
        lang.store(x_ptr + offsets, x, mask=mask)

    assert aliased.constexpr_names == {"BLOCK"}
    _assert_passes_check(aliased)
    _assert_passes_check(mixed)


def test_aliased_language_calls_are_named_as_written():
    """Report a call by the name the kernel source uses."""

    @sw.jit
    def aliased(x_ptr, n, BLOCK: lang.constexpr):
        offsets = lang.program_id(0) * BLOCK + lang.arange(0, BLOCK)
        _ = lang.load(x_ptr + offsets)

    assert _reason(aliased) == (
        "lang.load requires one address, mask=..., and other=...; both "
        "keywords are required"
    )


def test_a_different_object_spelled_sl_is_not_the_language_module():
    """Match the module object, not the conventional spelling."""

    def capture():
        sl = types.SimpleNamespace(program_id=None)

        @sw.jit
        def kernel(x_ptr):
            _ = sl.program_id(0)

        return kernel

    @sw.jit
    def parameter_shadow(lang):
        _ = lang.program_id(0)

    @sw.jit
    def local_shadow(x_ptr, BLOCK: sl.constexpr):
        lang = BLOCK
        _ = lang.program_id(0)

    suffix = (
        "a kernel can call only program_id, arange, load, and store, each "
        "through a name bound to the swage.language module, such as sl.load"
    )
    signature = {"x_ptr": sl.pointer(sl.float32)}
    assert capture().language_names == frozenset()
    assert _reason(capture(), signature=signature, constexprs={}) == (
        f"unsupported call 'sl.program_id'; {suffix}"
    )
    assert (
        _reason(parameter_shadow, signature={"lang": sl.int32}, constexprs={})
        == f"unsupported call 'lang.program_id'; {suffix}"
    )
    assert _reason(local_shadow, signature=signature) == (
        "cannot assign to 'lang'; the name is bound to the swage.language "
        "module"
    )


_ALIAS_ASSIGNMENT_SOURCE = """
import swage as sw
import swage.language as sl


@sw.jit
def kernel(x_ptr, n, BLOCK: sl.constexpr):
    {body}
"""


@pytest.mark.parametrize(
    ("body", "line"),
    [
        ("sl = 1", 8),
        ("offsets = sl.program_id(0) * BLOCK\n    sl = offsets", 9),
    ],
    ids=["first-statement", "after-a-call"],
)
def test_assigning_to_the_language_alias_is_reported_at_the_assignment(
    tmp_path, body, line
):
    """Point at the assignment, not at the annotation that reads correctly."""
    kernel = _kernel_from_source(
        tmp_path, _ALIAS_ASSIGNMENT_SOURCE.format(body=body)
    )

    with pytest.raises(sw.CompilationError) as caught:
        kernel.emit_mlir(signature=SIGNATURE, constexprs={"BLOCK": 8})

    assert kernel.constexpr_names == {"BLOCK"}
    assert str(caught.value) == (
        f"{kernel.filename}:{line}:5: kernel: cannot assign to 'sl'; the "
        "name is bound to the swage.language module"
    )


_BINDING_SOURCE = """
import swage as sw
{binding}


@sw.jit
def kernel(x_ptr):
    _ = sl.program_id(0)
"""


def test_source_digest_tracks_what_the_language_name_is_bound_to(tmp_path):
    """Keep one spelling with two meanings from sharing compiled work."""
    module = "import swage.language as sl"
    impostor = "import types\nsl = types.SimpleNamespace(program_id=None)"
    first = _kernel_from_source(
        tmp_path, _BINDING_SOURCE.format(binding=module), "first"
    )
    second = _kernel_from_source(
        tmp_path, _BINDING_SOURCE.format(binding=module), "second"
    )
    other = _kernel_from_source(
        tmp_path, _BINDING_SOURCE.format(binding=impostor), "other"
    )

    assert first.source_digest == second.source_digest
    assert first.source_digest != other.source_digest


def test_unsupported_calls_list_the_accepted_calls():
    """Name the call as written and say what a kernel may call."""

    @sw.jit
    def unknown_attribute(x_ptr, n, BLOCK: sl.constexpr):
        offsets = sl.program_id(0) * BLOCK + sl.arange(0, BLOCK)
        _ = sl.exp(offsets)

    @sw.jit
    def bare_import(x_ptr, n, BLOCK: sl.constexpr):
        _ = program_id(0)

    @sw.jit
    def dotted_path(x_ptr, n, BLOCK: sl.constexpr):
        _ = sw.language.program_id(0)

    suffix = (
        "a kernel can call only program_id, arange, load, and store, each "
        "through a name bound to the swage.language module, such as sl.load"
    )
    assert _reason(unknown_attribute) == f"unsupported call 'sl.exp'; {suffix}"
    assert _reason(bare_import) == f"unsupported call 'program_id'; {suffix}"
    assert _reason(dotted_path) == (
        f"unsupported call 'sw.language.program_id'; {suffix}"
    )


_F32_OPERATION_SOURCE = """
import swage as sw
import swage.language as sl


@sw.jit
def kernel(x_ptr, n, BLOCK: sl.constexpr):
    offsets = sl.program_id(0) * BLOCK + sl.arange(0, BLOCK)
    mask = offsets < n
    x = sl.load(x_ptr + offsets, mask=mask, other=0.0)
    sl.store(x_ptr + offsets, {expression}, mask=mask)
"""


@pytest.mark.parametrize(
    ("expression", "reason"),
    [
        (
            "x - x",
            "'-' is not supported on float vectors; only '+' and '*' are",
        ),
        (
            "x / x",
            "'/' is not supported on float vectors; only '+' and '*' are",
        ),
        (
            "x * offsets",
            "'*' on a float vector requires another float vector",
        ),
        (
            "x + offsets",
            "'+' on a float vector requires another float vector",
        ),
        (
            "x + 1.0",
            "float literal 1.0 is unsupported here; a float literal is "
            "accepted only as the other= value of load",
        ),
    ],
    ids=["sub", "div", "mul-index", "add-index", "add-literal"],
)
def test_f32_diagnostics_name_what_the_kernel_wrote(
    tmp_path, expression, reason
):
    """Say which float operation is missing instead of blaming integers."""
    kernel = _kernel_from_source(
        tmp_path, _F32_OPERATION_SOURCE.format(expression=expression)
    )

    assert _reason(kernel) == reason


def test_index_operator_diagnostics_name_the_operator_as_written():
    """Report an unsupported index operator by its source symbol."""

    @sw.jit
    def subtraction(x_ptr, n, BLOCK: sl.constexpr):
        _ = sl.arange(0, BLOCK) - 1

    @sw.jit
    def floor_division(x_ptr, n, BLOCK: sl.constexpr):
        _ = sl.arange(0, BLOCK) // 2

    suffix = "index arithmetic supports only '+' and '*'"
    assert _reason(subtraction) == f"unsupported binary operator '-'; {suffix}"
    assert _reason(floor_division) == (
        f"unsupported binary operator '//'; {suffix}"
    )


def test_arange_diagnostic_says_which_argument_is_fixed():
    """Tell a kernel that used another width name what the form requires."""

    @sw.jit
    def other_width(x_ptr, n, WIDTH: sl.constexpr):
        _ = sl.program_id(0) * WIDTH + sl.arange(0, WIDTH)

    @sw.jit
    def other_start(x_ptr, n, BLOCK: sl.constexpr):
        _ = sl.arange(1, BLOCK)

    suffix = (
        "is unsupported; the start must be the literal 0 and the end must "
        "be the compile-time parameter named BLOCK"
    )
    assert _reason(other_width, constexprs={"WIDTH": 8}) == (
        f"sl.arange(0, WIDTH) {suffix}"
    )
    assert _reason(other_start) == f"sl.arange(1, BLOCK) {suffix}"


def test_load_diagnostic_says_both_keywords_are_required():
    """Do not suggest that mask and other can be omitted."""

    @sw.jit
    def no_keywords(x_ptr, n, BLOCK: sl.constexpr):
        offsets = sl.program_id(0) * BLOCK + sl.arange(0, BLOCK)
        _ = sl.load(x_ptr + offsets)

    assert _reason(no_keywords) == (
        "sl.load requires one address, mask=..., and other=...; both "
        "keywords are required"
    )


def test_launch_validation_applies_the_parameter_checks():
    """Reject at launch what emission rejects in the parameter list."""
    from swage import _runtime

    @sw.jit
    def add_kernel(x_ptr, y_ptr, output_ptr, n: int, BLOCK: sl.constexpr):
        pid = sl.program_id(0)
        offsets = pid * BLOCK + sl.arange(0, BLOCK)
        mask = offsets < n
        x = sl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = sl.load(y_ptr + offsets, mask=mask, other=0.0)
        sl.store(output_ptr + offsets, x + y, mask=mask)

    with pytest.raises(
        sw.CompilationError,
        match="unsupported annotation 'int' on parameter 'n'",
    ):
        _runtime._launch_descriptors(add_kernel, sl.float32)
