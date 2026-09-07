# python/tests/mlir/test_low_precision_runtime.py
"""Low-precision numerical and specialization contracts on CPU and CUDA."""

import pytest
import swage as sw
import swage.language as sl

torch = pytest.importorskip("torch")
pytest.importorskip("mlir_swage._mlir_libs._swageDialectsNanobind")


@sw.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n, BLOCK: sl.constexpr):
    """Add equal-dtype vectors without promoting their storage."""
    pid = sl.program_id(0)
    offsets = pid * BLOCK + sl.arange(0, BLOCK)
    mask = offsets < n
    x = sl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = sl.load(y_ptr + offsets, mask=mask, other=0.0)
    sl.store(output_ptr + offsets, x + y, mask=mask)


@pytest.fixture(params=("cpu", "cuda"))
def backend(request):
    """Exercise CPU independently of CUDA availability."""
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip(reason="CUDA unavailable")
    return request.param


def _launch(x, y, output, backend, n=None):
    if n is None:
        n = x.numel()
    add_kernel.launch(
        arguments={"x_ptr": x, "y_ptr": y, "output_ptr": output, "n": n},
        constexprs={"BLOCK": 128},
        grid=((n + 127) // 128,),
        backend=backend,
    )


def _assert_encoding(actual, expected):
    """Require exact non-NaN storage, including subnormals and signed zeros."""
    actual = actual.cpu()
    expected = expected.cpu()
    torch.testing.assert_close(
        actual.float(), expected.float(), rtol=0, atol=0, equal_nan=True
    )
    non_nan = ~torch.isnan(expected.float())
    width = expected.element_size()
    actual_bits = actual.view(torch.uint8).reshape(-1, width)
    expected_bits = expected.view(torch.uint8).reshape(-1, width)
    assert torch.equal(actual_bits[non_nan], expected_bits[non_nan])


@pytest.mark.parametrize("dtype_name", ("float8_e4m3fn", "float8_e5m2"))
def test_fp8_add_every_encoding_pair(backend, dtype_name):
    """Cover all 65,536 FP8 pairs, including rounding and special values."""
    dtype = getattr(torch, dtype_name)
    encodings = torch.arange(256, dtype=torch.int16).to(torch.uint8)
    x_cpu = encodings.repeat_interleave(256).view(dtype)
    y_cpu = encodings.repeat(256).view(dtype)
    expected = (x_cpu.float() + y_cpu.float()).to(dtype)
    x = x_cpu.to(backend)
    y = y_cpu.to(backend)
    output = torch.empty_like(x)

    _launch(x, y, output, backend)

    _assert_encoding(output, expected)


def test_fp16_add_encodings_and_rounding_boundaries(backend):
    """Preserve half encodings and round normal/subnormal/overflow ties."""
    encodings = torch.arange(65536, dtype=torch.int32).to(torch.int16)
    values = encodings.view(torch.float16)
    left = torch.tensor(
        [
            1.0,
            1.0 + 2**-10,
            2**-14,
            2**-24,
            65504.0,
            65504.0,
            -65504.0,
            -65504.0,
            0.0,
            -0.0,
            float("inf"),
            float("nan"),
        ],
        dtype=torch.float16,
    )
    right = torch.tensor(
        [
            2**-11,
            2**-11,
            -(2**-24),
            2**-24,
            8.0,
            16.0,
            -8.0,
            -16.0,
            -0.0,
            -0.0,
            -float("inf"),
            1.0,
        ],
        dtype=torch.float16,
    )
    x_cpu = torch.cat((values, left))
    y_cpu = torch.cat((torch.zeros_like(values), right))
    expected = (x_cpu.float() + y_cpu.float()).to(torch.float16)
    x = x_cpu.to(backend)
    y = y_cpu.to(backend)
    output = torch.empty_like(x)

    _launch(x, y, output, backend)

    _assert_encoding(output, expected)


def test_dtype_switches_preserve_strides_tail_and_in_place_add(backend):
    """One kernel must never reuse another dtype's artifact or byte stride."""
    dtypes = (
        torch.float32,
        torch.float16,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    )
    n = 129
    for dtype in (*dtypes, *reversed(dtypes)):
        storage = torch.full((n + 8,), -42.0).to(dtype).to(backend)
        output = storage[3 : 3 + n]
        x = (torch.arange(n + 2).float() % 17 / 8).to(dtype).to(backend)[1:-1]
        y = (torch.arange(n).float() % 13 / 4).to(dtype).to(backend)
        expected = storage.cpu().clone()
        expected[3 : 3 + n] = (x.cpu().float() + y.cpu().float()).to(dtype)
        _launch(x, y, output, backend)
        _assert_encoding(storage, expected)

        # A repeated same-dtype call uses current storage and permits aliasing.
        expected[3 : 3 + n] = (
            expected[3 : 3 + n].float() + y.cpu().float()
        ).to(dtype)
        _launch(output, y, output, backend)
        _assert_encoding(storage, expected)

        # Empty work still validates, but must not touch any byte.
        _launch(x, y, output, backend, n=0)
        _assert_encoding(storage, expected)


@pytest.mark.parametrize("n", (0, 3))
def test_warm_launch_rejects_mixed_dtypes_without_writes(backend, n):
    """A supported dtype is still invalid when it differs from its peers."""
    x = torch.ones(3, dtype=torch.float16, device=backend)
    output = torch.empty_like(x)
    _launch(x, x, output, backend)
    _launch(x, x, output, backend)
    output.fill_(-7)
    other = x.to(torch.float8_e4m3fn)

    with pytest.raises(TypeError):
        _launch(x, other, output, backend, n=n)

    _assert_encoding(output, torch.full((3,), -7, dtype=torch.float16))
