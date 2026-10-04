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


@sw.jit
def multiply_kernel(x_ptr, y_ptr, output_ptr, n, BLOCK: sl.constexpr):
    """Multiply equal-dtype vectors without promoting their storage."""
    pid = sl.program_id(0)
    offsets = pid * BLOCK + sl.arange(0, BLOCK)
    mask = offsets < n
    x = sl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = sl.load(y_ptr + offsets, mask=mask, other=0.0)
    sl.store(output_ptr + offsets, x * y, mask=mask)


OPERATIONS = {
    "add": (add_kernel, torch.add),
    "multiply": (multiply_kernel, torch.mul),
}


@pytest.fixture(params=("cpu", "cuda"))
def backend(request):
    """Exercise CPU independently of CUDA availability."""
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip(reason="CUDA unavailable")
    return request.param


def _launch(x, y, output, backend, n=None, operation="add", block=128):
    if n is None:
        n = x.numel()
    kernel, _ = OPERATIONS[operation]
    kernel.launch(
        arguments={"x_ptr": x, "y_ptr": y, "output_ptr": output, "n": n},
        constexprs={"BLOCK": block},
        grid=((n + block - 1) // block,),
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


def _multiply_oracle(x, y):
    """Multiply through f32 and round once to the input storage dtype."""
    return (x.float() * y.float()).to(x.dtype)


def _fp8_oracle(x, y, operation, dtype):
    """Preserve non-saturating E4M3FN overflow across PyTorch versions."""
    wide = operation(x.float(), y.float())
    expected = wide.to(dtype)
    if dtype is torch.float8_e4m3fn:
        expected.view(torch.uint8)[wide.abs() > 464.0] = 0x7F
    return expected


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("dtype_name", ("float8_e4m3fn", "float8_e5m2"))
def test_fp8_every_encoding_pair(backend, dtype_name, operation):
    """Cover all 65,536 FP8 pairs, including rounding and special values."""
    dtype = getattr(torch, dtype_name)
    encodings = torch.arange(256, dtype=torch.int16).to(torch.uint8)
    x_cpu = encodings.repeat_interleave(256).view(dtype)
    y_cpu = encodings.repeat(256).view(dtype)
    _, torch_operation = OPERATIONS[operation]
    expected = _fp8_oracle(x_cpu, y_cpu, torch_operation, dtype)
    x = x_cpu.to(backend)
    y = y_cpu.to(backend)
    output = torch.empty_like(x)

    _launch(x, y, output, backend, operation=operation)

    _assert_encoding(output, expected)


@pytest.mark.parametrize("operation", OPERATIONS)
def test_fp16_encodings_and_rounding_boundaries(backend, operation):
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
            1.0 + 2**-10,
            -(1.0 + 2**-10),
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
            1.5,
            1.5,
        ],
        dtype=torch.float16,
    )
    x_cpu = torch.cat((values, left))
    y_cpu = torch.cat((torch.zeros_like(values), right))
    _, torch_operation = OPERATIONS[operation]
    expected = torch_operation(x_cpu.float(), y_cpu.float()).to(torch.float16)
    x = x_cpu.to(backend)
    y = y_cpu.to(backend)
    output = torch.empty_like(x)

    _launch(x, y, output, backend, operation=operation)

    _assert_encoding(output, expected)


def test_multiply_fp16_every_encoding_and_factor(backend):
    """Multiply every binary16 encoding by 19 boundary factors."""
    finfo = torch.finfo(torch.float16)
    factors = torch.tensor(
        [
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.5,
            -0.5,
            1.5,
            -1.5,
            2.0,
            -2.0,
            1.0 - 2**-11,
            1.0 + 2**-10,
            2**-24,
            finfo.smallest_normal,
            finfo.max,
            -finfo.max,
            float("inf"),
            -float("inf"),
            float("nan"),
        ],
        dtype=torch.float16,
    )
    encodings = torch.arange(65536, dtype=torch.int32).to(torch.int16)
    values = encodings.view(torch.float16)
    x_cpu = values.repeat_interleave(factors.numel())
    y_cpu = factors.repeat(values.numel())
    expected = _multiply_oracle(x_cpu, y_cpu)
    x = x_cpu.to(backend)
    y = y_cpu.to(backend)
    output = torch.empty_like(x)

    _launch(x, y, output, backend, operation="multiply")

    _assert_encoding(output, expected)


def test_multiply_fp32_seeded_pairs_and_rounding_boundaries(backend):
    """Cover seeded raw pairs and mantissa rounding across every exponent."""
    generator = torch.Generator().manual_seed(0x5A6E)
    random_bits = torch.randint(
        0, 2**32, (2, 65536), dtype=torch.int64, generator=generator
    ).to(torch.int32)
    x_random, y_random = random_bits.view(torch.float32)
    exponents = torch.arange(256, dtype=torch.int64)
    signs = torch.tensor([0, 1], dtype=torch.int64)
    mantissas = torch.tensor(
        [0x000000, 0x3FFFFF, 0x400000, 0x400001, 0x7FFFFF],
        dtype=torch.int64,
    )
    boundary_bits = (
        (signs[:, None, None] << 31)
        | (exponents[None, :, None] << 23)
        | mantissas[None, None, :]
    ).reshape(-1)
    x_boundary = boundary_bits.to(torch.int32).view(torch.float32)
    y_boundary = torch.full_like(x_boundary, 1.5)
    x_cpu = torch.cat((x_random, x_boundary))
    y_cpu = torch.cat((y_random, y_boundary))
    expected = _multiply_oracle(x_cpu, y_cpu)
    x = x_cpu.to(backend)
    y = y_cpu.to(backend)
    output = torch.empty_like(x)

    _launch(x, y, output, backend, operation="multiply")

    _assert_encoding(output, expected)


@pytest.mark.parametrize(
    "dtype",
    (
        torch.float32,
        torch.float16,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ),
)
@pytest.mark.parametrize("block", (1, 31, 32, 33, 128, 256, 512, 1024))
def test_multiply_launch_boundaries(backend, dtype, block):
    """Cover empty, immediate, tail, and multi-block launch boundaries."""
    for n in (0, 1, block - 1, block, block + 1, 2 * block + 1):
        x_cpu = (torch.arange(n).float() % 11 - 5).to(dtype)
        y_cpu = (torch.arange(n).float() % 7 - 3).to(dtype)
        x = x_cpu.to(backend)
        y = y_cpu.to(backend)
        output = torch.empty_like(x)

        _launch(
            x,
            y,
            output,
            backend,
            operation="multiply",
            block=block,
        )

        _assert_encoding(output, _multiply_oracle(x_cpu, y_cpu))


@pytest.mark.parametrize(
    "dtype",
    (
        torch.float32,
        torch.float16,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ),
)
def test_multiply_million_element_tail(backend, dtype):
    """Cover a large multi-block launch with a partial final block."""
    n = 1_000_003
    x_cpu = (torch.arange(n).float() % 17 - 8).to(dtype)
    y_cpu = (torch.arange(n).float() % 13 - 6).to(dtype)
    x = x_cpu.to(backend)
    y = y_cpu.to(backend)
    output = torch.empty_like(x)

    _launch(x, y, output, backend, operation="multiply", block=256)

    _assert_encoding(output, _multiply_oracle(x_cpu, y_cpu))


@pytest.mark.parametrize(
    "dtype",
    (
        torch.float32,
        torch.float16,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ),
)
@pytest.mark.parametrize("n", (1, 31, 33))
@pytest.mark.parametrize("alias", ("none", "x", "y", "both"))
def test_multiply_offset_views_guards_and_aliases(backend, dtype, n, alias):
    """Preserve guard storage and snapshot aliased inputs before writes."""
    capacity = n + 3
    x_storage = torch.full((capacity + 4,), -7.0, dtype=dtype, device=backend)
    x = x_storage[1 : capacity + 1]
    x.copy_((torch.arange(capacity, device=backend).float() % 9 - 4).to(dtype))
    storages = [x_storage]
    if alias == "both":
        y = x
    else:
        y_storage = torch.full(
            (capacity + 6,), -9.0, dtype=dtype, device=backend
        )
        y = y_storage[2 : capacity + 2]
        y.copy_(
            (torch.arange(capacity, device=backend).float() % 5 - 2).to(dtype)
        )
        storages.append(y_storage)
    if alias == "none":
        output_storage = torch.full(
            (capacity + 8,), -11.0, dtype=dtype, device=backend
        )
        output = output_storage[3 : capacity + 3]
        storages.append(output_storage)
    elif alias == "x" or alias == "both":
        output = x
    else:
        output = y
    expected_values = _multiply_oracle(x[:n].cpu().clone(), y[:n].cpu().clone())
    guarded = output._base
    expected_storages = [storage.cpu().clone() for storage in storages]
    start = output.storage_offset()
    output_index = next(
        index for index, storage in enumerate(storages) if storage is guarded
    )
    expected_storages[output_index][start : start + n] = expected_values

    _launch(x, y, output, backend, n=n, operation="multiply", block=32)

    for storage, expected_storage in zip(storages, expected_storages):
        _assert_encoding(storage, expected_storage)


@pytest.mark.parametrize("operation", OPERATIONS)
def test_dtype_switches_preserve_strides_tail_and_in_place(backend, operation):
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
        _, torch_operation = OPERATIONS[operation]
        expected[3 : 3 + n] = torch_operation(
            x.cpu().float(), y.cpu().float()
        ).to(dtype)
        _launch(x, y, output, backend, operation=operation)
        _assert_encoding(storage, expected)

        # A repeated same-dtype call uses current storage and permits aliasing.
        expected[3 : 3 + n] = (
            torch_operation(expected[3 : 3 + n].float(), y.cpu().float())
        ).to(dtype)
        _launch(output, y, output, backend, operation=operation)
        _assert_encoding(storage, expected)

        # Empty work still validates, but must not touch any byte.
        _launch(x, y, output, backend, n=0, operation=operation)
        _assert_encoding(storage, expected)


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("n", (0, 3))
def test_warm_launch_rejects_mixed_dtypes_without_writes(backend, n, operation):
    """A supported dtype is still invalid when it differs from its peers."""
    x = torch.ones(3, dtype=torch.float16, device=backend)
    output = torch.empty_like(x)
    _launch(x, x, output, backend, operation=operation)
    _launch(x, x, output, backend, operation=operation)
    output.fill_(-7)
    other = x.to(torch.float8_e4m3fn)

    with pytest.raises(TypeError):
        _launch(x, other, output, backend, n=n, operation=operation)

    _assert_encoding(output, torch.full((3,), -7, dtype=torch.float16))


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("dtype", (torch.float16, torch.float32))
def test_float_edge_matrix_matches_storage_oracle(backend, dtype, operation):
    """Cover signs, zeros, subnormals, RNE, overflow, infinities, and NaNs."""
    finfo = torch.finfo(dtype)
    smallest_subnormal = 2**-24 if dtype == torch.float16 else 2**-149
    values = torch.tensor(
        [
            0.0,
            -0.0,
            finfo.smallest_normal,
            -finfo.smallest_normal,
            smallest_subnormal,
            -smallest_subnormal,
            1.0,
            -1.0,
            1.5,
            -1.5,
            1.0 + finfo.eps,
            1.0 + 2 * finfo.eps,
            finfo.max,
            -finfo.max,
            float("inf"),
            -float("inf"),
            float("nan"),
        ],
        dtype=dtype,
    )
    x_cpu = values.repeat_interleave(values.numel())
    y_cpu = values.repeat(values.numel())
    _, torch_operation = OPERATIONS[operation]
    expected = torch_operation(x_cpu.float(), y_cpu.float()).to(dtype)
    x = x_cpu.to(backend)
    y = y_cpu.to(backend)
    output = torch.empty_like(x)

    _launch(x, y, output, backend, operation=operation)

    _assert_encoding(output, expected)


@pytest.mark.parametrize("operation", OPERATIONS)
def test_invalid_geometry_does_not_write(backend, operation):
    """Reject a grid inconsistent with n before touching output storage."""
    kernel, _ = OPERATIONS[operation]
    x = torch.ones(129, device=backend)
    output = torch.full_like(x, -7)

    with pytest.raises(ValueError, match="grid must equal"):
        kernel.launch(
            arguments={
                "x_ptr": x,
                "y_ptr": x,
                "output_ptr": output,
                "n": 129,
            },
            constexprs={"BLOCK": 128},
            grid=(1,),
            backend=backend,
        )

    _assert_encoding(output, torch.full((129,), -7.0))


@pytest.mark.parametrize("operation", OPERATIONS)
def test_invalid_device_does_not_write(backend, operation):
    """Reject a tensor on the wrong device before touching output storage."""
    output = torch.full((3,), -7.0, device=backend)
    wrong_device = "meta" if backend == "cpu" else "cpu"
    x = torch.ones(3, device=wrong_device)
    y = torch.ones_like(output)

    with pytest.raises(TypeError):
        _launch(x, y, output, backend, operation=operation)

    _assert_encoding(output, torch.full((3,), -7.0))
