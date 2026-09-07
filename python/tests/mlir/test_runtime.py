# python/tests/mlir/test_runtime.py
"""Real CUDA tests for the fixed vector-add launch boundary."""

import gc
import weakref

import pytest
import swage as sw
import swage.language as sl

torch = pytest.importorskip("torch")
pytest.importorskip("mlir_swage._mlir_libs._swageDialectsNanobind")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"
)


@sw.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n, BLOCK: sl.constexpr):  # noqa: D103
    pid = sl.program_id(0)
    offsets = pid * BLOCK + sl.arange(0, BLOCK)
    mask = offsets < n
    x = sl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = sl.load(y_ptr + offsets, mask=mask, other=0.0)
    sl.store(output_ptr + offsets, x + y, mask=mask)


def _run_cuda_probe(ptx, entry, kinds, values, grid, block, *, native):
    from mlir_swage._mlir_libs._swageDialectsNanobind import (
        swage as native_swage,
    )
    from swage import _abi, _cuda_backend

    driver = _cuda_backend._get_driver()
    module, function = driver.load(ptx, entry)
    stream = torch.cuda.current_stream().cuda_stream
    try:
        if native:
            native_swage._launch_cuda_kernel(
                kinds, values, grid, block, 0, stream, function
            )
        else:
            contract = _abi.KernelContract(
                version=2,
                backend="cuda",
                entry=entry,
                launch=_abi.KernelLaunch(model="spmd-grid", block=block),
                arguments=tuple(
                    _abi.KernelArgument(
                        kind=kind, origin="user", source_index=index
                    )
                    for index, kind in enumerate(kinds)
                ),
            )
            native_launch = driver._native_launch
            driver._native_launch = None
            try:
                driver.launch_entry(
                    function, contract, (kinds, values), grid, stream
                )
            finally:
                driver._native_launch = native_launch
        torch.cuda.synchronize()
    finally:
        driver.module_unload(module)


def _launch(x, y, output, n, block=128):
    add_kernel.launch(
        arguments={
            "x_ptr": x,
            "y_ptr": y,
            "output_ptr": output,
            "n": n,
        },
        constexprs={"BLOCK": block},
        grid=((n + block - 1) // block,),
    )


@pytest.mark.parametrize("n", [0, 1, 127, 128, 129, 4097])
def test_vector_add_matches_pytorch(n):
    """Compute empty, boundary, partial-block, and multi-block inputs."""
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    output = torch.empty_like(x)

    _launch(x, y, output, n)

    torch.testing.assert_close(output, torch.add(x, y))


def test_launch_uses_non_default_current_stream():
    """Follow stream switches and order writes before downstream consumers."""
    x = torch.zeros(129, device="cuda")
    y = torch.ones_like(x)
    output = torch.empty_like(x)
    _launch(x, y, output, 129)
    _launch(x, y, output, 129)
    torch.cuda.synchronize()
    streams = (torch.cuda.Stream(), torch.cuda.Stream())

    try:
        for value, stream in enumerate((*streams, *streams), start=2):
            with torch.cuda.stream(stream):
                x.fill_(value)
                output.fill_(float("nan"))
                _launch(x, y, output, 129)
                observed = output.clone()
            stream.synchronize()
            torch.testing.assert_close(
                observed, torch.full_like(observed, value + 1)
            )
    finally:
        torch.cuda.synchronize()


def test_repeated_launches_and_argument_release():
    """Reuse compiled state without retaining tensor arguments."""

    def launch_local():
        x = torch.randn(129, device="cuda")
        y = torch.randn(129, device="cuda")
        output = torch.empty_like(x)
        references = (weakref.ref(x), weakref.ref(y), weakref.ref(output))
        for _ in range(3):
            _launch(x, y, output, 129)
        return references

    try:
        references = launch_local()
        gc.collect()
        assert all(reference() is None for reference in references)
    finally:
        torch.cuda.synchronize()
    gc.collect()

    assert all(reference() is None for reference in references)


@pytest.mark.parametrize(
    ("mutation", "error_type"),
    [
        ("length", ValueError),
        ("rank", TypeError),
        ("stride", ValueError),
        ("dtype", TypeError),
    ],
)
def test_warm_launch_revalidates_same_tensor_metadata(mutation, error_type):
    """Never trust shape, strides, or dtype cached by tensor identity."""
    x = torch.arange(258, device="cuda", dtype=torch.float32)
    y = torch.ones_like(x)
    output = torch.empty_like(x)
    _launch(x, y, output, 129)
    _launch(x, y, output, 129)
    torch.cuda.synchronize()

    if mutation == "length":
        x.resize_(1)
    elif mutation == "rank":
        x.unsqueeze_(0)
    elif mutation == "stride":
        x.as_strided_((129,), (2,))
    else:
        x.data = x.to(torch.float64)
    output.fill_(-1)
    try:
        with pytest.raises(error_type):
            _launch(x, y, output, 129)
        torch.testing.assert_close(output, torch.full_like(output, -1))
    finally:
        torch.cuda.synchronize()


def test_warm_launch_preserves_torch_function_mode():
    """Do not bypass caller-owned tensor behavior on a warm shortcut."""
    x = torch.ones(129, device="cuda")
    y = torch.ones_like(x)
    output = torch.empty_like(x)
    for _ in range(2):
        _launch(x, y, output, 129)
    torch.cuda.synchronize()
    output.fill_(-1)

    class PointerFailure(RuntimeError):
        pass

    class RejectPointers(torch.overrides.TorchFunctionMode):
        def __torch_function__(self, function, types, args=(), kwargs=None):
            if function is torch.Tensor.data_ptr:
                raise PointerFailure("caller rejects pointer access")
            return function(*args, **(kwargs or {}))

    try:
        with RejectPointers(), pytest.raises(PointerFailure):
            _launch(x, y, output, 129)
        torch.testing.assert_close(output, torch.full_like(output, -1))
    finally:
        torch.cuda.synchronize()


def test_warm_launch_preserves_tensor_instance_overrides():
    """Honor live instance overrides before requesting a pointer."""
    x = torch.ones(129, device="cuda")
    y = torch.ones_like(x)
    output = torch.empty_like(x)
    for _ in range(2):
        _launch(x, y, output, 129)
    torch.cuda.synchronize()
    output.fill_(-1)

    class PointerFailure(RuntimeError):
        pass

    def reject_pointer():
        raise PointerFailure("instance rejects pointer access")

    x.data_ptr = reject_pointer
    try:
        with pytest.raises(PointerFailure):
            _launch(x, y, output, 129)
        torch.testing.assert_close(output, torch.full_like(output, -1))
    finally:
        torch.cuda.synchronize()


def test_warm_launch_reads_mutated_mappings_and_tensor_storage():
    """Read current pointers, aliases, n, grid, and block on every call."""
    x = torch.arange(514, device="cuda", dtype=torch.float32)
    y = torch.ones_like(x)
    output = torch.empty_like(x)
    arguments = {"x_ptr": x, "y_ptr": y, "output_ptr": output, "n": 129}
    constexprs = {"BLOCK": 128}
    for _ in range(2):
        add_kernel.launch(arguments=arguments, constexprs=constexprs, grid=(2,))
    torch.cuda.synchronize()

    # Keep the object but replace its storage, including a nonzero offset.
    storage = torch.arange(1028, device="cuda", dtype=torch.float32)
    x.set_(storage[17:531])
    replacement = torch.full_like(y, 7)
    arguments["y_ptr"] = replacement
    for n, block in ((257, 128), (1, 128), (0, 128), (513, 256), (129, 128)):
        output.fill_(-1)
        arguments["n"] = n
        constexprs["BLOCK"] = block
        add_kernel.launch(
            arguments=arguments,
            constexprs=constexprs,
            grid=((n + block - 1) // block,),
        )
        expected = torch.full_like(output, -1)
        expected[:n] = x[:n] + replacement[:n]
        torch.testing.assert_close(output, expected)

    # Mutate that same mapping to an exact output/input alias.
    arguments["output_ptr"] = x
    expected = x.clone()
    expected[:129] += replacement[:129]
    add_kernel.launch(arguments=arguments, constexprs=constexprs, grid=(2,))
    torch.testing.assert_close(x, expected)


@pytest.mark.parametrize("capture", [False, True], ids=["eager", "graph"])
def test_warm_launch_survives_capture_and_cache_eviction(monkeypatch, capture):
    """Reload evicted eager work without invalidating captured work."""
    from swage import _cuda_backend, _runtime

    monkeypatch.setenv("SWAGE_MEMORY_CACHE_ENTRIES", "1")
    monkeypatch.setattr(_runtime, "_memory_cache_entries", None)
    monkeypatch.setattr(_cuda_backend, "_memory_cache_entries", None)
    first = sw.jit(add_kernel.__wrapped__)
    second = sw.jit(add_kernel.__wrapped__)
    x = torch.ones(257, device="cuda")
    y = torch.full_like(x, 2)
    output = torch.empty_like(x)
    other_output = torch.empty_like(x)
    arguments = {"x_ptr": x, "y_ptr": y, "output_ptr": output, "n": 257}
    other_arguments = {**arguments, "output_ptr": other_output}
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    try:
        with torch.cuda.stream(stream):
            for _ in range(2):
                first.launch(
                    arguments=arguments, constexprs={"BLOCK": 128}, grid=(3,)
                )
        stream.synchronize()
        graph = None
        if capture:
            # A pending eager module may be revived by a capture. Retirement
            # must not insert its completion fences into that graph.
            with _cuda_backend._cuda_lock:
                _cuda_backend._evict_loaded_locked(0)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                first.launch(
                    arguments=arguments, constexprs={"BLOCK": 128}, grid=(3,)
                )

        # A second specialization evicts the original module. Repeating it
        # polls deferred unloads before the original callable is used again.
        for _ in range(2):
            second.launch(
                arguments=other_arguments,
                constexprs={"BLOCK": 256},
                grid=(2,),
            )
            torch.cuda.synchronize()
        torch.testing.assert_close(other_output, x + y)
        # Force another miss even if a broken warm path neglected to poll
        # retired entries: the first module must now unload or remain pinned.
        other_output.fill_(float("nan"))
        second.launch(
            arguments=other_arguments, constexprs={"BLOCK": 64}, grid=(5,)
        )
        torch.testing.assert_close(other_output, x + y)
        x.fill_(5)
        if graph is not None:
            output.fill_(float("nan"))
            graph.replay()
            torch.testing.assert_close(output, x + y)

        output.fill_(float("nan"))
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            first.launch(
                arguments=arguments, constexprs={"BLOCK": 128}, grid=(3,)
            )
        stream.synchronize()
        torch.testing.assert_close(output, x + y)
        if graph is not None:
            y.fill_(11)
            output.fill_(float("nan"))
            graph.replay()
            torch.testing.assert_close(output, x + y)
    finally:
        torch.cuda.synchronize()


def test_launch_rejects_invalid_runtime_inputs():
    """Fail closed for unsafe pointers, bounds, grids, and blocks."""
    x = torch.ones(4, device="cuda")
    output = torch.empty_like(x)
    arguments = {
        "x_ptr": x,
        "y_ptr": x,
        "output_ptr": output,
        "n": 4,
    }
    for _ in range(2):
        add_kernel.launch(
            arguments=arguments, constexprs={"BLOCK": 128}, grid=(1,)
        )
    torch.cuda.synchronize()

    arguments["y_ptr"] = torch.empty(4)
    with pytest.raises(TypeError):
        add_kernel.launch(
            arguments=arguments,
            constexprs={"BLOCK": 128},
            grid=(1,),
        )
    arguments["y_ptr"] = x
    arguments["n"] = 5
    with pytest.raises(ValueError):
        add_kernel.launch(
            arguments=arguments,
            constexprs={"BLOCK": 128},
            grid=(1,),
        )
    arguments["n"] = 4
    with pytest.raises(ValueError):
        add_kernel.launch(
            arguments=arguments,
            constexprs={"BLOCK": 128},
            grid=(2,),
        )
    limit = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).max_threads_per_block
    with pytest.raises(ValueError):
        add_kernel.launch(
            arguments=arguments,
            constexprs={"BLOCK": limit + 1},
            grid=(1,),
        )


def test_native_launcher_runs_the_fixed_kernel():
    """Dispatch one launch through the compiled path, not ctypes."""
    from mlir_swage._mlir_libs._swageDialectsNanobind import (
        swage as native_swage,
    )
    from swage import _abi, _cuda_backend

    n, block = 1000, 128
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    output = torch.full((n,), -777.0, device="cuda")
    module = add_kernel.emit_mlir(
        arguments={"x_ptr": x, "y_ptr": y, "output_ptr": output, "n": n},
        constexprs={"BLOCK": block},
    )
    major, minor = torch.cuda.get_device_capability()
    _, ptx, contract_json = native_swage._compile_ptx(
        module,
        kernel_name="add_kernel",
        block_size=block,
        target=f"sm_{major}{minor}",
    )
    contract = _abi.parse_kernel_contract(contract_json)
    assert contract.entry == "add_kernel"
    assert contract.launch.block == (block, 1, 1)
    bound = _abi.bind_kernel_contract(contract, user=(x, y, output, n))
    kinds, values = _abi.materialize_launch_arguments(bound)
    assert kinds == ("ptr", "ptr", "ptr", "i32")
    assert values == (
        x.data_ptr(),
        y.data_ptr(),
        output.data_ptr(),
        n,
    )
    driver = _cuda_backend._get_driver()
    _, function = driver.load(ptx, contract.entry)

    native_swage._launch_cuda_kernel(
        kinds,
        values,
        ((n + block - 1) // block, 1, 1),
        (block, 1, 1),
        0,
        torch.cuda.current_stream().cuda_stream,
        function,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(output, x + y)


def test_native_launcher_preserves_mixed_argument_order():
    """Keep interleaved pointer and scalar parameters in caller order."""
    from mlir_swage._mlir_libs._swageDialectsNanobind import (
        swage as native_swage,
    )
    from swage import _cuda_backend

    ptx = """
.version 6.0
.target sm_50
.address_size 64

.visible .entry mixed_order(
    .param .u64 input,
    .param .u32 increment,
    .param .u64 output
)
{
    .reg .b32 %r<3>;
    .reg .b64 %rd<3>;
    ld.param.u64 %rd1, [input];
    ld.param.u32 %r1, [increment];
    ld.param.u64 %rd2, [output];
    ld.global.u32 %r2, [%rd1];
    add.s32 %r2, %r2, %r1;
    st.global.u32 [%rd2], %r2;
    ret;
}
"""
    input_value = torch.tensor([7], dtype=torch.int32, device="cuda")
    output = torch.zeros_like(input_value)
    driver = _cuda_backend._get_driver()
    _, function = driver.load(ptx, "mixed_order")

    native_swage._launch_cuda_kernel(
        ("ptr", "i32", "ptr"),
        (input_value.data_ptr(), 5, output.data_ptr()),
        (1, 1, 1),
        (1, 1, 1),
        0,
        torch.cuda.current_stream().cuda_stream,
        function,
    )
    torch.cuda.synchronize()
    assert output.item() == 12


@pytest.mark.parametrize(
    ("kinds", "values", "error_type"),
    [
        (
            ("ptr",),
            (),
            ValueError,
        ),
        (
            ("u64",),
            (0,),
            ValueError,
        ),
        (
            ("ptr",),
            (-1,),
            ValueError,
        ),
        (
            ("ptr",),
            (1 << 64,),
            ValueError,
        ),
        (
            ("i32",),
            (-1,),
            ValueError,
        ),
        (
            ("i32",),
            (1 << 32,),
            ValueError,
        ),
        (
            ("ptr",),
            ("not-an-integer",),
            TypeError,
        ),
    ],
)
def test_native_launcher_rejects_invalid_arguments(kinds, values, error_type):
    """Reject malformed ordered ABI values before entering the driver."""
    from mlir_swage._mlir_libs._swageDialectsNanobind import (
        swage as native_swage,
    )

    torch.zeros(1, device="cuda")
    with pytest.raises(error_type):
        native_swage._launch_cuda_kernel(
            kinds,
            values,
            (1, 1, 1),
            (128, 1, 1),
            0,
            torch.cuda.current_stream().cuda_stream,
            0,
        )


def test_native_launcher_surfaces_driver_errors():
    """Preserve driver failures instead of classifying them as unavailable."""
    from mlir_swage._mlir_libs._swageDialectsNanobind import (
        swage as native_swage,
    )

    torch.zeros(1, device="cuda")
    with pytest.raises(RuntimeError) as raised:
        native_swage._launch_cuda_kernel(
            ("ptr", "i32"),
            (0, 0),
            (1, 1, 1),
            (128, 1, 1),
            0,
            torch.cuda.current_stream().cuda_stream,
            0,
        )
    assert type(raised.value) is RuntimeError


@pytest.mark.parametrize("native", [True, False], ids=["native", "ctypes"])
def test_cuda_scalar_probe_preserves_every_contract_kind(native):
    """Store every version-2 physical kind as its exact ABI byte pattern."""
    from swage import _abi

    ptx = """
.version 6.0
.target sm_50
.address_size 64

.visible .entry scalar_probe(
    .param .u64 output,
    .param .u8 value_i1,
    .param .u8 value_i8,
    .param .u16 value_i16,
    .param .u32 value_i32,
    .param .u64 value_i64,
    .param .u16 value_f16,
    .param .u16 value_bf16,
    .param .u32 value_f32,
    .param .u64 value_f64
)
{
    .reg .b16 %rs<4>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<4>;
    ld.param.u64 %rd1, [output];
    st.global.u64 [%rd1+0], %rd1;
    ld.param.u8 %r1, [value_i1];
    st.global.u8 [%rd1+8], %r1;
    ld.param.u8 %r2, [value_i8];
    st.global.u8 [%rd1+9], %r2;
    ld.param.u16 %rs1, [value_i16];
    st.global.u16 [%rd1+10], %rs1;
    ld.param.u32 %r3, [value_i32];
    st.global.u32 [%rd1+12], %r3;
    ld.param.u64 %rd2, [value_i64];
    st.global.u64 [%rd1+16], %rd2;
    ld.param.u16 %rs2, [value_f16];
    st.global.u16 [%rd1+24], %rs2;
    ld.param.u16 %rs3, [value_bf16];
    st.global.u16 [%rd1+26], %rs3;
    ld.param.u32 %r4, [value_f32];
    st.global.u32 [%rd1+28], %r4;
    ld.param.u64 %rd3, [value_f64];
    st.global.u64 [%rd1+32], %rd3;
    ret;
}
"""
    output = torch.zeros(40, dtype=torch.uint8, device="cuda")
    bound = (
        _abi.BoundArgument("ptr", output),
        _abi.BoundArgument("i1", 1),
        _abi.BoundArgument("i8", 0xA5),
        _abi.BoundArgument("i16", 0xBEEF),
        _abi.BoundArgument("i32", 0x89ABCDEF),
        _abi.BoundArgument("i64", 0x0123456789ABCDEF),
        _abi.BoundArgument("f16", 1.5),
        _abi.BoundArgument("bf16", -2.25),
        _abi.BoundArgument("f32", 3.5),
        _abi.BoundArgument("f64", -4.75),
    )
    kinds, values = _abi.materialize_launch_arguments(bound)
    widths = (8, 1, 1, 2, 4, 8, 2, 2, 4, 8)
    expected = b"".join(
        value.to_bytes(width, byteorder="little")
        for value, width in zip(values, widths)
    )

    _run_cuda_probe(
        ptx,
        "scalar_probe",
        kinds,
        values,
        (1, 1, 1),
        (1, 1, 1),
        native=native,
    )

    assert bytes(output.cpu().tolist()) == expected


@pytest.mark.parametrize("native", [True, False], ids=["native", "ctypes"])
def test_cuda_three_dimensional_geometry_reaches_every_thread(native):
    """Launch a 2x3x2 grid of 4x2x2 blocks with 192 unique outputs."""
    from swage import _abi

    ptx = """
.version 6.0
.target sm_50
.address_size 64

.visible .entry geometry_probe(.param .u64 output)
{
    .reg .b32 %r<10>;
    .reg .b64 %rd<4>;
    ld.param.u64 %rd1, [output];
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ctaid.y;
    mov.u32 %r3, %ctaid.z;
    mov.u32 %r4, %tid.x;
    mov.u32 %r5, %tid.y;
    mov.u32 %r6, %tid.z;
    mad.lo.u32 %r7, %r3, 3, %r2;
    mad.lo.u32 %r7, %r7, 2, %r1;
    mad.lo.u32 %r8, %r6, 2, %r5;
    mad.lo.u32 %r8, %r8, 4, %r4;
    mad.lo.u32 %r9, %r7, 16, %r8;
    mul.wide.u32 %rd2, %r9, 4;
    add.s64 %rd3, %rd1, %rd2;
    st.global.u32 [%rd3], %r9;
    ret;
}
"""
    output = torch.full((192,), -1, dtype=torch.int32, device="cuda")
    kinds, values = _abi.materialize_launch_arguments(
        (_abi.BoundArgument("ptr", output),)
    )

    _run_cuda_probe(
        ptx,
        "geometry_probe",
        kinds,
        values,
        (2, 3, 2),
        (4, 2, 2),
        native=native,
    )

    torch.testing.assert_close(
        output.cpu(), torch.arange(192, dtype=torch.int32)
    )
