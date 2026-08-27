# python/tests/mlir/test_runtime.py
"""Real CUDA tests for the fixed vector-add launch boundary."""

import gc
import weakref

import pytest
import swage as sw
import swage.language as sl
import torch

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
    """Queue work on the selected PyTorch stream without synchronizing."""
    x = torch.randn(129, device="cuda")
    y = torch.randn(129, device="cuda")
    output = torch.empty_like(x)
    stream = torch.cuda.Stream()

    with torch.cuda.stream(stream):
        _launch(x, y, output, 129)
    stream.synchronize()

    torch.testing.assert_close(output, x + y)


def test_repeated_launches_and_argument_release():
    """Reuse compiled state without retaining tensor arguments."""
    def launch_local():
        x = torch.randn(129, device="cuda")
        y = torch.randn(129, device="cuda")
        output = torch.empty_like(x)
        references = (weakref.ref(x), weakref.ref(y), weakref.ref(output))
        _launch(x, y, output, 129)
        _launch(x, y, output, 129)
        torch.cuda.current_stream().synchronize()
        return references

    references = launch_local()
    gc.collect()

    assert all(reference() is None for reference in references)


def test_launch_rejects_invalid_runtime_inputs():
    """Fail closed for unsafe pointers, bounds, grids, and blocks."""
    x = torch.empty(4, device="cuda")
    output = torch.empty_like(x)
    arguments = {
        "x_ptr": x,
        "y_ptr": x,
        "output_ptr": output,
        "n": 4,
    }

    with pytest.raises(TypeError, match="must be a CUDA tensor"):
        add_kernel.launch(
            arguments={**arguments, "y_ptr": torch.empty(4)},
            constexprs={"BLOCK": 128},
            grid=(1,),
        )
    with pytest.raises(ValueError, match="exceeds tensor length"):
        add_kernel.launch(
            arguments={**arguments, "n": 5},
            constexprs={"BLOCK": 128},
            grid=(1,),
        )
    with pytest.raises(ValueError, match="grid must equal"):
        add_kernel.launch(
            arguments=arguments,
            constexprs={"BLOCK": 128},
            grid=(2,),
        )
    limit = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).max_threads_per_block
    with pytest.raises(ValueError, match="exceeds device limit"):
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
    from swage import _runtime

    n, block = 1000, 128
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    output = torch.full((n,), -777.0, device="cuda")
    module = add_kernel.emit_mlir(
        arguments={"x_ptr": x, "y_ptr": y, "output_ptr": output, "n": n},
        constexprs={"BLOCK": block},
    )
    major, minor = torch.cuda.get_device_capability()
    _, ptx = native_swage._compile_ptx(
        module, kernel_name="add_kernel", block_size=block,
        target=f"sm_{major}{minor}",
    )
    driver = _runtime._get_driver()
    _, function = driver.load(ptx, "add_kernel")

    native_swage._launch_kernel(
        function,
        (n + block - 1) // block,
        block,
        torch.cuda.current_stream().cuda_stream,
        (x.data_ptr(), y.data_ptr(), output.data_ptr()),
        (n,),
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(output, x + y)


def test_native_launcher_surfaces_driver_errors():
    """Report a failed cuLaunchKernel with the stable message shape."""
    from mlir_swage._mlir_libs._swageDialectsNanobind import (
        swage as native_swage,
    )

    torch.zeros(1, device="cuda")
    with pytest.raises(RuntimeError, match="cuLaunchKernel failed"):
        native_swage._launch_kernel(
            0, 1, 128, torch.cuda.current_stream().cuda_stream, (0,), (0,)
        )
