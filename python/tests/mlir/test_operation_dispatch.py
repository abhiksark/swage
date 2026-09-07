# python/tests/mlir/test_operation_dispatch.py
"""Arithmetic source identity and both CUDA dispatch lanes."""

import logging

import pytest
import swage as sw
import swage.language as sl
from swage import _cuda_backend, _runtime

torch = pytest.importorskip("torch")
pytest.importorskip("mlir_swage._mlir_libs._swageDialectsNanobind")


def _kernels():
    # Identical names and signatures make the AST operation the distinction.
    @sw.jit
    def elementwise(x_ptr, y_ptr, output_ptr, n, BLOCK: sl.constexpr):
        offsets = sl.program_id(0) * BLOCK + sl.arange(0, BLOCK)
        mask = offsets < n
        x = sl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = sl.load(y_ptr + offsets, mask=mask, other=0.0)
        sl.store(output_ptr + offsets, x + y, mask=mask)

    addition = elementwise

    @sw.jit
    def elementwise(x_ptr, y_ptr, output_ptr, n, BLOCK: sl.constexpr):
        offsets = sl.program_id(0) * BLOCK + sl.arange(0, BLOCK)
        mask = offsets < n
        x = sl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = sl.load(y_ptr + offsets, mask=mask, other=0.0)
        sl.store(output_ptr + offsets, x * y, mask=mask)

    return addition, elementwise


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float16, torch.float8_e4m3fn, torch.float8_e5m2],
)
def test_same_name_operations_have_distinct_reusable_artifacts(
    backend, dtype, monkeypatch, caplog
):
    """Alternate identical signatures across new kernel instances and caches."""
    if backend == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    monkeypatch.setattr(_runtime, "_artifact_cache", _runtime.OrderedDict())
    caplog.set_level(logging.DEBUG, logger="swage.runtime")
    x = torch.full((129,), 2.0, device=backend).to(dtype)
    y = torch.full((129,), 3.0, device=backend).to(dtype)
    output = torch.empty_like(x)
    kernels = _kernels()
    assert kernels[0].__name__ == kernels[1].__name__
    assert kernels[0].source_digest != kernels[1].source_digest
    for pair in (kernels, kernels, _kernels()):
        for kernel, expected in zip(pair, (5.0, 6.0)):
            kernel.launch(
                arguments={
                    "x_ptr": x,
                    "y_ptr": y,
                    "output_ptr": output,
                    "n": 129,
                },
                constexprs={"BLOCK": 128},
                grid=(2,),
                backend=backend,
            )
            assert torch.equal(
                output.float(), torch.full_like(x.float(), expected)
            )
    artifacts = tuple(_runtime._artifact_cache.values())
    assert len(artifacts) == 2
    assert artifacts[0].key != artifacts[1].key
    assert artifacts[0].identity != artifacts[1].identity
    assert artifacts[0].contract_json == artifacts[1].contract_json
    assert artifacts[0].lowered != artifacts[1].lowered
    assert any("memory-hit" in record.message for record in caplog.records)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("native", [True, False], ids=["prepared", "ctypes"])
def test_multiply_stream_switches_and_graph_replay(native, monkeypatch):
    """Follow current streams through multiplication capture and replay."""
    driver = _cuda_backend._get_driver()
    if not native:
        monkeypatch.setattr(driver, "_native_launch", None)
        monkeypatch.setattr(driver, "_native_fixed_launcher", None)
    _, kernel = _kernels()
    x = torch.full((129,), 2.0, device="cuda")
    y = torch.full_like(x, 3.0)
    output = torch.empty_like(x)
    arguments = {"x_ptr": x, "y_ptr": y, "output_ptr": output, "n": 129}

    def launch():
        kernel.launch(arguments=arguments, constexprs={"BLOCK": 128}, grid=(2,))

    launch()
    launch()
    torch.cuda.synchronize()
    if native:
        assert kernel._cuda_fast_launch is not None
        assert kernel._cuda_fast_launch() is not None
    else:
        assert kernel._cuda_fast_launch is None
    streams = (torch.cuda.Stream(), torch.cuda.Stream())
    for value, stream in enumerate((*streams, *streams), start=4):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            x.fill_(value)
            launch()
            observed = output.clone()
        stream.synchronize()
        assert torch.equal(observed, torch.full_like(observed, value * 3))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=streams[0]):
        launch()
    x.fill_(-2)
    output.fill_(0)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(output, torch.full_like(output, -6))


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_combined_expression_is_rejected_without_writes(backend):
    """Frontend arithmetic chains must fail at the public launch boundary."""
    if backend == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    @sw.jit
    def chained(x_ptr, y_ptr, output_ptr, n, BLOCK: sl.constexpr):
        offsets = sl.program_id(0) * BLOCK + sl.arange(0, BLOCK)
        mask = offsets < n
        x = sl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = sl.load(y_ptr + offsets, mask=mask, other=0.0)
        sl.store(output_ptr + offsets, x * y + x, mask=mask)

    x = torch.full((129,), 2.0, device=backend)
    y = torch.full_like(x, 3.0)
    output = torch.full_like(x, -7.0)
    with pytest.raises(ValueError, match="one floating-point add or multiply"):
        chained.launch(
            arguments={"x_ptr": x, "y_ptr": y, "output_ptr": output, "n": 129},
            constexprs={"BLOCK": 128},
            grid=(2,),
            backend=backend,
        )
    assert torch.equal(output, torch.full_like(output, -7.0))
