# python/tests/mlir/test_operation_dispatch.py
"""Arithmetic source identity and both CUDA dispatch lanes."""

import logging
import os
import pathlib
import subprocess
import sys
import textwrap

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
@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float16, torch.float8_e4m3fn, torch.float8_e5m2],
)
def test_multiply_stream_switches_and_graph_replay(native, dtype, monkeypatch):
    """Follow current streams through multiplication capture and replay."""
    driver = _cuda_backend._get_driver()
    if not native:
        monkeypatch.setattr(driver, "_native_launch", None)
        monkeypatch.setattr(driver, "_native_fixed_launcher", None)
    _, kernel = _kernels()
    runtime_launch = _runtime.launch
    runtime_launches = 0

    def counted_launch(*args, **kwargs):
        nonlocal runtime_launches
        runtime_launches += 1
        return runtime_launch(*args, **kwargs)

    monkeypatch.setattr(_runtime, "launch", counted_launch)
    x = torch.full((129,), 2.0, device="cuda").to(dtype)
    y = torch.full_like(x, 3.0)
    output = torch.empty_like(x)
    arguments = {"x_ptr": x, "y_ptr": y, "output_ptr": output, "n": 129}

    def launch():
        kernel.launch(arguments=arguments, constexprs={"BLOCK": 128}, grid=(2,))

    launch()
    if native:
        for _ in range(3):
            before = runtime_launches
            launch()
            if runtime_launches == before:
                break
        else:
            pytest.fail("prepared CUDA path did not become active")
        assert kernel._cuda_fast_launch is not None
        assert kernel._cuda_fast_launch() is not None
    else:
        launch()
        assert kernel._cuda_fast_launch is None
        assert runtime_launches == 2
    torch.cuda.synchronize()
    x = torch.full((129,), 4.0, device="cuda").to(dtype)
    y = torch.full((129,), 3.0, device="cuda").to(dtype)
    output = torch.empty((129,), device="cuda", dtype=dtype)
    arguments.update(x_ptr=x, y_ptr=y, output_ptr=output)
    before = runtime_launches
    launch()
    torch.cuda.synchronize()
    assert torch.equal(output, torch.full_like(output, 12))
    assert runtime_launches == before + (not native)
    streams = (torch.cuda.Stream(), torch.cuda.Stream())
    for value, stream in enumerate((*streams, *streams), start=5):
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
    artifact_keys = tuple(_runtime._artifact_cache)
    assert artifact_keys
    _runtime._artifact_cache.clear()
    assert not any(key in _runtime._artifact_cache for key in artifact_keys)
    x.fill_(-2)
    y.fill_(4)
    output.fill_(0)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(output, torch.full_like(output, -8))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_same_name_operations_reuse_eight_persistent_artifacts_in_child(
    tmp_path, monkeypatch, caplog
):
    """Reuse all add/multiply PTX without compiling in process two."""
    identity = _runtime._cached_identity()
    if not identity["clean"]:
        if os.environ.get("SWAGE_REQUIRE_PERSISTENT_CACHE_TEST") == "1":
            pytest.fail("persistent cache regression requires a clean build")
        pytest.skip("dirty builds intentionally disable persistent caching")
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("SWAGE_CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(_runtime, "_artifact_cache", _runtime.OrderedDict())
    caplog.set_level(logging.DEBUG, logger="swage.runtime")

    for kernel, expected in zip(_kernels(), (5.0, 6.0)):
        for dtype in (
            torch.float32,
            torch.float16,
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ):
            x = torch.full((129,), 2.0, device="cuda").to(dtype)
            y = torch.full((129,), 3.0, device="cuda").to(dtype)
            output = torch.empty_like(x)
            kernel.launch(
                arguments={
                    "x_ptr": x,
                    "y_ptr": y,
                    "output_ptr": output,
                    "n": 129,
                },
                constexprs={"BLOCK": 128},
                grid=(2,),
            )
            torch.cuda.synchronize()
            assert torch.equal(
                output.float(), torch.full_like(x.float(), expected)
            )

    assert sum(
        record.message.startswith("compile ") for record in caplog.records
    ) == 8
    assert len(list(cache_dir.glob("*/metadata.json"))) == 8
    script = textwrap.dedent(
        """
        import logging

        import torch
        from swage import _cuda_backend
        from test_operation_dispatch import _kernels

        def fail_compile(*_args, **_kwargs):
            raise AssertionError("process two compiled instead of using PTX")

        class Hits(logging.Handler):
            def __init__(self):
                super().__init__(logging.DEBUG)
                self.count = 0

            def emit(self, record):
                self.count += record.getMessage().startswith("persistent-hit")

        _cuda_backend.CUDA_BACKEND.compile = fail_compile
        hits = Hits()
        logger = logging.getLogger("swage.runtime")
        logger.addHandler(hits)
        logger.setLevel(logging.DEBUG)
        for kernel, expected in zip(_kernels(), (5.0, 6.0)):
            for dtype in (
                torch.float32,
                torch.float16,
                torch.float8_e4m3fn,
                torch.float8_e5m2,
            ):
                x = torch.full((129,), 2.0, device="cuda").to(dtype)
                y = torch.full((129,), 3.0, device="cuda").to(dtype)
                output = torch.empty_like(x)
                kernel.launch(
                    arguments={
                        "x_ptr": x,
                        "y_ptr": y,
                        "output_ptr": output,
                        "n": 129,
                    },
                    constexprs={"BLOCK": 128},
                    grid=(2,),
                )
                torch.cuda.synchronize()
                assert torch.equal(
                    output.float(), torch.full_like(x.float(), expected)
                )
        assert hits.count == 8
        """
    )
    environment = os.environ.copy()
    test_dir = pathlib.Path(__file__).resolve().parent
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(test_dir), environment.get("PYTHONPATH")))
    )
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        cwd=tmp_path,
        env=environment,
        timeout=60,
    )


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


def _boundary_launch(backend, operation, warm, dtype=torch.float32):
    if backend == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    _runtime._artifact_cache.clear()
    kernel = _kernels()[operation]

    def launch(tensors, n=129):
        kernel.launch(
            arguments=dict(
                zip(("x_ptr", "y_ptr", "output_ptr", "n"), (*tensors, n))
            ),
            constexprs={"BLOCK": 128},
            grid=((n + 127) // 128,),
            backend=backend,
        )

    if warm:
        tensors = [
            torch.full((129,), value, device=backend, dtype=dtype)
            for value in (2.0, 3.0, -7.0)
        ]
        for _ in range(2):
            launch(tensors)
    return launch


@pytest.mark.parametrize(
    ("backend", "operation", "dtype", "warm"),
    [
        ("cpu", 0, torch.float32, False),
        ("cpu", 1, torch.float32, False),
        ("cuda", 0, torch.float32, False),
        ("cuda", 1, torch.float32, False),
        ("cuda", 0, torch.float32, True),
        ("cuda", 1, torch.float8_e4m3fn, True),
    ],
)
def test_partial_active_overlap_is_rejected(backend, operation, dtype, warm):
    """Reject shifted active overlap before compilation or writes."""
    launch = _boundary_launch(backend, operation, warm, dtype)
    for position, name in enumerate(("x_ptr", "y_ptr")):
        for input_start, output_start in ((1, 129), (129, 1)):
            storage = torch.full((260,), -7.0, device=backend, dtype=dtype)
            tensors = [
                torch.full_like(storage[:129], value) for value in (2, 3)
            ]
            tensors[position] = storage[input_start : input_start + 129]
            tensors.append(storage[output_start : output_start + 129])
            before = storage.clone()
            artifacts = tuple(_runtime._artifact_cache)
            with pytest.raises(ValueError, match=f"overlap.*{name}"):
                launch(tensors)
            assert tuple(_runtime._artifact_cache) == artifacts
            assert torch.equal(
                storage.view(torch.uint8), before.view(torch.uint8)
            )


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
@pytest.mark.parametrize("operation", [0, 1], ids=["add", "multiply"])
def test_negative_metadata_is_rejected_without_writes(backend, operation):
    """Reject every unresolved argument, including on a prepared CUDA call."""
    launch = _boundary_launch(backend, operation, warm=backend == "cuda")
    for position, name in enumerate(("x_ptr", "y_ptr", "output_ptr")):
        storage = [
            torch.full((129,), value, device=backend)
            for value in (2.0, 3.0, -7.0)
        ]
        tensors = list(storage)
        tensors[position] = storage[position]._neg_view()
        assert tensors[position].is_contiguous()
        assert tensors[position].is_neg()
        before = [tensor.clone() for tensor in storage]
        artifacts = tuple(_runtime._artifact_cache)
        for n in (129, 0):
            with pytest.raises(ValueError, match=f"{name}.*negative"):
                launch(tensors, n)
            assert tuple(_runtime._artifact_cache) == artifacts
            for tensor, expected in zip(storage, before):
                assert torch.equal(tensor, expected)
