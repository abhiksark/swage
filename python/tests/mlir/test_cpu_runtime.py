"""Real Native CPU tests for the canonical fixed-vector launch boundary."""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import swage as sw
import swage.language as sl
import torch
from swage import _cuda_backend, _runtime


@sw.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n, BLOCK: sl.constexpr):  # noqa: D103
    pid = sl.program_id(0)
    offsets = pid * BLOCK + sl.arange(0, BLOCK)
    mask = offsets < n
    x = sl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = sl.load(y_ptr + offsets, mask=mask, other=0.0)
    sl.store(output_ptr + offsets, x + y, mask=mask)


def _reset_runtime():
    _runtime._artifact_cache.clear()
    _runtime._compilations.clear()
    add_kernel.__dict__.pop("_specialization_memo", None)


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
        backend="cpu",
    )


def test_native_cpu_boundaries_and_process_artifact_reuse(monkeypatch):
    """Run every fixed boundary and compile one shared nonzero artifact."""
    from mlir_swage._mlir_libs._swageDialectsNanobind import (
        swage as native_swage,
    )

    _reset_runtime()
    original = native_swage._compile_fixed_host
    compiles = []

    def compile_host(*args, **kwargs):
        compiles.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(native_swage, "_compile_fixed_host", compile_host)
    for n in [0, 1, 127, 128, 129, 4097]:
        x = torch.randn(n)
        y = torch.randn(n)
        output = torch.empty_like(x)
        _launch(x, y, output, n)
        torch.testing.assert_close(output, torch.add(x, y))
        assert len(compiles) == (0 if n == 0 else 1)


def test_native_cpu_first_compile_is_coalesced_before_concurrent_invoke(
    monkeypatch,
):
    """Publish one eagerly initialized engine to two concurrent callers."""
    from mlir_swage._mlir_libs._swageDialectsNanobind import (
        swage as native_swage,
    )

    _reset_runtime()
    original = native_swage._compile_fixed_host
    compile_lock = threading.Lock()
    compile_count = 0
    callers = threading.Barrier(2)

    def compile_host(*args, **kwargs):
        nonlocal compile_count
        with compile_lock:
            compile_count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(native_swage, "_compile_fixed_host", compile_host)

    def run(seed):
        generator = torch.Generator().manual_seed(seed)
        x = torch.randn(4097, generator=generator)
        y = torch.randn(4097, generator=generator)
        output = torch.empty_like(x)
        callers.wait()
        _launch(x, y, output, x.numel())
        torch.testing.assert_close(output, x + y)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run, seed) for seed in (11, 29)]
        for future in futures:
            future.result()

    assert compile_count == 1


def test_cpu_validation_and_failures_never_consult_cuda(monkeypatch):
    """Reject CUDA tensors and CPU compilation failures without fallback."""
    _reset_runtime()

    def unexpected(*_args, **_kwargs):
        pytest.fail("CPU routing consulted the CUDA adapter")

    monkeypatch.setattr(_cuda_backend.CUDA_BACKEND, "compile", unexpected)
    monkeypatch.setattr(_cuda_backend.CUDA_BACKEND, "lease", unexpected)
    monkeypatch.setattr(_cuda_backend.CUDA_BACKEND, "launch", unexpected)

    x = torch.empty(4)
    with pytest.raises(TypeError, match="must be a CPU tensor"):
        _launch(x, x, torch.empty(4, device="meta"), 4)

    from swage import _cpu_backend

    monkeypatch.setattr(
        _cpu_backend.CPU_BACKEND,
        "compile",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("host compile failed")
        ),
    )
    with pytest.raises(RuntimeError, match="host compile failed"):
        _launch(x, x, torch.empty_like(x), 4)
