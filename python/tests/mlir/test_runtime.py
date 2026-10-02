# python/tests/mlir/test_runtime.py
"""Real CUDA tests for the fixed vector-add launch boundary."""

import gc
import threading
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
    # The inputs are produced on the default stream. Finish them before
    # another stream reads them; launch itself adds no cross-stream order.
    torch.cuda.synchronize()
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


def test_launch_rejects_a_lazy_negation_view():
    """Refuse a view whose storage holds the opposite of what it shows."""
    base = torch.ones(129, device="cuda")
    if not hasattr(base, "_neg_view"):
        pytest.skip("this PyTorch cannot build a contiguous negation view")
    view = base._neg_view()
    y = torch.ones(129, device="cuda")
    output = torch.full((129,), -777.0, device="cuda")
    assert view.is_neg() and view.is_contiguous()
    assert view.data_ptr() == base.data_ptr()

    with pytest.raises(ValueError, match="'x_ptr' must not be a lazy negation"):
        _launch(view, y, output, 129)
    with pytest.raises(ValueError, match="'output_ptr' must not be a lazy"):
        _launch(base, y, view, 129)
    _launch(view.resolve_neg(), y, output, 129)

    torch.cuda.synchronize()
    assert torch.all(base == 1.0)
    assert torch.all(output == 0.0)


def test_launch_rejects_tensors_that_require_grad():
    """Refuse a tensor autograd tracks, because a launch records nothing.

    A leaf, a result with a grad function, and a leaf under `no_grad` all
    require grad. The detached tensor shares their storage and is admitted.
    """
    leaf = torch.ones(129, device="cuda", requires_grad=True)
    y = torch.ones(129, device="cuda")
    output = torch.full((129,), -777.0, device="cuda")

    with pytest.raises(ValueError, match="'x_ptr' must not require grad"):
        _launch(leaf, y, output, 129)
    with pytest.raises(ValueError, match="'y_ptr' must not require grad"):
        _launch(y, leaf * 2, output, 129)
    with pytest.raises(ValueError, match="'output_ptr' must not require grad"):
        _launch(y, y, leaf, 129)
    with torch.no_grad():
        with pytest.raises(ValueError, match="'x_ptr' must not require grad"):
            _launch(leaf, y, output, 129)
    torch.cuda.synchronize()
    assert torch.all(output == -777.0)
    assert torch.all(leaf == 1.0)

    _launch(leaf.detach(), y, output, 129)

    torch.cuda.synchronize()
    assert torch.all(output == 2.0)


def test_launch_advances_the_output_version_for_autograd():
    """Make autograd refuse a value that a launch overwrote.

    The product saves the output for its backward pass. PyTorch cannot see
    the kernel store, so without the advance the backward pass would use
    the overwritten output and return a wrong gradient without an error.
    """
    x = torch.arange(129, dtype=torch.float32, device="cuda")
    y = torch.ones(129, device="cuda")
    output = torch.zeros(129, device="cuda")
    weights = torch.ones(129, device="cuda", requires_grad=True)
    loss = (weights * output).sum()
    versions = [tensor._version for tensor in (x, y, output)]

    _launch(x, y, output, 129)

    torch.cuda.synchronize()
    assert torch.equal(output, x + y)
    assert [x._version, y._version] == versions[:2]
    assert output._version == versions[2] + 1
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        loss.backward()


def test_captured_launch_advances_the_output_version_at_capture_only():
    """Advance the counter in the launch call, which a replay does not run."""
    x = torch.arange(129, dtype=torch.float32, device="cuda")
    y = torch.ones(129, device="cuda")
    output = torch.zeros(129, device="cuda")
    _launch(x, y, output, 129)
    torch.cuda.synchronize()
    version = output._version

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _launch(x, y, output, 129)
    assert output._version == version + 1

    output.zero_()
    graph.replay()
    torch.cuda.synchronize()

    assert output._version == version + 2
    assert torch.equal(output, x + y)


def test_launch_accepts_inference_tensors():
    """Launch on tensors that have no version counter to advance."""
    with torch.inference_mode():
        x = torch.arange(129, dtype=torch.float32, device="cuda")
        y = torch.ones(129, device="cuda")
        output = torch.zeros(129, device="cuda")
        assert output.is_inference()

        _launch(x, y, output, 129)

        torch.cuda.synchronize()
        assert torch.equal(output, x + y)


@pytest.mark.parametrize("shift", [0, 1, 128, -1, -128])
@pytest.mark.parametrize("overlapped", ["x_ptr", "y_ptr"])
def test_launch_rejects_an_output_that_overlaps_an_input(overlapped, shift):
    """Refuse an output sharing memory with a buffer the kernel reads."""
    buffer = torch.arange(512, device="cuda", dtype=torch.float32)
    expected = buffer.clone()
    shared = buffer[128:257]
    output = buffer[128 + shift:257 + shift]
    other = torch.ones(129, device="cuda")
    x, y = (shared, other) if overlapped == "x_ptr" else (other, shared)

    with pytest.raises(
        ValueError,
        match=f"'output_ptr' must not overlap argument '{overlapped}'",
    ):
        _launch(x, y, output, 129)

    torch.cuda.synchronize()
    assert torch.equal(buffer, expected)


def test_launch_accepts_adjacent_slices_and_one_tensor_for_both_inputs():
    """Run buffers that only touch, and two inputs that share memory."""
    buffer = torch.arange(258, device="cuda", dtype=torch.float32)
    x, output = buffer[:129], buffer[129:]
    expected = x + x

    _launch(x, x, output, 129)

    torch.cuda.synchronize()
    assert torch.equal(output, expected)
    assert torch.equal(x, torch.arange(129, device="cuda", dtype=torch.float32))


def _fresh_process_cache(monkeypatch, cache):
    """Start from an empty in-process cache on the cache root `cache`."""
    from swage import _runtime

    monkeypatch.setenv("SWAGE_CACHE_DIR", str(cache))
    monkeypatch.setattr(_runtime, "_ptx_cache", {})
    monkeypatch.setattr(_runtime, "_cache_off", {})
    for name in (
        "SWAGE_CACHE_MAX_ENTRIES",
        "SWAGE_CACHE_READ_ONLY",
        "SWAGE_NO_COMPILE",
    ):
        monkeypatch.delenv(name, raising=False)
    return _runtime


def test_no_compile_mode_launches_from_the_cache_and_refuses_a_miss(
    tmp_path, monkeypatch
):
    """Launch a published kernel without compiling, and refuse another."""
    cache = tmp_path / "cache"
    _runtime = _fresh_process_cache(monkeypatch, cache)
    x = torch.randn(129, device="cuda")
    y = torch.randn(129, device="cuda")
    output = torch.empty_like(x)
    _launch(x, y, output, 129, block=64)
    torch.cuda.synchronize()
    (entry,) = cache.iterdir()

    def compile_nothing(*_arguments, **_keywords):
        raise AssertionError("compiled although SWAGE_NO_COMPILE=1")

    monkeypatch.setattr(_runtime, "_ptx_cache", {})
    monkeypatch.setattr(_runtime, "_compile_native", compile_nothing)
    monkeypatch.setenv("SWAGE_NO_COMPILE", "1")
    warm = torch.full_like(x, -777.0)
    untouched = torch.full_like(x, -777.0)

    _launch(x, y, warm, 129, block=64)
    with pytest.raises(
        RuntimeError,
        match=(
            "SWAGE_NO_COMPILE=1 refuses to compile kernel 'add_kernel': "
            "no entry [0-9a-f]{64} in "
        ),
    ):
        _launch(x, y, untouched, 129, block=256)

    torch.cuda.synchronize()
    torch.testing.assert_close(warm, x + y)
    assert torch.all(untouched == -777.0)
    assert [path.name for path in cache.iterdir()] == [entry.name]


def test_read_only_mode_launches_without_writing_the_cache(
    tmp_path, monkeypatch
):
    """Compile and launch while leaving a missing cache root missing."""
    cache = tmp_path / "cache"
    _fresh_process_cache(monkeypatch, cache)
    monkeypatch.setenv("SWAGE_CACHE_READ_ONLY", "1")
    x = torch.randn(129, device="cuda")
    y = torch.randn(129, device="cuda")
    output = torch.empty_like(x)

    for _ in range(2):
        _launch(x, y, output, 129, block=32)

    torch.cuda.synchronize()
    torch.testing.assert_close(output, x + y)
    assert not cache.exists()


def test_cache_bound_evicts_real_entries(tmp_path, monkeypatch):
    """Keep the two newest specializations and recompile an evicted one."""
    cache = tmp_path / "cache"
    _runtime = _fresh_process_cache(monkeypatch, cache)
    monkeypatch.setenv("SWAGE_CACHE_MAX_ENTRIES", "2")
    x = torch.randn(129, device="cuda")
    y = torch.randn(129, device="cuda")
    output = torch.empty_like(x)
    keys = []

    for block in (32, 64, 128):
        _launch(x, y, output, 129, block=block)
        keys.append(list(_runtime._ptx_cache)[-1])
    torch.cuda.synchronize()
    assert sorted(path.name for path in cache.iterdir()) == sorted(keys[1:])

    monkeypatch.setattr(_runtime, "_ptx_cache", {})
    output.fill_(-777.0)
    _launch(x, y, output, 129, block=32)

    torch.cuda.synchronize()
    torch.testing.assert_close(output, x + y)
    assert sorted(path.name for path in cache.iterdir()) == sorted(
        [keys[2], keys[0]]
    )


def test_environment_report_describes_this_device(tmp_path, monkeypatch):
    """Report the loaded bindings, the driver, the target, and the cache."""
    import mlir_swage._mlir_libs._swageDialectsNanobind as extension
    from swage import env

    cache = tmp_path / "cache"
    _fresh_process_cache(monkeypatch, cache)
    major, minor = torch.cuda.get_device_capability()

    report = env.report()

    assert report["swage_file"] == sw.__file__
    assert report["mlir_swage_file"] == extension.__file__
    assert report["cuda_driver"]
    assert report["target"].startswith(f"sm_{major}{minor} (")
    assert report["cache_dir"] == str(cache)
    assert report["cache"] == (
        "active (reads and writes; 0 of at most 1024 entries)"
    )
    assert report["compile_on_miss"] == "allowed"
    assert not cache.exists()


@pytest.mark.parametrize("block", [128, 48])
def test_launch_works_on_a_thread_that_has_not_used_cuda(block):
    """Make PyTorch's context current on the thread and create no other.

    A new thread has no current CUDA context. The launch gives it the
    context of the validated device, for a kernel the process has loaded
    (block 128) and for one it compiles and loads on that thread (block 48).
    """
    from swage import _runtime

    x = torch.randn(129, device="cuda")
    y = torch.randn(129, device="cuda")
    output = torch.full_like(x, -777.0)
    _launch(x, y, torch.empty_like(x), 129)
    torch.cuda.synchronize()
    driver = _runtime._get_driver()
    main_context = driver.current_context()
    seen = {}

    def launch_first_thing():
        try:
            try:
                seen["before"] = driver.current_context()
            except RuntimeError as error:
                seen["before"] = str(error)
            _launch(x, y, output, 129, block=block)
            seen["after"] = driver.current_context()
        except Exception as error:  # Reported by the assertion below.
            seen["error"] = error

    worker = threading.Thread(target=launch_first_thing)
    worker.start()
    worker.join()
    torch.cuda.synchronize()

    assert "error" not in seen, seen
    assert seen["before"] == "PyTorch has no current CUDA context"
    # Context ids are unique for the life of the process, so an equal id is
    # the context PyTorch already had, not a new one.
    assert seen["after"] == main_context
    assert driver.current_context() == main_context
    torch.testing.assert_close(output, x + y)


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
