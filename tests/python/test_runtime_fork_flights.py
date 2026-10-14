# tests/python/test_runtime_fork_flights.py
"""Bounded LLVM-free regressions for compile flights inherited by fork."""

import os
import pathlib
import subprocess
import sys

import pytest
import swage

_FORK_FLIGHT_SCRIPT = """
import contextlib
import os
import signal
import sys
import threading
import types
import warnings

from swage import _abi, _artifact, _runtime
from swage import _segmented_runtime as execution

signal.alarm(10)
warnings.simplefilter("ignore", DeprecationWarning)
mode, gap = sys.argv[1:]
parent = os.getpid()
calls = []
contract = _abi.serialize_kernel_contract(_abi.KernelContract(
    2, "cuda", "kernel", _abi.KernelLaunch("spmd-grid", (128, 1, 1)),
    (
        _abi.KernelArgument("ptr", "user", 0, access="read"),
        _abi.KernelArgument("ptr", "user", 1, access="read"),
        _abi.KernelArgument("ptr", "user", 2, access="write"),
        _abi.KernelArgument("i32", "user", 3),
    ),
))


def compile_kernel(*_args, **_options):
    calls.append(os.getpid())
    return "lowered", f"ptx of {os.getpid()}", contract


adapter = types.SimpleNamespace(
    name="cuda", artifact_format="ptx", persistent_cache=False,
    compile=compile_kernel,
)
specialization = {
    "kernel": "kernel", "backend": "cuda", "format": "ptx",
    "target": "sm_86",
    "descriptors": ["ptr<f32>", "ptr<f32>", "ptr<f32>", "i32"],
    "codegen": {"lowering": "fixed", "block_size": 128, "options": []},
}
_runtime._cached_identity = lambda: {}
_artifact.selected = lambda: None


def request(key):
    if mode == "fixed":
        return _runtime._compile_cached(
            adapter, specialization, "kernel", 128, object, key=key,
        )
    return execution._compile_once(
        compile_kernel, key, module=object(), kernel_name="kernel",
        block_size=128, target="sm_86",
    )


warm = request("warm")
calls.clear()
paused = threading.Event()
release = threading.Event()
results = []
errors = []


def pause_parent():
    if os.getpid() == parent:
        paused.set()
        assert release.wait(8), "parent build was never released"


if gap == "before-place":
    compiling = _runtime._compile_lock.compiling

    @contextlib.contextmanager
    def delayed_place():
        pause_parent()
        with compiling():
            yield

    _runtime._compile_lock.compiling = delayed_place
elif gap == "before-publication":
    checked = execution._checked_kernel

    def delayed_publication(*args):
        kernel = checked(*args)
        assert not _runtime._compile_lock._compiles
        pause_parent()
        return kernel

    execution._checked_kernel = delayed_publication


def run_request():
    try:
        results.append(request("shared"))
    except BaseException as error:
        errors.append(error)


def hold_flight_mutex():
    with _runtime._flights_lock:
        pause_parent()


builder = threading.Thread(
    target=hold_flight_mutex if gap == "flight-mutex" else run_request,
    daemon=True,
)
builder.start()
assert paused.wait(5), "parent did not reach the requested fork gap"
if gap != "flight-mutex":
    flights = (
        _runtime._compile_flights if mode == "fixed" else execution._flights
    )
    flight, = flights.values()
    wait = flight.wait
    waiting = threading.Event()

    def observed_wait(*args, **kwargs):
        waiting.set()
        return wait(*args, **kwargs)

    flight.wait = observed_wait
    waiter = threading.Thread(target=run_request, daemon=True)
    waiter.start()
    assert waiting.wait(5), "second parent caller did not coalesce"

child = os.fork()
if child == 0:
    # No native compiler or CUDA driver is called in this process.
    signal.alarm(3)
    try:
        assert request("warm") is warm, "child lost an inherited warm kernel"
        kernel = request("shared")
        assert kernel.image == f"ptx of {os.getpid()}"
        assert request("shared") is kernel
    except BaseException:
        os._exit(2)
    os._exit(0)

_, status = os.waitpid(child, 0)
release.set()
builder.join(5)
assert not builder.is_alive(), "parent builder did not finish"
if gap != "flight-mutex":
    waiter.join(5)
    assert not waiter.is_alive(), "parent waiter did not finish"
    assert not errors, errors
    assert len(results) == 2 and results[0] is results[1]
else:
    results.append(request("shared"))
assert calls == [parent], calls
assert request("shared") is results[0]
assert request("warm") is warm
print("parent coalescing and caches preserved")
if os.WIFSIGNALED(status):
    print("child ended by signal", os.WTERMSIG(status))
else:
    print("child exited", os.WEXITSTATUS(status))
"""

_FORK_IN_BUILDER_SCRIPT = """
import os
import signal
import sys
import threading
import warnings

from swage import _runtime
from swage import _segmented_runtime as execution

signal.alarm(10)
warnings.simplefilter("ignore", DeprecationWarning)
flights = (
    _runtime._compile_flights if sys.argv[1] == "fixed" else execution._flights
)
parent = os.getpid()
child = None
started = threading.Event()
release = threading.Event()
cache = {}
results = []


def successor():
    def build():
        started.set()
        assert release.wait(5)
        cache["shared"] = "child value"
        return cache["shared"]

    results.append(_runtime._single_flight(
        flights, "shared", lambda: cache.get("shared"), build,
    ))


def fork_in_build():
    global child, worker
    child = os.fork()
    if child == 0:
        signal.alarm(3)
        worker = threading.Thread(target=successor, daemon=True)
        worker.start()
        assert started.wait(2), "child waited on the parent's flight"
    return "original value"


try:
    assert _runtime._single_flight(
        flights, "shared", lambda: None, fork_in_build,
    ) == "original value"
    if os.getpid() != parent:
        # The original builder must not remove or signal its successor.
        assert "shared" in flights
        assert not flights["shared"].is_set()
        release.set()
        worker.join(2)
        assert results == ["child value"]
        assert not flights
        os._exit(0)
except BaseException:
    if os.getpid() != parent:
        os._exit(2)
    raise

_, status = os.waitpid(child, 0)
assert not flights
print("child exited", os.waitstatus_to_exitcode(status))
"""


def _run_fork_script(script, *arguments):
    """Run a fork regression with a bound outside the child alarm too."""
    environment = dict(
        os.environ,
        PYTHONPATH=str(pathlib.Path(swage.__file__).parents[1]),
        SWAGE_NO_COMPILE="0",
    )
    environment.pop("SWAGE_ARTIFACT_DIR", None)
    return subprocess.run(
        [sys.executable, "-c", script, *arguments],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork")
@pytest.mark.parametrize(
    ("mode", "gap"),
    [
        ("fixed", "before-place"),
        ("segmented", "before-place"),
        ("segmented", "before-publication"),
        ("fixed", "flight-mutex"),
        ("segmented", "flight-mutex"),
    ],
)
def test_child_retries_same_key_across_flight_publication_gaps(mode, gap):
    """Reset orphaned child flights while retaining parent coalescing."""
    completed = _run_fork_script(_FORK_FLIGHT_SCRIPT, mode, gap)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == (
        "parent coalescing and caches preserved\nchild exited 0\n"
    )


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork")
@pytest.mark.parametrize("mode", ["fixed", "segmented"])
def test_forking_builder_does_not_finish_a_new_child_flight(mode):
    """Let the inherited builder unwind without retiring its successor."""
    completed = _run_fork_script(_FORK_IN_BUILDER_SCRIPT, mode)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "child exited 0\n"
