# python/tests/mlir/test_exit_and_fork.py
"""Interpreter exit and fork while a native compile is in flight.

The native compiler releases the GIL, and the runtime holds its cold-path
lock for a whole compile. An interpreter that finalizes under a compile
crashes, and a child forked under one inherits a lock that nobody releases.
The runtime registers an exit handler and fork handlers against both.

Each test runs `_PROGRAM` in fresh interpreters, where a thread keeps
compiling through the runtime while the main thread exits or forks. Nothing
here uses a GPU.
"""

import importlib.util
import json
import os
import pathlib
import subprocess
import sys

import pytest
import swage as sw

# Exits per test. Without the exit handler about one exit in four crashes,
# so all of them succeed by chance once in a thousand runs.
_EXITS = 24
_FORKS = 5
# The program imports as little as it can. An interpreter that takes longer
# to finalize than a compile takes leaves the compiling thread waiting for
# the GIL by the time the process tears down, and the crash does not show.
_PROGRAM = """
import json
import os
import signal
import sys
import threading
import time
import warnings

from mlir_swage._mlir_libs._swageDialectsNanobind import swage as native
from swage import _segmented_qualification as qualification

text = qualification._semantic_module("sum")
started = threading.Event()
stop = threading.Event()


def compile_variant(label):
    return qualification._compile_once(
        native._compile_persistent_segmented_reduction_ptx,
        text + f"// {label}\\n",
        kernel_name="segmented_sum",
        target="sm_86",
    )


def keep_compiling():
    index = 0
    while not stop.is_set():
        index += 1
        compile_variant(f"variant {index}")
        started.set()


if sys.argv[1] == "exit":
    # The main thread returns while a daemon thread compiles.
    threading.Thread(target=keep_compiling, daemon=True).start()
    started.wait()
    time.sleep(float(sys.argv[2]))
else:
    # Children that compile are forked while another thread compiles.
    thread = threading.Thread(target=keep_compiling)
    thread.start()
    started.wait()
    warnings.simplefilter("ignore", DeprecationWarning)
    outcomes = []
    for attempt in range(int(sys.argv[2])):
        time.sleep(0.013)
        child = os.fork()
        if child == 0:
            # SIGALRM ends a child that waits for a lock nobody releases.
            signal.alarm(10)
            ptx = compile_variant(f"child {attempt}")
            os._exit(0 if ".entry segmented_sum" in ptx else 3)
        _, status = os.waitpid(child, 0)
        if os.WIFSIGNALED(status):
            outcomes.append(f"signal {os.WTERMSIG(status)}")
        else:
            outcomes.append(f"exit {os.WEXITSTATUS(status)}")
    stop.set()
    thread.join()
    print(json.dumps(outcomes))
"""


def _program(arguments, tmp_path):
    """Return the command and the environment that run `_PROGRAM`."""
    bindings = importlib.util.find_spec("mlir_swage._mlir_libs")
    bindings_parent = pathlib.Path(
        list(bindings.submodule_search_locations)[0]
    ).parents[1]
    environment = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(
            [str(pathlib.Path(sw.__file__).parents[1]), str(bindings_parent)]
        ),
        SWAGE_CACHE_DIR=str(tmp_path / "cache"),
        CUDA_VISIBLE_DEVICES="",
    )
    environment.pop("SWAGE_NO_COMPILE", None)
    return [sys.executable, "-c", _PROGRAM, *arguments], environment


def test_interpreter_exit_during_a_compile_is_clean(tmp_path):
    """Exit normally at every moment of a compile on a daemon thread."""
    outcomes = []
    # A few at a time, so that a loaded machine does not stretch the exits.
    for first in range(0, _EXITS, 4):
        processes = []
        for index in range(first, first + 4):
            # The delays cover a whole compile, which takes up to 20 ms.
            command, environment = _program(
                ["exit", str(0.05 * index / _EXITS)], tmp_path
            )
            processes.append(
                subprocess.Popen(
                    command,
                    env=environment,
                    cwd=tmp_path,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            )
        for process in processes:
            _, errors = process.communicate(timeout=300)
            outcomes.append((process.returncode, errors.strip()[-200:]))

    assert outcomes == [(0, "")] * _EXITS


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork")
def test_children_forked_during_compiles_can_compile(tmp_path):
    """Compile in a child that was forked while its parent compiled."""
    command, environment = _program(["fork", str(_FORKS)], tmp_path)

    completed = subprocess.run(
        command,
        env=environment,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout.splitlines()[-1]) == ["exit 0"] * _FORKS
