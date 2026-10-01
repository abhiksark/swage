# benchmarks/benchmark_provenance.py
"""Provenance block shared by the benchmark harnesses.

A record names its source revision, but a revision does not say which
native library produced the PTX, which machine ran it, or what else was
running on the GPU. This module collects those facts so that every harness
writes the same block.

Nothing here fails a benchmark. A fact that cannot be read is recorded as
None with the reason, so a reader can tell an exclusive GPU from an unknown
one. PyTorch, Triton, and the native bindings are imported by the caller or
inside functions, never when this module is imported.
"""

import hashlib
import importlib.metadata
import os
import pathlib
import subprocess
import time
from datetime import datetime, timezone

_GPU_FIELDS = (
    "uuid",
    "name",
    "driver_version",
    "compute_mode",
    "pstate",
    "temperature.gpu",
    "power.draw",
    "power.limit",
    "clocks.current.graphics",
    "clocks.current.memory",
    "utilization.gpu",
)
_PROCESS_FIELDS = ("gpu_uuid", "pid", "process_name", "used_memory")
_SAME_READING = 1e-6


def cpu_model(cpuinfo=pathlib.Path("/proc/cpuinfo")):
    """Return the host processor model, or None when it is not reported."""
    try:
        lines = cpuinfo.read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        name, _, value = line.partition(":")
        if name.strip() == "model name":
            return value.strip()
    return None


def package_version(name):
    """Return an installed package version without importing the package."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def native_sha256(paths):
    """Hash the native library files that produce the PTX.

    Args:
        paths: Library paths; several may be links to one file.

    Returns:
        The SHA-256 of each distinct file by its resolved path.
    """
    digests = {}
    for path in sorted({pathlib.Path(path).resolve() for path in paths}):
        digest = hashlib.sha256()
        with path.open("rb") as library:
            for chunk in iter(lambda: library.read(1 << 20), b""):
                digest.update(chunk)
        digests[str(path)] = digest.hexdigest()
    return digests


def device_uuid(torch, device):
    """Return the device UUID as nvidia-smi spells it, or None."""
    uuid = getattr(torch.cuda.get_device_properties(device), "uuid", None)
    return None if uuid is None else f"GPU-{uuid}"


def _query(run, kind, fields):
    """Return the rows of one nvidia-smi query, split into fields."""
    result = run(
        [
            "nvidia-smi",
            f"--query-{kind}={','.join(fields)}",
            "--format=csv,noheader",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(
            f"nvidia-smi --query-{kind} exited {result.returncode}: {detail}"
        )
    return [line for line in result.stdout.splitlines() if line.strip()]


def gpu_state(uuid, *, pid, run=subprocess.run):
    """Sample the measured GPU and the other compute processes on it.

    Args:
        uuid: UUID of the measured device as nvidia-smi spells it, or None
            when PyTorch does not report one. Without a UUID the state is
            read only when the machine has a single GPU.
        pid: Process id of the benchmark itself, left out of the list.
        run: ``subprocess.run`` or a stand-in.

    Returns:
        ``gpu`` with the driver version, clocks, power, and temperature;
        ``other_compute_processes`` with one entry per other process, an
        empty list when there was none; ``error``; and the sample time. When
        nvidia-smi cannot answer, ``gpu`` and ``other_compute_processes``
        are None and ``error`` says why.
    """
    state = {
        "sampled_at": datetime.now(timezone.utc).isoformat(),
        "gpu": None,
        "other_compute_processes": None,
        "error": None,
    }
    try:
        gpus = [
            dict(zip(_GPU_FIELDS, row.split(", "), strict=True))
            for row in _query(run, "gpu", _GPU_FIELDS)
        ]
        if uuid is None and len(gpus) != 1:
            raise RuntimeError(
                f"the measured device has no UUID and nvidia-smi lists "
                f"{len(gpus)} GPUs"
            )
        measured = gpus[0]["uuid"] if uuid is None else uuid
        processes = []
        for row in _query(run, "compute-apps", _PROCESS_FIELDS):
            gpu_uuid, process_id, rest = row.split(", ", 2)
            name, _, memory = rest.rpartition(", ")
            if gpu_uuid == measured and int(process_id) != pid:
                processes.append(
                    {
                        "pid": int(process_id),
                        "process_name": name,
                        "used_memory": memory,
                    }
                )
        state["gpu"] = next(
            (gpu for gpu in gpus if gpu["uuid"] == measured), None
        )
        state["other_compute_processes"] = processes
    except (OSError, RuntimeError, ValueError) as error:
        state["error"] = f"{type(error).__name__}: {error}"
    return state


def other_compute_process_seen(before, after):
    """Return whether another compute process was seen on the GPU.

    Returns:
        True when either sample lists one, False when both samples are
        empty, and None when a sample is missing and none was seen, because
        an unreadable GPU is not an exclusive GPU.
    """
    samples = [
        state["other_compute_processes"] for state in (before, after)
    ]
    if any(samples):
        return True
    if any(sample is None for sample in samples):
        return None
    return False


def record_loaded_ptx(driver):
    """Record a hash of every PTX module the driver loads from now on.

    The hash is taken when the module is loaded, so an entry that the
    runtime later evicts from a cache is still in the record.

    Args:
        driver: The runtime CUDA driver wrapper. Its ``load(ptx,
            kernel_name)`` is replaced by a recording pass-through.

    Returns:
        The live list of ``kernel``, ``sha256``, and ``bytes`` entries.
    """
    loaded = []
    load = driver.load

    def recording_load(ptx, kernel_name):
        text = ptx.encode()
        loaded.append(
            {
                "kernel": kernel_name,
                "sha256": hashlib.sha256(text).hexdigest(),
                "bytes": len(text),
            }
        )
        return load(ptx, kernel_name)

    driver.load = recording_load
    return loaded


def swage_build():
    """Identify the Swage build of this process and record what it loads.

    This is the only function of the benchmarks that reaches into the
    runtime for provenance: the native library paths, the linked LLVM, and
    the driver whose ``load`` receives every PTX module.

    Returns:
        The Swage part of the provenance block. ``loaded_ptx`` is the live
        list from ``record_loaded_ptx``.
    """
    import swage
    from mlir_swage._mlir_libs import _swageDialectsNanobind as extension
    from swage import _runtime

    return {
        "swage": swage.__version__,
        "llvm_pin": _runtime._compiler_identity()["llvm"],
        "llvm_linked": getattr(extension.swage, "__llvm_version__", None),
        "cuda_driver": _runtime.driver_version(),
        "native_sha256": native_sha256(_runtime._native_libraries()),
        "loaded_ptx": record_loaded_ptx(_runtime._get_driver()),
    }


def start(torch, build, *, run=subprocess.run):
    """Start the provenance block of one benchmark process.

    Args:
        torch: The PyTorch module, with CUDA initialized.
        build: ``swage_build()``.
        run: ``subprocess.run`` or a stand-in.

    Returns:
        The block, with the GPU state sampled before the measurement. Pass
        it to ``finish`` after the measurement.
    """
    device = torch.cuda.current_device()
    uuid = device_uuid(torch, device)
    return {
        "gpu": torch.cuda.get_device_name(device),
        "gpu_uuid": uuid,
        "cpu_model": cpu_model(),
        "pytorch": torch.__version__,
        "triton": package_version("triton"),
        **build,
        "gpu_state_before": gpu_state(uuid, pid=os.getpid(), run=run),
    }


def finish(block, *, run=subprocess.run):
    """Complete a block after the measurement and return it.

    The GPU state is sampled again, the loaded PTX entries are put in a
    stable order without repeats, and ``other_compute_process_seen``
    summarizes the two samples.
    """
    block["gpu_state_after"] = gpu_state(
        block["gpu_uuid"], pid=os.getpid(), run=run
    )
    distinct = {
        (entry["kernel"], entry["sha256"]): entry
        for entry in block["loaded_ptx"]
    }
    block["loaded_ptx"] = [distinct[key] for key in sorted(distinct)]
    block["other_compute_process_seen"] = other_compute_process_seen(
        block["gpu_state_before"], block["gpu_state_after"]
    )
    return block


def smallest_step(values):
    """Return the finest gap between distinct timer readings, or None.

    A timer returns multiples of its tick, so the smallest gap between two
    different readings is an upper bound of the tick. Readings that differ
    by less than one part in a million are one reading: float arithmetic on
    the same tick count does not always give the same last digit.
    """
    distinct = sorted(set(values))
    gaps = [
        later - earlier
        for earlier, later in zip(distinct, distinct[1:])
        if later - earlier > _SAME_READING * abs(later)
    ]
    return min(gaps, default=None)


def clock_tick_us(clock=time.perf_counter_ns, reads=10_000):
    """Return the smallest advance of a nanosecond clock, in microseconds.

    Back-to-back reads include the cost of the call, so the result is an
    upper bound of the clock tick. None when the clock never advanced.
    """
    readings = [clock() for _ in range(reads)]
    advances = [
        later - earlier
        for earlier, later in zip(readings, readings[1:])
        if later > earlier
    ]
    return min(advances) / 1_000.0 if advances else None
