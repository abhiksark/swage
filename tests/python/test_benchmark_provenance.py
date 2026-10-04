# tests/python/test_benchmark_provenance.py
"""Tests for the provenance block shared by the benchmark harnesses."""

import hashlib
import importlib
import pathlib
import random
import struct
import subprocess
import types

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_UUID = "GPU-1b08842a-4c6e-32aa-53de-5b3f8553388b"
_OTHER_UUID = "GPU-00000000-0000-0000-0000-000000000000"
_GPU_ROW = (
    f"{_UUID}, NVIDIA RTX A6000, 580.178.04, Default, P2, 59, 68.68 W, "
    "300.00 W, 1800 MHz, 7601 MHz, 9 %"
)
_OTHER_GPU_ROW = (
    f"{_OTHER_UUID}, NVIDIA GeForce RTX 5090, 580.178.04, Default, P8, 40, "
    "20.00 W, 575.00 W, 210 MHz, 405 MHz, 0 %"
)


@pytest.fixture
def provenance(monkeypatch):
    """Import the shared module as the benchmark scripts do."""
    monkeypatch.syspath_prepend(str(_ROOT / "benchmarks"))
    return importlib.import_module("benchmark_provenance")


def _nvidia_smi(gpu_rows, process_rows, calls=None):
    """Return a fake ``subprocess.run`` that answers the two queries."""

    def run(command, **options):
        if calls is not None:
            calls.append((command, options))
        query = command[1].split("=")[0]
        rows = gpu_rows if query == "--query-gpu" else process_rows
        return subprocess.CompletedProcess(
            command, 0, "".join(f"{row}\n" for row in rows), ""
        )

    return run


def _stub_torch(uuid="1b08842a-4c6e-32aa-53de-5b3f8553388b"):
    """Return the device surface the provenance block reads."""
    properties = types.SimpleNamespace()
    if uuid is not None:
        properties.uuid = uuid
    return types.SimpleNamespace(
        __version__="2.12.0+cu130",
        cuda=types.SimpleNamespace(
            current_device=lambda: 0,
            get_device_name=lambda device: "NVIDIA RTX A6000",
            get_device_properties=lambda device: properties,
        ),
    )


def _sysfs(root, governors, driver="amd-pstate-epp"):
    """Write a fake CPU sysfs tree; a governor of None has no file."""
    for index, governor in enumerate(governors):
        policy = root / f"cpu{index}" / "cpufreq"
        policy.mkdir(parents=True, exist_ok=True)
        if governor is None:
            (policy / "scaling_governor").unlink(missing_ok=True)
        else:
            (policy / "scaling_governor").write_text(f"{governor}\n")
    # Directories beside the CPUs that are not CPUs.
    for name in ("cpufreq", "cpuidle"):
        (root / name).mkdir(exist_ok=True)
    if driver is not None and governors:
        (root / "cpu0" / "cpufreq" / "scaling_driver").write_text(f"{driver}\n")
        (
            root / "cpu0" / "cpufreq" / "energy_performance_preference"
        ).write_text("balance_performance\n")
    return root


def test_cpu_frequency_policy_counts_the_governor_of_every_cpu(
    provenance, tmp_path
):
    """Record the frequency governor the host side of a launch ran under."""
    root = _sysfs(tmp_path, ["powersave"] * 4)

    assert provenance.cpu_frequency_policy(root) == {
        "governors": {"powersave": 4},
        "scaling_driver": "amd-pstate-epp",
        "energy_performance_preference": "balance_performance",
    }


def test_cpu_frequency_policy_records_what_it_cannot_read_as_unknown(
    provenance, tmp_path
):
    """Count a CPU without a governor file instead of leaving it out."""
    root = _sysfs(
        tmp_path / "mixed",
        ["powersave", "performance", None, "powersave"],
        driver=None,
    )

    assert provenance.cpu_frequency_policy(root) == {
        "governors": {"performance": 1, "powersave": 2, "unknown": 1},
        "scaling_driver": None,
        "energy_performance_preference": None,
    }
    assert provenance.cpu_frequency_policy(tmp_path / "missing") == {
        "governors": {},
        "scaling_driver": None,
        "energy_performance_preference": None,
    }


@pytest.mark.parametrize(
    ("before", "after", "unchanged"),
    [
        ({"powersave": 4}, {"powersave": 4}, True),
        ({"powersave": 4}, {"performance": 4}, False),
        ({"powersave": 4}, {"performance": 1, "powersave": 3}, False),
        ({"powersave": 3, "unknown": 1}, {"powersave": 3, "unknown": 1}, None),
        ({}, {}, None),
        ({"powersave": 4}, {}, None),
    ],
)
def test_governor_unchanged_is_tri_state(provenance, before, after, unchanged):
    """Say the governor held, changed, or could not be read."""
    assert (
        provenance.cpu_governor_unchanged(
            {"governors": before}, {"governors": after}
        )
        is unchanged
    )


def test_block_states_the_governor_before_and_after(provenance, tmp_path):
    """Sample the governor around the run and keep a change once seen."""
    root = _sysfs(tmp_path, ["powersave"] * 2)
    quiet = _nvidia_smi([_GPU_ROW], [])
    build = {"loaded_ptx": []}

    block = provenance.start(_stub_torch(), build, run=quiet, cpu_root=root)
    assert block["cpu_frequency_before"]["governors"] == {"powersave": 2}
    assert "cpu_governor_unchanged" not in block
    provenance.finish(block, run=quiet, cpu_root=root)
    assert block["cpu_frequency_after"] == block["cpu_frequency_before"]
    assert block["cpu_governor_unchanged"] is True

    _sysfs(root, ["performance"] * 2)
    provenance.finish(block, run=quiet, cpu_root=root)
    assert block["cpu_frequency_after"]["governors"] == {"performance": 2}
    assert block["cpu_governor_unchanged"] is False

    # A later sample that matches the first does not undo the change.
    _sysfs(root, ["powersave"] * 2)
    provenance.finish(block, run=quiet, cpu_root=root)
    assert block["cpu_governor_unchanged"] is False


def test_cpu_model_reads_the_first_model_name(provenance, tmp_path):
    """Record the processor that ran the host side of every launch."""
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text(
        "processor\t: 0\n"
        "model name\t: AMD Ryzen 9 7900X 12-Core Processor\n"
        "processor\t: 1\n"
        "model name\t: another model\n"
    )

    assert provenance.cpu_model(cpuinfo) == (
        "AMD Ryzen 9 7900X 12-Core Processor"
    )


def test_cpu_model_is_none_when_the_kernel_does_not_report_one(
    provenance, tmp_path
):
    """Leave the field empty instead of guessing off Linux."""
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("processor\t: 0\n")

    assert provenance.cpu_model(cpuinfo) is None
    assert provenance.cpu_model(tmp_path / "missing") is None


def test_package_version_does_not_import_the_package(provenance):
    """Read an installed version, and None for a package that is absent."""
    assert provenance.package_version("pytest") == pytest.__version__
    assert provenance.package_version("swage-no-such-package") is None


def test_native_sha256_hashes_each_library_file_once(provenance, tmp_path):
    """Identify the native build by content, not by size and mtime."""
    extension = tmp_path / "_swageDialectsNanobind.so"
    library = tmp_path / "libSwagePythonCAPI.so.22.1"
    extension.write_bytes(b"extension")
    library.write_bytes(b"library")
    (tmp_path / "libSwagePythonCAPI.so").symlink_to(library.name)

    assert provenance.native_sha256(
        [tmp_path / "libSwagePythonCAPI.so", library, extension]
    ) == {
        str(extension): hashlib.sha256(b"extension").hexdigest(),
        str(library): hashlib.sha256(b"library").hexdigest(),
    }


def test_gpu_state_reports_the_device_and_the_other_processes(provenance):
    """Say what else was computing on the measured GPU, and its clocks."""
    calls = []
    run = _nvidia_smi(
        [_OTHER_GPU_ROW, _GPU_ROW],
        [
            f"{_UUID}, 4242, python3, 512 MiB",
            f"{_UUID}, 777, ninja, clang++, 1024 MiB",
            f"{_OTHER_UUID}, 888, python3, 256 MiB",
        ],
        calls,
    )

    state = provenance.gpu_state(_UUID, pid=4242, run=run)

    assert state["error"] is None
    assert state["gpu"] == {
        "uuid": _UUID,
        "name": "NVIDIA RTX A6000",
        "driver_version": "580.178.04",
        "compute_mode": "Default",
        "pstate": "P2",
        "temperature.gpu": "59",
        "power.draw": "68.68 W",
        "power.limit": "300.00 W",
        "clocks.current.graphics": "1800 MHz",
        "clocks.current.memory": "7601 MHz",
        "utilization.gpu": "9 %",
    }
    assert state["other_compute_processes"] == [
        {
            "pid": 777,
            "process_name": "ninja, clang++",
            "used_memory": "1024 MiB",
        }
    ]
    assert state["sampled_at"]
    assert [command[0] for command, _ in calls] == ["nvidia-smi"] * 2
    assert calls[0][0][2] == "--format=csv,noheader"
    assert all(
        options == {"check": False, "capture_output": True, "text": True}
        for _, options in calls
    )


def test_gpu_state_distinguishes_an_exclusive_gpu_from_an_unknown_one(
    provenance,
):
    """Report an empty list when nothing else ran, None when unknown."""
    exclusive = provenance.gpu_state(
        _UUID,
        pid=4242,
        run=_nvidia_smi([_GPU_ROW], [f"{_UUID}, 4242, python3, 512 MiB"]),
    )
    assert exclusive["other_compute_processes"] == []
    assert exclusive["error"] is None

    def mismatch(command, **options):
        return subprocess.CompletedProcess(
            command,
            18,
            "",
            "Failed to initialize NVML: Driver/library version mismatch\n",
        )

    unknown = provenance.gpu_state(_UUID, pid=4242, run=mismatch)
    assert unknown["gpu"] is None
    assert unknown["other_compute_processes"] is None
    assert "Driver/library version mismatch" in unknown["error"]


def test_gpu_state_survives_a_machine_without_nvidia_smi(provenance):
    """Degrade to an explained gap instead of failing the benchmark."""

    def missing(command, **options):
        raise FileNotFoundError(2, "No such file or directory", "nvidia-smi")

    state = provenance.gpu_state(_UUID, pid=1, run=missing)

    assert state["gpu"] is None
    assert state["other_compute_processes"] is None
    assert "nvidia-smi" in state["error"]


def test_gpu_state_without_a_uuid_keeps_a_single_gpu(provenance):
    """Fall back to the only device when PyTorch cannot name its UUID."""
    state = provenance.gpu_state(
        None,
        pid=1,
        run=_nvidia_smi([_GPU_ROW], [f"{_UUID}, 9, python3, 1 MiB"]),
    )
    assert state["gpu"]["name"] == "NVIDIA RTX A6000"
    assert [row["pid"] for row in state["other_compute_processes"]] == [9]

    ambiguous = provenance.gpu_state(
        None, pid=1, run=_nvidia_smi([_GPU_ROW, _OTHER_GPU_ROW], [])
    )
    assert ambiguous["gpu"] is None
    assert ambiguous["other_compute_processes"] is None
    assert "2 GPUs" in ambiguous["error"]


def test_device_uuid_uses_the_nvidia_smi_spelling(provenance):
    """Match the PyTorch device with its nvidia-smi row."""
    assert provenance.device_uuid(_stub_torch(), 0) == _UUID
    assert provenance.device_uuid(_stub_torch(uuid=None), 0) is None


def test_loaded_ptx_is_hashed_when_the_driver_loads_it(provenance):
    """Record every PTX module at load time, whatever is evicted later."""
    loads = []

    class Driver:
        def load(self, ptx, kernel_name):
            loads.append((ptx, kernel_name))
            return "module", f"function {kernel_name}"

    driver = Driver()
    loaded = provenance.record_loaded_ptx(driver)

    assert loaded == []
    assert driver.load("ptx text", "segmented_sum") == (
        "module",
        "function segmented_sum",
    )
    assert loads == [("ptx text", "segmented_sum")]
    assert loaded == [
        {
            "kernel": "segmented_sum",
            "sha256": hashlib.sha256(b"ptx text").hexdigest(),
            "bytes": 8,
        }
    ]


def test_loaded_ptx_sees_the_private_segmented_load_path(provenance):
    """Stay attached to the load call the prepared reductions really make."""
    from swage import _segmented_qualification

    class Driver:
        def current_context(self):
            return 1

        def load(self, ptx, kernel_name):
            return "module", "function"

    driver = Driver()
    loaded = provenance.record_loaded_ptx(driver)
    for _ in range(2):
        _segmented_qualification._load_once(driver, "warp ptx", "segmented_sum")
    _segmented_qualification._load_once(driver, "cta ptx", "segmented_sum")

    assert [entry["sha256"] for entry in loaded] == [
        hashlib.sha256(text).hexdigest() for text in (b"warp ptx", b"cta ptx")
    ]


def test_smallest_step_is_the_finest_gap_between_distinct_values(provenance):
    """Estimate a timer tick from the values the timer returned."""
    assert provenance.smallest_step([3.072, 3.104, 3.104, 3.2, 0.0]) == (
        pytest.approx(0.032)
    )
    assert provenance.smallest_step([5.0, 5.0]) is None
    assert provenance.smallest_step([]) is None
    # Two float spellings of one reading are not a gap.
    assert provenance.smallest_step([0.1 + 0.2, 0.3, 0.332]) == pytest.approx(
        0.032
    )


def test_timer_tick_is_the_coarse_step_most_readings_sit_on(provenance):
    """Report the grid a timer favours, not its finest possible reading.

    CUDA event readings are multiples of 32 ns, yet most of them are also
    multiples of 1.024 us. A ratio of two medians is then only as fine as
    the coarse step, whatever the finest gap between two readings is.
    """
    rng = random.Random(7)
    coarse = [1.024 * rng.randint(60, 70) for _ in range(90)]
    fine = [0.032 * rng.randint(1900, 2200) for _ in range(10)]
    as_float32 = struct.Struct("f")

    def reading(value):
        # Readings arrive as float32 milliseconds, as the event timer
        # returns them, and are scaled to microseconds afterwards.
        (milliseconds,) = as_float32.unpack(as_float32.pack(value / 1000))
        return milliseconds * 1000

    readings = [reading(value) for value in coarse + fine]

    assert provenance.smallest_step(readings) == pytest.approx(0.032, rel=1e-3)
    assert provenance.timer_tick(readings) == pytest.approx(1.024, rel=1e-4)
    # Half of the readings on the coarse grid is still far above chance.
    half = [reading(value) for value in coarse[:10] + fine]
    assert provenance.timer_tick(half) == pytest.approx(1.024, rel=1e-4)


def test_timer_tick_stays_fine_without_a_coarse_grid(provenance):
    """Keep the finest step when readings spread evenly over it."""
    rng = random.Random(11)
    readings = [0.032 * rng.randint(1900, 2200) for _ in range(200)]

    assert provenance.timer_tick(readings) == pytest.approx(0.032, rel=1e-4)
    assert provenance.timer_tick([1.024 * n for n in range(60, 70)]) == (
        pytest.approx(1.024)
    )


def test_timer_tick_of_too_few_readings(provenance):
    """Bound the tick by a lone reading; report None without any."""
    assert provenance.timer_tick([5.0, 5.0]) == 5.0
    assert provenance.timer_tick([0.0, 0.0]) is None
    assert provenance.timer_tick([]) is None


def test_clock_tick_is_the_smallest_advance_of_back_to_back_reads(provenance):
    """Measure the host clock instead of trusting its nominal resolution."""
    start = 250_000_000_000_000
    ticks = iter(start + tick for tick in [0, 0, 50, 50, 90, 200, 200])

    assert provenance.clock_tick_us(lambda: next(ticks), reads=7) == 0.04
    assert provenance.clock_tick_us(lambda: 7, reads=5) is None


def test_block_carries_every_provenance_field(
    provenance, monkeypatch, tmp_path
):
    """Assemble the fields a reader needs to place a measurement."""
    monkeypatch.setattr(provenance, "cpu_model", lambda: "Test CPU")
    monkeypatch.setattr(provenance, "package_version", lambda name: "3.7.0")
    monkeypatch.setattr(provenance.os, "getpid", lambda: 4242)
    loaded = []
    build = {
        "swage": "0.5.1",
        "llvm_pin": "llvmorg-22.1.8",
        "llvm_linked": "22.1.8",
        "cuda_driver": "13.0",
        "native_sha256": {"/build/lib.so": "a" * 64},
        "loaded_ptx": loaded,
    }
    quiet = _nvidia_smi([_GPU_ROW], [f"{_UUID}, 4242, python3, 512 MiB"])
    shared = _nvidia_smi(
        [_GPU_ROW],
        [f"{_UUID}, 4242, python3, 512 MiB", f"{_UUID}, 9, python3, 1 MiB"],
    )

    root = _sysfs(tmp_path, ["powersave"] * 2)
    block = provenance.start(_stub_torch(), build, run=quiet, cpu_root=root)
    loaded.extend(
        [
            {"kernel": "segmented_sum", "sha256": "b" * 64, "bytes": 2},
            {"kernel": "segmented_sum", "sha256": "a" * 64, "bytes": 1},
            {"kernel": "segmented_sum", "sha256": "b" * 64, "bytes": 2},
        ]
    )
    assert "gpu_state_after" not in block
    finished = provenance.finish(block, run=shared, cpu_root=root)

    assert finished is block
    sampled = ("gpu_state", "cpu_frequency")
    assert {
        key: block[key] for key in block if not key.startswith(sampled)
    } == {
        "gpu": "NVIDIA RTX A6000",
        "gpu_uuid": _UUID,
        "cpu_model": "Test CPU",
        "pytorch": "2.12.0+cu130",
        "triton": "3.7.0",
        "swage": "0.5.1",
        "llvm_pin": "llvmorg-22.1.8",
        "llvm_linked": "22.1.8",
        "cuda_driver": "13.0",
        "native_sha256": {"/build/lib.so": "a" * 64},
        "loaded_ptx": [
            {"kernel": "segmented_sum", "sha256": "a" * 64, "bytes": 1},
            {"kernel": "segmented_sum", "sha256": "b" * 64, "bytes": 2},
        ],
        "other_compute_process_seen": True,
        "cpu_governor_unchanged": True,
    }
    assert block["cpu_frequency_before"] == block["cpu_frequency_after"]
    assert block["gpu_state_before"]["other_compute_processes"] == []
    assert [
        row["pid"]
        for row in block["gpu_state_after"]["other_compute_processes"]
    ] == [9]
    assert block["gpu_state_before"]["gpu"]["driver_version"] == "580.178.04"


@pytest.mark.parametrize(
    ("before", "after", "seen"),
    [
        ([], [], False),
        ([], None, None),
        (None, None, None),
        (None, [{"pid": 9}], True),
        ([{"pid": 9}], [], True),
    ],
)
def test_other_compute_process_seen_is_tri_state(
    provenance, before, after, seen
):
    """Say yes, no, or unknown; never report unknown as an exclusive GPU."""
    assert (
        provenance.other_compute_process_seen(
            {"other_compute_processes": before},
            {"other_compute_processes": after},
        )
        is seen
    )


def test_block_can_be_finished_again_after_more_loads(provenance):
    """Keep recording for a harness that rewrites its record per row."""

    class Driver:
        def load(self, ptx, kernel_name):
            return "module", "function"

    driver = Driver()
    build = {"loaded_ptx": provenance.record_loaded_ptx(driver)}
    quiet = _nvidia_smi([_GPU_ROW], [])
    shared = _nvidia_smi([_GPU_ROW], [f"{_UUID}, 9, python3, 1 MiB"])
    block = provenance.start(_stub_torch(), build, run=quiet)

    driver.load("first", "segmented_sum")
    provenance.finish(block, run=shared)
    assert [entry["bytes"] for entry in block["loaded_ptx"]] == [5]
    assert block["other_compute_process_seen"] is True

    driver.load("second kernel", "segmented_max")
    provenance.finish(block, run=quiet)
    assert [entry["kernel"] for entry in block["loaded_ptx"]] == [
        "segmented_max",
        "segmented_sum",
    ]
    # A process seen during an earlier row stays seen.
    assert block["other_compute_process_seen"] is True
    assert block["gpu_state_after"]["other_compute_processes"] == []
