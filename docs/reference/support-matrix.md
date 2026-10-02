<!-- docs/reference/support-matrix.md -->

# Support Matrix

This page lists the environments that Swage is tested in, the environments
that its package metadata or compiler accepts without a test, and the
environments about which nothing is known. Every row names its evidence: a
workflow file, a test, or a committed record.

Swage is pre-alpha. A tested row reports what the workflows run today. It is
not a compatibility promise for a later release.

## How to read the tables

The Python, PyTorch, and driver tables give each row one of four statuses:

- **Tested**: a workflow runs a test tier in this environment.
- **Admitted**: the package metadata or the compiler accepts this
  environment, and no workflow runs a test in it.
- **Unknown**: nothing accepts, rejects, or tests this environment.
- **Rejected**: installation, configuration, or a launch stops with an
  error.

The GPU table uses the three terms that the `target` line of
`python -m swage.env` prints, with the same meaning:

- **Qualified**: the GPU tier executes on this target.
- **Admitted, not qualified**: the compiler emits PTX for this target, and
  no test has executed that PTX.
- **Not admitted**: compilation rejects this target.

A tested or qualified row names the tier that ran. The tiers are:

| Tier | Workflow and job | Covers | Where it runs |
|---|---|---|---|
| Pure Python | `ci-python`, `test` | `tests/python`, `ruff check .`, and the package build | GitHub-hosted runner, on every pull request and every push to `main` |
| Native | `ci-cpp`, `build-and-test` | The lit suite, the C++ unit tests, the committed dialect reference, and `python/tests/mlir` with every CUDA test skipped | GitHub-hosted runner, on every pull request and every push to `main` |
| GPU | `ci-gpu`, `runtime-qualification` | The whole `python/tests/mlir` directory including the CUDA tests, one of which runs the segmented example, and the fixed vector-add example | One self-hosted machine, on `main` only, weekly and on request |

The hosted jobs install one hash-locked tool set from `requirements-ci.txt`.
`ci-cpp` also defines a `format` job, which checks the C and C++ sources
with `clang-format`, and a `sanitizers` job, which runs the lit suite and
the C++ unit tests under AddressSanitizer and UndefinedBehaviorSanitizer.
These two jobs have not yet run on hosted CI, and no row below relies on
them.

All tiers run on Linux x86-64. The native build requires Linux x86-64 and
the exact LLVM release in `cmake/llvm-version.txt`; CMake rejects any other
release. The pure Python package on another platform is unknown.

## Python

| Python | Status | Evidence |
|---|---|---|
| 3.9 and older | Rejected | `requires-python = ">=3.10"` in `pyproject.toml`; the build scripts stop when neither `python` nor `python3` is 3.10 or newer |
| 3.10 | Tested in the pure Python tier. Admitted in the native and GPU tiers | `ci-python` matrix; accepted by `requires-python`, and no workflow builds the bindings or launches on it |
| 3.11 and 3.12 | Admitted | Accepted by `requires-python`; no workflow uses them |
| 3.13 | Tested in all three tiers | `ci-python` matrix; `ci-cpp` builds and tests the bindings on 3.13; the committed A6000 records name 3.13.13 on the GPU runner |
| 3.14 and newer | Admitted | Accepted by `requires-python`, which has no upper bound; no workflow uses them |

`ci-gpu` uses the Python installed on the runner and does not pin it.

## PyTorch

PyTorch is optional. Metadata inference, launch, and the segmented calls
need it. Capture, the kernel-language check, the environment report, and
emission with an explicit signature do not. The segmented calls also need
`numpy`, which no extra of the package declares; the binding requirements
of the native build install it.

| PyTorch | Status | Evidence |
|---|---|---|
| Not installed | Tested in the pure Python tier, and for explicit-signature emission in the native tier | `requirements-ci.txt`, which `ci-python` installs, has no PyTorch; `python/tests/mlir/test_examples.py` runs the emit-only example with the PyTorch import blocked |
| Older than 2.6 | Rejected at launch and at a segmented call | `launch()`, `segment_reduce`, and `segment_softmax` raise a `RuntimeError` that names the version found, before any kernel is compiled or enqueued; `tests/python/test_runtime.py` and `tests/python/test_segments.py`. Compile-only emission does not run this check |
| 2.6 through 2.11 | Admitted | Inside the declared `torch>=2.6` in `pyproject.toml`, which a test keeps equal to the launch check; no workflow runs these releases, and the floor itself has not been run |
| 2.12.0+cu130 | Tested in the GPU tier | The build on the GPU runner, named by the committed A6000 records; `ci-gpu` prints the version and does not pin it |
| 2.13.0+cpu | Tested in the native tier, with every CUDA test skipped | Pinned in `ci-cpp` |
| Other 2.12 and 2.13 builds, and newer releases | Admitted | The declared range has no upper bound, and the launch check refuses no newer release; no workflow runs them |

## CUDA driver

| Driver | Status | Evidence |
|---|---|---|
| CUDA driver API 13.0 | Tested in the GPU tier | The committed A6000 records store `cuda_driver: 13.0`, the value `cuDriverGetVersion` reports on the GPU runner |
| Any other driver | Unknown | No minimum driver version is documented or checked, and no test covers another driver |

The records store the driver API level, not the NVIDIA driver package
version. `python -m swage.env` prints the same value as `cuda_driver`.

## GPU architecture

A launch targets the active device exactly. The admitted processor list and
the rejection rules are stated in
[Runtime and Environment](runtime-environment.md).

| Target | Status | Evidence |
|---|---|---|
| `sm_86` | Qualified | The GPU tier executes on one NVIDIA RTX A6000, the GPU runner |
| `sm_80`, `sm_87`, `sm_88`, `sm_89`, `sm_90`, `sm_100`, `sm_101`, `sm_103`, `sm_110`, `sm_120`, and `sm_121` | Admitted, not qualified | `python/tests/mlir/test_target_compile.py` emits PTX for the public kernel, and for an identity sum through every private segmented entry point, on each of them in the native tier without a device. No test executes on one |
| Below `sm_80`, and a processor the pinned LLVM does not define, such as `sm_85` | Not admitted | Compiler admission; the same test file checks that every other `sm_` value from 80 through 129 is refused |

`sm_86` is the only qualified target, for the public kernel, for the public
segmented calls, and for the private segmented paths. Committed records
under `benchmarks/results/` also
report benchmark runs and one correctness run on an NVIDIA GeForce RTX 5090
(`sm_120`), taken outside the workflows. They are recorded evidence and do
not make `sm_120` qualified; see [Benchmarks](../internals/benchmarks.md).

## Segmented calls

`swage.segment_reduce` and `swage.segment_softmax` have a narrower tested
surface than the tables above suggest for the package as a whole:

| Aspect | Status | Evidence |
|---|---|---|
| Argument checks that need no native build, and the wheel-only error | Tested in the pure Python tier | `tests/python/test_segments.py`, with a stand-in for PyTorch |
| Argument and offsets validation on host tensors | Tested in the native tier | The tests of `python/tests/mlir/test_public_segments.py` that need no GPU |
| Results, schedules, streams, threads, graph capture, and resource use | Tested in the GPU tier on `sm_86` | The CUDA tests of `python/tests/mlir/test_public_segments.py` and the example test in `test_examples.py` |
| f32 values with int32 offsets, rank one | The only admitted data model | Other dtypes and ranks raise a `TypeError`; the same file |
| A second GPU on one host | Unknown | The calls require the current device; the tests run on a host with one GPU |
| Gradients | Rejected | `values` that require grad raise a `ValueError`; the same files |

The GPU tests of the segmented calls ran on one NVIDIA RTX A6000 with
PyTorch 2.12.0+cu130 and Python 3.13, on the branch that added the calls.
They have not run through the `ci-gpu` workflow, which runs on `main` only.

## Stability

Versions follow semantic versioning in the `0.x` range, where any release
may change public behavior. The
[changelog](https://github.com/abhiksark/swage/blob/main/CHANGELOG.md)
records each change, and the
[security policy](https://github.com/abhiksark/swage/blob/main/SECURITY.md)
states that only the latest `main` receives fixes.

Continue with [Installation](../getting-started/installation.md) for the
build prerequisites, or run `python -m swage.env` and compare its output
with the rows above before reporting a problem.
