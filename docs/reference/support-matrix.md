<!-- docs/reference/support-matrix.md -->

# Support Matrix

This page lists the environments that Swage is tested in, the environments
that its package metadata or compiler accepts without a test, and the
environments about which nothing is known. Every row names its evidence: a
workflow file, a test, or a committed record.

Swage is pre-alpha. A tested row reports what the workflows run today. It is
not a compatibility promise for a later release.

## How to read the tables

Each row has one of four statuses:

- **Tested**: a workflow runs a test tier in this environment.
- **Admitted**: the package metadata or the compiler accepts this
  environment, and no workflow runs a test in it.
- **Unknown**: nothing accepts, rejects, or tests this environment.
- **Rejected**: installation, configuration, or compilation stops with an
  error.

A tested row names the tier that ran. The three tiers are:

| Tier | Workflow | Covers | Where it runs |
|---|---|---|---|
| Pure Python | `ci-python` | `tests/python`, `ruff check .`, and the package build | GitHub-hosted runner, on every pull request and every push to `main` |
| Native | `ci-cpp` | The lit suite, the C++ unit tests, and `python/tests/mlir` with every CUDA test skipped | GitHub-hosted runner, on every pull request and every push to `main` |
| GPU | `ci-gpu` | The whole `python/tests/mlir` directory including the CUDA tests, and the fixed vector-add example | One self-hosted machine, on `main` only, weekly and on request |

All three tiers run on Linux x86-64. The native build requires Linux x86-64
and the exact LLVM release in `cmake/llvm-version.txt`; CMake rejects any
other release. The pure Python package on another platform is unknown.

## Python

| Python | Status | Evidence |
|---|---|---|
| 3.9 and older | Rejected | `requires-python = ">=3.10"` in `pyproject.toml` |
| 3.10 | Tested in the pure Python tier. Unknown in the native and GPU tiers | `ci-python` matrix |
| 3.11 and 3.12 | Admitted | Accepted by `requires-python`; no workflow uses them |
| 3.13 | Tested in all three tiers | `ci-python` matrix; `ci-cpp` builds and tests the bindings on 3.13; the committed A6000 records name 3.13.13 on the GPU runner |
| 3.14 and newer | Admitted | Accepted by `requires-python`, which has no upper bound; no workflow uses them |

`ci-gpu` uses the Python installed on the runner and does not pin it.

## PyTorch

PyTorch is optional. Metadata inference and launch need it; capture, the
environment report, and emission with an explicit signature do not.

| PyTorch | Status | Evidence |
|---|---|---|
| Not installed | Tested in the pure Python tier, and for explicit-signature emission in the native tier | `ci-python` installs the `dev` extra, which has no PyTorch; `python/tests/mlir/test_examples.py` runs the emit-only example with the PyTorch import blocked |
| Older than 2.6 | Unknown | Outside the declared `torch>=2.6`; no test covers it |
| 2.6 through 2.11 | Admitted | Inside the declared `torch>=2.6` in `pyproject.toml`; no workflow runs them, and the floor itself is not tested |
| 2.12.0+cu130 | Tested in the GPU tier | The build on the GPU runner, named by the committed A6000 records; `ci-gpu` prints the version and does not pin it |
| 2.13.0+cpu | Tested in the native tier, with every CUDA test skipped | Pinned in `ci-cpp` |
| Other 2.12 and 2.13 builds, and newer releases | Admitted | The declared range has no upper bound; no workflow runs them |

## CUDA driver

| Driver | Status | Evidence |
|---|---|---|
| CUDA driver API 13.0 | Tested in the GPU tier | The committed A6000 records store `cuda_driver: 13.0`, the value `cuDriverGetVersion` reports on the GPU runner |
| Any other driver | Unknown | No minimum driver version is documented, and no test covers another driver |

The records store the driver API level, not the NVIDIA driver package
version. `python -m swage.env` prints the same value as `cuda_driver`.

## GPU architecture

A launch targets the active device exactly. The admitted processor list and
the rejection rules are stated in
[Runtime and Environment](runtime-environment.md).

| Target | Status | Evidence |
|---|---|---|
| Below `sm_80` | Rejected | Compiler admission; `python/tests/mlir/test_codegen.py` |
| A processor the pinned LLVM does not define, such as `sm_85` | Rejected | Compiler admission; `python/tests/mlir/test_codegen.py` |
| `sm_86` | Tested in the GPU tier | NVIDIA RTX A6000, the GPU runner |
| `sm_80` and `sm_121` | Admitted. The native tier compiles for them without a device, and no test runs on one | `python/tests/mlir/test_codegen.py` emits PTX for both; `python/tests/mlir/test_segmented_codegen.py` emits PTX for `sm_80` |
| `sm_87`, `sm_88`, `sm_89`, `sm_90`, `sm_100`, `sm_101`, `sm_103`, and `sm_110` | Admitted | On the admitted list; no workflow compiles for them or runs on them |
| `sm_120` | Admitted | On the admitted list; no workflow compiles for it or runs on it. Committed records under `benchmarks/results/` report benchmark runs and one correctness run on an NVIDIA GeForce RTX 5090, taken outside the workflows. They are recorded evidence, not a tested status; see [Benchmarks](../internals/benchmarks.md) |

The private segmented paths are qualified on `sm_86` only.

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
