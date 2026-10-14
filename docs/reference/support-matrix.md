<!-- docs/reference/support-matrix.md -->

# Support Matrix

This page lists the environments that Swage is tested in, the environments
that its package metadata or compiler accepts without a test, and the
environments about which nothing is known. Every row names its evidence: a
workflow file, a test, or a committed record.

Swage is experimental. A tested row reports what the workflows are
configured to run on the current source tree. It is not a compatibility
promise for a later release, and `v0.5.2`, the release these rows describe,
is not yet published.

## How to read the tables

The Python, PyTorch, driver, and wheel tables give each row one of four
statuses:

- **Tested**: a workflow runs a test tier in this environment.
- **Admitted**: the package metadata or the compiler accepts this
  environment, and no workflow runs a test in it.
- **Unknown**: nothing accepts, rejects, or tests this environment.
- **Rejected**: installation, configuration, or a launch stops with an
  error.

The GPU table uses three terms that match the `backends.cuda` object of
`python -m swage.env --json`:

- **Qualified**: the GPU tier executes on this target; `qualified` is
  `true` on the NVIDIA RTX A6000.
- **Admitted, not qualified**: the compiler emits PTX for this target, and
  no test has executed that PTX; `qualified` is `false`.
- **Not admitted**: compilation rejects this target; `available` is
  `false` with the reason `CUDA target is not admitted by the pinned
  compiler`.

A tested or qualified row names the tier that ran. The tiers are:

| Tier | Workflow and job | Covers | Where it runs |
|---|---|---|---|
| Pure Python | `ci-python`, `test` | `tests/python`, `ruff check .`, and the source distribution build and metadata check, on CPython 3.10, 3.11, 3.12, and 3.13 | GitHub-hosted runner, on every pull request and every push to `main` |
| Native | `ci-cpp`, `build-and-test` | The lit suite, the C++ unit tests, the committed dialect reference, and `python/tests/mlir` with every CUDA test skipped, from the build tree; then a local CPython 3.13 wheel, built and installed into a fresh environment, with the CPU checks and the whole `python/tests/mlir` suite run against it | GitHub-hosted runner, on every pull request and every push to `main` |
| Release wheels | `publish-pypi`, `wheels` | One `manylinux_2_28` wheel per CPython 3.10 to 3.13, built and repaired in the digest-pinned image; against each installed wheel, `python -m swage.env --check cpu`, the CPU, low-precision, and dispatch runtime tests, and the CPU smoke script, with PyTorch 2.6.0+cpu | GitHub-hosted runner, on a pushed release tag and on manual dispatch |
| GPU | `ci-gpu`, `runtime-qualification` and `fixed-runtime-slo`; `publish-pypi`, `gpu` | From the build tree, the whole `python/tests/mlir` directory including the CUDA tests, one of which runs the segmented example, and the fixed vector-add example. From an installed wheel, the CUDA health check, the CUDA smoke script, `scripts/qualify_installed_segments.sh`, and the benchmark gates | One self-hosted machine with an NVIDIA RTX A6000: `ci-gpu` on `main` only, weekly and on request; the release job on a release run |

The hosted jobs install one hash-locked tool set from `requirements-ci.txt`.
`ci-cpp` also defines a `format` job, which checks the C and C++ sources
with `clang-format`, and a `sanitizers` job, which runs after
`build-and-test` on pull requests and runs the lit suite and the C++ unit
tests under AddressSanitizer and UndefinedBehaviorSanitizer against the
uninstrumented LLVM. The `sanitizers-instrumented-llvm` job of
`.github/workflows/sanitizers.yml` builds an instrumented LLVM and runs on
pushes to `main` and weekly. No row below relies on the format or sanitizer
jobs.

All tiers run on Linux x86-64. The native build and the wheels require
Linux x86-64 and the exact LLVM release in `cmake/llvm-version.txt`; CMake
rejects any other release. A frontend-only install on another platform is
unknown.

## Python

| Python | Status | Evidence |
|---|---|---|
| 3.9 and older | Rejected | `requires-python = ">=3.10,<3.14"` in `pyproject.toml`; the build scripts stop when neither `python` nor `python3` is 3.10 or newer |
| 3.10, 3.11, and 3.12 | Tested in the pure Python and release wheel tiers | `ci-python` matrix; the `wheels` job of `publish-pypi` builds one wheel per version and runs the CPU checks against it. No workflow launches on CUDA with them |
| 3.13 | Tested in every tier | `ci-python` matrix; `ci-cpp` builds and tests the bindings and a local wheel on 3.13; the release and GPU jobs install a 3.13 wheel; the committed A6000 records name 3.13.13 on the GPU runner |
| 3.14 and newer | Rejected | `requires-python` excludes them, and no wheel is built for them |

The `runtime-qualification` job of `ci-gpu` uses the Python installed on the
runner and does not pin it. Free-threaded CPython and PyPy have no wheel.

## PyTorch

PyTorch is optional. Metadata inference, either launch backend, and the
segmented calls need it. Capture, the kernel-language check, the
environment report, and emission with an explicit signature do not. The
segmented calls and `python -m swage.compile` also need `numpy`, which the
`pytorch` extra declares without a version bound. The binding requirements
of the native build install it too, and the tests run with `numpy` 2.1.2. A
segmented call without it raises a `BackendUnavailableError` with the code
`numpy-unavailable`; `tests/python/test_segments.py`.

| PyTorch | Status | Evidence |
|---|---|---|
| Not installed | Tested in the pure Python tier, and for explicit-signature emission in the native tier | `requirements-ci.txt`, which `ci-python` installs, has no PyTorch; `python/tests/mlir/test_examples.py` runs the emit-only example with the PyTorch import blocked |
| Older than 2.6 | Rejected at launch and at a segmented call | `launch()`, `segment_reduce`, and `segment_softmax` raise a `BackendUnavailableError` with the code `pytorch-unsupported` that names the version found, before any kernel is compiled or enqueued; `tests/python/test_runtime.py` and `tests/python/test_segments.py`. Compile-only emission does not run this check |
| 2.6.0+cpu | Tested in the release wheel tier | The `wheels` job of `publish-pypi` installs it before the CPU checks of each wheel |
| 2.7 through 2.11 | Admitted | Inside the declared `torch>=2.6,<3` in `pyproject.toml`; no workflow runs these releases |
| 2.12.0+cu130 | Tested in the GPU tier | The build on the GPU runner, named by the committed A6000 records; `ci-gpu` prints the version and does not pin it |
| 2.13.0+cpu | Tested in the native tier, with every CUDA test skipped | Pinned in `ci-cpp` |
| Other 2.x builds | Admitted | Inside the declared range; no workflow runs them |

## CUDA driver

| Driver | Status | Evidence |
|---|---|---|
| CUDA driver API 13.0 | Tested in the GPU tier | The committed A6000 records store `cuda_driver: 13.0`, the value `cuDriverGetVersion` reports on the GPU runner |
| Any other driver | Unknown | No test covers another driver |

The records store the driver API level, not the NVIDIA driver package
version. `python -m swage.env --json` reports the same value as
`cuda_driver`.

## GPU architecture

A launch targets the active device exactly. The admitted processor list and
the rejection rules are stated in
[Runtime and Environment](runtime-environment.md).

| Target | Status | Evidence |
|---|---|---|
| `sm_86` | Qualified | The GPU tier executes on one NVIDIA RTX A6000, the GPU runner |
| `sm_80`, `sm_87`, `sm_88`, `sm_89`, `sm_90`, `sm_100`, `sm_101`, `sm_103`, `sm_110`, `sm_120`, and `sm_121` | Admitted, not qualified | `python/tests/mlir/test_target_compile.py` emits PTX for the public kernel, and for an identity sum through every private segmented entry point, on each of them in the native tier without a device. No test executes on one |
| Below `sm_80`, and a processor the pinned LLVM does not define, such as `sm_85` | Not admitted | Compiler admission; the same test file checks that every other `sm_` value from 80 through 129 is refused |

`sm_86` is the only qualified target, for the fixed kernels, for the public
segmented calls, and for the private segmented paths. Committed records
under `benchmarks/results/` also report benchmark runs and one correctness
run on an NVIDIA GeForce RTX 5090 (`sm_120`), taken outside the workflows.
They are recorded evidence and do not make `sm_120` qualified; see
[Benchmarks](../internals/benchmarks.md).

## Native wheel

`swage-compiler` is one native wheel per CPython version;
[Installation](../getting-started/installation.md) describes how to install
it and how to build one locally.

| Environment | Status | Evidence |
|---|---|---|
| `manylinux_2_28` x86-64 wheels for CPython 3.10 to 3.13 | Tested in the release wheel tier | The `wheels` job of `publish-pypi` builds and repairs each in the digest-pinned `manylinux_2_28_x86_64` image and runs the CPU checks against each installed wheel; the `aggregate` job requires a byte-identical rebuild of the 3.13 wheel |
| The released 3.13 wheel on the NVIDIA RTX A6000 | Tested in the GPU tier | The `gpu` job of `publish-pypi` installs it and runs the CUDA health check, the fixed runtime tests, the CUDA smoke script with a second-process cache hit, `scripts/qualify_installed_segments.sh`, and the benchmark gates |
| A local CPython 3.13 wheel built on a hosted runner (`linux_x86_64`) | Tested in the native tier | The `build-and-test` job of `ci-cpp` builds it, installs it into a fresh environment, and runs the CPU checks and `python/tests/mlir` against it outside the checkout |
| A local CPython 3.13 wheel built on the GPU runner | Tested in the GPU tier | The `fixed-runtime-slo` job of `ci-gpu` builds it with the command of [Installation](../getting-started/installation.md#build-a-local-native-wheel) and runs the CUDA health check, the CUDA smoke script, `scripts/qualify_installed_segments.sh`, and the benchmark gates |
| macOS, Windows, musl, AArch64, free-threaded, or PyPy | Rejected | No wheel is built for them; `--only-binary=:all:` makes installation stop |
| Bindings built for another `swage` version, or that record none | Rejected | `swage` raises when it first uses the bindings; `tests/python/test_native_identity.py`, `python/tests/mlir/test_native_identity.py` |

## Segmented calls

`swage.segment_reduce` and `swage.segment_softmax` have a narrower tested
surface than the tables above suggest for the package as a whole:

| Aspect | Status | Evidence |
|---|---|---|
| Argument checks that need no native bindings, and the error without bindings | Tested in the pure Python tier | `tests/python/test_segments.py`, with a stand-in for PyTorch |
| Argument and offsets validation on host tensors | Tested in the native tier | The tests of `python/tests/mlir/test_public_segments.py` that need no GPU |
| Results, schedules, streams, threads, graph capture, and resource use | Tested in the GPU tier on `sm_86` | The CUDA tests of `python/tests/mlir/test_public_segments.py` and the example test in `test_examples.py` |
| The kinds `"sum"`, `"max"`, `"min"`, and `"mean"` | Tested in the GPU tier on `sm_86` | Each kind runs the differential suite, the long segments, the empty segments, and its special values in `python/tests/mlir/test_public_segments.py`; a mean is also compared, bit for bit, with the sum of the same call divided by the length, and with the exactly rounded mean under its bound |
| float32 and float64 values of `segment_reduce`, rank one | Tested in the GPU tier on `sm_86` | Each dtype runs the differential suite, the long segments, the empty segments, and the special values of each kind; a float64 sum is compared with the exactly rounded sum; other dtypes and ranks raise a `TypeError`; the same file |
| `[N, D]` values of `segment_reduce` | Tested in the GPU tier on `sm_86` | `python/tests/mlir/test_segment_columns.py`: every kind and both dtypes agree with `torch.segment_reduce` along axis 0 at 1, 3, 64, 129, 200, and 1024 columns, equal the CPU oracle bit for bit, and are exact on values that depend on the row and on the column; `[N, 1]` takes the rank-one schedules and `[N, 0]` launches nothing; `python/tests/mlir/test_segmented_bounds.py` launches the kernel below the Python validation with row ranges and feature counts that validation rejects; values of rank three or above raise a `TypeError` |
| float32 values of `segment_softmax`, rank one | Tested in the GPU tier on `sm_86` | The differential suite, one long segment, the empty segments, and the special values in `python/tests/mlir/test_public_segments.py`; float64 values raise a `TypeError` that states the reason, the missing 64-bit `exp2` of the device |
| `[N, D]` values of `segment_softmax`, float32 | Tested in the GPU tier on `sm_86` | `python/tests/mlir/test_segment_columns.py`: every column agrees with float64 `torch.softmax` along the rows of its segment inside the bound of the softmax page at `k = n - 1`, at 1, 3, 64, 129, 200, and 1024 columns and for a segment of 100,003 rows, agrees with the CPU oracle within the derived tolerance, and follows `torch.softmax` on NaN and infinities per column; `[N, 1]` runs the rank-one kernel and `[N, 0]` launches nothing; `python/tests/mlir/test_segmented_bounds.py` launches the kernel below the Python validation with row ranges and feature counts that validation rejects and with an output of fewer rows than the values; float64 values and values of rank three or above raise a `TypeError` |
| int32 and int64 offsets | Tested in the GPU tier on `sm_86` | int64 offsets are refused by their 64-bit values, give the bits of int32 offsets on every distribution of the suite, and reach a kernel only as a retained private int32 copy; other offset dtypes raise a `TypeError`; the same file |
| Both calls from an installed native wheel | Tested in the GPU tier on `sm_86` | `scripts/qualify_installed_segments.sh` runs `test_public_segments.py`, `test_segment_columns.py`, and `test_artifact.py` from a copy, with `PYTHONPATH` unset, against the installed wheel, and fails on any skipped test; the `gpu` job of `publish-pypi` and the `fixed-runtime-slo` job of `ci-gpu` run it |
| A second GPU on one host | Unknown | The calls require the current device; the tests run on a host with one GPU |
| Gradients | Rejected | `values` that require grad raise a `ValueError`; the same files |
| Tensors created under `torch.inference_mode()` | Tested in the GPU tier on `sm_86` | `values`, `offsets`, and `out` are inference tensors in `python/tests/mlir/test_public_segments.py` |
| An install without `numpy` | Rejected | A `BackendUnavailableError` with the code `numpy-unavailable`, with the bindings and with an artifact; `tests/python/test_segments.py` and `tests/python/test_artifact.py` block the import |

The GPU tests of the segmented calls ran on one NVIDIA RTX A6000 with
PyTorch 2.12.0+cu130 and Python 3.13, on the branch that added the calls.
The GPU workflows run on `main` and on release runs only; the GPU rows above
describe what those workflows are configured to run.

## Artifacts

`python -m swage.compile` writes the kernels of the two segmented calls
ahead of time, and `SWAGE_ARTIFACT_DIR` makes a process run the calls from
the result.
[Running Without the Compiler](../user-guide/deployment.md) describes both.
These rows use the statuses of the tables above, and one more:
**Checked by hand** means the commands were run once on one machine and no
workflow repeats them.

| Aspect | Status | Evidence |
|---|---|---|
| Writing an artifact for each admitted processor, with no device and no PyTorch | Tested in the native tier | `python/tests/mlir/test_artifact.py` writes one per processor, and one in a process where PyTorch cannot be imported and no device is visible |
| Writing an artifact from an installed native wheel | Tested in the native tier | The `build-and-test` job of `ci-cpp` runs `python/tests/mlir/test_artifact.py` against the installed CPython 3.13 wheel, which carries the runtime library that the command copies |
| Selection, verification, the trust rule, and every refusal | Tested in the pure Python tier | `tests/python/test_artifact.py`, with a stand-in for the runtime library |
| The classifier of the runtime library | Tested in the native tier | `unittests/RuntimeTest.cpp` and `python/tests/mlir/test_segmented_classification.py` compare it with the compiler's classifier |
| Both calls from an artifact, in a process that cannot import `mlir_swage` | Tested in the GPU tier on `sm_86` | `python/tests/mlir/test_artifact.py`: no LLVM or MLIR library is mapped, the results agree with PyTorch and float64 references, and they equal the compiled path bit for bit |
| Both calls from an artifact, in a process in which `mlir_swage` is importable | Tested in the GPU tier on `sm_86` | The same file: no LLVM or MLIR library is mapped until the process launches the fixed vector add, which compiles |
| Both calls from an artifact with an installed native wheel and no checkout on any path | Tested in the GPU tier on `sm_86` | `scripts/qualify_installed_segments.sh` runs the same file against the installed wheel |
| Both calls from an artifact with a frontend-only package, PyTorch 2.12.0+cu130, `numpy` 2.1.2, and no checkout or build tree on any path | Checked by hand | Run once in a fresh virtual environment on the Linux x86-64 machine with the NVIDIA RTX A6000, Python 3.13, on the branch that added the feature |
| An artifact for another target than the device | Rejected | A `RuntimeError` before any kernel is loaded; the same file |
| An artifact directory owned by another account | Tested with a simulated owner. Checked by hand under a second user id, without a GPU | `tests/python/test_artifact.py` reports another owner and another effective user to the loader. Once, in a container without a GPU, a process under another user id loaded an artifact that was mounted read-only and owned by the account that wrote it, and classified with its runtime library; no kernel was launched there |
| A read-only artifact directory | Tested in the pure Python tier | The same file removes every write permission before loading |
| An artifact on a host of another machine, such as AArch64 | Unknown | No runtime library was built for another machine. The loader refuses an artifact whose library names another machine; the same files |
| An artifact of format version 1, which an earlier `swage` wrote | Rejected | A `RuntimeError` that names both versions and says to write the artifact again; `tests/python/test_artifact.py` |
| An artifact loaded by a `swage` of another source revision | Rejected when a program text differs, otherwise unknown | The loader compares the digest of each program text and the launch description of each kernel, and nothing else of the two revisions |

The GPU rows ran on one NVIDIA RTX A6000 with PyTorch 2.12.0+cu130 and
Python 3.13, on the branch that added the feature. The GPU workflows run on
`main` and on release runs only; the GPU rows above describe what those
workflows are configured to run.

## Stability

Versions follow semantic versioning in the `0.x` range, where any release
may change public behavior. The
[changelog](https://github.com/abhiksark/swage/blob/main/CHANGELOG.md)
records each change, and the
[security policy](https://github.com/abhiksark/swage/blob/main/SECURITY.md)
states that only the latest `main` receives fixes.

Continue with [Installation](../getting-started/installation.md) for the
build prerequisites, or run `python -m swage.env --json` and compare its
output with the rows above before reporting a problem.
