<!-- docs/getting-started/troubleshooting.md -->

# Troubleshooting

Start by recording the complete environment and checking only the component
you intend to use:

```bash
python -m swage.env --json
python -m swage.env --json --check native
python -m swage.env --json --check cpu
# Select this check for CUDA instead of CPU:
python -m swage.env --json --check cuda
```

Include the JSON, the selected backend, and the failing command in bug
reports. The report works without native bindings or PyTorch. Without
`--check`, missing components do not change the zero exit status. A
selected unavailable component exits with status 1 while still printing the
complete report; malformed command-line options exit non-zero. The
[Environment report](../reference/runtime-environment.md#environment-report)
describes the fields. These identify the code and the build that the
command saw:

- `source.file` and `source.revision`: the `swage` package that was
  imported, and the revision of its checkout, with `-dirty` for a modified
  tree.
- `native.available` and `native.error`: whether the native bindings can be
  used, and why not.
- `native.bindings`: the `swage` version and source revision the bindings
  were built for, the LLVM release they link (`llvm_linked`), the extension
  file that was loaded, and the `problem` that refuses them, if any.
- `backends.cpu` and `backends.cuda`: whether each backend is available and
  the reason when it is not. For CUDA, `target` is the processor of the
  current device and `qualified` says whether it is the qualified NVIDIA
  RTX A6000 (`sm_86`).

These pages describe the implemented `v0.5.2` contract, whose publication
and production qualification remain pending. The latest released tag is
`v0.5.1`, a pure Python wheel;
[Installation](installation.md#what-the-released-051-wheel-lacks) lists
what it lacks.

## `mlir_swage` cannot be imported

The report shows this state as `native.available: false` with
`native.error: "native-unavailable"`. A `v0.5.2` native wheel includes
self-contained private `mlir_swage` bindings. Do not install an unrelated
`mlir` package to fix it. Check the installed version, interpreter,
platform, and `native.error` against the
[Support Matrix](../reference/support-matrix.md). Install a verified wheel
for that ABI using [Installation](installation.md); `--only-binary=:all:`
avoids silently attempting a source build on an unsupported platform. A
frontend-only editable install has no native bindings by design, and
neither has the released `0.5.1` wheel.

For a source build, use exact LLVM/MLIR 22.1.8 and the build-tree package:

```bash
ninja -C build check-swage-python
PYTHONPATH=build/python_packages python -m swage.env --json --check native
PYTHONPATH=build/python_packages python your_script.py
```

`native.bindings.file` names the extension that was loaded, which shows the
build tree or the installed wheel it came from. If CMake reports missing
MLIR Python bindings, rebuild the pinned toolchain with bindings enabled and
reconfigure Swage:

```bash
SWAGE_LLVM_PYTHON_BINDINGS=ON ./scripts/build_llvm.sh
cmake -G Ninja -S . -B build \
    -DMLIR_DIR=/path/to/lib/cmake/mlir \
    -DLLVM_DIR=/path/to/lib/cmake/llvm \
    -DSWAGE_PYTHON_BINDINGS=ON
```

## `mlir_swage` is rejected

Bindings that load but were not built for the `swage` that is loaded show
`native.error: "native-mismatch"`, and `native.bindings.problem` gives the
reason, which names both versions and both locations. Emission and launch
raise a `RuntimeError` with the same text. Rebuild the bindings from the
sources of that `swage`, or install a wheel that holds both. The bindings
also record the digest of the frontend sources they were built beside.
Outside a checkout, bindings built beside another frontend are refused in
the same way. In a checkout, bindings whose native sources differ from the
revision they were built from are refused, and bindings whose `swage`
sources alone differ are used with a warning; rebuild the bindings to remove
either.
[Frontend and bindings](../reference/runtime-environment.md#frontend-and-bindings)
states the rule.

## PyTorch or CUDA is unavailable

Metadata inference, either launch backend, and the segmented calls require
`torch>=2.6,<3`, available through the optional `pytorch` extra. Explicit
signature MLIR emission does not. CUDA additionally requires a
CUDA-enabled PyTorch build, an admitted current GPU, and the driver's
`libcuda.so.1`; there is no runtime requirement for the CUDA toolkit
compiler. Install the appropriate PyTorch build rather than switching
backends automatically. CPU and CUDA never fall back to each other.

`BackendUnavailableError` is a public `SwageError` (and `RuntimeError`) with
stable string `code`, `backend`, and `remediation` attributes. Use the
code, not message matching, to distinguish prerequisite failures:

| Code | Action |
|---|---|
| `native-unavailable` | Install a matching verified native wheel or build the exact pinned bindings. |
| `pytorch-unavailable` | Install the optional PyTorch runtime in the same interpreter. |
| `pytorch-unsupported` | Install PyTorch 2.6 or newer; the error names the version found. |
| `numpy-unavailable` | Install `numpy`, which the `pytorch` extra declares; the segmented calls need it. |
| `cuda-unavailable` | Select a CUDA-enabled PyTorch build and ensure the intended GPU is visible. |
| `cuda-driver-unavailable` | Install or expose the NVIDIA driver and `libcuda.so.1`. |
| `cuda-context-unavailable` | Establish the intended current CUDA context through PyTorch; Swage never creates or switches one. |

The error's `backend` identifies the attempted component (`native`, `cpu`,
or `cuda`); `remediation` carries the suggested action. Unsupported target,
dtype, shape, grid, source, cache-integrity, or driver-call failures are
not reclassified as availability errors. `CompilationError` keeps
source-located frontend messages, including failures to infer PyTorch
metadata. See [Exceptions](../reference/swage.md#exceptions) for the full
distinction.

`backends.cuda.qualified` is `true` only on the NVIDIA RTX A6000 (`sm_86`),
the required release-qualification configuration. On another admitted
target it is `false`: the compiler emits PTX for the device, and no test
has executed it there. A target that the compiler does not admit makes
`backends.cuda.available` false, with the reason
`CUDA target is not admitted by the pinned compiler`. A successful health
check is not performance qualification. Compare `torch_cuda_build` with
`cuda_driver` separately; the [Support Matrix](../reference/support-matrix.md)
lists the driver the tests run with.

## The wrong `swage` checkout is imported

Editable installs can point Python at another worktree. For installed-wheel
checks, leave the checkout and remove `PYTHONPATH` rather than overriding
the installed package with source files. Inspect the import path:

```bash
python -c 'import pathlib, swage; print(pathlib.Path(swage.__file__).resolve())'
```

`source.file` in the report is the same file, and `source.revision` names
its checkout. Compare the revision with `git rev-parse --short=12 HEAD` in
the intended worktree; a `-dirty` suffix means tracked files differ from
that commit, and `null` means `swage` was imported from outside a Swage
checkout and without a build record.

For intended source development, install that checkout with
`python -m pip install -e ".[dev]" -Cwheel.cmake=false`. Native tests need
both the source package and the build-tree bindings; prefer
`ninja -C build check-swage-python` because CMake supplies that
environment.

## The cache is not reused

Persistent caching applies only to CUDA PTX; CPU executables remain
process-local. Read the `cache` object of the report, run with the same
environment as the failing command:

- `directory` is the cache root.
- `state` reads `active` when the process reads and publishes entries,
  `off (<reason>)` when it compiles without them, `rejected (<reason>)`
  when the cache root is unsafe and every lookup raises, and
  `unknown (<error>)` when a cache variable has a value that a launch
  rejects.
- `compile_on_miss` reads `allowed`, or `refused (SWAGE_NO_COMPILE=1)`.

A process uses the cache when it can identify its frontend sources and its
native libraries and none of them changed after the process started; a
clean checkout is not required. Set `SWAGE_CACHE_DIR` to a fresh isolated
directory and rerun. Do not weaken cache checks or remove a broad shared
cache while diagnosing a failure. The `swage.runtime` logger reports
`memory-hit`, `persistent-hit`, `compile`, and backend launch events at
DEBUG level, without cache paths, tensor contents, pointers, or PTX. Swage
configures no handlers and prints nothing by default. Compiler dumps are
separate and are not sanitized logs; treat them as trusted, potentially
sensitive native artifacts under the
[security policy](https://github.com/abhiksark/swage/blob/main/SECURITY.md).

## A launch refuses to compile

With `SWAGE_NO_COMPILE=1`, a kernel that is not cached raises a
`swage.SwageError` instead of compiling. The message starts with
`SWAGE_NO_COMPILE=1 refuses to compile kernel '<name>'` and gives the reason:
a missing cache entry, or the reason the cache is off. The key includes the
kernel source, the block size, the backend and device target, the `swage`
sources, and the native libraries. Launch the kernel once with the same
values in a process that may compile, or unset the variable. When the
message gives a reason the cache is off, `cache.state` shows the same
reason.

[Runtime and Environment](../reference/runtime-environment.md) owns the
rules for persistent reuse, cache location, and entry validation.

## A segmented call is refused

`swage.segment_reduce` and `swage.segment_softmax` raise before anything is
enqueued in these cases:

- The native bindings are missing and no artifact is selected. The call
  raises `BackendUnavailableError` with the code `native-unavailable`,
  after the argument checks that need no bindings. Install the native wheel,
  use a source build, or select an artifact directory.
- `SWAGE_NO_COMPILE=1` is set and the process does not hold the kernel. The
  segmented kernels are never in the persistent cache, so a process that
  starts with the variable set cannot run a segmented call that has work to
  do, unless an artifact is selected. Unset the variable for such a
  process.
- The current stream is capturing a CUDA graph. Make the call outside the
  capture.
- `numpy` cannot be imported. The call raises `BackendUnavailableError`
  with the code `numpy-unavailable`. Install it, for example through the
  `pytorch` extra, which declares it.
- `out` is given while the call records a gradient. Call without `out`,
  or under `torch.no_grad()`.

[Segmented Calls](../user-guide/segmented-calls.md#where-a-call-is-refused)
lists every refusal and the reason for it.

## An artifact directory is rejected

With `SWAGE_ARTIFACT_DIR` set, the two segmented calls run from that
directory and never compile in its place. The `artifact` field of the
report shows what a call would raise:

```text
artifact: rejected (<reason>)
```

The reason names the directory. These are the common ones:

- A file or the directory `is writable by its group or by other users`. A
  copy made under a umask of `002` sets that permission. Remove it with
  `chmod -R go-w` on the directory.
- A file `does not match its manifest`. The copy is incomplete or a file
  was changed; copy the artifact again.
- The artifact `holds kernels for` another target than `the current device
  needs`. Write an artifact for the processor that the report shows as
  `backends.cuda.target` on the serving host.
- The artifact `was compiled from another` program than this `swage` runs.
  The artifact and the installed package come from different source
  revisions; write the artifact again with the `swage` that loads it.
- The runtime library `was built for` another machine. An artifact runs
  only on hosts of the machine its runtime library was built for.

[Running Without the Compiler](../user-guide/deployment.md#refusals) lists
every refusal.

## The build uses the wrong LLVM/MLIR

Compare the selected install with `cmake/llvm-version.txt`
(`llvmorg-22.1.8`). The report shows `llvm_pin`, the release the build
record or the checkout names, and `native.bindings.llvm_linked`, the release
the imported bindings link, so a matching build shows `llvmorg-22.1.8` and
`22.1.8`. CMake requires an exact version match. Reconfigure only the
affected Swage build directory with matching `MLIR_DIR` and `LLVM_DIR`
paths; do not update the project pin to fit a local toolchain. Wheel builds
additionally require the `Release` build type and an explicit source
identity; a normal build-tree build is a separate mode. Follow
[Installation](installation.md#build-from-source).

Once setup works, the [Quickstart](quickstart.md) exercises explicit CPU or
CUDA selection and the segmented calls. Exact launch, cache, lifetime, and
ABI rules live in [swage](../reference/swage.md) and
[Runtime and Environment](../reference/runtime-environment.md).
