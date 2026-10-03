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

Include the JSON, selected backend, and failing command in bug reports. The
report works without native bindings or PyTorch. Without `--check`, missing
components do not change the zero exit status. A selected unavailable
component exits one while still printing the complete report; malformed CLI
options exit nonzero. A metadata warning in `native.error` need not mean that
native compilation is unavailable. See the
[report schema](../reference/runtime-environment.md#environment-report).

These pages describe the implemented v0.5.2 contract, whose publication and
production qualification remain pending. The latest released tag is v0.5.1;
do not expect its older artifacts to have the new native-wheel contents.

## `mlir_swage` cannot be imported

A v0.5.2 native wheel includes self-contained private `mlir_swage` bindings.
Do not install an unrelated `mlir` package to fix it. Check the installed
version, interpreter, platform, and `native.error` against the
[support matrix](../reference/runtime-environment.md#support-matrix).
Install a verified wheel for that ABI using [Installation](installation.md);
`--only-binary=:all:` avoids silently attempting a source build on an
unsupported platform. A frontend-only editable install intentionally omits
native bindings.

For a source build, use exact LLVM/MLIR 22.1.8 and the build-tree package:

```bash
ninja -C build check-swage-python
PYTHONPATH=build/python_packages python -m swage.env --json --check native
PYTHONPATH=build/python_packages python your_script.py
```

If CMake reports missing MLIR Python bindings, rebuild the pinned toolchain
with bindings enabled and reconfigure Swage:

```bash
SWAGE_LLVM_PYTHON_BINDINGS=ON ./scripts/build_llvm.sh
cmake -G Ninja -S . -B build \
    -DMLIR_DIR=/path/to/lib/cmake/mlir \
    -DLLVM_DIR=/path/to/lib/cmake/llvm \
    -DSWAGE_PYTHON_BINDINGS=ON
```

## PyTorch or CUDA is unavailable

Metadata inference and launch require `torch>=2.6,<3`, available through the
optional `pytorch` extra. Explicit-signature MLIR emission does not. CUDA
additionally requires a CUDA-enabled PyTorch build, an admitted current GPU,
and the driver's `libcuda.so.1`; there is no runtime toolkit-compiler
requirement. Install the appropriate PyTorch build rather than switching
backends automatically. CPU and CUDA never fall back to each other.

`BackendUnavailableError` is a public `SwageError` (and `RuntimeError`) with
stable string `code`, `backend`, and `remediation` attributes. Use the code,
not message matching, to distinguish prerequisite failures:

| Code | Action |
|---|---|
| `native-unavailable` | Install a matching verified native wheel or build the exact pinned bindings. |
| `pytorch-unavailable` | Install the optional PyTorch runtime in the same interpreter. |
| `cuda-unavailable` | Select a CUDA-enabled PyTorch build and ensure the intended GPU is visible. |
| `cuda-driver-unavailable` | Install or expose the NVIDIA driver and `libcuda.so.1`. |
| `cuda-context-unavailable` | Establish the intended current CUDA context through PyTorch; Swage never creates or switches one. |

The error's `backend` identifies the attempted component (`native`, `cpu`, or
`cuda`); `remediation` carries the actionable suggestion. Unsupported target,
dtype, shape, grid, source, cache-integrity, or driver-call failures are not
reclassified as availability errors. `CompilationError` retains source-located
frontend messages, including failures to infer PyTorch metadata. See
[Exceptions](../reference/swage.md#exceptions) for the full distinction.

The A6000/`sm_86` hardware boundary is the required release-qualification
configuration; other admitted targets remain best-effort. A successful health
check is not performance qualification. Compare `torch_cuda_build` with
`cuda_driver` separately and apply the driver floor in the
[runtime reference](../reference/runtime-environment.md#support-matrix).

## The wrong `swage` checkout is imported

Editable installs can point Python at another worktree. For installed-wheel
checks, leave the checkout and remove `PYTHONPATH` rather than overriding the
installed package with source files. Inspect the import path:

```bash
python -c 'import pathlib, swage; print(pathlib.Path(swage.__file__).resolve())'
```

For intended source development, install that checkout with
`python -m pip install -e ".[dev]" -Cwheel.cmake=false`. Native tests need both
the source package and build-tree bindings; prefer
`ninja -C build check-swage-python` because CMake supplies that environment.

## The cache is not reused

Persistent caching applies only to CUDA PTX. CPU executables remain
process-local. Native wheels use the validated `_build_info.json` resource;
source builds fall back to clean checkout identity only when that resource is
absent. A dirty identity disables persistence. Malformed packaged metadata is
reported in `native.error` and disables persistent cache reads and writes,
but does not itself disable native compilation or process-local reuse. Do not
edit metadata to fabricate a clean identity; rebuild or reinstall a verified
artifact. See [native build identity](../reference/runtime-environment.md#native-build-identity).

Set `SWAGE_CACHE_DIR` to a fresh isolated directory and rerun. Do not weaken
cache checks or remove a broad shared cache while diagnosing a failure.
[Opt-in logging](../reference/runtime-environment.md#opt-in-logging) exposes
`memory-hit`, `persistent-hit`, `compile`, and backend launch events without
cache paths, tensor contents, pointers, or PTX. Swage configures no handlers
and prints nothing by default. Compiler dumps are separate and not sanitized
logs; treat them as trusted, potentially sensitive native artifacts under the
[security policy](https://github.com/abhiksark/swage/blob/main/SECURITY.md).

## The build uses the wrong LLVM/MLIR

Compare the selected install to `cmake/llvm-version.txt` (`llvmorg-22.1.8`).
CMake requires an exact version match. Reconfigure only the affected Swage
build directory with matching `MLIR_DIR` and `LLVM_DIR` paths; do not update
the project pin to fit a local toolchain. Wheel builds additionally require
Release mode and explicit source identity; a normal build-tree build is a
separate mode. Follow [Installation](installation.md#build-from-source).

Once setup works, the [Quickstart](quickstart.md) exercises explicit CPU or
CUDA selection. Exact launch, cache, lifetime, and ABI rules live in
[swage](../reference/swage.md) and
[Runtime and Environment](../reference/runtime-environment.md).
