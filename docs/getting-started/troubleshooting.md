<!-- docs/getting-started/troubleshooting.md -->

# Troubleshooting

Most setup failures come from crossing the published-package, native-build,
or CUDA execution boundaries. Start by recording the environment:

```bash
python -m swage.env
```

Include that output and the failing command in bug reports.

## `mlir_swage` cannot be imported

`mlir_swage` is not included in the `swage-compiler` wheel. Build Swage
against the pinned LLVM/MLIR install, then use the build-tree package:

```bash
ninja -C build check-swage-python
PYTHONPATH=build/python_packages python your_script.py
```

If configuration reports that MLIR Python bindings are missing, rebuild the
pinned toolchain with bindings enabled and reconfigure Swage:

```bash
SWAGE_LLVM_PYTHON_BINDINGS=ON ./scripts/build_llvm.sh
cmake -G Ninja -S . -B build \
    -DMLIR_DIR=/path/to/lib/cmake/mlir \
    -DLLVM_DIR=/path/to/lib/cmake/llvm \
    -DSWAGE_PYTHON_BINDINGS=ON
```

## PyTorch or CUDA is unavailable

Metadata inference and launch require the optional PyTorch dependency:

```bash
python -m pip install "swage-compiler[pytorch]"
python -m swage.env
```

A CUDA launch additionally requires Linux, a CUDA-enabled PyTorch build, a
supported NVIDIA GPU, and `libcuda.so.1` from the installed driver. Swage does
not use the CUDA toolkit compiler on the production path.

## The wrong `swage` checkout is imported

Editable installs can point Python at another worktree. Pin the intended
checkout explicitly while testing:

```bash
PYTHONPATH="$PWD/python" python -c \
    'import pathlib, swage; print(pathlib.Path(swage.__file__).resolve())'
PYTHONPATH="$PWD/python" python -m pytest tests/python -q
```

Native tests need both the source package and build-tree bindings in their
configured path. Prefer `ninja -C build check-swage-python` because CMake
provides that environment.

## The cache is not reused

Persistent caching is deliberately disabled for a dirty or unidentified
compiler checkout. Clean, identified builds may reuse verified disk entries;
other builds use only process-local reuse. Set `SWAGE_CACHE_DIR` to isolate a
run when investigating cache behavior.

Unsafe, incomplete, corrupt, or specialization-mismatched entries fail
closed. Remove only the specific isolated cache directory you created for the
investigation, then rerun the command.

## The build uses the wrong LLVM/MLIR

Compare `cmake/llvm-version.txt` with the selected install. Delete or
reconfigure only the affected Swage build directory, then pass matching
`MLIR_DIR` and `LLVM_DIR` paths. Do not update the project pin to fit a local
toolchain.

Once setup works, the [Quickstart](../quickstart.md) exercises the supported
path. Exact launch and cache contracts live in [Public Python API](../reference/public-python-api.md)
and [Runtime and Environment](../reference/runtime-environment.md).
