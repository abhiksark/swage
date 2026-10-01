<!-- docs/getting-started/troubleshooting.md -->

# Troubleshooting

Most setup failures come from crossing the published-package, native-build,
or CUDA execution boundaries. Start by recording the environment:

```bash
python -m swage.env
```

Include that output and the failing command in bug reports. The `revision`,
`swage_file`, `llvm_linked`, `mlir_swage_file`, and `backends` lines identify
the checkout and the native build that the command saw.

## `mlir_swage` cannot be imported

The environment report shows this state as:

```text
llvm_linked: None
mlir_swage_file: None
backends: {'mlir': 'unavailable (build-tree mlir_swage bindings not importable)'}
```

`mlir_swage` is not included in the `swage-compiler` wheel. Build Swage
against the pinned LLVM/MLIR install, then use the build-tree package:

```bash
ninja -C build check-swage-python
PYTHONPATH=build/python_packages python your_script.py
```

Run the report with the same `PYTHONPATH` to confirm the bindings import.
`mlir_swage_file` names the extension that was loaded, which shows the build
tree it came from:

```text
llvm_linked: 22.1.8
mlir_swage_file: /path/to/swage/build/python_packages/mlir_swage/_mlir_libs/_swageDialectsNanobind.cpython-313-x86_64-linux-gnu.so
backends: {'mlir': 'available (linked LLVM 22.1.8)'}
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

A CUDA launch additionally requires Linux, a CUDA-enabled PyTorch build, and
`libcuda.so.1` from the installed driver. See
[Runtime and Environment](../reference/runtime-environment.md) for exact
target-admission and zero-work rules. Swage does not use the CUDA toolkit
compiler on the production path.

A launch on a PyTorch older than 2.6 fails before any work starts:

```text
RuntimeError: Swage launch requires PyTorch 2.6 or newer; found PyTorch 2.5.1
```

The `target` line of the report says whether the current device is one the
GPU tests execute on. `sm_86 (qualified)` is the only qualified target. A
line such as `sm_89 (admitted, not qualified)` means that the compiler emits
PTX for the device and that no test has executed it there.

## The wrong `swage` checkout is imported

Editable installs can point Python at another worktree. Pin the intended
checkout explicitly while testing:

```bash
PYTHONPATH="$PWD/python" python -c \
    'import pathlib, swage; print(pathlib.Path(swage.__file__).resolve())'
PYTHONPATH="$PWD/python" python -m pytest tests/python -q
```

The `swage_file` line of `python -m swage.env` is the file that was
imported, and the `revision` line names its checkout. Compare the revision
with `git rev-parse --short=12 HEAD` in the intended worktree; a `-dirty`
suffix means tracked files differ from that commit, and `None` means `swage`
was imported from outside a git checkout.

Native tests need both the source package and build-tree bindings in their
configured path. Prefer `ninja -C build check-swage-python` because CMake
provides that environment.

## The cache is not reused

Read the `cache_dir` and `cache` lines of `python -m swage.env`, run with the
same environment as the failing command. `active` means the process reads
published entries; `off (<reason>)` gives the reason it compiles without
them:

```text
cache_dir: /home/user/.cache/swage
cache: off (the native compiler libraries are not found)
compile_on_miss: allowed
```

Set `SWAGE_CACHE_DIR` to a fresh, isolated directory and rerun the command. If
reuse still does not occur, include the command, error, and `python -m
swage.env` output in the report. Do not weaken cache checks or remove a broad
shared cache while diagnosing the symptom.

## A launch refuses to compile

With `SWAGE_NO_COMPILE=1` a kernel that is not cached raises instead of
compiling:

```text
RuntimeError: SWAGE_NO_COMPILE=1 refuses to compile kernel 'add_kernel': no entry <key> in /home/user/.cache/swage
```

The key includes the kernel source, the block size, the device target, the
`swage` sources, and the native libraries. Launch the kernel once with the
same values in a process that may compile, or unset the variable. When the
message gives a reason the cache is off instead of a missing key, the `cache`
line of the report shows the same reason.

[Runtime and Environment](../reference/runtime-environment.md) owns the rules
for persistent reuse, cache location, and entry validation.

## The build uses the wrong LLVM/MLIR

Compare `cmake/llvm-version.txt` with the selected install. `python -m
swage.env` prints both sides: `llvm_pin` is the repository tag and
`llvm_linked` is the release the imported bindings were built against, so a
matching build shows `llvmorg-22.1.8` and `22.1.8`. Delete or reconfigure only
the affected Swage build directory, then pass matching `MLIR_DIR` and
`LLVM_DIR` paths. Do not update the project pin to fit a local
toolchain.

Once setup works, the [Quickstart](quickstart.md) exercises the supported
path. Exact launch and cache contracts live in [swage](../reference/swage.md)
and [Runtime and Environment](../reference/runtime-environment.md).
