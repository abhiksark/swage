<!-- docs/internals/compiler-tools.md -->

# Compiler Tools and Passes

Swage provides one optimizer driver and a small registered pass surface. Every
segmented lowering mode can be run from the driver, including the private
split stages and the experimental persistent mode, so that each lowering can
be inspected and tested as text. As a first roundtrip,
`swage-opt` can parse, verify, and print a test module from the native
MLIR surface, which is broader than the public Python kernel language:

```bash
./build/bin/swage-opt test/Dialect/Swage/roundtrip.mlir
```

## `swage-opt`

`swage-opt` is built by the native CMake project and follows the standard
`mlir-opt` command shape:

```bash
./build/bin/swage-opt input.mlir
./build/bin/swage-opt --help
```

It registers the `swage` and `swage_plan` dialects plus the upstream dialects
used by current test and lowering paths. It also registers upstream MLIR
passes and the upstream dialect extensions those passes look up, so the
pipeline the code generation C API builds runs from text:

```bash
./build/bin/swage-opt input.mlir \
  --pass-pipeline='builtin.module(swage-segmented-reduction-to-gpu{block-size=128},gpu.module(convert-scf-to-cf,convert-gpu-to-nvvm{index-bitwidth=64}))'
```

A pipeline in the driver ends there. The two steps the C API runs
afterwards, the replacement of libdevice calls and PTX emission, are
functions of the C API and not registered passes.

## Registered Swage passes

| Pass argument | Options | Current admitted purpose |
|---|---|---|
| `--swage-fixed-block-to-gpu` | required positive `block-size` | Lower the canonical fixed vector-add shape to one GPU x-thread per lane |
| `--swage-segmented-reduction-to-scf` | optional `function` | Lower every admitted private segmented sum, max, or fused softmax function to sequential SCF and memref operations |
| `--swage-segmented-reduction-to-gpu` | required `block-size` from 1 to 1024 whose warp count, `ceil(block-size / 32)`, is a power of two; optional `use-task-ids`; optional `fused-mixed`, requires block size 128; optional `persistent`, requires block size 512; optional `function` | Lower every admitted private segment function to a GPU kernel module. `use-task-ids` cannot be combined with `fused-mixed` or `persistent` |
| `--swage-to-plan` | `warp-max-elements`, default 32; `cta-chunk-elements`, default 4096; optional `function` | Add one private planning companion for every capture-free, single-stage f32 sum or max function |
| `--swage-split-segmented-reduction-to-gpu` | optional `merge`; optional `function` | Lower every admitted private capture-free, single-stage f32 sum or max function to the split partial kernel, or to the split merge kernel when `merge` is set |

Planning limits must satisfy:

```text
0 < warp-max-elements <= cta-chunk-elements <= INT32_MAX
```

The planning pass preserves each admitted semantic function and adds one
private companion with `swage_plan.classify` for it. It does not lower a
general task graph or inspect runtime offset contents.

## Functions and symbols

A segment function is a `func.func` that holds an operation of the `swage`
dialect and declares its arguments with `swage.role`, as
[Textual Swage IR](../language/swage-ir.md#argument-roles) describes. The
four segmented passes treat a module the same way:

- A pass lowers every segment function of the module and leaves the other
  functions as they are. A module without a segment function is left
  unchanged.
- `function=<name>` restricts a pass to the function of that name. The pass
  fails when the name is not a function of the module or names a function
  without Swage operations.
- A pass admits every function it will lower before it changes any of them,
  so a module that is rejected is left as it was.

A GPU lowering replaces a segment function by a `gpu.module` named
`<kernel>_module`, where `<kernel>` is the function name, followed by
`__partial` or `__merge` for a split stage. Before it changes anything, the
pass requires that nothing in the module refers to the function, that
`<kernel>_module` is not defined, and, for a split stage, that `<kernel>` is
not defined. The sequential lowering rewrites a function in place, so a
function it lowers may have callers.

The code generation C API passes its `kernelName` as `function` and then
selects the `gpu.module` named `<kernel>_module`, so a module with several
segment functions compiles one kernel per call.

## Private segmented modes

The split pass emits one stage per run: the partial kernel by default and the
merge kernel with `merge`. Both stages admit private capture-free,
single-stage f32 sum/max programs with optional map chains and emit
512-thread kernels whose names carry a `__partial` or `__merge` suffix. Only
the partial stage evaluates the element program.

The GPU pass also accepts `persistent`, which requires `block-size=512` and
emits the experimental persistent queue kernel for the identity f32 sum
described in [Persistent Execution](persistent-execution.md). It cannot be
combined with `fused-mixed`, which requires block size 128.

`use-task-ids` selects the task-ID ABI of the pure warp and pure CTA
kernels. The pass rejects it together with `fused-mixed` and together with
`persistent`, with a diagnostic, because the fused and persistent kernels
have ABIs of their own and always load segment IDs from their own task
buffers.

These modes are registered so that their lowerings can be inspected and
tested from the driver. Registration does not change their status. Split
execution remains private qualification. Persistent execution remains a
private experiment whose predeclared performance gate failed:
[ADR-0018](../adr/ADR-0018-private-persistent-task-queue.md) remains proposed
and no current release status depends on that path. Native runtime code
constructs the same passes through compiler factories instead of pass
arguments.

The driver and passes expose the tested compiler surface, not a general
optimizer pipeline.

## Using the libraries from another CMake project

`cmake --install` of a Swage build tree installs the headers under
`include/swage` and `include/swage-c`, the libraries, and a CMake package
under `lib/cmake/swage`. A consumer loads it with
`find_package(Swage REQUIRED CONFIG)`:

- The imported targets are `MLIRSwage`, `MLIRSwagePlan`,
  `MLIRSwageFixedBlockToGPU`, `MLIRSwageSegmentedReduction`, and
  `SwageCAPI`.
- The targets carry no include directories, as the MLIR targets do not, so
  the consumer adds `SWAGE_INCLUDE_DIRS`, `MLIR_INCLUDE_DIRS`, and
  `LLVM_INCLUDE_DIRS`.
- The package finds MLIR itself and reports Swage as not found when that
  MLIR is not the release Swage was built against.

No workflow builds a consumer project against the installed package, so the
package is an install rule and not a tested interface. Swage itself does not
build in its source directory: configuration stops when the build directory
is the source directory.

## Test tools and lit features

The lit suite needs `swage-opt` from the build tree and `FileCheck`, `not`,
and `count` from the LLVM install. Four tests under
`test/Conversion/SwageToCPU` also execute lowered code. They declare
`REQUIRES: mlir-runner`, and lit reports them as unsupported unless the LLVM
install holds all four of `mlir-opt`, `mlir-runner`, `libmlir_runner_utils`,
and `libmlir_c_runner_utils`. A pinned install built by
`scripts/build_llvm.sh` holds them.

Continue with [Verification](verification.md) for the evidence behind each
boundary. For data flow go back to
[Compiler Pipeline](compiler-pipeline.md), and for admitted segmented
modules and ABIs to [Segmented Reductions](segmented-reductions.md).
