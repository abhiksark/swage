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
passes.

## Registered Swage passes

| Pass argument | Options | Current admitted purpose |
|---|---|---|
| `--swage-fixed-block-to-gpu` | required positive `block-size` | Lower the canonical fixed vector-add shape to one GPU x-thread per lane |
| `--swage-segmented-reduction-to-scf` | none | Lower an admitted private segmented sum, max, or fused softmax program to sequential SCF and memref operations |
| `--swage-segmented-reduction-to-gpu` | required `block-size` from 1 to 1024 whose warp count, `ceil(block-size / 32)`, is a power of two; optional `use-task-ids`; optional `fused-mixed`; optional `persistent`, requires block size 512 | Lower an admitted private segmented program to GPU form; fused mixed mode requires block size 128 |
| `--swage-to-plan` | `warp-max-elements`, default 32; `cta-chunk-elements`, default 4096 | Add one private planning companion for a capture-free, single-stage f32 sum or max |
| `--swage-split-segmented-reduction-to-gpu` | optional `merge` | Lower an admitted private capture-free, single-stage f32 sum or max to the split partial kernel, or to the split merge kernel when `merge` is set |

Planning limits must satisfy:

```text
0 < warp-max-elements <= cta-chunk-elements <= INT32_MAX
```

The planning pass preserves the admitted semantic function and adds one
private companion with `swage_plan.classify`. It does not lower a general
task graph or inspect runtime offset contents.

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

These modes are registered so that their lowerings can be inspected and
tested from the driver. Registration does not change their status. Split
execution remains private qualification. Persistent execution remains a
private experiment whose predeclared performance gate failed:
[ADR-0018](../adr/ADR-0018-private-persistent-task-queue.md) remains proposed
and no current release status depends on that path. Native runtime code
constructs the same passes through compiler factories instead of pass
arguments.

The driver and passes expose the tested compiler surface, not a general
optimizer pipeline. Continue with [Compiler Pipeline](compiler-pipeline.md)
for data flow, [Swage Dialect](swage-dialect.md) for semantic operations, or
[Segmented Reductions](segmented-reductions.md) for admitted
segmented modules and ABIs.
