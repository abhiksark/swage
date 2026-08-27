<!-- docs/internals/compiler-tools.md -->

# Compiler Tools and Passes

Swage provides one optimizer driver and a small registered pass surface. The
split partial and merge conversions used by private split qualification are
compiler factories, not command-line passes. As a first roundtrip,
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
| `--swage-segmented-reduction-to-gpu` | required positive `block-size`; optional `use-task-ids`; optional `fused-mixed` | Lower an admitted private segmented program to GPU form; fused mixed mode requires block size 128 |
| `--swage-to-plan` | `warp-max-elements`, default 32; `cta-chunk-elements`, default 4096 | Add one private planning companion for the canonical identity segmented sum |

Planning limits must satisfy:

```text
0 < warp-max-elements <= cta-chunk-elements <= INT32_MAX
```

The planning pass preserves the admitted semantic function and adds one
private companion with `swage_plan.classify`. It does not lower a general
task graph or inspect runtime offset contents.

## Private compiler factories

Native runtime code also constructs split partial and split merge lowering
passes directly. These factories admit only the private split identity-sum
shape and emit 512-thread partial or merge kernels. They are intentionally
not registered as `swage-opt` arguments.

The driver and passes expose the tested compiler surface, not a general
optimizer pipeline. Continue with [Compiler Pipeline](../internals/compiler-pipeline.md)
for data flow, [Swage Dialect](swage-dialect.md) for semantic operations, or
[Segmented Reductions](segmented-reductions.md) for admitted
segmented modules and ABIs.
