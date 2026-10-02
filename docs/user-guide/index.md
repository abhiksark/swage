<!-- docs/user-guide/index.md -->

# User Guide

The user guide explains how Swage models ragged data and how to use the
public surface. The two meet in one place: two functions run a fixed
reduction or a softmax over segments. The first page describes the segment
storage model, and the second calls those two functions on it. The next two
pages write and launch the one kernel the kernel language accepts, a
fixed-block vector add that uses no segment, because there is no public
segment syntax. The last page returns to the model and states which parts
of it are public.

Status labels are load-bearing everywhere in this documentation. Public
today is supported application surface. Private qualification is tested
contributor machinery, not public API. Planned work has not passed a
public gate.

Runnable snippets in this guide state one of three requirement tiers:

- **wheel-only**: the pure Python `swage` package with no native build, as
  the `swage-compiler` wheel installs it. Enough to import `swage`, capture
  kernels, check a kernel against the kernel language, and run
  `python -m swage.env`. The released `0.5.1` wheel predates the check of
  the kernel body; [Installation](../getting-started/installation.md) lists
  what it lacks.
- **native build**: the build-tree `mlir_swage` package from
  [Installation](../getting-started/installation.md). Enough to emit and
  inspect MLIR without a GPU.
- **CUDA GPU**: the native build, a CUDA-enabled PyTorch build, an admitted
  NVIDIA GPU, and the installed driver. Required to launch a kernel and to
  run a segmented call.

Three committed examples use the native build:
`examples/emit_fixed_vector_add.py` runs at the native-build tier, and
`examples/fixed_vector_add.py` and `examples/segment_reduce.py` run at the
CUDA GPU tier. The [Support Matrix](../reference/support-matrix.md) lists
the versions each tier is tested with.

Read the guide in this order:

1. [Ragged Data](ragged-data.md): the segment storage model and its
   offsets contract. Public today for the two segmented calls.
2. [Segmented Calls](segmented-calls.md): `segment_reduce` and
   `segment_softmax`, their results, their cost, and their limits. Public
   today.
3. [Writing Kernels](writing-kernels.md): capture, the kernel language,
   and compile-only emission of the fixed-block kernel. Public today.
4. [Launching Kernels](launching.md): what happens between `launch()`
   and the GPU for that kernel. Public today.
5. [Execution Model](execution-model.md): segments, tasks, and tiles,
   and which of them each status covers.

Continue with [Ragged Data](ragged-data.md), or jump to the
[API reference](../reference/index.md) for exact contracts.
