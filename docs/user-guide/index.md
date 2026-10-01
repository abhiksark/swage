<!-- docs/user-guide/index.md -->

# User Guide

The user guide explains how Swage models ragged data and how to use the
public surface. Those two do not meet yet: segments are not usable from
Python. The first page describes the segment storage model, which only
private qualification executes today. The next two pages write and launch
the one public kernel, a fixed-block vector add that uses no segment. The
last page returns to the model and states which parts of it are public.

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
- **CUDA GPU**: a CUDA-enabled PyTorch build, an admitted NVIDIA GPU, and
  the installed driver. Required to launch.

Two committed examples use the native build:
`examples/emit_fixed_vector_add.py` runs at the native-build tier, and
`examples/fixed_vector_add.py` runs at the CUDA GPU tier. The
[Support Matrix](../reference/support-matrix.md) lists the versions each
tier is tested with.

Read the guide in this order:

1. [Ragged Data](ragged-data.md): the segment storage model and its
   offsets contract. Private qualification.
2. [Writing Kernels](writing-kernels.md): capture, the kernel language,
   and compile-only emission of the fixed-block kernel. Public today.
3. [Launching Kernels](launching.md): what happens between `launch()`
   and the GPU for that kernel. Public today.
4. [Execution Model](execution-model.md): segments, tasks, and tiles,
   and which of them each status covers.

Continue with [Ragged Data](ragged-data.md), or jump to the
[API reference](../reference/index.md) for exact contracts.
