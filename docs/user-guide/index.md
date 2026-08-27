<!-- docs/user-guide/index.md -->

# User Guide

The user guide explains how Swage thinks about ragged data and how to use
the supported public surface. It reads in order; each page builds on the
one before it.

Runnable snippets in this guide state one of three requirement tiers:

- **wheel-only**: the published `swage-compiler` package, no native build.
  Enough to import `swage`, capture kernels, and run
  `python -m swage.env`.
- **native build**: the build-tree `mlir_swage` package from
  [Installation](../getting-started/installation.md). Enough to emit and
  inspect MLIR without a GPU.
- **CUDA GPU**: a CUDA-enabled PyTorch build, an admitted NVIDIA GPU, and
  the installed driver. Required to launch.

Status labels are load-bearing everywhere in this documentation. Public
today is supported application surface. Private qualification is tested
contributor machinery, not public API. Planned work has not passed a
public gate.

Read the guide in this order:

1. [Ragged Data](ragged-data.md): the storage model behind everything.
2. [Writing Kernels](writing-kernels.md): capture, the kernel language,
   and compile-only emission.
3. [Launching Kernels](launching.md): what happens between `launch()`
   and the GPU.
4. [Execution Model](execution-model.md): segments, tasks, and tiles,
   and how execution machinery grows from them.

Continue with [Ragged Data](ragged-data.md), or jump to the
[API reference](../reference/index.md) for exact contracts.
