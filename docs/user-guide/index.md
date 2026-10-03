<!-- docs/user-guide/index.md -->

# User Guide

The user guide explains how Swage thinks about ragged data and how to use
the supported public surface. It reads in order; each page builds on the
one before it.

This guide describes the v0.5.2 native-wheel contract; publication and release
qualification are pending, and v0.5.1 remains the latest released tag. See
[Installation](../getting-started/installation.md) before using version-pinned
install commands.

Runnable snippets state their requirements:

- **wheel-only**: a v0.5.2 native wheel, including private `mlir_swage`,
  needs no local compiler build. It can import `swage`, capture kernels,
  emit MLIR with an explicit signature, and run `python -m swage.env --json`
  without PyTorch or a GPU.
- **Native CPU**: that wheel plus PyTorch CPU tensors for explicit
  `backend="cpu"` launch.
- **CUDA GPU**: that wheel, a CUDA-enabled PyTorch build, an admitted NVIDIA
  GPU, and the installed driver for explicit `backend="cuda"` launch.

A [source build](../getting-started/installation.md#build-from-source) can
supply native bindings instead of a wheel. A frontend-only editable install
can capture source and report the environment, but cannot emit or launch
without those bindings. The authoritative
[support matrix](../reference/runtime-environment.md#support-matrix) defines
supported ABIs and qualified versus best-effort CUDA targets. CUDA is the
default selection; neither backend silently falls back.

Status labels are load-bearing everywhere. Public is the application surface,
not a claim that an unreleased artifact is qualified. Private qualification
is tested contributor machinery, not public API, and its Python modules are
excluded from native wheels. Planned work has not passed a public gate.

Read the guide in this order:

1. [Ragged Data](ragged-data.md): the storage model behind everything.
2. [Writing Kernels](writing-kernels.md): capture, the kernel language,
   and compile-only emission.
3. [Launching Kernels](launching.md): what happens between `launch()`
   and the explicitly selected backend.
4. [Execution Model](execution-model.md): segments, tasks, and tiles,
   and how execution machinery grows from them.

Continue with [Ragged Data](ragged-data.md), or jump to the
[API reference](../reference/index.md) for exact contracts.
