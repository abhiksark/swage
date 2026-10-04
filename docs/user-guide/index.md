<!-- docs/user-guide/index.md -->

# User Guide

The user guide explains how Swage models ragged data and how to use the
public surface. The two meet in one place: two functions run a fixed
reduction or a softmax over segments. The first page describes the segment
storage model, the second calls those two functions on it, and the third
serves them from kernels that were compiled ahead of time. The next two
pages write and launch the kernels the kernel language accepts, a fixed-block
vector add or multiply that uses no segment, because there is no public
segment syntax. The last page returns to the model and states which parts
of it are public.

This guide describes the `v0.5.2` native-wheel contract. Its publication and
release qualification are pending, and `v0.5.1` remains the latest released
tag. Read [Installation](../getting-started/installation.md) before using a
version-pinned install command.

Status labels are load-bearing everywhere in this documentation. Public is
the application surface, not a claim that an unreleased artifact is
qualified. Private qualification is tested contributor machinery, not
public API. Planned work has not passed a public gate.

Runnable snippets in this guide state one of three requirement tiers:

- **wheel-only**: the `v0.5.2` native wheel, which includes the private
  `mlir_swage` bindings, with no PyTorch and no GPU. It can import `swage`,
  capture kernels, emit MLIR with an explicit signature, and run
  `python -m swage.env --json`. With `numpy` added, it can also write an
  artifact with `python -m swage.compile`.
- **Native CPU**: that wheel plus PyTorch, for an explicit `backend="cpu"`
  launch on CPU tensors.
- **CUDA GPU**: that wheel, a CUDA-enabled PyTorch build, `numpy`, an
  admitted NVIDIA GPU, and the installed driver, for an explicit
  `backend="cuda"` launch and for the two segmented calls.

A [source build](../getting-started/installation.md#build-from-source) can
supply the native bindings instead of a wheel, through
`PYTHONPATH=build/python_packages`. A frontend-only editable install has no
bindings: it can capture source, check a kernel against the kernel language,
and report the environment, but it cannot emit or launch, and it runs a
segmented call only from an artifact directory. The
[Support Matrix](../reference/support-matrix.md) lists the versions each
tier is tested with, and the qualified and best-effort CUDA targets. CUDA is
the default backend of `launch()`; neither backend falls back to the other.

Four committed examples follow the guide: `examples/emit_fixed_vector_add.py`
runs at the wheel-only tier, `examples/fixed_vector_add.py` and
`examples/fixed_vector_multiply.py` run at the Native CPU or CUDA GPU tier,
and `examples/segment_reduce.py` runs at the CUDA GPU tier.

Read the guide in this order:

1. [Ragged Data](ragged-data.md): the segment storage model and its
   offsets contract. Public today for the two segmented calls.
2. [Segmented Calls](segmented-calls.md): `segment_reduce` and
   `segment_softmax`, their results, their cost, and their limits. Public
   today.
3. [Running Without the Compiler](deployment.md): compiling the kernels of
   those two calls ahead of time and serving the calls from the result.
   Public today.
4. [Writing Kernels](writing-kernels.md): capture, the kernel language,
   and compile-only emission of the fixed-block kernels. Public today.
5. [Launching Kernels](launching.md): what happens between `launch()` and
   the explicitly selected backend for those kernels. Public today.
6. [Execution Model](execution-model.md): segments, tasks, and tiles,
   and which of them each status covers.

Continue with [Ragged Data](ragged-data.md), or jump to the
[API reference](../reference/index.md) for exact contracts.
