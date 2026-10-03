<!-- docs/reference/benchmarking.md -->

# Benchmarking

Native wheels ship one operational benchmark entry point:

```bash
python -m swage.bench vector-add --output fixed-runtime-slo.json
```

This is the frozen CUDA **float32 vector-add** workload, not a general kernel
benchmark API. The source tree also supports public fixed-vector multiplication
and low-precision execution, but neither is a benchmark selector. Segmented
comparison remains source-only private qualification; this CLI does not expose
segmented launch. No benchmark helper is added to the package exports or public
type stubs.

## Prerequisites

Install a native `swage-compiler` wheel containing the self-contained private
`mlir_swage` compiler, alongside CUDA-enabled PyTorch and an accessible NVIDIA
GPU and driver. Follow [Installation](../getting-started/installation.md) and
the [runtime support matrix](runtime-environment.md#support-matrix). A
frontend-only editable installation is not sufficient. The v0.5.2 source
contract remains pending publication and release qualification.

Run measurements outside the checkout, with `PYTHONPATH` unset, so `swage`
resolves from the installed wheel. The benchmark rejects a nonempty
`PYTHONPATH` or a `swage` package outside the interpreter's site-packages.
For example, from an external working directory:

```bash
env -u PYTHONPATH python -m swage.env --json --check cuda
env -u PYTHONPATH python -m swage.bench vector-add \
  --output fixed-runtime-slo.json
```

Help does not import PyTorch or `mlir_swage` and works without either optional
runtime dependency:

```bash
python -m swage.bench --help
python -m swage.bench vector-add --help
```

## Command and exit behavior

The public syntax is exactly:

```text
python -m swage.bench vector-add --output PATH [--enforce]
```

`--output` is required. There are no workload, sample-count, dtype, backend,
or threshold tuning options. A missing command or output, an unknown command
or option, or a segmented selector is a usage error with exit status **2**.
Help exits **0** without measuring anything.

The command creates the output parent directories and writes a sorted,
indented JSON record to `PATH`, replacing an existing file there. It then
prints a compact, key-sorted JSON summary to stdout containing `output`,
`passed`, and `valid`.

- Without `--enforce`, a valid measurement exits **0** even if a timing gate
  fails; inspect `passed` and the individual gates.
- With `--enforce`, a valid measurement exits **0** only if every frozen gate
  passes. Enforcement requires the device name **exactly NVIDIA RTX A6000**
  and target **exactly `sm_86`**; another admitted NVIDIA target is not
  interchangeable qualification evidence.
- Invalid measurement or correctness failure exits **1**, with or without
  enforcement. Caught execution errors retain the partial record, including
  an `error` object and any measurements already collected. Failed gates
  also retain their raw evidence. This retention does not cover parser
  errors, process termination, or an unwritable output destination.

## Frozen evidence contract

Records retain `schema_version: 1` and `benchmark: "fixed-runtime"`, the fixed
configuration, timestamp, enforcement and hardware-qualification flags,
correctness flags, raw measurements, and gates. Environment and native build
identity are recorded as execution reaches those checks. `qualified` denotes
only the exact A6000/`sm_86` hardware match, not release qualification;
`valid` denotes complete correct evidence, and `passed` denotes the timing
and memory gate result.

The workload and inclusive thresholds remain frozen:

| Measurement | Method | Passing threshold |
|---|---|---|
| Cold compile/load/first synchronized CUDA launch | Five fresh child processes and unique empty caches; `n=129`, `BLOCK=128` | Median <=250 ms; maximum <=400 ms |
| Warm host dispatch | `n=129`, `BLOCK=128`; 200 warmups and 20 batches of 500 launches, synchronized at batch boundaries | Median <=15 microseconds/call; p95 <=20 microseconds/call |
| Large-vector throughput | `n=2^18` and `2^20`, `BLOCK=256`; 25 warmups and 100 rotating interleaved CUDA-event samples of 32 launches, against `torch.add(out=...)` | Swage/PyTorch median ratio <=1.50 at each size |
| Native compiler memory | Linux `/proc/self/status` RSS increase from PyTorch/tensor setup to first compile/launch | Maximum <=512 MiB |

Correctness preflights precede every timing section. Cold children are
re-entered through this same installed module. Compilation, loading, and
first launch remain one cold measurement; there is no invented planning
field or segmented evidence in this schema.

A passing benchmark is one item of evidence, **not independent release
qualification**. The trusted release workflow must still qualify the actual
repaired installed artifact, its source/build identity, CPU and CUDA
correctness, persistent cache reuse, reproducibility, security, and publication
gates. Historical addition results do not qualify multiplication or
low-precision performance. See [Verification](../internals/verification.md)
for the complete release boundary and retained evidence.
