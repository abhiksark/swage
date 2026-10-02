<!-- docs/user-guide/deployment.md -->

# Running Without the Compiler

A process normally compiles the kernels of `swage.segment_reduce` and
`swage.segment_softmax` when it first needs them. That requires the native
`mlir_swage` package and loads LLVM into the process. This page describes
the other way to run the two calls: a build host compiles the kernels ahead
of time into an artifact directory, and the process that serves the calls
loads that directory. The serving process needs neither `mlir_swage` nor
LLVM.

Like the two calls, this is newer than the released `0.5.1` wheel.

## What this delivers and what it does not

It delivers three things:

- A command, `python -m swage.compile`, that compiles every kernel the two
  calls can launch for one named NVPTX processor and writes them to a new
  directory with a manifest. It needs the native build. It needs no GPU and
  no PyTorch, and the target does not have to be the processor of the build
  host.
- A loader in the pure Python package. With `SWAGE_ARTIFACT_DIR` set, the
  two calls take their kernels from the directory, compile nothing, and do
  not import `mlir_swage`.
- The same results. On the same device, a call from an artifact returns the
  bits that the compiled path returns for the same inputs.

It does not deliver these:

- A runtime without Python or PyTorch. PyTorch still owns the tensors, the
  CUDA context, and the current stream, and the calls still need `numpy`.
  The runtime library that an artifact carries has a C interface for
  classifying offsets and enqueuing a kernel, but no C code loads an
  artifact or a CUDA module, and nothing calls that interface except the
  Python loader.
- Machine code for the GPU. An artifact holds PTX text, and the CUDA driver
  compiles PTX when a module is loaded, as it does on the compiled path.
- Any other kernel. An artifact holds the fixed programs of the two
  calls. The public `launch()` of the fixed vector add is not served from an
  artifact, and neither is any private helper.
- A lower cost per call. A call from an artifact does the host work of a
  compiled call: it copies the offsets to the host and classifies them.
- A host of another architecture. The runtime library has been built and
  run on Linux x86-64 only; [The runtime library](#the-runtime-library)
  states what another machine would need.

## Write an artifact on the build host

Run the command where the native package is importable, for the processor
of the device that will run the kernels:

```bash
PYTHONPATH=build/python_packages \
    python -m swage.compile --target sm_86 --output /srv/swage/sm_86
```

```text
artifact: /srv/swage/sm_86
format_version: 2
target: sm_86
programs: segmented_sum, segmented_max, segmented_min, segmented_mean, segmented_sum_f64, segmented_max_f64, segmented_min_f64, segmented_mean_f64, ragged_softmax
kernels: 33
runtime: libSwageRuntime.so (x86_64)
manifest_sha256: <64 hexadecimal digits>
```

The command takes these options:

| Option | Meaning |
|---|---|
| `--target` | The NVPTX processor to compile for, one of the processors [Runtime and Environment](../reference/runtime-environment.md#launch-lifecycle) lists. Required. |
| `--output` | The directory to create. It must not exist, and its parent must. Required. |
| `--program` | `sum`, `max`, `min`, `mean`, `sum_f64`, `max_f64`, `min_f64`, `mean_f64`, or `softmax`. A kind alone names the reduction over float32 values, and the kind with `_f64` the one over float64 values. Repeat it to include several. Without it, all nine are included. |
| `--runtime-library` | A `libSwageRuntime.so` to ship in place of the one of the native build. See [The runtime library](#the-runtime-library). |

These rules hold for every run:

- The target must be the processor of the serving device. On that host,
  `python -m swage.env` prints it as `target`. One artifact holds the
  kernels of one target.
- The directory is written once. The command stages it beside `--output`
  and renames it into place, so a failed run leaves nothing behind and no
  reader sees part of an artifact.
- The directory and its files are created without write permission for the
  group and for other users, whatever the umask is, because the loader
  requires that.
- A target that the compiler rejects ends the command with the diagnostic
  of the compiler and exit status 1.
- The command refuses to run while `SWAGE_ARTIFACT_DIR` is set, and with
  `SWAGE_NO_COMPILE=1` it stops at the first kernel it would compile.

## Run from an artifact on the serving host

The serving environment holds the pure `swage` package of the same source
revision as the build host, PyTorch 2.6 or newer with CUDA, and `numpy`. It
does not need `mlir_swage`. Copy the directory with a tool that keeps the
permission bits, and select it:

```bash
export SWAGE_ARTIFACT_DIR=/srv/swage/sm_86
python -m swage.env
```

The last line of the report describes the selected artifact:

```text
artifact: /srv/swage/sm_86 (format 2, target sm_86, 33 kernels of segmented_sum, segmented_max, segmented_min, segmented_mean, segmented_sum_f64, segmented_max_f64, segmented_min_f64, segmented_mean_f64, ragged_softmax, written by swage 0.5.1 at revision <revision>)
```

It reads `none (SWAGE_ARTIFACT_DIR is unset)` without the variable, and
`rejected (<reason>)` for a directory that a call would refuse.

With an artifact selected, a call behaves as
[Segmented Calls](segmented-calls.md) describes, with these differences:

- The directory is read and verified at the first call of the process, at
  the point where a call would otherwise require the native bindings: after
  the argument checks and before the result is allocated or the offsets are
  copied. Later calls use what the process holds, so a change to the
  directory is not seen until the process starts again.
- Every kernel comes from the artifact. Nothing is compiled, so the calls
  also run with `SWAGE_NO_COMPILE=1`.
- The offsets are classified by the runtime library of the artifact, which
  produces the task records that the compiler's classifier produces.
- The variable is read at every call. Naming another directory loads that
  artifact, and unsetting it returns the process to compiling.

If `mlir_swage` is importable in a process that has an artifact selected,
the kernels and the classification still come from the artifact, and so
does the launcher: the CUDA driver wrapper takes it from the runtime
library of the artifact and does not import `mlir_swage`. A process that
runs only the two calls therefore loads no LLVM whether or not `mlir_swage`
is installed, provided `SWAGE_ARTIFACT_DIR` is set before the process first
uses the driver. A `launch()` of the fixed vector add in the same process
still compiles, so it imports `mlir_swage` and loads LLVM at that point.

## What the directory holds

An artifact for all nine programs holds thirty-five files:

| File | Contents |
|---|---|
| `manifest.json` | What the artifact is and how each kernel is launched |
| `<program>.<role>.ptx` | One kernel: four roles for each of the eight reductions, which are `segmented_sum`, `segmented_max`, `segmented_min`, and `segmented_mean` over float32 values and the same four names with `_f64` over float64 values, and one for `ragged_softmax` |
| `libSwageRuntime.so` | The runtime library: the task classifier and a launcher |

The roles of a reduction are `cta` for the pure CTA schedule, `mixed` for
the fused kernel, and `partial` and `merge` for split segments. The `merge`
kernel of a mean takes one buffer more than that of the other reductions:
the range records of the partial tasks, from which it reads the length of
each split segment. The softmax
has one `cta` kernel. These are the kernels a call can launch, and the
loader requires each of them for every program the artifact lists.

The manifest is JSON. This one is shortened to the first kernel:

```json
{
  "format_version": 2,
  "swage_version": "0.5.1",
  "source_revision": "0123456789abcdef0123456789abcdef01234567",
  "llvm_version": "22.1.8",
  "target": "sm_86",
  "target_description": {
    "subgroup_width": 32,
    "cta_block_threads": 128,
    "split_block_threads": 512
  },
  "planning": {
    "warp_max_elements": 32,
    "cta_chunk_elements": 4096
  },
  "runtime": {
    "file": "libSwageRuntime.so",
    "sha256": "<64 hexadecimal digits>",
    "machine": "x86_64",
    "abi_version": 1
  },
  "programs": [
    {
      "name": "segmented_sum",
      "sha256": "<64 hexadecimal digits>",
      "small_element_program": true
    }
  ],
  "kernels": [
    {
      "program": "segmented_sum",
      "role": "cta",
      "entry": "segmented_sum",
      "block_size": 128,
      "file": "segmented_sum.cta.ptx",
      "sha256": "<64 hexadecimal digits>",
      "arguments": [
        {"role": "values", "type": "const float*"},
        {"role": "offsets", "type": "const int32_t*"},
        {"role": "output", "type": "float*"},
        {"role": "task_ids", "type": "const int32_t*"},
        {"role": "value_count", "type": "int32_t"},
        {"role": "task_count", "type": "int32_t"},
        {"role": "segment_count", "type": "int32_t"}
      ]
    }
  ]
}
```

The fields mean the following:

| Field | Meaning |
|---|---|
| `format_version` | The version of this layout. The loader reads version 2 and refuses any other, also version 1, which an earlier `swage` wrote: write such an artifact again. |
| `swage_version`, `source_revision`, `llvm_version` | What the native build that compiled the kernels recorded about itself: the `swage` version, the source revision, which ends in `-dirty` for a modified tree, and the LLVM release it links. They are information; the loader does not compare them. |
| `target` | The NVPTX processor of every kernel. |
| `target_description` | The widths the kernels were compiled for: the threads of one subgroup, of a block of the `cta` and `mixed` kernels, and of a block of the `partial` and `merge` kernels. The loader requires the two block widths this `swage` launches with. |
| `planning` | The limits the reductions were admitted under: the longest segment of warp work and the longest range of one CTA task. They are the limits `segment_reduce` plans with. |
| `runtime` | The runtime library: its file, its SHA-256 digest, the machine it was built for, and the version of its C interface. |
| `programs` | Each program by the name of its kernel function, with the SHA-256 digest of the program text it was compiled from. A reduction also records what the planning admission of the build host returned, which the schedule selection of a call reads. |
| `kernels` | Each kernel: its program and role, the entry name in the PTX, the threads per block it must be launched with, its file and the SHA-256 digest of that file, and its launch arguments in order. |

An argument has a role and a C type. A pointer type is a device pointer,
`const` marks a buffer the kernel only reads, and `int32_t` is a value
passed by value. The values, the results, and the partial results of a
float64 program are `double` buffers; its offsets and task records are
`int32_t`, as for float32.
[Task Execution](../internals/task-execution.md) and
[Split Execution](../internals/split-execution.md) state what each
argument holds. The manifest describes kernels that only the two calls
launch; it makes none of those internal contracts public.

## Refusals

Each condition below raises a `RuntimeError` that names the directory, and
nothing is compiled in its place. The first group is found when the
artifact is loaded, at the first call:

- The directory does not exist, is not a directory, or has no
  `manifest.json`.
- The manifest is not valid JSON, lacks a field, or has another
  `format_version`.
- A file the manifest lists is missing, or its SHA-256 digest differs from
  the manifest.
- A program lacks one of its kernels, or a kernel has another entry name,
  block size, or argument list than this `swage` launches it with.
- The runtime library was built for another machine, has another interface
  version, or cannot be loaded.
- The directory or one of its files breaks the
  [trust rule](#trust).

The second group is found at a call, before a kernel is loaded or enqueued:

- The artifact was written for another target than the current device.
- The artifact does not hold the program of the call, for example an
  artifact written with `--program sum` under a call for a maximum or for a
  sum of float64 values.
- The program text of the artifact differs from the one this `swage` runs,
  which happens when the artifact and the package come from different
  source revisions. Write the artifact again with the `swage` that loads
  it.

The private qualification helpers also compile nothing while an artifact is
selected: a helper that asks for a kernel outside the artifact is refused.

## Trust

A process executes what an artifact holds: the PTX on the GPU and the
runtime library in the process. Naming the directory is therefore the trust
decision, as putting a directory on `PYTHONPATH` is. The loader adds these
checks:

- It does not compare the owner of the directory with the current user. An
  artifact is normally written by one account and read by another, and it
  may sit on a read-only file system.
- It refuses a directory, a manifest, a kernel file, or a runtime library
  that has the group-write or the other-write permission bit. A copy made
  under a umask of `002` has that bit; `chmod -R go-w` removes it.
- It follows symbolic links, for the directory and for each file, and
  applies the rule to what a link leads to.
- It takes only plain file names from the manifest, so a manifest cannot
  name a file outside the directory.
- It verifies every digest before it loads anything.

The digests detect damage and a partial copy. They do not authenticate an
artifact, because the manifest is not signed: whoever can write the
directory can replace all of it. The loader does not check the parent
directories and does not read access control lists.
[SECURITY.md](https://github.com/abhiksark/swage/blob/main/SECURITY.md)
states the rule with the rest of the threat model.

## The runtime library

`libSwageRuntime.so` is a small C library that links against the C library
only. It holds a second implementation of the compiler's task classifier
and one function that enqueues a kernel through `libcuda.so.1`, which it
loads at the first launch. The loader calls it through `ctypes`, so one
library serves every Python version. The native build places the library
in the `mlir_swage` package, and `python -m swage.compile` copies it from
there. Its interface is `include/swage-c/Runtime.h`.

The library is machine code for the host, not for the GPU, so an artifact
runs only on hosts of the machine its library was built for. The manifest
records that machine, and the loader refuses the artifact on another one.
Only an x86-64 library has been built and run.

For a serving host of another machine, such as an AArch64 host with an
`sm_87` device, two steps are needed, and neither has been tried:

1. Compile `lib/Runtime/SwageRuntime.c`, which needs a C11 compiler for
   that machine and nothing from LLVM, into a shared library. It includes
   only `include/swage-c/Runtime.h`.
2. Pass that library to the command with `--runtime-library`. The command
   reads the machine from the ELF header of the library, records it in the
   manifest, and ships the file. It names `x86_64` and `aarch64` and
   refuses a library of any other machine.

## Evidence

- `python/tests/mlir/test_artifact.py` writes artifacts with the command,
  for every admitted processor and without a device, and compares each
  shipped kernel with an independent compile and each manifest entry with
  the PTX it describes. On the RTX A6000 (`sm_86`) it runs both calls from
  an artifact in a process that cannot import `mlir_swage`, checks that the
  process maps no LLVM or MLIR library, compares the results with
  `torch.segment_reduce`, `torch.softmax`, and float64 references, and
  requires the bits of the compiled path. The cases include the mean and
  the float64 reductions, whose sums are compared with exactly rounded
  ones. A second process, in which `mlir_swage` is importable, runs both
  calls from the artifact with no LLVM or MLIR library mapped and then
  launches the fixed vector add.
- `tests/python/test_artifact.py` covers selection, every refusal, and the
  trust rule, without the native build.
- `unittests/RuntimeTest.cpp` and
  `python/tests/mlir/test_segmented_classification.py` hold the classifier
  of the runtime library to the compiler's classifier, on seeded layouts
  and on every refusal.

These tests ran on the branch that added the feature. The trusted GPU
workflow has not executed them. The [Support Matrix](../reference/support-matrix.md#artifacts)
lists what is tested, what was checked by hand, and what is unknown.

Continue with [Writing Kernels](writing-kernels.md). That page turns to the
kernel language, which has no segment syntax: the one kernel it accepts is
a fixed-block vector add. For the order of the checks a call makes on an
artifact, see
[Runtime and Environment](../reference/runtime-environment.md#artifacts).
