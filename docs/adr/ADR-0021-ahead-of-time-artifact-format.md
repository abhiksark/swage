<!-- docs/adr/ADR-0021-ahead-of-time-artifact-format.md -->
# ADR-0021: Ahead-of-time artifact format

- Status: accepted
- Date: 2026-10-02
- Amended: 2026-10-03, format version 2

## Context

`swage.segment_reduce` and `swage.segment_softmax` compile their kernels in
the process that calls them. That needs the native `mlir_swage` package and
loads LLVM and MLIR into the process. A serving process often cannot carry a
compiler: the image is built elsewhere, the account may not write a cache,
and a compile on the first request is not acceptable.

The persistent cache does not answer this. Its key includes the frontend
sources and the native libraries of the process that reads it, an entry
must be owned by the current user, and the segmented kernels are not in it.

The kernels of the two calls are a closed set for a given `swage`: a fixed
list of programs, each with a fixed list of kernels for one NVPTX processor.
None of them depends on the tensors of a call. They can be compiled once,
on a build host, for a named processor.

## Decision

An artifact is one directory that a build host writes and a serving process
reads. `python -m swage.compile` writes it, and `SWAGE_ARTIFACT_DIR` selects
it. This record fixes format version 2. "Format version 1" at the end
states what the first format held and why it was replaced.

### What an artifact holds

- `manifest.json`, which describes everything else.
- One PTX text file per kernel, named `<program>.<role>.ptx`. A reduction
  has the roles `cta`, `mixed`, `partial`, and `merge`. The softmax has the
  role `cta`.
- `libSwageRuntime.so`, a C library that classifies offsets into task
  records and enqueues one kernel through `libcuda.so.1`. It needs nothing
  from LLVM.

The programs and the roles of each program are the kernel table of the
`swage` that writes the artifact: every program a public call can run, and
for each program the kernels a call of it can launch. The format does not
fix the list of programs. A later `swage` may add a program to its table
without a new format version, and an artifact that was written before does
not hold it. The reductions over f64 values were added that way: each is a
program of its own, named with the suffix `_f64`, with the four roles of a
reduction and with `double` buffers in its argument lists. So were the two
programs of a mean, `segmented_mean` and `segmented_mean_f64`. Their
`merge` kernel takes one buffer more than the merge of the other
reductions, the range records of the partial tasks, which the argument
list of that kernel states in the manifest like any other.

The loader requires every kernel of a listed program, so that a missing
kernel is found when the artifact is loaded and not at the first batch that
needs it.

### The manifest

The manifest is a JSON object with these fields:

| Field | Contents |
|---|---|
| `format_version` | The integer `2`. |
| `swage_version`, `source_revision`, `llvm_version` | What the native build that compiled the kernels recorded about itself. |
| `target` | The NVPTX processor of every kernel, such as `sm_86`. |
| `target_description` | `subgroup_width`, `cta_block_threads`, and `split_block_threads`, the widths of the target description the kernels were compiled for. |
| `planning` | `warp_max_elements` and `cta_chunk_elements`, the limits the reductions were admitted under. |
| `runtime` | The runtime library: `file`, `sha256`, the `machine` it was built for, and the `abi_version` of its C interface. |
| `programs` | A list. Each entry has the `name` of the kernel function and the `sha256` of the program text. A reduction also has `small_element_program`, the answer of planning admission on the build host. |
| `kernels` | A list. Each entry has `program`, `role`, the `entry` name in the PTX, the `block_size` of the launch, `file`, `sha256`, and `arguments`, the launch arguments in order as a `role` and a C `type`. |

[Running Without the Compiler](../user-guide/deployment.md#what-the-directory-holds)
shows one manifest and states each field for a user.

### What the loader verifies

The directory is read once per process, at the first call that would
otherwise need the bindings. Before anything is loaded, the loader requires
all of the following:

- The manifest is a JSON object with `format_version` equal to `2`, and
  every field has its JSON type.
- The two block widths of `target_description` are the ones this `swage`
  launches the kernels with, and the subgroup width divides the CTA block
  into whole subgroups.
- Every program is one this `swage` runs.
- Every kernel belongs to a listed program and has a role of the kernel
  table, at most once.
- Every kernel has the entry name, the block size, and the argument list
  this `swage` launches it with.
- Every kernel of every listed program is present.
- Every kernel file and the runtime library have the SHA-256 digest the
  manifest states.
- The runtime library was built for the machine of the host, states the
  interface version this `swage` calls, and reports the same version once
  it is loaded.

Three more conditions depend on a call and are checked at the call, before
a kernel is loaded or enqueued:

- The `target` is the processor of the current device.
- The digest of the program text this `swage` would compile equals the one
  in the manifest.
- For a reduction, the planning limits of the call equal the ones in the
  manifest.

Every failed check raises a `RuntimeError` that names the directory.
Nothing is compiled in place of an artifact that cannot serve a call: a
process that selected an artifact said it does not compile.

The three version fields are information. The loader does not compare
them, because the program digest already binds an artifact to the program
texts of the `swage` that loads it, and the kernel table binds it to the
launch arguments.

### The trust rule

A process executes what an artifact holds: the PTX on the GPU and the
runtime library in the process. Naming the directory is therefore the trust
decision, as putting a directory on `PYTHONPATH` is. The loader adds these
checks:

- It does not compare the owner of the directory with the current user. An
  artifact is normally written by one account and read by another, and it
  may sit on a read-only file system.
- It refuses a directory, a manifest, a kernel file, or a runtime library
  that has the group-write or the other-write permission bit.
- It follows symbolic links, for the directory and for each file, and
  applies the rule to what a link leads to.
- It takes only plain file names from the manifest, so a manifest cannot
  name a file outside the directory.
- It verifies every digest before it loads anything.

This differs from the rule of the persistent cache, which a process writes
for itself and which therefore requires entries owned by the current user.

The digests detect damage and a partial copy. They do not authenticate an
artifact: the manifest is not signed, so whoever can write the directory or
one of its parents can replace the artifact as a whole. The loader does not
check the parent directories and does not read access control lists.

### How it is written

The command stages the directory beside its destination and renames it into
place, so a failed run leaves nothing and no reader sees part of an
artifact. It creates the directory and its files without write permission
for the group and for other users, whatever the umask is. It refuses an
existing destination, and it refuses to run while `SWAGE_ARTIFACT_DIR` is
set.

### How a process uses it

The artifact stands in for the native bindings of the private runner. It
answers each kernel request from its files, returns the recorded answer of
planning admission, and classifies offsets with its runtime library, which
produces the records of the compiler's classifier. On the same device a call
from an artifact returns the bits of the compiled path.

A driver that is created while an artifact is selected takes its launcher
from the runtime library and does not import the bindings. A process that
runs the two calls from an artifact therefore maps no LLVM or MLIR library,
whether or not `mlir_swage` is importable.

## What is out of scope

- A runtime without Python or PyTorch. PyTorch owns the tensors, the CUDA
  context, and the current stream, and the calls need `numpy`. The runtime
  library has a C interface, and no C code loads an artifact or a CUDA
  module.
- Machine code for the GPU. An artifact holds PTX text, which the CUDA
  driver compiles when a module is loaded.
- Any other kernel. The public `launch()` of the fixed vector add and the
  private helpers are not served from an artifact.
- A lower cost per call. A call from an artifact does the host work of a
  compiled call.
- Authentication. The format has no signature.
- Compatibility across source revisions. An artifact is written again with
  the `swage` that loads it. A later format changes `format_version`, and a
  loader refuses a version it does not read.
- Hosts other than Linux x86-64. The manifest records the machine of the
  runtime library, and the command accepts a library that was built for
  AArch64. No such library has been built or run.

## Consequences

A serving image needs the pure `swage` package, PyTorch, `numpy`, and one
directory per GPU processor. It needs no `mlir_swage` and no LLVM.

The kernel table is duplicated on purpose: the writer compiles from it and
the loader checks a manifest against it. A change to a launch argument list,
a block width, or a program text makes older artifacts unloadable, and the
loader says so before a launch. Tests hold the table to the kernel layouts
of the compiler and to the PTX each kernel declares.

## Format version 1

The first format differed from version 2 in two ways:

- A reduction listed a fifth role, `warp`: the pure warp kernel of the
  private prepared path, which no public call launches. The loader required
  it all the same.
- The manifest recorded no widths. The loader took the block widths from
  its kernel table and the subgroup width from the launch width of the
  `warp` kernel.

Version 2 drops the kernel and records the three widths in
`target_description`. The runner needs the subgroup width to compute the
grid of the fused kernel, which serves one warp task per subgroup of a
block, and no kernel of the version 2 table is launched one subgroup wide.

A loader of version 2 refuses a version 1 artifact, and the error says to
write the artifact again with the `swage` that loads it.

## Evidence

`python/tests/mlir/test_artifact.py` writes artifacts for every admitted
processor, compares each shipped kernel with an independent compile, and on
the RTX A6000 (`sm_86`) runs both calls from an artifact in a process that
maps no LLVM or MLIR library, once with `mlir_swage` unimportable and once
with it importable. `tests/python/test_artifact.py` covers selection, every
refusal above, and the trust rule without the native build.
