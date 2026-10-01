<!-- docs/reference/runtime-environment.md -->

# Runtime and Environment

The public runtime executes only canonical fixed vector add. Private
qualification reuses the same CUDA Driver wrapper for admitted segmented
modules. Both paths validate their complete host-visible boundary before
reading pointers, allocating private storage, compiling, or launching.

## Launch lifecycle

The public path requires three contiguous rank-one `torch.float32` CUDA
tensors on the current device. `n` is a nonnegative i32 no larger than any
tensor, `BLOCK` is a positive integer within the active device limit, and
`grid` must equal `(ceildiv(n, BLOCK),)`. Validation also checks the canonical
parameter names and ordered ABI before reading pointers or starting compiler
work.

For `n == 0`, the required grid is `(0,)`, and the validated launch returns
without compilation, cache access, module loading, or enqueue. Other launches
target the active device exactly and admit the NVPTX processors that the
pinned LLVM release defines, currently `sm_80`, `sm_86`, `sm_87`, `sm_88`,
`sm_89`, `sm_90`, `sm_100`, `sm_101`, `sm_103`, `sm_110`, `sm_120`, and
`sm_121`; a device below compute capability 8.0, or one the pinned release
does not define, is rejected during compilation. The runtime then
emits semantic MLIR, lowers and emits PTX in process through LLVM NVPTX, loads
the module through `libcuda.so.1`, and enqueues asynchronously on the current
PyTorch CUDA stream.

Swage does not invoke NVRTC or a subprocess compiler. It does not copy or
cast tensors, change devices, create a CUDA context, synchronize, or select a
fallback backend. Loaded functions are reused per specialization and CUDA
context. Tensor storage remains owned by PyTorch, and submitted tensors are
retained through `record_stream()`.

Emitted kernels also pin their own launch width: the PTX carries a
`.reqntid` directive matching the specialized block size, so a launch
whose block dimension differs from the specialization fails at the driver
instead of running with the wrong geometry.

Validation fails closed before specialization, compilation, or private
allocation. Nonzero work specializes and checks the cache, then compiles and
loads when needed.

<div class="doc-figure" tabindex="0" markdown="1">

![Fail-closed validation, current-stream launch, and tensor retention](../assets/diagrams/runtime-lifecycle.svg)

</div>

*The validated runtime lifecycle, including zero work and stream retention. [Open the full-size figure](../assets/diagrams/runtime-lifecycle.svg).*

Dispatch reaches the driver through a compiled entry point. The nanobind
`_launch_kernel` binding builds the driver argument array without
per-launch ctypes marshaling, resolves `libcuda.so.1` with `dlopen` once
per process, and deliberately holds the GIL across the microsecond
enqueue. When the compiled bindings are absent, a ctypes path submits the
same driver call with the same error shape and a slower per-launch cost.

<div class="doc-figure" tabindex="0" markdown="1">

![The compiled nanobind launch lane beside the ctypes fallback lane](../assets/figures/dispatch-path.svg)

</div>

*The two launch dispatch lanes and when each is taken. [Open the full-size figure](../assets/figures/dispatch-path.svg).*

## Specialization and cache

The specialization key contains normalized source, kernel name, ordered ABI
descriptors, sorted compile-time values, exact compute capability, code
generation options, frontend identity, native compiler identity, dialect
version, and LLVM version.

The two compiler identities are read from the files on disk, not from the
checkout they came from:

- Frontend identity is the SHA-256 digest of every Python source file in the
  installed `swage` package, taken in sorted name order. Only regular files
  count; names that start with a dot, dangling symlinks, and directories are
  skipped.
- Native compiler identity is the file name, size, and modification time of
  the nanobind extension and the `libSwagePythonCAPI` library in
  `mlir_swage/_mlir_libs`.

Editing a frontend file or rebuilding the native libraries therefore changes
the key, and entries written by the earlier compiler are no longer matched.
The native identity reads file metadata, not file contents: a library
replaced by one with the same name, size, and modification time is not
detected.

A process runs the code it loaded, and nothing records which bytes that was.
The persistent cache is therefore used only when the files on disk are known
to be the loaded code, which requires all of the following:

- The native bindings are found and at least one frontend source is found.
- Every frontend source file can be read.
- Every frontend source file and native library was last changed before the
  process started.
- The process start time is available from `/proc/self/stat`.

A file changed after the process started may differ from the code that was
loaded earlier, so the process neither reads nor publishes entries. The check
runs before the lookup and again before an entry is published, where the
files are also compared with the identity in the key. An edit or a rebuild
that lands during a compile is therefore not published under either key. The
checkout state is not part of the check: the cache does not require a git
checkout or a clean working tree.

The start time is read when `swage` is imported. A forked child therefore
inherits the start time of the process that imported the package and is not
judged by the later time of the fork. This covers `os.fork`, the
`multiprocessing` fork start method, and a fork server that preloaded
`swage`: when a file changed after the parent loaded the code, the child
neither reads nor publishes. A spawned child imports the package itself and
uses its own start time.

The check has these limits:

- A file written less than one clock tick (10 ms) before the process started
  counts as changed, because the start time has that resolution.
- A file system whose clock differs from the host clock, or a step of the
  host clock while the process runs, shifts the comparison by that amount.

The cache root is selected in this order:

1. `SWAGE_CACHE_DIR`;
2. `$XDG_CACHE_HOME/swage`;
3. `~/.cache/swage`.

Each persistent entry is a directory named by the key digest that contains
`metadata.json`, `lowered.mlir`, and `kernel.ptx`. Cache directories use
user-only permissions and files use mode `0600`.

Entries are published atomically. A writer stages the three files in a
`.staging-` directory inside the cache root and renames that directory to the
entry name, so a concurrent process sees either no entry or a complete one.
When several processes compile the same key, the first rename wins and the
other processes discard their staged copy and use the published entry. A
writer that is killed leaves only its staging directory, which lookups never
read. Staging directories and old entries are not removed automatically.

Metadata and content digests are verified before module load. An entry
directory that lacks any of the three files is treated as a miss. It is
renamed aside and removed, unless another process completed it in the
meantime, and the key is compiled and published again. Symlinked,
world-writable, unreadable, corrupt, or specialization-mismatched entries,
entries not owned by the current user, and a regular file in place of an
entry directory are rejected with an error. These are treated as evidence of
tampering and always fail the launch.

A cache that cannot be used never fails a launch. When the identity check
does not pass, or the cache root cannot be inspected, created, or written,
the compiled kernel is kept and reused within the process, and one
`RuntimeWarning` per process names the file or directory and the cause. When
only writing fails, entries that are already published are still read. An
incomplete entry that cannot be removed from such a root stays a miss for its
own key and does not affect lookups of other keys.

<div class="doc-figure" tabindex="0" markdown="1">

![Key composition feeding cache verification, where rejected entries raise and a plain miss recompiles](../assets/figures/specialization-key-cache.svg)

</div>

*Specialization key fields and the verify-or-reject cache path. [Open the full-size figure](../assets/figures/specialization-key-cache.svg).*

## Debug dumps

Set either dump switch to the string `1`:

```bash
export SWAGE_DUMP_MLIR=1
export SWAGE_DUMP_PTX=1
export SWAGE_DUMP_DIR=/path/to/output
```

The default dump directory is `swage-dumps` under the current directory.
Dump files are named by specialization digest and receive the same safe,
atomic write treatment as cache artifacts.

## Environment report

```bash
python -m swage.env
```

The command reports the Swage version and checkout revision, Python version,
platform, PyTorch version, the CUDA version used to build PyTorch, actual CUDA
driver version when available, CUDA availability, GPU name and compute
capability, repository LLVM pin when discoverable, the LLVM version the native
bindings were linked against, and backend status. It exits cleanly when
optional components are absent and reports them as unavailable.

With the build-tree bindings on the path, the report looks like this:

```text
swage: 0.5.1
revision: 0123456789ab
python: 3.13.13
platform: Linux-6.8.0-138-generic-x86_64-with-glibc2.35
torch: 2.12.0+cu130
torch_cuda_build: 13.0
cuda_driver: 13.0
cuda: True
gpu: {'name': 'NVIDIA RTX A6000', 'compute_capability': '8.6'}
llvm_pin: llvmorg-22.1.8
llvm_linked: 22.1.8
backends: {'mlir': 'available (linked LLVM 22.1.8)'}
```

Three fields identify the code and the native build:

- `revision` is the abbreviated git HEAD of the checkout that `swage` was
  imported from, with `-dirty` appended when tracked files are modified. It
  is `None` outside a git checkout, for example in a wheel install.
- `llvm_linked` is the LLVM version compiled into the `mlir_swage` extension.
  It comes from the native library, not from `cmake/llvm-version.txt`, and
  is `None` when the bindings cannot be imported or were built before they
  recorded a version.
- `backends` records whether `mlir_swage` imports in the reporting process.
  It reads `available (linked LLVM <version>)` when the import succeeds and
  `unavailable (build-tree mlir_swage bindings not importable)` when it
  fails.

`llvm_pin` is the release tag in `cmake/llvm-version.txt` and `llvm_linked`
is a bare version, so a build against the pinned release shows
`llvmorg-22.1.8` and `22.1.8`. Any other pair means the bindings were built
against a different LLVM install than the repository pins.

Continue with [swage](swage.md) for the public call
surface, [Troubleshooting](../getting-started/troubleshooting.md) for common
boundary failures, or [Verification](../internals/verification.md)
for the tests behind runtime claims.
