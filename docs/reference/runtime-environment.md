<!-- docs/reference/runtime-environment.md -->

# Runtime and Environment

The public runtime executes canonical fixed vector add through `launch()`
and two fixed segmented programs through `swage.segment_reduce` and
`swage.segment_softmax`. The segmented calls wrap private qualification,
which reuses the same CUDA Driver wrapper for admitted segmented modules.
Every path validates its complete host-visible boundary before reading
pointers, allocating private storage, compiling, or launching.

## Launch lifecycle

The public path requires three contiguous rank-one `torch.float32` CUDA
tensors on the current device. `n` is a nonnegative i32 no larger than any
tensor, `BLOCK` is a positive integer within the active device limit, and
`grid` must equal `(ceildiv(n, BLOCK),)`. Validation also checks the canonical
parameter names and ordered ABI before reading pointers or starting compiler
work.

The kernel reads and writes tensor storage through raw pointers, so three
more checks apply to the tensors:

- No tensor may be a lazy negation view or a lazy conjugate view
  (`is_neg()` or `is_conj()`). Such a view shares the storage of its base
  and shows different values, so the kernel would compute with the base.
  Pass `tensor.resolve_neg()` or `tensor.resolve_conj()`.
- The output must not share any byte with either input, including an output
  that is the same tensor as an input. The comparison uses the full extent
  of each tensor, not only the first `n` elements. The two inputs may share
  memory with each other. The check cannot see two virtual mappings of one
  physical allocation.
- No tensor may require grad. A launch records no gradient, so a result
  computed from such a tensor would be cut from the autograd graph without
  an error. Pass `tensor.detach()` to launch without gradients.

A launch requires PyTorch 2.6 or newer, the floor that the `pytorch` extra
declares, a `torch.Tensor.record_stream` method, and the
`torch.autograd.graph.increment_version` function. All three are checked
before validation, so an older PyTorch fails with a `RuntimeError` that
names the version found, and no kernel is compiled or enqueued. No newer
release is refused. Compile-only emission does not run this check.

For `n == 0`, the required grid is `(0,)`, and the validated launch returns
without compilation, cache access, module loading, or enqueue. Other launches
target the active device exactly and admit the NVPTX processors that the
pinned LLVM release defines, currently `sm_80`, `sm_86`, `sm_87`, `sm_88`,
`sm_89`, `sm_90`, `sm_100`, `sm_101`, `sm_103`, `sm_110`, `sm_120`, and
`sm_121`; a device below compute capability 8.0, or one the pinned release
does not define, is rejected during compilation. The
[Support Matrix](support-matrix.md) says which of these targets the GPU
tests execute on. The runtime then
emits semantic MLIR, lowers and emits PTX in process through LLVM NVPTX, loads
the module through `libcuda.so.1`, and enqueues asynchronously on the current
PyTorch CUDA stream.

Swage does not invoke NVRTC or a subprocess compiler. It does not copy or
cast tensors, change devices, create a CUDA context, or select a fallback
backend. A thread that has not used CUDA has no current context. A launch
on such a thread makes the context of the validated current device current
there, as the first PyTorch CUDA call on the thread would. That context is
the one PyTorch already holds for the device, so none is created, and a
context that is already current is never replaced. A launch of a kernel
that is already loaded does not synchronize.
A launch that loads a kernel can synchronize the context once;
[Module lifetime](#module-lifetime) states when. Loaded functions are reused
per specialization and CUDA context. Tensor storage remains owned by
PyTorch, and submitted tensors are retained through `record_stream()`.

Autograd sees neither what a kernel reads nor what it stores. Two rules keep
that from giving a wrong gradient without an error:

- A tensor that requires grad is rejected, as stated above.
- After the enqueue, a launch advances the version counter of its output
  through `torch.autograd.graph.increment_version`, as an in-place PyTorch
  operation does. A backward pass that saved the output before the launch
  then raises instead of using the overwritten values. The counters of the
  inputs are not advanced.

The segmented calls and the private qualification helpers apply both rules
to their values and output. The advance changes host metadata only: it
enqueues nothing and does not wait for the device. It has two limits:

- A replayed CUDA graph runs no host code. The launch call that was
  captured advances the counter once, and a replay does not.
- An inference tensor has no version counter, so nothing is advanced for an
  output created under `torch.inference_mode()`.

Emitted kernels also pin their own launch width: the PTX carries a
`.reqntid` directive matching the specialized block size, so a launch
whose block dimension differs from the specialization fails at the driver
instead of running with the wrong geometry.

Validation fails closed before specialization, compilation, or private
allocation. Nonzero work specializes and checks the cache, then compiles and
loads when needed.

A `TypeError` or `ValueError` from a public launch keeps the message of the
check that failed and ends with the kernel and the place it is defined, as
in `grid must equal (2,) for n and BLOCK (in launch of kernel 'add_kernel',
defined at /path/to/kernels.py:12)`. A kernel whose parameter list has a
default value or an annotation other than `sl.constexpr` raises the
source-located `CompilationError` of the frontend before anything is
compiled. The private qualification helpers reject lazy negation and
conjugate views of their values, offsets, output, and task buffers in the
same way as the public launch. They also require the output to share no
memory with the values, the offsets, or caller-supplied task IDs.

A prepared private launch is bound to the storage it was prepared with. The
preparation records the data pointer, the element count, and the dtype of
the values, offsets, and output tensors, and each launch compares them
before anything is enqueued. In-place writes to values and output are fine
and are the way to reuse a prepared launch. A tensor that was rebound to
other storage, for example through `tensor.data = other`, raises a
`RuntimeError` that names the tensor, even when the new storage has the same
size, and the launch must be prepared again. PyTorch does not advance the
version counter for such a rebind, so the offsets check alone does not see
it. The comparison is host work only. It cannot see storage that was freed
and allocated again at the prepared address with the prepared count and
dtype.

A prepared private launch also compares the version counter of the offsets
tensor with the one recorded at preparation and raises a `RuntimeError`
when it moved, because the plan was built from the offsets as they were
then. The comparison is one host attribute read. It has these limits:

- Offsets created under `torch.inference_mode()` have no version counter.
  Preparation rejects them with a `ValueError`. Offsets created outside the
  context, or cloned outside it, are admitted, also when the launch is
  prepared and run inside it. The one-shot helpers `launch_gpu` and
  `launch_softmax_gpu` compare no counter and accept inference tensors.
- A write that PyTorch does not count is not detected: an in-place write
  through `offsets.data`, a write through a DLPack alias, and a write
  through a raw pointer by another library or another kernel. The launch
  proceeds without a diagnostic. The device-side bounds keep every access
  inside the buffers. A segment that the plan runs as one task is reduced
  over the new offsets, and a segment that the plan split keeps the ranges
  recorded at preparation, so the output can mix the old and the new
  layout.
- A write to another view of the same tensor is refused although the
  offsets did not change, because every view of a tensor shares one version
  counter. A launch into such a view counts as a write, since a launch
  advances the version counter of its output. The output of the prepared
  launch itself is exempt: preparation advances the version counter of the
  output once, sees whether the offsets counter moved with it, and allows
  for that at every launch.
- A replayed CUDA graph runs no host check.

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
A process that runs the segmented calls from an artifact without the
bindings has a third lane, which the figure below does not show: the
launcher of the runtime library of the artifact, called through `ctypes`.
It replaces the ctypes path when the artifact serves its first kernel, and
it raises the errors of the compiled launcher in the same words.

<div class="doc-figure" tabindex="0" markdown="1">

![The compiled nanobind launch lane beside the ctypes fallback lane](../assets/figures/dispatch-path.svg)

</div>

*The two launch dispatch lanes and when each is taken. [Open the full-size figure](../assets/figures/dispatch-path.svg).*

### Threads

A launch from several threads follows these rules:

- A launch of a kernel that the process has already compiled and loaded
  takes no lock. It reads the in-process caches and enqueues.
- The enqueue holds the GIL, as stated above.
- Compilation releases the GIL. While the native compiler lowers a kernel
  and emits PTX, other Python threads run.
- One process-wide lock serializes every compile and every load of a kernel
  that `launch()` or a private qualification helper starts. Two compiles
  therefore never run at the same time, and a compile on one thread does
  not delay a launch of a loaded kernel on another.
- The native binding also keeps two compiles off one MLIR context. A caller
  of the binding must not use a context from another thread while a compile
  of a module in that context runs.

### Interpreter exit and fork

The lock that serializes compiles and loads is held for a whole compile,
and the compiler runs without the GIL. Two process events need care:

- Interpreter exit. A daemon thread can be inside the compiler when the
  main thread ends, and a process that finalizes under a compile aborts or
  crashes. An exit handler therefore waits for the compile or load in
  flight and then keeps the lock, so that no other thread starts one. The
  wait is bounded at 5 seconds. A compile takes a few milliseconds, but the
  lock is also held while a load waits for the device, which nothing
  bounds, and a stuck thread must not hold the process open. After the
  bound the exit continues and the crash is possible again. A later exit
  handler that then needs a compile or a load gets a `RuntimeError` at
  once and does not wait again.
- Fork. A child forked while another thread compiles would inherit the lock
  in its taken state and wait for it forever at its first compile or load.
  `os.fork()` therefore waits for the compile or load in flight, and the
  parent and the child both release the lock. A fork taken on a thread
  that is itself compiling does not wait for itself.

These cases stay exposed:

- A compile that calls the native binding directly, without `launch()` or
  a private qualification helper, does not hold the lock. Exit and fork do
  not wait for it.
- `os._exit()`, a fatal signal, and a process that an embedding application
  tears down without running exit handlers skip the wait.
- A child forked from a process that has used CUDA cannot use CUDA. The
  fork handling covers the compiler and the caches, not the driver.

### Module lifetime

The process keeps compiled artifacts and loaded functions in two in-process
caches. Each keeps 128 entries. When one more entry is stored, the cache
forgets the entry it stored first, whether or not that kernel is still in
use. The next launch of a forgotten kernel reads it from the persistent
cache or compiles it, and loads it again. The private qualification helpers,
which the segmented calls run through, keep caches of their own with the
same bound.

A loaded CUDA module stays loaded for as long as something holds its
function handle: a cache entry, a prepared private launch, or a launch in
progress. Once nothing holds the handle, the module is queued for unloading.
The queue is emptied the next time the process loads a kernel in the same
CUDA context:

1. The context is synchronized once, so that launches of the queued modules
   that are still on a stream finish.
2. Each queued module is unloaded.

This is the only synchronization on the public path, and a launch of a
loaded kernel never reaches it. These rules bound it:

- A kernel that a public launch enqueues on a stream that is capturing a
  CUDA graph stays loaded for the rest of the process, because the graph
  can replay it at any time.
- A prepared private launch is not kept that way. The prepared object keeps
  its kernels loaded and must outlive every graph that captured it.
- A module of another CUDA context stays queued until that context is
  current.
- While the calling thread captures a CUDA graph, nothing is unloaded and
  the queue waits for a later load.
- A driver error during the synchronize or an unload is reported as one
  `RuntimeWarning` per load and is never raised, because the caller is
  loading an unrelated kernel. Every queued module is tried, and a module
  that was not unloaded stays queued for the next load, which synchronizes
  and tries again. When the warning filters turn warnings into errors, the
  report is written to standard error instead.

One limitation remains. A capture that is open on another thread cannot be
seen from the loading thread. If a load finds queued modules while another
thread captures a graph in the same context, the synchronize invalidates
that capture and fails, the modules stay queued, and the process warns
that it left `<count>` unused CUDA modules loaded. To avoid it, launch every
kernel once before any thread starts a capture, so that no kernel is loaded
while a capture is open.

## Segmented calls

`swage.segment_reduce` and `swage.segment_softmax` prepare and launch in
one call. [swage](swage.md#swagesegment_reduce) states their arguments, and
[Ragged Data](../user-guide/ragged-data.md#the-offsets-contract) states the
offsets contract. This section states what a call does to the device and
the process.

A call makes its checks in a fixed order, all before the first enqueue:

1. PyTorch: the release floor and the two functions that `launch()`
   requires.
2. Arguments that need no native build: `kind`, the tensor type of `values`
   and `offsets`, the grad state of `values`, the rank of `values` for
   `segment_reduce`, and every rule of `out`.
3. The selected artifact, or the native bindings. With
   `SWAGE_ARTIFACT_DIR` set, the directory is read and verified at the
   first call of the process, the bindings are not needed, and a directory
   that cannot be used stops the call here. Without the variable, a
   wheel-only install stops here with a `RuntimeError` that names the
   installation page.
4. `numpy`, which holds the host copy of the offsets. An install without
   it stops here with a `RuntimeError` that names `numpy` and the
   installation page.
5. CUDA graph capture.
6. The dtype of `values` for `segment_reduce`, which takes float32 and
   float64, and float64 values for `segment_softmax`, which refuses them
   with a reason of its own.
7. The shared validation of dtype, rank, layout, lazy views, the offsets on
   a host copy, and the device.

A result that the call allocates is allocated before step 7, in the dtype
of `values`, so a call that fails there has allocated and released one
tensor.

Preparation and launch follow these rules:

- The host copy of the offsets is made on every call and waits for the
  work already queued on the current stream. A call asks for no other
  synchronization, and it does not wait for the kernels it enqueued. A call
  that loads a kernel can synchronize the context once, as stated under
  [Module lifetime](#module-lifetime).
- No plan is kept between calls. A second call with the same offsets tensor
  copies and classifies it again.
- `segment_reduce` validates, classifies, and enqueues the `mixed` schedule
  of the private planned path in one step, with the default limits and
  automatic schedule selection. For the same batch it returns the bits of
  the prepared private `mixed` launch. `segment_softmax` runs the one-CTA
  path with a 128-thread block. Neither call takes a scheduling argument.
- `segment_reduce` runs the program of the dtype of `values`: float64
  values have kernels of their own, compiled and loaded like the float32
  ones, and nothing is cast. The schedule does not depend on the dtype.
- `segment_reduce` on `[N, D]` values with more than one column takes
  another path. It validates the offsets and enqueues one kernel, the
  column kernel of the kind and dtype, with one 128-thread block per
  segment. It admits no program for planning, classifies no segment,
  uploads no task record, and allocates no scratch. With int64 offsets it
  uploads the narrowed copy, as `segment_softmax` does. `[N, 1]` values
  take the rank-one path through a view, and `[N, 0]` values enqueue
  nothing.
- `segment_softmax` on `[N, D]` values with more than one column enqueues
  the column kernel of the softmax in the same way: one 128-thread block
  per segment, and a thread per column. `[N, 1]` values run the rank-one
  kernel through a view, and `[N, 0]` values enqueue nothing.
- `segment_reduce` prepares nothing it does not launch. A batch compiles
  and loads the fused kernel when it has segments of up to 4096 elements
  and the partial and merge kernels when it has longer ones. A batch that
  the selection rule sends to the pure CTA kernel compiles and loads that
  kernel alone. The pure warp kernel is never compiled.
- Kernels are enqueued on the stream that is current at the call, and
  `values`, `offsets`, and the result are retained through
  `record_stream()`.
- After the enqueue, the version counter of the result is advanced once.
  A call that enqueues nothing, because it is refused or because its batch
  has no segment, leaves the counter where it was. The counters of `values`
  and `offsets` are not advanced.
- The task records and the split scratch of a `segment_reduce` call are
  released when the call returns, and a call creates no CUDA event. Device
  memory and the number of loaded modules stay flat over repeated calls
  with new offsets.
- Compiled kernels are kept in the in-process caches of the private
  helpers, which [Module lifetime](#module-lifetime) describes, and are not
  written to the persistent cache. With an artifact selected, the same
  caches hold kernels that were read from the artifact, and nothing is
  compiled; [Artifacts](#artifacts) states the rules. The first
  `segment_reduce` call on a device whose batch goes to the pure CTA kernel
  also uploads the shared segment ids that
  [Task Execution](../internals/task-execution.md) describes, which hold
  4 MiB of device memory for the life of the process.

A prepared private launch guards the time between its preparation and a
later launch. A segmented call has no such time: it enqueues what it
classified before it returns, and the caller runs nothing in between. Four
guards of a prepared launch are therefore not part of a call:

- The version counter of the offsets is not compared, so `values`,
  `offsets`, and `out` may all be inference tensors, which have no counter.
- The storage of the tensors is not compared. The data pointers are read
  after validation and passed to the driver in the same call.
- The CUDA context is not compared. The kernels are loaded, or found
  loaded, in the context that is current at the call.
- No event orders the task records before the kernels. Both use the stream
  that is current at the call.

Everything else is checked on every call: the shared validation of step 6,
with the offsets on a host copy, and the refusals below. A write to the
offsets by another thread, or by a kernel on another stream, between the
host copy and the enqueue is not detected, and the result is then not a
validated one. The kernels clamp every range they load to the buffer it
indexes, so such a write cannot move an access outside the buffers.

Three conditions are refused with an error instead of being handled:

- A stream that is capturing a CUDA graph. The call raises a `RuntimeError`
  before the host copy, so the capture stays valid. A prepared private
  launch can be captured; a public call cannot, because a replay would not
  repeat its preparation.
- `values` that require grad, with a `ValueError`. A call records no
  gradient.
- `SWAGE_NO_COMPILE=1` with a kernel the process does not hold and no
  artifact selected; [Cache variables](#cache-variables) states the rule.

A fourth refusal exists only with `SWAGE_ARTIFACT_DIR` set: an artifact that
cannot be used, or that does not hold the kernels of the call, raises a
`RuntimeError`. [Artifacts](#artifacts) lists the cases.

A call works on a thread that has not used CUDA: its first PyTorch CUDA
operation makes the context of the current device current there.

Two errors that reach a caller of the public calls use the terms of the
code that raises them: the PyTorch errors speak of a launch, and the refusal
under `SWAGE_NO_COMPILE=1` refers to the private segmented path.

## Specialization and cache

The specialization key contains normalized source, kernel name, ordered ABI
descriptors, sorted compile-time values, exact compute capability, code
generation options, frontend identity, native compiler identity, and
dialect version.

The two compiler identities are read from the files on disk, not from the
checkout they came from:

- Frontend identity is the SHA-256 digest of every Python source file in the
  installed `swage` package, taken in sorted name order. Only regular files
  count; names that start with a dot, dangling symlinks, and directories are
  skipped.
- Native compiler identity is the file name and a content identity of the
  nanobind extension and of the `libSwagePythonCAPI` library in
  `mlir_swage/_mlir_libs`. The content identity is the ELF build id of the
  library, written as `build-id:<hex>`. The linker derives it from the
  linked contents, and the build of `mlir_swage` asks the linker for one. A
  library without a build id is identified by the SHA-256 digest of the
  whole file, written as `sha256:<hex>`. A symbolic link counts as the file
  it leads to.

Editing a frontend file or rebuilding the native libraries therefore changes
the key, and entries written by the earlier compiler are no longer matched.
File sizes and file times are not part of the key:

- The same libraries give the same key after a copy, an archive round trip,
  or an install that sets every file time to one value. A build-tree
  package and a wheel made from that build therefore share entries, since
  stripping a library keeps its build id.
- Libraries with other contents give another key, whatever their sizes and
  file times are.
- The LLVM version is not a separate field. The native identity is derived
  from the libraries that contain LLVM, and the pinned tag in
  `cmake/llvm-version.txt` exists only in a checkout, so a field read from
  it would give an installed package another key than a checkout.

Reading a build id takes about 10 microseconds per library. The digest of a
library without one is computed once per process, at the first launch that
looks up the persistent cache, and again only if the file changes: about
70 ms for the 114 MB compiler library of a stripped `Release` build. Both
were measured on the machine that qualifies the GPU tier.

A build id identifies what the linker produced. A library that is modified
after linking, for example by a binary patch, keeps its build id and is not
detected. Entries written before the key carried a content identity are no
longer matched; they stay in the cache root until the entry bound evicts
them.

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
read.

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

### Cache bound

The cache root holds at most 1024 entries. After a process publishes an
entry, it removes the entries published longest ago until the root is within
the bound. The entry just published always stays. Reading an entry does not
renew it, so the order is the order of publication, taken from the
modification time of each entry directory. An entry of the fixed vector-add
kernel holds about 3 KB of data and occupies 16 KB on a file system with
4 KB blocks, so the default bound is about 16 MB there.

The same pass removes staging directories older than one hour, which only a
killed writer leaves behind.

Only a directory that the current user owns and whose name is an entry digest
or a staging name is removed. Other files and directories under the cache
root are never touched. An entry is renamed aside before it is deleted, so a
concurrent lookup finds a complete entry or a miss. An evicted key is
compiled and published again when it is next needed.

When an old entry cannot be removed, the process stops publishing, warns
once, and keeps reading published entries.

### Cache variables

Four environment variables configure the cache. They are read at each lookup
that the process cache does not answer.

| Variable | Values | Effect |
|---|---|---|
| `SWAGE_CACHE_DIR` | A directory path | Selects the cache root. |
| `SWAGE_CACHE_MAX_ENTRIES` | A positive integer; default `1024` | Sets the most entries the cache root holds after a publish. |
| `SWAGE_CACHE_READ_ONLY` | `1`, or `0` or unset | With `1`, the process reads published entries and never changes the cache root. |
| `SWAGE_NO_COMPILE` | `1`, or `0` or unset | With `1`, a kernel that is not cached raises instead of compiling. |

With `SWAGE_CACHE_READ_ONLY=1` a miss still compiles, and the kernel is kept
and reused within the process. The process does not create the cache root,
publish, remove an incomplete entry, or evict, and it does not warn about
any of these. Unsafe and corrupt entries are still rejected with an error.

With `SWAGE_NO_COMPILE=1` a launch is served from the process cache or from a
published entry. On a miss it raises a `RuntimeError` before the kernel is
emitted, compiled, loaded, or enqueued. The error names the kernel and either
the missing key and the cache root, or the reason the process does not read
the cache. A process in this mode never publishes. It still removes an
incomplete entry that it finds, so add `SWAGE_CACHE_READ_ONLY=1` for a
process that must not change the cache root. To prepare such a process,
launch each kernel once with the same block size, on the same device target,
with the same `swage` sources and native libraries, in a process that may
compile.

`SWAGE_NO_COMPILE=1` also stops the public segmented calls and the private
qualification helpers they wrap. These keep compiled kernels in the process
only and never use the persistent cache, so with the switch set they launch
a kernel that the process already holds and raise the same `RuntimeError`
for any other. A process that starts with the switch set therefore cannot
run a segmented call, except on a batch without segments, which needs no
kernel, and except from an artifact: a kernel that is read from the
directory `SWAGE_ARTIFACT_DIR` selects is not compiled, so the switch does
not refuse it. Host planning admission is not a kernel compile and still
runs, once per program and pair of planning limits. `SWAGE_CACHE_DIR`,
`SWAGE_CACHE_MAX_ENTRIES`, and `SWAGE_CACHE_READ_ONLY` have no effect on
the segmented calls or the private helpers.

A value other than the ones listed is an error, not a default. A mistyped
variable raises a `ValueError` that names it at the first lookup, before
anything is compiled.

## Artifacts

`SWAGE_ARTIFACT_DIR` names a directory that `python -m swage.compile`
wrote. With it set, `swage.segment_reduce` and `swage.segment_softmax` run
from that directory.
[Running Without the Compiler](../user-guide/deployment.md) describes the
command, the files, and the manifest. This section states what the runtime
does.

The variable is read at every segmented call. Unset and empty mean that no
artifact is selected. A directory is read and verified when it is first
named, under the lock that serializes compiles and loads, and is then kept
for the process: later calls read nothing from it, and a change to its
files is not seen. Naming another directory loads that directory. A
directory that fails verification is read again at the next call.

Verification covers the whole directory before anything is used:

- The directory and every file it uses must exist, must be a directory and
  regular files after symbolic links are followed, and must not have the
  group-write or the other-write permission bit. The owner is not compared
  with the current user, and a read-only directory is admitted.
- The manifest must be JSON of format version 2 with every field of the
  expected type. An artifact of format version 1, which an earlier `swage`
  wrote, is refused and has to be written again.
- The block widths the manifest records must be the ones this `swage`
  launches the kernels with, and its subgroup width must divide the CTA
  block into whole subgroups.
- Each kernel file and the runtime library must have the SHA-256 digest the
  manifest states, and the manifest may name only files directly in the
  directory.
- Each kernel must be one that this `swage` requests, with the entry name,
  the block size, and the argument list it launches with, and each program
  must have all of its kernels. A missing kernel is therefore found here,
  not at the first batch that needs it.
- The runtime library must be built for the machine of the host, must have
  interface version 1 in the manifest and when asked, and must load.

A call then uses the artifact in place of the native bindings:

- Each kernel request is answered from the artifact, once per kernel, and
  kept in the in-process cache. The request is refused when the target of
  the current device is not the target of the artifact, when the artifact
  does not hold the program, and when the program text of the artifact has
  another SHA-256 digest than the text this `swage` would compile.
- The planning admission of a reduction is answered from the manifest, for
  the planning limits the manifest records. Native admission does not run.
- Offsets are classified by the runtime library, which admits and refuses
  what the native classifier does, with the same messages.
- The block widths, the subgroup width, and the planning defaults, which
  the runner otherwise reads from the native target description, are
  answered by the artifact from its manifest: the widths its kernels were
  compiled for and the limits its programs were admitted under.
- Nothing is compiled. A private qualification helper that asks for a
  kernel outside the artifact is refused in the same way.

Every refusal is a `RuntimeError` that names the directory, raised before a
kernel is loaded or enqueued. The process never compiles in place of an
artifact it cannot use.

An artifact concerns the segmented calls only. The public `launch()` of the
fixed vector add does not read it and behaves as the sections above state.
When `mlir_swage` is importable beside a selected artifact, the kernels and
the classification still come from the artifact. A driver wrapper that is
created while an artifact is selected takes its launcher from the runtime
library of the artifact and does not import the bindings, so a process that
runs only the segmented calls maps no LLVM or MLIR library whether or not
`mlir_swage` is importable. Three rules bound this:

- The launcher is chosen when the process first uses the driver. Set
  `SWAGE_ARTIFACT_DIR` before that. A driver that was created earlier keeps
  the launcher it took.
- A selected directory that cannot be used leaves the driver on the ctypes
  path. The bindings are not imported in its place.
- A `launch()` of the fixed vector add in such a process imports the
  bindings to compile, which maps the compiler libraries from then on, and
  enqueues through the launcher of the artifact.

## Test and development settings

One variable serves the tests and no public call.

`SWAGE_ORACLE_BUILD_DIR` names the Swage build directory from which the
private CPU oracle takes its tools. The oracle runs `bin/swage-opt` of that
directory and reads its `CMakeCache.txt` to find the LLVM install the build
was configured with, whose `mlir-opt`, `mlir-runner`, and runner libraries
it uses. Without the variable the directory is `build` in the checkout that
`swage` was imported from, which a `swage` installed from a wheel does not
have. The variable is read at every oracle call. A directory that lacks
either file raises a `RuntimeError` that names what is missing.

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

The command reports the Swage version, checkout revision, and package file,
Python version, platform, PyTorch version, the CUDA version used to build
PyTorch, actual CUDA driver version when available, CUDA availability, GPU
name and compute capability, the qualification of the device target,
repository LLVM pin when discoverable, the LLVM version the native bindings
were linked against, the native extension file, backend status, the state
of the persistent cache, and the selected artifact. It exits cleanly when
optional components are absent and reports them as unavailable.

With the bindings importable, here from a build tree, the report looks like
this:

```text
swage: 0.5.1
revision: 0123456789ab
swage_file: /home/user/swage/python/swage/__init__.py
python: 3.13.13
platform: Linux-6.8.0-138-generic-x86_64-with-glibc2.35
torch: 2.12.0+cu130
torch_cuda_build: 13.0
cuda_driver: 13.0
cuda: True
gpu: {'name': 'NVIDIA RTX A6000', 'compute_capability': '8.6'}
target: sm_86 (qualified)
llvm_pin: llvmorg-22.1.8
llvm_linked: 22.1.8
native_version: 0.5.1
native_revision: 0123456789abcdef0123456789abcdef01234567
mlir_swage_file: /home/user/swage/build/python_packages/mlir_swage/_mlir_libs/_swageDialectsNanobind.cpython-313-x86_64-linux-gnu.so
backends: {'mlir': 'available (linked LLVM 22.1.8)'}
cache_dir: /home/user/.cache/swage
cache: active (reads and writes; 12 of at most 1024 entries)
compile_on_miss: allowed
artifact: none (SWAGE_ARTIFACT_DIR is unset)
```

Seven fields identify the code and the native build:

- `revision` is the abbreviated git HEAD of the Swage checkout that `swage`
  was imported from, with `-dirty` appended when tracked files are
  modified. It is reported only when the package is the `python/swage`
  directory of a git checkout that also holds `cmake/llvm-version.txt`.
  Otherwise it is `None`: in a wheel install, and for a copy of the package
  vendored inside another repository, whose HEAD is not a Swage commit.
- `swage_file` is the `__init__.py` that `swage` was imported from. It
  tells two checkouts, or a checkout and a wheel install, apart.
- `llvm_linked` is the LLVM version compiled into the `mlir_swage` extension.
  It comes from the native library, not from `cmake/llvm-version.txt`, and
  is `None` when the bindings cannot be imported or were built before they
  recorded a version.
- `native_version` is the `swage` version the bindings were built for, and
  `native_revision` is the full git commit of the sources they were built
  from. The revision ends in `-dirty` when a tracked file differed from that
  commit, and reads `unknown` when the sources were not a git checkout. Both
  are compiled into the extension. They are `None` when the bindings cannot
  be imported or are refused.
- `mlir_swage_file` is the native extension that was loaded. `mlir_swage` is
  a namespace package with no file of its own, so the extension names the
  build tree or the installed wheel. It is `None` when the bindings cannot
  be imported or are refused.
- `backends` records whether `mlir_swage` imports in the reporting process.
  It reads `available (linked LLVM <version>)` when the import succeeds,
  `unavailable (mlir_swage bindings not importable)` when it fails, and
  `rejected (<reason>)` when the bindings load but were not built for this
  `swage`.

### Frontend and bindings

The `swage` package and the `mlir_swage` bindings are built from one source
tree and must match. The bindings record the `swage` version they were
built for, the source revision of the build, and a digest of the `swage`
sources beside the build: the SHA-256 of one line per Python source of
`python/swage` that gives its SHA-256 and its name. `swage` checks them
once per process, when the bindings are first imported or first used:

- Bindings built for another `swage` version are refused with a
  `RuntimeError` that names both versions and both locations. Emission and
  launch raise it, and an import of the extension fails with it as the
  cause.
- Bindings that record no version are refused in the same way. They come
  from a build that predates this check.
- Outside a git checkout, as for two installed packages, `swage` has no
  revision of its own, because the `swage-compiler` wheel records none. It
  compares its own sources with the digest the bindings recorded instead,
  and refuses bindings that were built beside other `swage` sources. The
  report shows `revision` and `native_revision` side by side.
- In a git checkout, with bindings built from a commit, `swage` compares
  the checkout with that commit. Native sources that differ from it refuse
  the bindings, and so does a commit the checkout does not have. A
  frontend that differs while the native sources do not is expected, since
  the frontend is edited and committed without a native rebuild: `swage`
  warns once with a `RuntimeWarning` and uses the bindings. A change to
  anything else, such as the documentation, is not reported.
- In a git checkout, with bindings built from a modified tree, whose
  revision ends in `-dirty`, or from sources without a revision, there is
  no commit to compare with. A `swage` source that changed since the build
  warns once.
- Bindings that record no digest of the `swage` sources come from a build
  that predates it. They are compared by revision only, and outside a
  checkout not at all.
- A `swage` package from before this check cannot refuse anything. When
  such a package uses bindings that carry the check, the bindings warn once
  that nothing verified the pair.

The check covers the public paths, `emit_mlir()` and `launch()`, and every
import of the extension made after `swage` was imported.

`cuda_driver` is the CUDA version that `libcuda.so.1` reports. It needs no
PyTorch, so a CPU-only PyTorch build beside an installed driver shows
`cuda: False` with a driver version. It is `None` when the library cannot be
loaded.

`target` is the NVPTX processor of the current CUDA device, followed by its
standing. It is `None` when PyTorch sees no CUDA device.

- `(qualified)`: the GPU tests execute on this target. Only `sm_86` is
  qualified, on one NVIDIA RTX A6000.
- `(admitted, not qualified)`: the compiler accepts the target and emits PTX
  for it, and no test has executed that PTX.
- `(not admitted)`: compilation rejects the target.

The [Support Matrix](support-matrix.md) uses the same three terms and adds
the Python, PyTorch, and driver versions that the tests run with.

Three fields describe the persistent cache as the reporting process would use
it. A process that started earlier, or that runs with other variables, can
differ.

- `cache_dir` is the cache root.
- `cache` reads `active (reads and writes; <count> of at most <bound>
  entries)` in the default mode and `active (reads only; <count> entries)`
  with `SWAGE_CACHE_READ_ONLY=1`. It reads `off (<reason>)` when the process
  compiles without the cache, `rejected (<reason>)` when the cache root is
  unsafe and every lookup raises, and `unknown (<error>)` when a cache
  variable has a value that a launch rejects.
- `compile_on_miss` reads `allowed`, or `refused (SWAGE_NO_COMPILE=1)`.

The report only reads. It does not create the cache root, remove anything
from it, or warn.

`artifact` describes the directory that `SWAGE_ARTIFACT_DIR` selects for
the segmented calls of the reporting process:

- `none (SWAGE_ARTIFACT_DIR is unset)` without the variable.
- The directory, followed by its manifest format, its target, the number
  of kernels and the programs they belong to, and the `swage` version and
  source revision that wrote it, when the directory passes verification.
  The report loads the runtime library of the artifact to verify it.
- `rejected (<reason>)` with the error a segmented call would raise.

`llvm_pin` is the release tag in `cmake/llvm-version.txt` and `llvm_linked`
is a bare version, so a build against the pinned release shows
`llvmorg-22.1.8` and `22.1.8`. Any other pair means the bindings were built
against a different LLVM install than the repository pins.

Continue with the [Support Matrix](support-matrix.md) for the environments
these rules are tested in. Use
[Troubleshooting](../getting-started/troubleshooting.md) for common boundary
failures, or [Verification](../internals/verification.md) for the tests
behind runtime claims.
