<!-- docs/reference/runtime-environment.md -->

# Runtime and Environment

The public runtime executes canonical fixed vector add and multiply through
`launch()`, on the Native CPU or the CUDA backend, and two fixed segmented
programs on CUDA through `swage.segment_reduce` and
`swage.segment_softmax`. The segmented calls wrap private qualification,
which reuses the same CUDA Driver wrapper for admitted segmented modules.
Every path validates its complete host-visible boundary before reading
pointers, allocating private storage, compiling, or launching.

## Dtypes and rounding

Canonical vector addition and multiplication accept contiguous rank-one
tensors with matching input and output dtypes:

| PyTorch dtype | Storage | Format |
|---|---|---|
| `torch.float32` | 32 bits | IEEE binary32; existing arithmetic unchanged |
| `torch.float16` | 16 bits | IEEE binary16 |
| `torch.float8_e4m3fn` | 8 bits | E4M3FN, finite values and NaNs, no infinities |
| `torch.float8_e5m2` | 8 bits | E5M2, including infinities and NaNs |

FP16 and FP8 values are widened inside the compiled kernel, combined in FP32,
then rounded to the output format using round-to-nearest, ties-to-even.
FP8 loads and stores use bytes with scalar software conversion on both
backends; native FP8 arithmetic or a newer GPU than `sm_86` is not required.
No temporary promoted tensors or host-side tensor casts are introduced.

Conversion is not saturating: FP16 and E5M2 overflow produce signed infinity;
E4M3FN overflow produces NaN. Subnormal values and signed zeros follow
the format's arithmetic. NaN payload and sign are not guaranteed.
The FP16 reference is `(x.float() op y.float()).to(x.dtype)`. FP8
qualification uses the same widened operation with a version-independent
format oracle. This matters for E4M3FN because PyTorch 2.13 saturates results
above the 464 overflow midpoint while Swage's documented non-saturating
conversion produces NaN.

Mixed dtypes, `bfloat16`, `float64`, and the FP8 FNUZ formats are rejected,
including for zero-length work. This does not expand the admitted operation
beyond one canonical vector addition or multiplication and does not add
low-precision segmented kernels. Operation chains, floating vector/scalar
arithmetic, broadcasting, and matrix multiplication remain unsupported.

## Launch lifecycle

The public path requires four runtime parameters followed by one constexpr
block parameter. In source order, the first three runtime values are
contiguous rank-one tensors on the selected backend with the same dtype, one
of those in [Dtypes and rounding](#dtypes-and-rounding), and the fourth, `n`,
is a nonnegative i32 no larger than any tensor. `BLOCK` is a positive
integer, and `grid` must equal `(ceildiv(n, BLOCK),)`. Validation checks
this shape before reading pointers or starting compiler work.

`backend="cuda"` is the default. It requires CUDA tensors on the current
device and a `BLOCK` within the limit of that device. `backend="cpu"`
selects the Native CPU backend. It requires CPU tensors and a `BLOCK` of at
most 1024, lowers the same admitted program to a host function, and runs it
through a process-local LLVM JIT before `launch()` returns. The CPU backend
has no CUDA context, stream, graph capture, module load, or persistent
cache. Neither backend falls back to the other. The rest of this section
describes the CUDA backend unless it names the CPU.

The kernel reads and writes tensor storage through raw pointers, so three
more checks apply to the tensors:

- No tensor may be a lazy negation view or a lazy conjugate view
  (`is_neg()` or `is_conj()`). Such a view shares the storage of its base
  and shows different values, so the kernel would compute with the base.
  Pass `tensor.resolve_neg()` or `tensor.resolve_conj()`.
- The output must not partially overlap either input within the first `n`
  elements. An output that starts at the same address as an input, as an
  in-place launch into one of its inputs does, is admitted, because every
  lane reads and writes one element index. The two inputs may share memory
  with each other. The check cannot see two virtual mappings of one
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
context that is already current is never replaced. A launch does not
synchronize the device, the context, or a stream: a module that is no
longer used is unloaded only once events show that its launches completed,
as [Module lifetime](#module-lifetime) states. Loaded functions are reused
per specialization and CUDA context. Tensor storage remains owned by
PyTorch, and submitted tensors are retained through `record_stream()`. A
CPU launch has completed when `launch()` returns.

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

*The CUDA runtime lifecycle, including zero work and stream retention. [Open the full-size figure](../assets/diagrams/runtime-lifecycle.svg).*

CUDA dispatch reaches the driver through a compiled entry point. The
nanobind `_launch_cuda_kernel` binding receives the argument kinds of the
launch contract in order with their values, builds the driver argument
array without per-launch ctypes marshaling, resolves `libcuda.so.1` with
`dlopen` once per process, and deliberately holds the GIL across the
microsecond enqueue. When the compiled bindings are absent, a ctypes path submits the
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
  takes no process-wide lock. It reads the in-process caches and enqueues.
- The enqueue holds the GIL, as stated above.
- Compilation releases the GIL. While the native compiler lowers a kernel
  and emits PTX, other Python threads run.
- Compiles of different kernels run at the same time: each compile parses
  or emits its module in an MLIR context of its own. A second request for a
  kernel that is being compiled waits for that compile and uses its result,
  so each kernel compiles once. When that compile fails, a waiting request
  compiles in its place.
- One process-wide lock, the cold-path lock, serializes the first load of a
  kernel into a CUDA context and the creation of the driver wrapper. A
  compile does not hold it; it holds a place that interpreter exit and
  `os.fork()` wait for.
- The native binding keeps two compiles off one MLIR context. A caller of
  the binding must not use a context from another thread while a compile
  of a module in that context runs.

### Interpreter exit and fork

A compile runs without the GIL, and a first load holds the cold-path lock
while it waits for the driver. Two process events need care:

- Interpreter exit. A daemon thread can be inside the compiler when the
  main thread ends, and a process that finalizes under a compile aborts or
  crashes. An exit handler therefore takes the cold-path lock, waits for
  every compile in flight, and keeps the lock, so that no other thread
  starts a compile or a load. The wait is bounded at 5 seconds. A compile
  takes a few milliseconds, but a load waits for the device, which nothing
  bounds, and a stuck thread must not hold the process open. After the
  bound the exit continues and the crash is possible again. A later exit
  handler that then needs a compile or a load gets a `RuntimeError` at
  once and does not wait again.
- Fork. A child forked while another thread loads would inherit the lock in
  its taken state and wait for it forever at its first compile or load.
  `os.fork()` therefore takes the lock and waits for every compile in
  flight, and the parent and the child both release the lock.

These cases stay exposed:

- A compile that calls the native binding directly, without `launch()` or
  a private qualification helper, holds no place. Exit and fork do not wait
  for it.
- `os._exit()`, a fatal signal, and a process that an embedding application
  tears down without running exit handlers skip the wait.
- A child forked from a process that has used CUDA cannot use CUDA. The
  fork handling covers the compiler and the caches, not the driver.

### Module lifetime

The process keeps compiled kernels and loaded CUDA modules in in-process
caches. The caches of `launch()` and the cache of loaded modules are
least-recently-used caches of 128 entries each; `SWAGE_MEMORY_CACHE_ENTRIES`
sets another positive bound, and any other value raises a `ValueError` at
the first launch that uses a cache. The private qualification helpers,
which the segmented calls run through, load their modules into the same
cache of loaded modules and keep their compiled kernels in a cache of their
own with a fixed bound of 128. A kernel that has left a cache is read again
from the persistent cache or compiled again, and loaded again, at its next
launch.

A launch holds a lease on its loaded module while it uses it, and a
prepared private launch holds leases on its kernels for as long as it
lives. A module that leaves the cache of loaded modules is retired. A
retired module is unloaded only when all of these hold:

- No lease holds it.
- No CUDA graph captured a launch of it. A module launched on a stream that
  is capturing a graph stays loaded for the rest of the process, because
  the graph can replay it at any time.
- An event recorded after its last launch on every stream that launched it
  has completed. A stream other than the legacy default stream is fenced
  right after each launch, because its owner may destroy it; the legacy
  default stream is fenced once the module is idle.

A later lease or launch in the same CUDA context unloads the retired
modules that are idle. It only records and queries events, so it never
waits for the device. Nothing is unloaded while the calling thread, or a
stream that launched the module, captures a graph. A module of another CUDA
context waits until that context is current.

A driver error while a retired module is fenced or unloaded is reported as
one `RuntimeWarning` per call, `Swage left <count> unused CUDA modules
loaded: <error>`, and is never raised, because the caller is working on an
unrelated kernel. That module stays loaded for the rest of the process,
and the other retired modules are still tried. When the warning filters
turn warnings into errors, the report is written to standard error instead.

A prepared private launch that a CUDA graph captured also lends the graph
its task storage, so the prepared object must outlive every graph that
captured it.

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
  synchronization, and it does not wait for the kernels it enqueued.
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
- `segment_reduce` on `[N, D]` values with more than one column runs the
  row-stripe tile of the kind and dtype. It classifies the rows of each
  segment with the default chunk limit divided by the column-group width
  `W` of the feature count, and one direct class. A segment of at most
  `4096 / W` rows is a task of the task-id kernel, with 128-thread blocks,
  and a longer one is cut into chunks for the partial and merge kernels,
  with 512-thread blocks. A batch without a split uploads no record. Each launch has one block per task and group of `W`
  columns. `[N, 1]` values take the rank-one path through a view, and
  `[N, 0]` values enqueue nothing.
- `segment_softmax` on `[N, D]` values with more than one column enqueues
  the task-id kernel of the softmax with one task per segment and one
  128-thread block per segment and group of `W` columns. It classifies
  nothing and uploads no task record. With int64 offsets it uploads the
  narrowed copy. `[N, 1]` values run the rank-one kernel through a view,
  and `[N, 0]` values enqueue nothing.
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
- An `out` while the call records a gradient, with a `ValueError`;
  [Gradients](../user-guide/segmented-calls.md#gradients) states the rule.
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

The specialization key contains normalized source, kernel name, backend,
artifact format, exact target (an NVPTX processor for CUDA, `native` for
the CPU), ordered ABI descriptors including the element dtype, sorted
compile-time values, code generation options, frontend identity, native
compiler identity, and dialect version.

Only the CUDA backend uses the persistent cache described below. A Native
CPU executable is kept in the process only.

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
`metadata.json`, `lowered.mlir`, and `kernel.ptx`. The metadata, version 4,
names the backend, the artifact format, the target, and the launch contract
of the kernel; an entry written with an earlier version is not reused. Cache
directories use user-only permissions and files use mode `0600`.

Entries are published atomically. A writer stages the three files in a
`.staging-` directory inside the cache root and renames that directory to the
entry name, so a concurrent process sees either no entry or a complete one.
When several processes compile the same key, the first rename wins and the
other processes discard their staged copy and use the published entry. A
writer that is killed leaves only its staging directory, which lookups never
read.

Metadata and the digests of the lowered MLIR, the PTX, and the launch
contract are verified before module load. An entry
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
fixed vector add and multiply does not read it and behaves as the sections above state.
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
- A CUDA `launch()` of the fixed vector add or multiply in such a process
  imports the bindings to compile, which maps the compiler libraries from then on, and
  enqueues through the launcher of the artifact.

## Test and development settings

Two variables serve the tests and no public call.

`SWAGE_ORACLE_BUILD_DIR` names the Swage build directory from which the
private CPU oracle takes its tools. The oracle runs `bin/swage-opt` of that
directory and reads its `CMakeCache.txt` to find the LLVM install the build
was configured with, whose `mlir-opt`, `mlir-runner`, and runner libraries
it uses. Without the variable the directory is `build` in the checkout that
`swage` was imported from, which a `swage` installed from a wheel does not
have. The variable is read at every oracle call. A directory that lacks
either file raises a `RuntimeError` that names what is missing.

`SWAGE_REQUIRE_PERSISTENT_CACHE_TEST=1` requires that the persistent cache
is usable in the run that sets it. The test that reuses persistent artifacts
in a second process skips when this process would not read or write the
cache, for example when its compiler files changed after it started; with
the variable set it fails instead and names the reason. The release workflow
sets it while it qualifies the installed wheel on the A6000, so that a
qualification never passes without the cache reuse it records. A dirty
checkout does not count as unusable: it reads and writes the cache like a
clean one.

## Opt-in logging

Swage uses `logging.getLogger("swage.runtime")` with no handlers, global
logging configuration, or default prints. Configure logging in the
application to see lazy DEBUG records:

```python
import logging

logging.basicConfig(level=logging.WARNING)
logging.getLogger("swage.runtime").setLevel(logging.DEBUG)
```

Cache decisions of `launch()` emit `memory-hit`, `persistent-hit`, or
`compile`. A successful CPU launch emits `launch complete`; a CUDA launch
emits `launch enqueued`, not a claim that asynchronous GPU work has
completed. Records contain only the event, backend, target, kernel name,
grid where applicable, and shortened artifact key. They exclude pointers,
tensor contents, source text, PTX, cache paths, and environment secrets.
Logging does not add synchronization.

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
python -m swage.env --json
python -m swage.env --json --check native
python -m swage.env --json --check cpu
python -m swage.env --json --check cuda
```

The command prints the dictionary that `swage.env.report()` returns, schema
version 2. The report never fails: a component that is absent or cannot be
probed is reported with the reason instead of raising, and each probe runs
on its own, so one failure does not hide the other facts. Without `--json`
the command prints one `key: value` line per top-level key, in the order of
the table below. `--json` prints one JSON object with sorted keys on one
line.

Without `--check` the command exits with status 0. With `--check native`,
`--check cpu`, or `--check cuda` it exits with status 0 when
`native.available`, `backends.cpu.available`, or `backends.cuda.available`
is true and with status 1 otherwise, and prints the whole report in both
cases. An unknown check name or another malformed argument exits with
status 2.

With the bindings importable from a build tree, on the qualified device,
`--json` prints a report like this one, formatted here for reading:

```json
{
  "artifact": "none (SWAGE_ARTIFACT_DIR is unset)",
  "backends": {
    "cpu": {"available": true, "reason": null},
    "cuda": {"available": true, "qualified": true, "reason": null, "target": "sm_86"}
  },
  "cache": {
    "compile_on_miss": "allowed",
    "directory": "/home/user/.cache/swage",
    "state": "active (reads and writes; 12 of at most 1024 entries)"
  },
  "cuda": true,
  "cuda_driver": "13.0",
  "gpu": {"compute_capability": "8.6", "name": "NVIDIA RTX A6000"},
  "implementation": "CPython",
  "llvm_pin": "llvmorg-22.1.8",
  "machine": "x86_64",
  "native": {
    "available": true,
    "bindings": {
      "file": "/home/user/swage/build/python_packages/mlir_swage/_mlir_libs/_swageDialectsNanobind.cpython-313-x86_64-linux-gnu.so",
      "llvm_linked": "22.1.8",
      "problem": null,
      "revision": "0123456789abcdef0123456789abcdef01234567",
      "version": "0.5.2"
    },
    "build_type": "Release",
    "error": null,
    "frontend_digest": "<64 hexadecimal digits>",
    "llvm_version": "llvmorg-22.1.8",
    "package_version": "0.5.2",
    "source_clean": true,
    "source_revision": "0123456789abcdef0123456789abcdef01234567"
  },
  "platform": "Linux-6.8.0-138-generic-x86_64-with-glibc2.35",
  "python": "3.13.13",
  "schema_version": 2,
  "source": {
    "file": "/home/user/swage/python/swage/__init__.py",
    "revision": "0123456789ab"
  },
  "swage": "0.5.2",
  "torch": "2.12.0+cu130",
  "torch_cuda_build": "13.0"
}
```

The top-level keys, in the order of the lines without `--json`:

| Key | Value |
|---|---|
| `schema_version` | The integer `2`. |
| `swage` | The version of the `swage` package. |
| `source` | The package file and the revision of its sources, described below. |
| `python` | The interpreter version. |
| `implementation` | The interpreter implementation, such as `CPython`. |
| `machine` | The machine type, such as `x86_64`. |
| `platform` | The platform string of the interpreter. |
| `torch` | The PyTorch version, or `null` without PyTorch. |
| `torch_cuda_build` | The CUDA version PyTorch was built with, or `null`. It is not the driver version. |
| `cuda_driver` | The CUDA version that `libcuda.so.1` reports, or `null` when the library cannot be loaded. |
| `cuda` | Whether PyTorch sees a CUDA device. It is not a verdict on the CUDA backend; `backends.cuda` is. |
| `gpu` | The `name` and `compute_capability` of the current CUDA device, or `null`. |
| `llvm_pin` | The LLVM release of the native build record, or of `cmake/llvm-version.txt` in a checkout without one, or `null`. |
| `native` | The native build record and the loaded bindings, described below. |
| `backends` | The availability of the `cpu` and `cuda` backends, described below. |
| `cache` | The persistent cache as the reporting process would use it, described below. |
| `artifact` | The artifact the segmented calls of the reporting process would run from, described below. |

`cuda_driver` needs no PyTorch, so a CPU-only PyTorch build beside an
installed driver shows `cuda: false` with a driver version.

### Source

`source.file` is the `__init__.py` that `swage` was imported from. It tells
two checkouts, or a checkout and a wheel install, apart.

`source.revision` is the first 12 hexadecimal digits of the source
revision, with `-dirty` appended when the sources differ from it:

- In a Swage checkout, where the package is the `python/swage` directory
  beside `cmake/llvm-version.txt`, it is the git HEAD, and a modified
  tracked file marks it dirty.
- Elsewhere, as in a wheel install, it is the revision of the native build
  record, which is dirty when the build recorded modified sources.
- It is `null` when neither identifies the sources: for a checkout that git
  cannot describe, and for a copy of the package vendored inside another
  repository without a build record, whose HEAD is not a Swage commit.

### Native build and bindings

`native` describes the native package in these fields:

| Field | Value |
|---|---|
| `available` | `true` when the bindings import and pair with this `swage`. |
| `error` | `null`, or the first problem found: the validation error of a build record that is malformed or unreadable, `native-unavailable` when the bindings cannot be imported, `native-probe-failed` when importing them raised another error, or `native-mismatch` when they import and are refused. |
| `package_version` | The `swage` version of the build record. |
| `source_revision` | The 40-digit source revision of the build record, or `null` for sources that were not a git checkout. |
| `source_clean` | Whether the build record states clean sources. |
| `frontend_digest` | The SHA-256 digest of the Python frontend that the build packaged. |
| `llvm_version` | The LLVM release of the build record, such as `llvmorg-22.1.8`. |
| `build_type` | The CMake build type of the build record. |
| `bindings` | The loaded extension, described below, or `null` when the bindings cannot be imported. |

The six build record fields come from `mlir_swage/_build_info.json`, which
the native build writes, and are `null` when the record is absent. A
malformed record leaves them `null` and puts its validation error in
`error`, which can stand beside `available: true`.

`native.bindings` describes the extension that was loaded:

- `version` is the `swage` version the bindings were built for.
- `revision` is the full source revision they were built from. It ends in
  `-dirty` when a tracked file differed from that revision, and reads
  `unknown` when the sources were not a git checkout.
- `llvm_linked` is the LLVM version compiled into the extension. It comes
  from the native library, not from `cmake/llvm-version.txt`.
- `file` is the extension that was loaded. `mlir_swage` is a namespace
  package with no file of its own, so the extension names the build tree or
  the installed wheel.
- `problem` is `null`, or why bindings that import are refused, in the
  words of the error that emission and launch raise.

`llvm_pin` is a release tag and `llvm_linked` is a bare version, so a build
against the pinned release shows `llvmorg-22.1.8` and `22.1.8`. Any other
pair means the bindings were built against a different LLVM install than
the repository pins.

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
- Outside a git checkout, as for an installed wheel, `swage` compares the
  digest of its own sources with the frontend digest the bindings recorded,
  the one `_build_info.json` records as well, and refuses bindings that
  were built beside other `swage` sources. The report shows
  `source.revision` and `native.bindings.revision` side by side.
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

### Backends

`backends.cpu` holds `available` and `reason`; `backends.cuda` holds
`available`, `qualified`, `target`, and `reason`. A `reason` is `null` when
the backend is available and otherwise joins every cause with `; `:

| Cause | `cpu` | `cuda` |
|---|---|---|
| `native-unavailable: install a supported native wheel`, when `native.available` is false | Yes | Yes |
| `pytorch-unavailable: install swage-compiler[pytorch]`, when PyTorch cannot be imported | Yes | Yes |
| `cuda-unavailable: select a CUDA-enabled PyTorch build`, when PyTorch sees no CUDA device | No | Yes |
| `PyTorch probe failed (<exception type>)`, when importing PyTorch or asking it about the device raised, which includes a missing PyTorch | No | Yes |
| `CUDA driver is unavailable` or `CUDA driver probe failed (<exception type>)`, when `libcuda.so.1` gives no version | No | Yes |
| `CUDA target is not admitted by the pinned compiler`, when the device is not one of the admitted NVPTX processors | No | Yes |

`backends.cuda.target` is the NVPTX processor of the current CUDA device,
such as `sm_86`, or `null` when PyTorch sees no CUDA device. The admitted
processors are `sm_80`, `sm_86`, `sm_87`, `sm_88`, `sm_89`, `sm_90`,
`sm_100`, `sm_101`, `sm_103`, `sm_110`, `sm_120`, and `sm_121`.

`backends.cuda.qualified` is `true` only for an `NVIDIA RTX A6000` at
`sm_86`, the hardware that the GPU tests and the release qualification run
on. It describes the device, not whether this build passed a release gate,
and it is independent of `available`: an admitted device that is not
qualified reports `available: true` and `qualified: false`. Neither field
runs a kernel. The [Support Matrix](support-matrix.md) adds the Python,
PyTorch, and driver versions that the tests run with.

### Cache and artifact

`cache` describes the persistent cache as the reporting process would use
it. A process that started earlier, or that runs with other variables, can
differ.

- `directory` is the cache root.
- `state` reads `active (reads and writes; <count> of at most <bound>
  entries)` in the default mode and `active (reads only; <count> entries)`
  with `SWAGE_CACHE_READ_ONLY=1`. It reads `off (<reason>)` when the process
  compiles without the cache and `rejected (<reason>)` when the cache root
  is unsafe and every lookup raises. When a cache variable has a value that
  a launch rejects, it reads `unknown (<error>)`, and `directory` and
  `compile_on_miss` are `null`.
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

Continue with the [Support Matrix](support-matrix.md) for the environments
these rules are tested in. Use
[Troubleshooting](../getting-started/troubleshooting.md) for common boundary
failures, or [Verification](../internals/verification.md) for the tests
behind runtime claims.
