<!-- docs/reference/runtime-environment.md -->

# Runtime and Environment

The public runtime executes canonical fixed vector add or multiply on an
explicitly selected CUDA or Native CPU backend. The private segmented runtime remains a
CUDA qualification surface; its sequential CPU lowering is a correctness
oracle, not the public Native CPU backend. Every path validates its complete
host-visible boundary before reading pointers, allocating private storage,
compiling, or launching.

This reference specifies the implemented **v0.5.2** fixed-vector contract,
whose publication and production qualification are pending. The latest
released tag remains v0.5.1. Packaging, diagnostics, errors, and typing do not
change fixed-kernel semantics or make private segmented execution public.

## Support matrix

| Area | v0.5.2 release contract |
|---|---|
| Platform | Linux x86-64, glibc **2.28 or newer**; `manylinux_2_28_x86_64` wheels. No musllinux, macOS, Windows, or aarch64 artifact. |
| Python | Regular-GIL CPython **3.10, 3.11, 3.12, 3.13** (`>=3.10,<3.14`); one native wheel per ABI. No PyPy or free-threaded ABI. |
| Native package | Public `swage` plus self-contained private `mlir_swage`, both native extensions, runtime shared libraries, build metadata, stubs, `py.typed`, MIT and LLVM licenses. No external `mlir` dependency, local LLVM tree, or `PYTHONPATH` needed. Private `swage/_segmented*.py` modules and bytecode are excluded. |
| PyTorch | Optional `pytorch` extra, **`torch>=2.6,<3`**, never bundled. Required for metadata inference and either launch backend; not for import or explicit-signature MLIR emission. CPU artifact gates use **2.6.0+cpu** on all four ABIs; current native CI also exercises **2.13.0+cpu** on CPython 3.13. |
| CPU | Explicit `backend="cpu"`, CPU tensors, and the native LLVM host JIT on every published ABI; synchronous and no CUDA prerequisite. |
| CUDA | Explicit `backend="cuda"` (also the API default), CUDA-enabled PyTorch, driver `libcuda.so.1`, and an admitted current device. No CUDA toolkit compiler at runtime. |
| CUDA release qualification | The required trusted installed-wheel gate is **NVIDIA RTX A6000 / `sm_86`**. Other admitted targets are best-effort and unqualified. Implementation or a health-check result alone is not passing release evidence. |
| Qualified-driver floor | For A6000/`sm_86`, the greater of **NVIDIA R455** (PTX 7.1) and the installed CUDA-enabled PyTorch build's documented minimum driver. |
| Source builds | Exact **LLVM/MLIR 22.1.8** (`llvmorg-22.1.8`); CMake rejects mismatches. Normal checkout builds retain build-tree bindings. Wheel builds from checkouts or extracted sdists require Release mode and explicit source provenance. |

Target admission is broader than release qualification: `sm_80`, `sm_86`,
`sm_87`, `sm_88`, `sm_89`, `sm_90`, `sm_100`, `sm_101`, `sm_103`, `sm_110`,
`sm_120`, and `sm_121` are admitted by the pinned compiler. Even another
`sm_86` model lacks the A6000 trusted hardware gate. No other target inherits
that qualification or its driver floor; newer targets may require newer
drivers. The broader segmented compiler remains experimental.

Use [Installation](../getting-started/installation.md) for wheel-first,
source-build, and downloaded-wheel checksum/attestation verification commands.
Only the latest released `0.x` line is supported, with best-effort general
support rather than an uptime or response-time SLA. Public incompatible
changes require one published minor release of deprecation unless security or
correctness requires immediate removal.

Kernel source and native compiler artifacts are trusted input; Swage is
**not a sandbox**. Bundled LLVM and cache validation do not make
attacker-controlled compilation safe. See the
[security policy](https://github.com/abhiksark/swage/blob/main/SECURITY.md).

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
contiguous rank-one tensors on the selected backend with the same supported
floating dtype, and the fourth is a nonnegative i32 no larger than any tensor.
The constexpr block is positive, and `grid` must equal its one-dimensional
ceiling division.
Parameter names are diagnostic labels; source order and the compiler-produced
contract define binding.

`backend="cuda"` is the default. It requires CUDA tensors on the current
device and a block within that device's limit. Nonzero launches target the
active device exactly and admit only the processors listed in the support
matrix. Unsupported targets are rejected during compilation. Admission does
not imply release qualification. The runtime emits PTX in process, loads it
through `libcuda.so.1`, and enqueues asynchronously on the current PyTorch
CUDA stream.

`backend="cpu"` is the only Native CPU selection. It requires CPU tensors and
a block no larger than 1024. Nonzero launches use the literal specialization
target `native`, lower the same admitted semantic program to LLVM dialect,
and synchronously invoke an eagerly initialized process-local LLVM JIT entry.
The CPU branch has no CUDA context, stream, event, graph-capture, module-load,
or persistent-cache behavior.

For a zero element count, the required grid is `(0,)`, and the fully validated
launch returns without compilation, cache access, executable loading, or
invocation. Invalid backend names and tensor/backend mismatches fail before
pointer reads or compilation. Swage never changes devices, copies or casts
tensors, creates a CUDA context, synchronizes CUDA work, or falls back to the
other backend.

CUDA loaded functions are reused per specialization and context. Their tensor
storage remains owned by PyTorch and is retained through `record_stream()`.
CPU execution is complete before `launch()` returns.

<div class="doc-figure" tabindex="0" markdown="1">

![Fail-closed validation, current-stream launch, and tensor retention](../assets/diagrams/runtime-lifecycle.svg)

</div>

*The CUDA runtime lifecycle, including zero work and stream retention.
[Open the full-size figure](../assets/diagrams/runtime-lifecycle.svg).*

CUDA dispatch reaches the driver through one ordered typed entry point. The
nanobind `_launch_cuda_kernel` binding receives the version-2 contract's exact
kind sequence and full three-axis grid/block geometry, creates type-correct
backing storage in that order, and builds CUDA's internal `kernelParams`
array. It resolves `libcuda.so.1` once per process and deliberately holds the
GIL across the microsecond enqueue. When the compiled bindings are absent, a
ctypes lane submits the same raw values and geometry with the same error
shape.

<div class="doc-figure" tabindex="0" markdown="1">

![The compiled nanobind launch lane beside the ctypes fallback lane](../assets/figures/dispatch-path.svg)

</div>

*The two launch dispatch lanes and when each is taken. [Open the full-size figure](../assets/figures/dispatch-path.svg).*

## Specialization and cache

The specialization request key contains normalized source, kernel name,
ordered ABI descriptors including each pointer's element dtype, sorted
compile-time values, backend, artifact
format, exact backend target, code-generation options, Swage revision,
dialect version, and LLVM version. The final artifact identity also includes
those backend fields and the validated contract digest.

Process-local artifacts for both backends use a least-recently-used cache with
128 entries by default. `SWAGE_MEMORY_CACHE_ENTRIES` may set another positive
decimal limit; invalid values fail when nonzero work first uses the cache.
Compilation is coalesced per specialization: equal requests share one result,
while unrelated specializations compile concurrently.

Only the known `(cuda, ptx)` pair may use the persistent cache, and only with
validated clean compiler identity as described below. CPU JIT executables are
always process-local. The cache root is selected in this order:

1. `SWAGE_CACHE_DIR`;
2. `$XDG_CACHE_HOME/swage`;
3. `~/.cache/swage`.

Each persistent entry contains `metadata.json`, `lowered.mlir`, and
`kernel.ptx`. Version-3 metadata names the backend, format, target,
specialization, and canonical version-2 launch contract. Contract,
lowered-MLIR, and PTX digests are verified before module load. Older cache
versions and symlinked, world-writable, incomplete, unreadable, corrupt,
wrong-backend, wrong-format, wrong-target, or specialization-mismatched
entries are rejected. Cache directories use user-only permissions; files use
mode `0600` and are replaced atomically.

Warm canonical fixed CUDA launches can use a native dispatch shortcut that
checks live tensor metadata and pointers, device, stream, current context,
and cache residency before enqueue. Changed or unsupported state resumes the
full validation path; an enqueue failure is never retried. PyTorch allocator
stream recording is retained for every tensor.

The following lifecycle is CUDA-only. Loaded-module lookup uses the same
bound, but lookup residency is separate from CUDA ownership. Prepared
executions hold explicit leases. For the context-owned legacy default stream,
retirement records a non-timing completion fence after eviction closes cache
admission and all leases are released. Caller-owned streams are fenced on
each submission because their handles may be destroyed before retirement.
Later activity polls completion events only in the exact current CUDA context
and unloads only after all events complete. Capture launches skip retirement
polling, and pending default-stream fences wait while that stream is capturing.
Polling never synchronizes or changes contexts. A module used during CUDA
graph capture is conservatively pinned until context destruction because
replay can outlive Python launch objects. PyTorch current-stream wrappers are
scoped to the imported torch module and bounded to 128 recently observed handles.

<div class="doc-figure" tabindex="0" markdown="1">

![Key composition feeding cache verification, where rejected entries raise and a plain miss recompiles](../assets/figures/specialization-key-cache.svg)

</div>

*Specialization key fields and the verify-or-reject cache path. [Open the full-size figure](../assets/figures/specialization-key-cache.svg).*

## Native build identity

Native wheels install the Python JSON resource `mlir_swage/_build_info.json`.
Its schema has exactly these fields:

| Field | Official v0.5.2 wheel value |
|---|---|
| `schema_version` | Integer `1`. |
| `package_version` | `"0.5.2"`. |
| `source_revision` | Exact 40-character lowercase hexadecimal source commit. |
| `source_clean` | Boolean `true` for official wheels; local native wheels may record `false`. |
| `llvm_version` | `"llvmorg-22.1.8"`. |
| `build_type` | `"Release"`. |

The runtime prefers validated packaged revision, cleanliness, and LLVM
identity over checkout Git discovery. This allows installed clean wheels to
reuse persistent CUDA artifacts without a source checkout. If the resource
is absent, the existing checkout-Git and `cmake/llvm-version.txt` discovery
applies. A dirty or unidentified source build permits only process-local reuse.

If the resource exists but is malformed or unreadable, the runtime does not
fall back to checkout identity or fabricate a cache key identity. Persistent
CUDA reads and writes are disabled; native compilation and process-local
reuse remain possible. The report exposes the validation error in
`native.error`, even if `native.available` remains true. Reinstall or rebuild
from verified source rather than editing metadata to force cache reuse.
This resource records identity, not a cryptographic signature: verify
downloaded checksums and signed build provenance separately.

## Opt-in logging

Swage uses `logging.getLogger("swage.runtime")` with no handlers, global
logging configuration, or default prints. Configure logging in the application
to see lazy DEBUG records:

```python
import logging

logging.basicConfig(level=logging.WARNING)
logging.getLogger("swage.runtime").setLevel(logging.DEBUG)
```

Cache decisions emit `memory-hit`, `persistent-hit`, or `compile`. Successful
CPU launch emits `launch complete`; CUDA emits `launch enqueued`, not a claim
that asynchronous GPU work has completed. Records contain only the event,
backend, target, sanitized kernel label, grid where applicable, and shortened
artifact key. They exclude pointers, tensor contents, source text, PTX,
cache paths, and environment secrets. Logging does not add synchronization.

## Debug dumps

Set either dump switch to the string `1`:

```bash
export SWAGE_DUMP_MLIR=1
export SWAGE_DUMP_PTX=1
export SWAGE_DUMP_DIR=/path/to/output
```

The default dump directory is `swage-dumps` under the current directory.
MLIR dumps apply to either backend; PTX dumps apply only to CUDA artifacts.
Dump files are named by specialization digest and receive the same safe,
atomic write treatment as cache artifacts.

## Environment report

```python
from swage.env import report

environment = report()
```

```bash
python -m swage.env
python -m swage.env --json
python -m swage.env --json --check native
python -m swage.env --json --check cpu
python -m swage.env --json --check cuda
```

`report()` returns a non-throwing dictionary, including when PyTorch or native
bindings are absent. Without `--json`, the CLI prints key/value lines;
`--json` prints exactly one sorted JSON object to stdout. Existing top-level
keys are preserved:

- `swage`, `python`, `platform`: package, interpreter, and platform facts.
- `torch`, `torch_cuda_build`: installed PyTorch and the CUDA version it was
  built against, or `null`; neither is the installed driver version.
- `cuda_driver`: the separately probed actual CUDA driver version, or `null`.
- `cuda`: PyTorch CUDA availability, not a complete backend-health verdict.
- `gpu`: GPU `name` and `compute_capability`, or `null`.
- `llvm_pin`: packaged LLVM version first, checkout pin second, or `null`.
- `backends`: structured CPU and CUDA availability, described below.

Schema version 1 adds `schema_version` (integer `1`), `implementation`,
`machine`, and `native`:

```text
native = {
  available, error, package_version, source_revision,
  source_clean, llvm_version, build_type
}
backends.cpu = {available, reason}
backends.cuda = {available, qualified, target, reason}
```

Availability and qualification fields are booleans; unknown native identity
fields are `null`. `native.error` is a sanitized import/probe or metadata
validation error, or `null`. Backend `reason` is a diagnostic string, or
`null` when available; CUDA `target` is an `sm_*` string or `null`. Native
imports, metadata, PyTorch, driver loading, and target admission are probed
independently so a failure does not hide other facts.

`backends.cuda.qualified` identifies the A6000/`sm_86` release hardware
configuration, not whether this particular artifact passed release gates.
It is independent of backend availability. An admitted non-qualified GPU
may report `available: true, qualified: false`. Neither this field nor a
successful check runs vector arithmetic, measures SLOs, or establishes production
qualification.

Without `--check` the CLI exits zero. A selected check exits zero only if
that component is available, and one otherwise, still emitting the complete
report. Invalid check names or malformed CLI arguments exit nonzero. A
metadata validation warning alone can coexist with available native bindings
and a successful check; it disables persistent caching, not native execution.
Inspect `native.error` as well as the exit status when verifying provenance.

Prerequisite exceptions at the launch boundary use stable
[`BackendUnavailableError`](swage.md#swagebackendunavailableerror) codes and
`code`, `backend`, and `remediation` attributes. The report itself reports
failures instead of raising them and never prints tensor values, pointers,
source text, cache contents, or environment secrets.

Continue with [swage](swage.md) for the public call
surface, [Troubleshooting](../getting-started/troubleshooting.md) for common
boundary failures, or [Verification](../internals/verification.md)
for the tests behind runtime claims.
