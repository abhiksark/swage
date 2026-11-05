<!-- benchmarks/results/segment-design-20261104-98b17f3/runtime-proposals.md -->
# Runtime proposals for a programmable segment compiler

This is a proposal for the owner to judge, not an implementation decision. It reviews source revision `98b17f38d7feccb8a31d7e45b10db510db5aeed1` and the local 2026-11-04 comparison. Proposed names below illustrate lifecycle boundaries; they do not prescribe the frontend syntax being investigated separately.

The recommended first experiment is **A: compile the program independently, snapshot the layout once, and bind execution separately**. It makes existing private programmable lowering useful without making caller mutation a hidden cache-coherence problem. **B: keep layouts transient and consolidate checked one-shot execution** is the competing design for workloads whose offsets change on every invocation. B remains useful as the convenience path of A if its cost is measured honestly. Neither proposal implies automatic differentiation of arbitrary programs.

## What is established, and what is not

**Observed:** The local public API lost all 17 fresh-offset comparisons against the fixed Triton baseline. Prepared mixed kernels won 4/6 against that fixed baseline and 3/6 against the best configuration selected after the sweep. Fresh calls include validation, classification, allocation, launch and synchronization; prepared measurements exclude preparation and use graph replay. The report explicitly does not establish performance for softmax, gradients, other reduction kinds, float64, other GPUs or held-out seeds. See the [comparison report](../triton-engineering-a6000-sm86-98b17f3/README.md), source lines 4, 19 and 91.

**Source facts:** A public reduction performs the following work:

| Step | Source | Consequence |
|---|---|---|
| Inspect arguments, allocate or select result, select fixed program, reject graph capture | `python/swage/_segments.py:111-139` | The public entry is an operator dispatcher, not a general compiled-program entry. |
| Copy device offsets to CPU, validate them, narrow int64 if needed | `python/swage/_segmented_validation.py:241-268` | There is a CPU decision dependent on current device data on every call. |
| Admit program, classify lengths and select launch families | `python/swage/_segmented_qualification.py:902-978` | Program admission can be memoized, but layout work still repeats. |
| Resolve compiled kernels, upload records, allocate partial scratch, bind arguments, enqueue | `python/swage/_segmented_qualification.py:981-1073` | A public call combines invariant and per-invocation work. |
| Retain storage on streams, advance output version, release leases | `python/swage/_segmented_qualification.py:148-200` | This work is correctness machinery, not disposable overhead. |

**Hypothesis:** Repeated Python orchestration, the device-to-host dependency, layout classification and metadata transfer together explain much of the fresh-call gap. The comparison does not isolate their contributions. Subtracting graph-replay time from public latency is not a valid estimate of any one component. An instrumented trace can locate work, but only matched uninstrumented ablations can quantify its benefit.

**Diagnostic evidence added during this investigation:** The root reviewer profiled 100 calls over 8,192 segments of length 16. The [profile record](public-call-profile.json) reports 19.804 ms total under cProfile; `_validate_shapes` accumulated 5.315 ms, `torch.tensor` used 2.981 ms self time, `Tensor.cpu` used 2.091 ms self time, and `_bind` accumulated 2.512 ms. Fifty unprofiled synchronized calls had a median of 111.955 us. Profiling overhead, nested spans and waits prevent adding these figures or treating them as an attribution benchmark. They support investigating binding/upload/validation together; they do not establish that the host synchronization alone causes the gap. This proposal agent ran no GPU work.

**Existing reuse:** `_prepare_planned_reduction` already prepares programmable capture-free reductions, stores bound pointers, retains kernels and uploads task records (`python/swage/_segmented_qualification.py:506-843`; ADR-0019). It also binds the caller's offsets and compares their version counters. Those counters explicitly cannot detect `.data`, foreign-pointer or external-kernel writes, and a CUDA graph replay does not run the check (`python/swage/_segmented_validation.py:335-351`). This private prepared helper is a useful implementation seam, but its mutation contract should not become the default public ownership contract unchanged.

## Shared prerequisite: native program facts and the checked tensor boundary

Both alternatives need the compiler to return facts about the selected, verified function: ordered semantic arguments, element type/rank, output shape category, admitted schedule families, whether a layout must cover every row, and any stage dependencies. The frontend supplies the function's actual MLIR object. The facts must come from native verification and the same argument specifications used by lowering.

Today `_program_element` searches text with a regex and `_parameter_roles` searches all role occurrences (`python/swage/_segmented_programs.py:47-79`); `_user_arguments` depends on that result (`python/swage/_segmented_runtime.py:265-275`). This is tolerable for closed, internally generated strings, but is a poor public trust boundary for caller-written programs, multi-function modules, reordered attributes or comments. Do not repair it with a larger regex or a second Python program IR. Add a narrow native query for one selected verified function, reuse its immutable result during compilation and binding, and retire text inspection from the new path.

The physical `KernelContract` already carries ordered scalar kinds, origins, binding keys and pointer access. This is the right owner for the physical ABI, as ADR-0025 requires; its pointer kind currently does not carry pointee dtype, shape or extent (`python/swage/_abi.py:25-46`; `docs/adr/ADR-0025-compiler-generated-kernel-contracts.md:18-54`). Keep the existing contract and binder rather than creating an unrelated universal launch wrapper.

**Necessary now:** return function-scoped semantic facts from verified IR and feed those facts to the existing checked tensor validator. For the first one-input, one-output programs, the facts identify f32/f64, rank one/two, reduction-versus-map output extent, and coverage requirements. Bind those facts to the same selected compiled program, accept tensors rather than public raw pointers, and keep offsets/task/scratch allocations under the existing trusted helpers. Programmable AOT is outside this first slice. `_validate_shapes` already checks dtype, rank, features, offsets, aliases, device and counts. Bind only after those checks, using the current version-2 physical contract. This gives a typed tensor boundary without making a new physical ABI version a prerequisite for useful programs. A small verified work-mapping field is justified separately where the host otherwise duplicates task-to-grid geometry; it should follow the compiler proposal's bounded mapping rather than a general formula language.

**Conditional later:** multiple inputs, stage-summary buffers and caller-supplied workspaces may justify versioned buffer descriptors in `KernelContract`: element storage kind, contiguous rank, explicit count references, fixed record width and access. Examples are values sized by `value_count`, offsets by `segment_count + 1`, and row scratch by `partial_count * feature_count`. Represent admitted shapes explicitly and check multiplication overflow. Do not invent a general expression language, strides API, allocator protocol or arbitrary pointer arithmetic. Raw-pointer users still have to tell the truth about allocation capacity; metadata is an ABI consistency guarantee, not a proof of arbitrary memory validity.

If that extension is selected, the compiler remains its source (`include/swage/Support/KernelContract.h`, `lib/Support/KernelContract.cpp`), and Python parsing, C API extraction, native binding, artifact derivation and tests change together. New versions fail closed. Do not reinterpret an old pointer-only contract as carrying verified extents. Until staged/multi-input execution actually needs it, keep this work out of the critical path.

## Alternative A: compiled program, owned layout, prepared execution

### Proposed API and meaning

```python
# Proposed API; kernel syntax is a separate design decision.
program = sw.compile_segments(
    sum_of_squares,
    signature={"values": (torch.float32, 1)},
    target="sm_86",
)
execution = program.prepare(values=values, offsets=offsets, out=output)
execution.run()
```

`program` is independent of offsets and storage addresses. It owns the verified semantic program, native facts, target identity and admitted kernel family. A caller-written sum of squares can use the existing single-stage planned path. A captured reduction followed by normalization can use a directly fused family only if native admission establishes that family; unsupported splitting is rejected during preparation, not discovered by attempting a launch and changing policy after failure. Compiling eagerly for this experiment makes the preparation/no-compile boundary explicit; later laziness is unnecessary until measurements justify it.

Preparation creates an internal layout that is an **owned snapshot**, not an assertion that the caller's offsets will remain unchanged. For device offsets, preparation waits for the current stream's preceding writes, copies one snapshot for host validation, and uploads the validated, narrowed int32 offsets that kernels will actually read. Both int32 and int64 inputs use the owned copy. The device allocation is not exposed through a mutable public tensor property. Keep a validated host copy or compact sufficient statistics when that lets multiple programs share classification; do not retain both full offsets and all possible task lists without a measured need. CPU-resident offsets could be a later admitted input, but are not necessary for the initial device-input experiment.

The row count inferred from the bound values and the segment count are fixed layout facts. Program facts decide whether the last offset must equal `rows` (a fully covering map output) or may be smaller (a reduction). Layout construction can validate monotonicity and the upper bound; program preparation adds the coverage requirement. Feature count, dtype, target and program determine scheduling and therefore belong to preparation, not the abstract layout. One internal layout can eventually serve sum of squares and normalization, or several feature widths, without implying that their task lists or scratch layouts are identical. A separately public `sw.segment_layout(offsets, rows=N)` object is optional: expose it only when real callers need to share layouts across programs. The first interface can keep the separation internal and offer just `program.prepare(...)` for explicit reuse plus `program.run(...)` for one-shot convenience. The convenience call creates a transient snapshot; it must not secretly cache mutable offsets by tensor identity.

`execution` is one binding of a program and layout to values/output storage on one CUDA context and one stream. It owns task records, mutable scratch, leases and the readiness/completion events. It holds tensor storage alive and checks dtype, shape, layout, device, lazy-view flags, aliasing and exact storage identity before each eager run. In-place changes to values are allowed after ordinary stream synchronization; rebinding storage is refused. A second binding creates a new execution with independent scratch. This is a limited initial concurrency model that avoids adding a general workspace pool.

### Ownership and state

```text
compile program -> verified immutable program and kernel family
snapshot layout -> validated host facts and private device offsets
prepare binding -> validated buffers, task records, scratch, leases, readiness
run             -> enqueue on bound stream, retain storage, update output version
close           -> prohibit new runs, defer resources until queued work completes
```

The plan's meaning is unaffected by subsequent mutation of the original offsets, including foreign writes that PyTorch does not count. Mutation concurrent with the snapshot is still a caller data race; ownership cannot repair unordered input production. The new snapshot semantics must be explicit because the private helper currently rejects some caller mutations instead of ignoring them.

Do not equate immutable Python fields with immutable GPU memory. Hold the owned device allocation and expose neither its pointer nor a mutable alias in the public interface. Internal tests can intentionally corrupt it, but supported callers cannot accidentally invalidate it. Values/output remain user buffers, and retaining their storage is different from merely retaining a Python tensor that can later be rebound.

### Graph execution and autograd

First qualify eager prepared execution. The next bounded graph experiment can be `graph = execution.capture(); graph.replay()`, where the returned managed graph owns the execution and storage references, checks fixed bindings before every replay and advances the output version on that host replay boundary. It must be created after compilation, allocation and task readiness, and must reject simultaneous use of one scratch workspace. The first version should stay on its bound stream, with separate bindings for concurrent streams. A failed enqueue poisons the execution until synchronization establishes completion; it must not hand uncertain scratch to another invocation.

Arbitrary external capture is a separate obligation. Host validation does not replay inside a CUDA graph. PyTorch 2.12 documents fixed replay addresses and shapes, no CPU replay, and prohibition of CPU/GPU synchronization during capture. A graph wrapper does not automatically make the call suitable for full-network `torch.compile` or compiled autograd. [PyTorch CUDA graph constraints](https://docs.pytorch.org/docs/2.12/notes/cuda.html#constraints).

The existing `_autograd` implementation hardcodes derivatives of fixed sum/max/min/mean and softmax (`python/swage/_autograd.py:246-317`). It is not the derivative of an arbitrary `reduce(x * x)`. The first programmable execution must reject `requires_grad` unless a separately verified derivative has been provided by the compiler or another explicit, tested mechanism. Preserve all gradients of the existing operator APIs. Adding a custom-op registration may later improve integration, but does not itself derive gradients or remove host validation; ADR-0024 already identifies that separate decision (`docs/adr/ADR-0024-gradients-of-the-segmented-calls.md:140-148`).

### Smallest implementation slices

1. Return native facts for one verified segment function; add mismatch and multi-function tests. Bind a sum-of-squares program using current private native emission, without promising new syntax yet.
2. Introduce an internal owned layout snapshot and prove mutation isolation for int32/int64, inference tensors, `.data`, pointer writes and cross-stream producers. Reuse the current classifier and kernel families.
3. Separate prepared binding from layout; start with rank-one f32, one reduction, eager inference and one bound stream. Verify empty/split/mixed cases and allocation/launch failures. Reject everything outside this slice explicitly.
4. Add rank-two and directly fused captured programs only after their native admission paths and checked tensor extents are wired. Preserve the distinction between captured direct execution and capture-free split execution.
5. If the reuse measurements pass, qualify managed graph replay and resource retirement. Adapt existing operator autograd to layouts only as a separate slice with first/second derivative tests.

Relative scope: native facts and an eager owned snapshot touch the frontend/native query, existing Python validation and prepared binding. They reuse current kernels and physical contracts. Managed graph ownership adds a separate state/lifetime boundary, and arbitrary-program differentiation is a substantially larger compiler capability. Keep those later obligations from delaying a useful eager program path.

### Kill criteria

Measure compilation separately, then layout creation `P`, binding `B`, prepared eager execution `E`, and competing one-shot execution `O` under identical synchronization boundaries. Report the break-even reuse count from `P + B + R*E < R*O`; if `O <= E`, there is no break-even at that boundary. Graph replay is a separate pairwise experiment, including any input/output copies the proposed graph API needs.

Predeclare before optimization: reject the design for a target workload if its observed reuse count is below measured break-even, if the plan fails any mutation/lifetime/alias test, or if repeated create/run/destroy leaves an increasing live-memory baseline after synchronization. Use five held-out seeds, both rank forms, mixed and extreme-tail layouts, and three independent processes. For the first private performance gate, require at least a 20% amortized improvement over the current checked public path at the workload's actual reuse count, no more than 5% kernel-time regression against the existing matching prepared family, and report all losses against tuned Triton. These thresholds are proposed decision rules, not measured outcomes.

## Alternative B: transient layouts and a checked native dispatch transaction

### Proposed API and state

```python
# The same compiled program object; offsets are consumed anew every time.
program = sw.compile_segments(kernel, signature=signature, target="sm_86")
program.run(values=values, offsets=fresh_offsets, out=output)
```

Each call owns a transient execution record. It validates tensor metadata, snapshots and validates current offsets, chooses only an admitted schedule, allocates records/scratch and enqueues. It retains all referenced storage until completion, then retires the record. Nothing is cached by caller tensor identity or version. Kernel compilation and native semantic facts are cached by the compiled program identity, not by the mutable layout.

The first experiment should consolidate the existing checks/classification/binding into one native transaction, while preserving the current schedule and arithmetic. Python still handles PyTorch ownership and metadata where necessary; a native API should accept an explicit checked descriptor rather than import a second tensor framework or assume bare pointers imply ownership. Use the existing runtime C classifier and launch layer. Output allocation remains a caller/PyTorch responsibility, with `out=` supplied in latency measurements. A C++ rewrite of every Python helper is not required.

This alternative offers a simpler fresh-input API and no persistent plan lifetime for callers to manage. It may remove Python dispatch and repeated metadata assembly, but it retains a host decision on current offsets. It is not a graph-safe design while rejection must happen synchronously before a result-writing kernel.

### Host versus device validation experiment

A second, conditional B experiment replaces the full offset readback with a device snapshot-and-validate stage and a small status readback. Its contract is:

1. Validate tensor rank/dtype/device/capacity on the host.
2. Validate all offset values and store a private snapshot on the device, preserving int64 until bounds are proven. Set a bounded status code and the first invalid index. There are no result writes in this phase.
3. Read status and raise on any failure. Only after success may numerical kernels run against that exact snapshot.
4. Execute either the existing classified schedule or a separately admitted direct family. A direct family avoids host task construction, but is an explicit design choice, not fallback after failed classification.

This reduces transfer volume in principle, but adds a kernel and retains a synchronization point. A safe global monotonicity check must inspect boundaries between validation tiles. A plan-record allocation cannot depend on an unknown device count without a bounded capacity strategy. Start with direct kernels and no dynamic queue if this variant is tested at all. Rank-one many-tiny cases may become slower because the validator's launch cost exceeds the saved transfer. Long segments can make direct execution slower and change its rounding tree. The existing public numerical bounds must not be claimed for an arbitrarily long direct loop without a new proof and tests.

A device status flag alone is not a solution to immediate errors. Returning a normal output tensor while silently leaving an unchecked status buffer behind would weaken the contract. Truly asynchronous fresh-layout validation requires a different public result/error model, such as a checked completion token that must be resolved before consuming results, or an admitted device producer whose guarantees establish the layout. Neither belongs in the first implementation. Device asserts also change error timing and context-failure behavior and should not replace the current actionable exceptions.

### Smallest implementation slices and kill criteria

1. Add per-phase profiling around existing source paths without modifying timed benchmark variants. Establish the share of time in host waiting, validation/classification, binding and launch.
2. Add one native checked transaction for rank-one f32 sum-of-squares, retaining the current host snapshot, classifier and kernels. Reuse A's native facts, current tensor validator and physical contracts. Verify identical rejection order where it is contractual and unchanged output on prelaunch failures.
3. Only if full offset readback is material after step 2, prototype the device validation/status variant and compare total call cost, not just validation kernel time.
4. Keep full program, dtype, rank, stream and gradient admission explicit. Do not route the existing public operator through a different numerical tree merely to improve this experiment.

Relative scope: the native-transaction ablation changes host orchestration and bindings while preserving kernels. The device validator adds a GPU program, status ABI, snapshot ownership and new failure ordering, so it is materially larger. The largest risk is that retained synchronization dominates, making a native port maintenance work with little user benefit.

Proposed gate: stop widening B if it fails to improve median fresh-call latency by at least 20% in 12/17 matched cases, or regresses any case by more than 5%, across three processes. Test held-out layouts before revising those thresholds. This would establish a useful reduction in overhead, not parity with Triton. Before claiming a competitive fresh-call path, require a separate ratio gate against the matched tuned baseline; the present 1.18x-7.80x losses show why 20% alone is insufficient.

## Correctness matrix for either alternative

| Case | Required behavior or test |
|---|---|
| Empty offsets; negative/decreasing offsets; final offset beyond rows; int64 overflow | Reject before numerical output writes; retain precise first-failure diagnostics. |
| Empty batch; empty segments; zero features; uncovered tail | Preserve current identities and no-work behavior; full map output requires covering layout. |
| f32 versus f64; rank and feature mismatch; offset dtype | Compiler facts and typed buffer checks agree; no reinterpretation, cast or backend fallback. |
| Caller changes original offsets after snapshot, including untracked foreign writes | A remains tied to its owned snapshot; B snapshots each call anew. Concurrent unsynchronized mutation during snapshot remains a caller race. |
| Values/output rebound, freed or reshaped between runs | Eager/managed replay rejects changed bindings; retained allocation prevents a stale pointer from outliving storage. |
| Partial aliases, lazy views, output sharing a version counter without sharing bytes | Keep exact byte-overlap rejection; validate storage semantics; advance written output without relying on offset counters for ownership. |
| Two streams or threads use one execution's scratch | A initially refuses off-stream reuse; distinct executions can overlap. B owns per-call scratch. Future serialization must use completion events, not a Python lock alone. |
| Producer on another stream | Require an explicit dependency before snapshot/run; allocator retention is not execution ordering. |
| Plan destroyed before queued work finishes; graph survives plan; failed enqueue | Hold storage/modules until completion or graph lifetime ends; poison uncertain execution; exercise teardown after failure. |
| CUDA graph replay after mutation or cache eviction | Owned offsets and module/storage references survive; stable binding checks occur at managed replay, never assumed to be captured CPU work. |
| NaN, infinities, both zero signs, subnormals, cancellation and huge segments | Compare exact max/min semantics and schedule-specific sum bounds; do not substitute PyTorch's different tie/zero behavior silently. |
| Arbitrary element expression with requires_grad | Reject until a valid derivative is available; do not reuse a fixed operator's backward. |
| Existing public operator gradients and AOT execution | Preserve first/second derivatives and no-compile/artifact refusal; do not advertise programmable AOT until the artifact format supports selected programs. |

PyTorch documents that `record_stream` prevents allocator reuse before stream work finishes; it does not make producers and consumers ordered. Keep storage retention or equivalent explicit event ownership. [PyTorch 2.12 record_stream](https://docs.pytorch.org/docs/2.12/generated/torch.Tensor.record_stream.html). PyTorch also documents that raw-pointer writes require explicit version updates and inference tensors have no version counter. This supports snapshot ownership rather than a promise that counters detect all mutation. [PyTorch 2.12 increment_version](https://docs.pytorch.org/docs/2.12/generated/torch.autograd.graph.increment_version.html).

## Cross-review: strongest objections and responses

**Objection: a new full typed kernel-contract version delays the useful compiler without fixing an immediate admission problem.** Agreed for the first one-input programs. The existing validator plus function-scoped native facts can establish dtype, shape and allocation bounds before the existing physical binder. The contract extension above is conditional on stage buffers, multiple inputs or exposed workspaces, not a launch requirement for sum-of-squares or directly fused normalization. Explicit work mapping remains a smaller distinct correctness need because block dimensions alone do not describe how many tasks a block serves.

**Objection: three mandatory public objects turn a simple segment program into a resource-management API.** Agreed. Separate ownership internally first. A caller can start with `program.run(values=..., offsets=...)`, and use `execution = program.prepare(values=..., offsets=..., out=...)` only for reuse. A public reusable layout handle earns its place only when multiple programs share one layout or layout construction comes from a distinct producer. Preparation always snapshots offsets; there is no hidden identity cache. The tradeoff is explicit: convenience pays current-layout validation, reuse pays preparation once and keeps the original snapshot.

**Objection: the compiler proposal's direct captured route could accidentally become a fallback after split rejection.** The compiler must report admitted families before preparation. A captured normalization program selects its direct family from those facts; split is not attempted. A capture-free reduction may select a planned family under its verified constraints. Unsupported operations and ABI mismatches remain errors, not reasons to try another family. Return an inspectable chosen mapping on the execution so the limitation is visible.

**Objection: a graph-friendly prepared API is being mistaken for complete PyTorch integration.** The first useful language can be eager and refuse gradient recording for arbitrary programs. Fixed operator autograd remains intact. Managed graph replay, arbitrary external capture, compiled autograd and `torch.func` are independent integration gates. A correct owned layout solves one lifetime problem; it does not solve derivative generation or PyTorch's graph semantics.

**Compiler proposal assessment:** the single-input route is the smallest complete test of programmability. Energy and normalization exercise distinct admitted native paths and cannot be reduced to adding operator selectors. Its weighted-pooling extension changes real source operands/roles and should follow evidence that users need a second input. The proposed explicit bounded work mapping is preferable to duplicating grid formulas in Python, but can be introduced incrementally for supported shapes without a universal GPU layout algebra. See [compiler proposals](compiler-proposals.md).

## Decision proposed for the owner

Prioritize A's private programmable eager slice, with native program facts, the existing checked tensor boundary and an owned layout snapshot. It directly tests the compiler's value: the same caller-written semantics can execute under verified different task decompositions, while preparation cost is visible and reusable. Keep B as the measured fresh-layout convenience path, and do not require a device planner before the compiler admits useful programs. If representative applications never reuse layouts, A does not solve their latency problem; that evidence should change the priority rather than be hidden by a prepared-only benchmark.

Only this proposal file was written. No compiler/runtime code, API, artifact format or tracked test changed, and no GPU workload ran for this proposal. Follow-up decisions are API/lifetime approval, an agreed programmable subset, and predeclared workload/reuse gates. No issue was filed. Graph integration, differentiation of arbitrary programs, multi-GPU behavior and performance improvements remain unimplemented and unverified.
