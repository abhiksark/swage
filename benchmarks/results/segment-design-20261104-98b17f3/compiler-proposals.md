<!-- benchmarks/results/segment-design-20261104-98b17f3/compiler-proposals.md -->

# Compiler design alternatives for a programmable segment compiler

Review basis: current source at `98b17f38d7feccb8a31d7e45b10db510db5aeed1` on 2026-11-04. These are proposals for the owner's decision. No production implementation or performance claim follows from this document.

## Finding and recommendation

The compiler's strongest reusable part already exists: one symbolic segment, pure per-element regions, reductions with explicit captures, and lowering to bounded GPU work. The shortest route to a useful programmable compiler is to let callers express programs that this machinery already admits. Sum of squares and normalization by a segment sum are useful first examples. Neither requires a new reduction kind, a generic combiner, a general task DAG, or a new intermediate representation outside MLIR.

There are two credible options:

1. **Expose the existing single-input program model first.** Compile caller-written maps, reductions, and terminal stores through verified semantic MLIR. Single-stage reductions use the existing direct and split schedules. Programs with captured reduction results use the existing direct fused kernel. This is the recommended first implementation because it tests whether programming the compiler is useful before expanding its planner.
2. **Build a bounded two-input, two-stage compiler before exposing the language.** Include weighted pooling and dependent reductions that can split across blocks. This serves a broader workload immediately but requires real changes to semantic operands, plan regions, scratch ownership, launch orchestration, and numerical qualification. It is an alternative investment decision, not a prerequisite for option 1.

The owner's priority is programmability. Optimizing fixed built-in operators alone would not meet it. Conversely, exposing every construct accepted by the semantic dialect would overstate the admitted lowering surface.

## What the current implementation actually supplies

| Source evidence | Implemented behavior | Design consequence |
|---|---|---|
| `lib/Conversion/SwageToPlan/Admission.cpp:80` | Element regions admit constants, add/subtract/multiply/divide, maximum/minimum, and f32 `exp2`; every result has the program's element type. | A useful first language can compile expressions without changing native arithmetic admission. It cannot promise arbitrary Python or every MLIR arithmetic operation. |
| `Admission.cpp:491` | Region captures must be scalar results of `swage.reduce`. | Segment-derived normalization is structurally expressible. Runtime scalar parameters and arbitrary scalar expressions captured from outside the region are additional capabilities. |
| `Admission.cpp:675` | Split planning accepts exactly one capture-free reduction and no terminal map-store. | Split softmax is a planner extension, not a missing spelling of the current language. |
| `Admission.cpp:542` | The scalar epilogue is exactly extent, index cast, conversion to float, and division of one reduction. | General scalar postprocessing, even a final multiply by a scale, needs a deliberate extension. |
| `lib/Conversion/SwageToPlan/SwageToPlan.cpp:154` and `:192` | Direct tasks reuse consumer regions; split merges use the reduction kind and do not repeat the element transform. | Keep this decomposition rule. An element expression belongs in partial kernels, and a scalar epilogue belongs after the complete reduction. |
| `include/swage/Conversion/SwageToPlan/Admission.h:33` | `SegmentABI` has one values argument, one offsets argument, one output, and counts. | Weighted pooling requires a second values source and corresponding contract changes. |
| `lib/Dialect/SwagePlan/IR/SwagePlanDialect.cpp:218` | The second task-region argument is implicitly interpreted as extent. | Cross-kernel captured stage inputs cannot be added as unnamed extra block arguments. Their schema needs explicit arity and roles. |
| `lib/Conversion/SwagePlanToGPU/ConsumerPatterns.cpp:89` and `lib/Conversion/SwagePlanToSCF/SwagePlanToSCF.cpp:203` | GPU and CPU reuse the consumer patterns and reduction helpers. | Reuse is valuable, but their agreement is not independent proof of shared expression or combine semantics. |
| `python/swage/_segmented_programs.py:47` and `:71` | The private runner infers element type and parameter roles with regular expressions over qualification text. | Do not make this the public compiler boundary. Return function-scoped facts from verified native IR. |

ADRs 0001, 0008, 0019, 0020, 0022, 0023, and 0025 support preserving the current separation: source meaning in Swage MLIR, physical task decisions in SwagePlan, compiler-owned entry signatures, and explicit captures.

## First three caller-written programs

These examples describe source meaning. Public API spelling belongs to the separate frontend/runtime proposal; no new decorator name is selected here.

### 1. Segment energy: sum of squares

```text
segment_energy(values, offsets):
    s = segment(values, offsets)
    return reduce_sum(map(s, lambda x: x * x))
```

The essential existing semantic IR is:

```mlir
%energy = swage.reduce %s kind<sum> : !swage.segment<f32> -> f32 {
^bb0(%x: f32):
  %square = arith.mulf %x, %x : f32
  swage.yield %square : f32
}
```

The frontend emits the existing roles, segment IDs, segment construction, and scalar store around it. The single-stage planner already handles this shape, including a split and a merge that sums partial energies rather than squaring them again. The program also works per column for `[N, D]` values within the existing rank-two model.

This proves user expressions pass through the compiler. A public wrapper whose implementation chooses among eighteen fixed built-in modules would not establish that.

### 2. Segment normalization: divide each value by its segment total

```text
normalize_by_sum(values, offsets):
    s = segment(values, offsets)
    total = reduce_sum(s)
    return map(s, lambda x: x / total)
```

Existing semantic IR expresses the dependency explicitly:

```mlir
%total = swage.reduce %s kind<sum> : !swage.segment<f32> -> f32 {
^bb0(%x: f32):
  swage.yield %x : f32
}
swage.map_store %s, %out captures(%total : f32)
    : !swage.segment<f32>, memref<?xf32> {
^bb0(%x: f32, %den: f32):
  %normalized = arith.divf %x, %den : f32
  swage.yield %normalized : f32
}
```

The direct kernel can execute the reduction, broadcast the completed scalar inside the block, then emit the final map. For f32 stable softmax, the existing `_softmax_text` demonstrates two reductions and a terminal map with captured maximum and denominator. Its explicit `exp2((x - max) * log2(e))` is an approximation contract, not an exact mathematical exponential.

Option 1 should admit this bounded family: one input segment, at most two reduction stages, and one scalar or map-store terminal. A zero denominator follows floating-point division semantics; the compiler must not silently insert an epsilon. Empty segments write nothing for a map terminal. Extrema and softmax retain the defined NaN/infinity behavior.

These programs use the direct fused schedule. The documented compilation decision may select that supported schedule; a failed split attempt must not silently change execution policy. An inspection result should show the selected schedule before execution. Long segments can therefore be slow under option 1, and that is an explicit scope limit.

### 3. Weighted pooling: sum products of corresponding inputs

```text
weighted_pool(values, weights, offsets):
    x = segment(values, offsets)
    w = segment(weights, offsets)
    return reduce_sum(zip_map(x, w, lambda value, weight: value * weight))
```

This is the first justified semantic extension. Start with two contiguous input buffers of identical shape and dtype sharing the same offsets and logical segment/column coordinates. Do not add broadcasting in the first version. Thus `[N, D]` values take `[N, D]` weights initially; per-row `[N]` weights are a separate extension.

One coherent implementation extends the segment operands of map/reduce/map-store from one to one-or-two, with element arguments followed by captures in region order. For example, the following is proposed syntax, not currently valid IR:

```mlir
%weighted = swage.reduce (%xs, %ws) kind<sum>
    : (!swage.segment<f32>, !swage.segment<f32>) -> f32 {
^bb0(%x: f32, %w: f32):
  %product = arith.mulf %x, %w : f32
  swage.yield %product : f32
}
```

Both operands must be proven to describe identical ranges. For the first version, require identical offsets SSA operands and segment/column IDs, not an expensive proof that two different offset arrays happen to match. Host validation establishes equal input row counts and feature counts. The partial emitter loads both values at the same logical index; the merge still consumes one scalar partial and needs no second values buffer.

This avoids a tuple-valued runtime segment type. It nevertheless changes ODS operands, region verification, map fusion, segment binding, source argument roles, and kernel layouts. Adding a standalone `zip_map` op without changing its consumers would not be complete: today's map fusion reduces a consumer to one underlying segment and one load.

## Option 1: expose current bounded programs

### Compiler work

1. Add direct native construction from the restricted source frontend for existing segment operations. Preserve source locations and `IsolatedFromAbove`; kernel bodies remain unexecuted Python syntax.
2. Expose a small immutable native program description: selected function, input/output dtype and rank, scalar-versus-map output extent, admitted schedules, reduction stages, and whether f32 exponential approximation occurs. Read these facts from the selected verified function, not all text in the module.
3. Keep current element operations, one input, one output, and current mean epilogue. Refuse runtime scalar captures, arbitrary scalar epilogues, generic combiners, user memory effects, cross-segment indexing, and unimplemented dtypes with source diagnostics.
4. Route one capture-free reduction to current direct/split machinery; route the bounded captured family to direct tasks. Existing operator APIs can remain unchanged.
5. For custom programs, refuse gradient recording until derivative behavior is implemented and tested. Reusing built-in softmax gradients for arbitrary expressions is not valid.

### Dependencies and verification

The frontend and runtime must provide owned compilation/preparation objects, native metadata extraction, checked input bindings, and lifetime/stream behavior. Reuse their current validation and compiler-generated contracts. A public JIT path should not serialize and reparse textual MLIR, per ADR-0001; qualification text and artifact inspection can remain textual interfaces.

Acceptance tests must include actual caller-written energy, mean-square, normalization, and stable-softmax programs; positive source-location checks; negative capture and side-effect cases; independent references; direct versus split where legal; map-fusion equivalence; and public error boundaries. Keep the current fixed operators' tests to detect regressions.

### Failure criteria

Stop expanding the language if these examples require kernel-name or source-text pattern matching, if native facts and runtime binding disagree, or if a valid program needs to execute Python to build its semantic IR. Reject release if malformed source changes IR before diagnostics, if a capture reads a different segment's summary, or if any mismatch is hidden by a loose tolerance.

For utility, require at least one small external caller-written example that uses an element expression absent from the built-ins and one captured normalization example. Do not claim general performance superiority from them. If direct captured programs are too slow for the owner's intended lengths, use that measured workload to choose option 2 rather than adding arbitrary optimizations.

## Option 2: bounded multi-input and staged execution

This option includes option 1's program facts and direct path, then adds weighted pooling and split dependent reductions. Its upper boundary should be explicit: at most two reduction stages over one logical segment range, at most two zipped inputs, and one terminal scalar or map-store. No loops over stages, arbitrary dependencies, cross-segment communication, persistent queue redesign, or user-defined combiner.

### Split softmax as the concrete staged proof

For the existing softmax meaning, a straightforward split plan is:

```text
1. partial_max(values, ranges)                  -> max_partials[P, D]
2. merge_max(max_partials, merges)              -> max_by_segment[S, D]
3. partial_sum_exp(values, ranges, segment_ids,
                   max_by_segment)             -> sum_partials[P, D]
4. merge_sum(sum_partials, merges)              -> sum_by_segment[S, D]
5. normalize(values, offsets, max_by_segment,
              sum_by_segment)                  -> output[N, D]
```

`D = 1` covers rank-one values. Every producer completes before a consumer on the selected CUDA stream. Partial records need a segment association so every chunk reads the completed maximum of its own segment. Existing partial records hold only `[begin, end]`; `partial_merge_ids` exist for persistent execution but are not currently an ordinary staged-input ABI. Either reuse a materialized segment-association array or add a versioned field deliberately. Do not infer segment identity from equal lengths or scratch offsets.

The planner must prove that the maximum seen by every partial denominator task is the full-segment maximum. Taking a separate maximum per chunk and merely adding the resulting denominators is incorrect. An online `(maximum, normalized_sum)` combine is a different numerical algorithm and should not be slipped into this implementation as an equivalent rewrite.

### Required plan changes

The task operations need explicit summary-buffer operands, indexed by the task's segment and column, with corresponding scalar block arguments. Extend capture verification to accept these verified stage-input arguments in plan IR; semantic captures remain explicit SSA results of earlier semantic reductions. Replace the current inference that two block arguments means `segment, extent` with an explicit operand/region schema. Validate source and destination element types, summary extent, producer order, and scratch sizes.

Keep the execution description in MLIR. A bounded orchestration function can use ordinary `func`/`scf` operations and runtime launch calls for the named plan functions. Lower it to the native host execution boundary, or first support a fixed compiler-declared staged template whose runtime performs a fixed sequence of launches. The latter is smaller but less extensible. Do not invent a Python object graph with expressions, branches, and dependency evaluation; that would recreate the second production IR prohibited by ADR-0001 and reiterated for launch metadata in ADR-0025. This host orchestration boundary is a major dependency and deserves its own implementation decision before promising generic staged programs.

For independent reductions, fuse input traversal only after each accumulator's original visitation and combination order is specified. For dependent reductions, the scalar dependency is a real barrier. A shared source buffer alone does not make fusion legal. Preserve the existing no-alias output obligation.

### Tests and failure criteria

In addition to option 1's tests, validate two-input range identity, mismatched shapes/dtypes, identical operands, disjoint input storage, and partial output aliasing rejection. Test weighted pooling against elementwise multiplication followed by independent segmentation, including split boundaries and ragged empties.

For staged softmax, use segments whose maximum lies in the last chunk, different maxima in every chunk, chunk permutations, empty segments, NaNs, infinities, and values near exponential underflow. Vary the segment's neighbors and batch position to catch incorrect summary indexing. Inject malformed producer and summary metadata before launch. Racecheck every stage and test resource release after partial compilation or launch failure.

Measure complete operation latency including plan preparation, scratch allocation, metadata transfer, and every launch, as well as repeated prepared execution. A five-kernel split can lose to one direct kernel. Do not make splitting the default unless it beats the existing direct path on a predeclared target workload and stays correct. If no target workload benefits after including those costs, retain direct captured programs and stop the staged expansion. Weighted pooling can still ship independently if it passes its own gates.

## Numerical contract needed by either option

The language should distinguish legal decomposition from a quantitative error promise. Sum means each transformed element contributes once to an allowed floating-point reduction tree, plus the defined identity; it does not mean exact real arithmetic or bounded relative error for all inputs. Finite inputs may overflow in an intermediate partial. A sum error bound requires stated hypotheses about intermediate overflow, underflow treatment, precision, and reduction depth. Element-transform error is additional to summation error. Exponential approximation is additional again.

Keep maximum/minimum semantics consistent with `arith.maximumf`/`arith.minimumf`, including NaNs and signed zeros. Do not claim NaN payload identity across schedules or architectures. Keep generic or integer combiners out until their algebra and overflow behavior are defined. MLIR's arithmetic specification explicitly distinguishes operation semantics, rounding, and fast-math permissions; a compiler needs to preserve those choices rather than treating every mathematical identity as a valid rewrite. [Primary source: MLIR arith dialect](https://mlir.llvm.org/docs/Dialects/ArithOps/).

For reproducibility, initially promise replay under the same compiled artifact, selected schedule, input shape/layout, and qualified target, with correctly ordered stream access and no concurrent input mutation or workspace races. Record schedule and tile identity in the compilation/preparation result. Do not add a broad `deterministic=True` claim or promise equality across different reduction trees. Existing rank-one selection depends on the batch and SM count (`_segmented_plan.py:237`); rank-two row tiling depends on feature count. Those dependencies should be visible rather than hidden under a generic determinism label.

Independent tests must separate exact integer-representable fixtures, floating cancellation, finite-input overflow, subnormals, signed zeros, NaNs/infinities, and exponential error. A shared CPU emitter cannot be the only oracle. A tolerance should be derived for its tested class and must not let NaN silently pass as a small error.

## Tile geometry and launch contracts

The current column width rule appears in `TargetDescription::columnGroupWidth`, the emitted kernel helper `Emission.cpp:214`, and Python `_column_group_width` at `_segmented_qualification.py:25`. Python also uses that width for chunk limits and launch grids. The kernel contract validates block dimensions, while these task-to-grid rules are separate. This duplicated knowledge is a design seam with demonstrated correctness consequences in the broader review's artifact metadata finding.

Immediate hardening is small: an artifact for the existing NVIDIA lowering must match its supported subgroup width and geometry invariants. Do not treat these as freely adjustable manifest knobs.

For either programmable option, make the compiled entry's work mapping explicit using a small versioned enum, for example `one_task` or `task_column_groups_v1`, plus the fixed grouping parameters it needs. Derive the launch item count from that verified mapping and checked runtime counts. Do not introduce a general expression language for grid formulas. Bind this metadata to artifact identity and fail closed if it contradicts the target, block dimensions, or compiled entry. The plan IR remains where tile choice belongs; semantic Swage IR still contains no hardware IDs.

Do not redesign SwagePlan as a universal GPU layout algebra now. An explicit two-choice column/row-stripe mapping suffices for the accepted rank-two model. More layouts need a workload and their own numerical/launch contract.

## Cross-review of compilation, immutable layouts, and execution ownership

The frontend/runtime proposal separates a compiled program, an owned immutable offsets layout, and an execution prepared for concrete tensors. These are ownership boundaries, not three mandatory public objects. After cross-review, the runtime proposal keeps the layout snapshot internal to `program.prepare(...)` initially; a separate public layout handle is optional if cross-program reuse justifies it. This separation is compatible with the native compiler, with the following obligations:

- A compiled program contains semantic facts and permitted kernel variants. Actual offsets are not part of semantic compilation. `D` need not enter the segment type; current row kernels already calculate the grouping width at runtime.
- A layout owns a validated offsets snapshot, not a promise that a borrowed tensor will stay unchanged. Construct and validate one owned snapshot, then derive the device int32 offsets from that exact validated snapshot. A host read and a later independent device clone of mutable caller data can disagree. Stream/event ordering must make the internal snapshot visible on the execution stream.
- Offset classification is not solely a property of the offset values. Rank-two chunk limits depend on `D` through the column grouping width. Direct-versus-split selection can depend on program work and target SM count. A layout may own lengths and immutable records, but any cached derived plan must include its effective chunk limit and schedule geometry in the key. Reusing the rank-one classification for the same offsets at `D = 64` is not valid.
- Mutable scratch belongs to a prepared execution, not the shared program or shared layout. Two executions may share immutable offsets and kernel leases; they must not race on partial results or stage summaries. Concurrent reuse of one execution requires explicit in-flight handling. A host mutex released after asynchronous enqueue is insufficient across CUDA streams.
- Prepared tensors can have their storage or metadata changed after preparation. At run time, validate dtype, shape, stride, device/context, pointers, and aliasing against the binding contract, or bind an object that owns an immutable storage contract. An offsets snapshot solves layout mutation, not tensor-resize hazards.
- Ahead-of-time compilation can emit all allowed variants, or preparation can compile missing allowed variants before the hot path. The API must state which operation may compile; `run()` should not unexpectedly compile or classify. A no-compile mode must reject missing variants.
- Public arbitrary-program derivatives remain rejected until separately derived and verified. A program that resembles a built-in expression is not sufficient evidence to attach the built-in derivative.

No extra compiler IR is needed for the first option. The immutable program facts and layout/execution ownership are runtime metadata and storage, while the program itself remains native MLIR. Staged option 2 needs the additional orchestration decision described above.

### A full buffer-contract version change is not a prerequisite

The runtime draft proposes a new typed buffer descriptor as shared prerequisite. That is useful future hardening, but I would narrow it for the first single-input, rank-one f32 eager experiment. Its safety obligations can be met by selected-function native facts plus the existing tensor validation and version-2 `KernelContract` binding:

1. Native facts establish the selected function's values dtype/rank, output category, argument order, and supported schedules.
2. The runtime validates actual tensors against those facts before reading addresses. The existing code already validates shape, dtype, contiguity, device, overlap, and offset bounds for this fixed buffer model.
3. Existing trusted helpers allocate owned int32 offsets, task records, and correctly sized scalar partial scratch. Checked sizes and capacities remain mandatory.
4. The version-2 contract binds the already-validated pointers and exact scalar kinds in compiler-owned order. Program facts belong to the same compiled-program object and cache identity as that contract; mixing facts from another function or specialization is rejected.

There is no public raw-pointer input and no arbitrary-program AOT format in this first slice. Consequently, a new C++/Python/native-binding/cache/artifact schema migration does not unlock either demonstrated program. The semantic facts are not a replacement physical ABI; they supply validation information for the existing narrow tensor model.

Reconsider a typed physical buffer schema when a concrete multi-input, staged-scratch, or compiler-free artifact boundary cannot be validated from those facts and the known allocation rules. Even then, identify the missing invariant before versioning every layer. The confirmed artifact geometry inconsistency should be fixed independently; it does not by itself require a universal typed buffer ABI.

This reduces option 1's mandatory work to a source-to-native bridge, a narrow native query, immutable offsets ownership, and prepared binding around existing checked execution. It preserves a clear path to stronger descriptors without making the first useful compiler example an infrastructure rewrite.

## Native feasibility checked in this investigation

The persisted reproducer is `compiler-native-check.py`, and its output is `compiler-native-results.json` in this directory. Run from the repository root:

```sh
PYTHONPATH=build/python_packages:python:python/tests/mlir python \
  benchmarks/results/segment-design-20261104-98b17f3/compiler-native-check.py \
  --output-dir /tmp/swage-native-recheck
```

This uses the existing qualification helpers to construct arbitrary named semantic modules. It does not exercise or establish a public Python segment frontend.

For values `[1, 3, 2, 6]` and offsets `[0, 2, 2, 4]`, including an empty segment:

| Program | Executed CPU oracle | NVPTX compilation for `sm_86` | Negative check |
|---|---|---|---|
| `caller_energy`, sum of squares | `[10, 0, 40]`, exact asserted result | Direct, split partial, and split merge; 5,081, 5,261, and 5,390 PTX bytes | All compiles preserve the source module |
| `caller_normalize`, value divided by segment sum | `[0.25, 0.75, 0.25, 0.75]`, exact asserted result | Direct; 5,321 PTX bytes | Split partial rejected before source mutation |

Arbitrary `caller_*` function names were used, so this evidence does not depend on fixed operator-name dispatch. CPU guard sentinels were checked by the existing oracle helper. Four native compiles and two executed CPU examples passed; the invalid split was rejected as expected. GPU kernels were not launched, and these small examples do not establish numerical or performance qualification of a future public frontend.

## Decision requested from the owner

Choose between delivering useful caller-written single-input programs first, or accepting the larger input/stage expansion before the initial programmable interface. My recommendation is option 1, followed by weighted pooling as the first semantic extension, and split dependent stages only when a real captured program demonstrates the direct path's limit.

This ordering preserves the existing compiler spine and makes the central proposition testable early: a caller writes one segment-local program, and the compiler emits and selects valid bounded GPU work for it.

Files changed by this investigation: this note, `compiler-native-check.py`, and its result JSON only. Semantic behavior changed: no production change. Checks run: two executed native CPU examples, four successful native PTX compiles for `sm_86`, one expected split rejection, and source-preservation assertions. GPU workloads run: none. Follow-up issues are proposals only; none were filed. Unchecked: implementation cost, GPU execution and performance of these examples, arbitrary-program derivatives, and targets beyond the existing qualified hardware.
