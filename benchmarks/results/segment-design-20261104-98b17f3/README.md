<!-- benchmarks/results/segment-design-20261104-98b17f3/README.md -->
# Swage design investigation: a useful programmable segment compiler

**Decision brief, 4 November 2026. Reviewed source: `98b17f38d7feccb8a31d7e45b10db510db5aeed1`.**

The recommended experiment is to expose a bounded caller-written segment language through the compiler that already exists, with a direct execution baseline and reusable owned layouts where applications benefit. Add aligned two-input programs next if a real workload needs them. Expand split execution of dependent stages only after measuring the direct path's limit. These are proposals for the owner to judge; no proposal has been adopted or implemented.

The central design problem is that Swage's public execution surface is an operator library while its native implementation already contains a small programmable segment compiler. Closing that gap is more aligned with the chosen goal than adding another scheduler. The present evidence supports a bounded experiment, not a general superiority or novelty claim.

[Open the interactive comparison](showcase.html). Supporting reviews: [compiler semantics](compiler-proposals.md), [frontend and runtime](runtime-proposals.md), and [independent strategy](strategy-proposals.md). Each review inspects a different boundary, then challenges the combined recommendation.

## 1. What should remain, and what should change

Keep the existing architecture: restricted Python source constructs native MLIR directly; semantic Swage IR describes logical segments; SwagePlan describes physical tasks; upstream MLIR/LLVM emits code; compiler-generated contracts describe kernel invocation. Runtime segment identity stays in SSA values. A variable-length segment remains a symbolic range that fixed-size work streams through. There is no reason here to add a Python expression IR, runtime-sized register array, or Triton production backend.

The distinction worth testing is **who derives the schedule**. A Swage caller should describe an element expression and its segment-local dependencies; the compiler should determine which decompositions preserve that meaning. Triton already has runtime loops, explicit device scheduling and configuration tuning. Swage cannot distinguish itself by claiming variable-length execution alone. [Triton loop API](https://triton-lang.org/main/python-api/generated/triton.language.range.html), [grouped GEMM device scheduling](https://triton-lang.org/main/getting-started/tutorials/08-grouped-gemm.html).

Separating semantics from scheduling also has substantial prior art. MLIR's Transform dialect controls transformations separately from the payload; Futhark's incremental flattening selects among mappings of nested parallelism. Its July 2026 account describes work toward nonuniform flattening and explicitly separates expressibility from performance; this report does not assert the current release status of that work. These are design references, not new dependencies or proof that Swage is redundant. [MLIR tutorial](https://mlir.llvm.org/docs/Tutorials/transform/), [Futhark paper](https://futhark-lang.org/publications/ppopp19.pdf), [Futhark development account](https://futhark-lang.org/blog/2026-07-31-full-flattening.html).

## 2. Design findings, ordered by consequence

| Finding | Evidence in current code | Consequence and proposed response |
|---|---|---|
| Public programmability ends before segment execution. | `python/swage/__init__.py:6-18` exposes fixed segmented calls; `language.py:76-99` has no segment operations. `docs/internals/compiler-pipeline.md:24-28` states the boundary. | Expose caller expressions through existing native operations, preserving source diagnostics. Adding more named built-ins alone does not test the compiler idea. |
| A valid semantic program does not imply every schedule is legal. | `lib/Conversion/SwageToPlan/Admission.cpp:491` admits earlier reduction captures; `:674` limits split planning to one capture-free reduction. | Return admitted schedule families as native program facts. Captured normalization uses an explicitly selected direct schedule. Do not discover policy by catching compiler failures. |
| Private textual metadata extraction is unsuitable for public caller programs. | `_segmented_programs.py:47-79` uses regexes for element type and roles. | Query the selected verified native function. Reuse existing tensor checks and kernel contracts; add only metadata required by the first program family. |
| Layout work and buffer execution are coupled. | `_segmented_validation.py:250` reads offsets to the CPU on each public call; `_segmented_qualification.py:847` validates, classifies, uploads records and binds execution. | Separate immutable program facts, owned layout facts and per-execution state. Measure fresh and reused layouts independently. |
| Version counters are an incomplete ownership contract. | `_segmented_validation.py:335-351` explicitly excludes `.data`, foreign writes, external kernels and graph replay. | Own the validated offsets snapshot used by the device. Retain storage and scratch until work completes. Exposing the private prepared helper unchanged is insufficient. |
| Input and output cardinality limits real programs. | `Admission.h:33` has one values source; `Admission.cpp:466` requires one output terminal; `:477` requires single-consumer maps. | Weighted pooling needs genuine aligned-input support. Multiple outputs, arbitrary scalars, fan-out and generic combiners are separate extensions, not incidental syntax work. |
| Kernel geometry still has duplicated knowledge. | NVIDIA subgroup width is fixed at `NVIDIATarget.cpp:44`; artifact validation at `_artifact.py:547` accepts other divisors; row grouping also appears in native and Python code. | Immediately require the supported NVIDIA width. As mappings expand, bind a small versioned work-mapping descriptor to the compiled entry, rather than a general formula interpreter. |
| Numerical and oracle guarantees need precision. | Current sum documentation promises bounded error without intermediate-overflow conditions; CPU/GPU consumer patterns are shared. | Specify legal reduction trees and assumptions separately from error bounds. Keep independent references and adversarial numerical cases. |

The geometry issue is a **confirmed conditional correctness defect**, not merely design preference. In the preceding review, changing only an artifact's subgroup width from 32 to 16 was accepted and made eight singleton reductions write only four outputs. The current writer emits 32; ordinary generated artifacts were not shown to fail. Fix validation and its execution regression before broadening artifact consumption. The [artifact reproducer](https://github.com/abhiksark/swage/blob/e43af738844a8034c0d3650d314b278d2788ae32/benchmarks/results/segment-design-20261104-98b17f3/artifact-subgroup-repro.py) compiles a temporary artifact and prints intact and mutated results; it intentionally demonstrates the defect in the reviewed revision.

The numerical issue is also concrete: finite f32/f64 inputs `[M, M, -M, -M]`, with `M` the largest finite value, produced NaN under the tested reduction tree; the reordered input `[M, -M, M, -M]` produced zero. Both have real sum zero. An unconditional finite error bound cannot cover that. This does not by itself invalidate an order-free floating-point reduction contract. The documentation must state its assumptions. `DESIGN.md:187-190` at the reviewed revision also incorrectly said the public segment calls were absent from main; this change corrects that bullet.

## 3. What the measurements establish

The [recorded Swage/Triton comparison](https://github.com/abhiksark/swage/blob/e43af738844a8034c0d3650d314b278d2788ae32/benchmarks/results/triton-engineering-a6000-sm86-98b17f3/README.md) used three independent processes per configuration, 30 samples per case, exact-checkable f32 values and an NVIDIA RTX A6000 (`sm_86`). It measured installed Triton 3.7.0 and PyTorch 2.12.0+cu130.

| Boundary | Result against the fixed Triton baseline | Interpretation |
|---|---|---|
| Fresh public calls, rank one | 0 wins / 9 cases; Swage/Triton ratios 1.97 to 7.80 | The tested public path does not make the kernel capability competitive here. |
| Fresh public calls, feature rows | 0 wins / 8 cases; ratios 1.18 to 6.56 | Wider rows narrow some gaps, but do not reverse this result. |
| Prepared rank-one kernels, graph replay | 4 wins / 6 cases; 3 wins / 6 against the fastest post-hoc sweep configuration | Some kernels are competitive once setup is excluded. This does not establish amortized benefit. |

These boundaries differ. Do not subtract a prepared graph time from fresh wall time to call the difference validation overhead. The comparison is synthetic, local engineering evidence, not release qualification or an end-to-end application result. The prepared driver wrote measurements and then failed a provenance type check; serialized child records were revalidated and summaries recovered. The failure is retained and the campaign gate did not pass. The supporting benchmark records remain in the linked evidence commit, outside this design-review diff. Their measured source revision is independent of the commit that archives them.

A new [diagnostic profile](https://github.com/abhiksark/swage/blob/e43af738844a8034c0d3650d314b278d2788ae32/benchmarks/results/segment-design-20261104-98b17f3/public-call-profile.json), with [reproducer](https://github.com/abhiksark/swage/blob/e43af738844a8034c0d3650d314b278d2788ae32/benchmarks/results/segment-design-20261104-98b17f3/profile_public_call.py), tested 8,192 segments of length 16 with exact quarter-valued inputs. Fifty unprofiled synchronized calls had a **111.955 microsecond median**. One hundred cProfile-instrumented calls took 19.804 milliseconds. Across those 100 calls, offset/shape validation occupied 5.315 ms cumulatively, `torch.tensor` 2.981 ms, tensor CPU transfer 2.091 ms, and binding 2.512 ms cumulatively. Correctness assertions passed.

Those spans overlap, include waits and suffer profiler overhead. They locate repeated work; they do not apportion a speedup. They argue against treating the host copy as the sole problem or rewriting the whole runtime in C++ without controlled ablations.

## 4. Three credible routes

| Route | Concrete result | Main investment | Strongest objection | Choose it when |
|---|---|---|---|---|
| **A. Expose bounded programs first** | Caller-written transformed reductions and captured normalization, with existing direct/split families where legal | Source emission, native program facts, checked execution and optional owned-layout reuse | Single-input inference programs may still be too narrow; direct dependent stages may be slow | You want early evidence that the compiler abstraction is useful. Recommended experiment. |
| **B. Add aligned inputs before stabilizing the interface** | A plus weighted pooling over two same-shape buffers sharing one layout | Operand/region schema, fusion, ABI, binding and independent qualification | A larger semantic change delays feedback; initial no-broadcast rule limits common `[N,D]` plus `[N]` weights | A real application already requires aligned multi-input composition. |
| **C. Prioritize fresh-layout execution** | Compiled caller programs with a consolidated checked one-shot dispatch path | Native validation/binding transaction; explicit direct policy; producer-consumer profiling | It retains a host decision for eager errors; direct kernels can lose on long tails | Offsets change on nearly every call and layout reuse is rare. |

The three supporting reviews use their own labels for the same choices:

| Route above | Compiler review | Runtime review | Strategy review |
|---|---|---|---|
| A. Expose bounded programs first | Option 1 | Alternative A | B |
| B. Add aligned inputs first | Option 2 (its weighted-pooling part) | Not covered | Weighted-pooling-first alternative |
| C. Prioritize fresh-layout execution | Not covered | Alternative B | C, fresh-call-first alternative |
| Fixed operators only (rejected above) | Not covered | Not covered | A |

These routes can share components, but their first deliverables differ. A fixed-operator-only plan is not a fourth recommendation: it would not meet the selected priority. A general task DAG, persistent queue, arbitrary combiners, automatic differentiation and tensor-language expansion are not prerequisites for any of the three.

The reviews' strongest challenge to A is utility, not feasibility: sum of squares already has straightforward Triton and transformed-reduction library implementations. The abstraction must earn its place by admitting new bodies without compiler edits, preserving semantics, and reducing user scheduling work at an acceptable measured cost. Another objection is API ceremony. A convenient checked call should remain available; explicit layout and execution objects belong to callers who need reuse and predictable lifetimes.

## 5. Programs that make the choice tangible

The following is **illustrative source meaning, not an implemented Python API**. No new decorator or final spelling has been selected.

```text
energy(segment):
    return sum(map(segment, x -> x * x))

rational_pool(segment):
    return sum(map(segment, x -> x*x / (1 + x*x)))

normalize(segment):
    total = sum(segment)
    return map(segment, x -> x / total)

weighted_pool(values_segment, weights_segment):
    return sum(zip_map(values_segment, weights_segment, (x, w) -> x*w))
```

| Program | Current native foundation | Proposed public work | Legal first execution |
|---|---|---|---|
| Energy and rational pooling | Existing arithmetic regions and capture-free reduction | Restricted AST emission and checked bindings | Direct or existing split; merge combines partial results without reapplying the element expression |
| Normalize by sum | Existing reduction capture plus map-store | Source captures, output extent, diagnostics | Direct fused program; zero denominator follows division semantics, no implicit epsilon |
| Stable softmax | Existing f32 two-reduction/capture structure | Caller-written expression, approximation documentation | Existing direct fused program; split dependencies are additional work |
| Weighted pooling | Missing second aligned source | Extend native operands, region verification, map fusion and contracts | Start with two equal-shape/equal-dtype inputs and identical offset/segment coordinates |

The compiler review ran a new [native feasibility check](compiler-native-check.py) with arbitrary names, proving this is not name-based built-in dispatch. `caller_energy([1,3,2,6], [0,2,2,4])` produced `[10,0,40]` on the CPU oracle and compiled direct/partial/merge PTX for `sm_86`. `caller_normalize` produced `[.25,.75,.25,.75]` and compiled direct PTX; split-partial compilation correctly rejected the dependency. Source IR remained unchanged. This establishes native construction, CPU behavior and compilation, **not GPU execution or public source support**. [Recorded results](compiler-native-results.json) and the compiler review retain the exact evidence.

Freeze the first advertised subset, then independently write two held-out bodies, for example `max(x*x/(1+x*x))` and `sum(x-max(x))`. They must work without changing production admission, kernel-name tables or runtime dispatch. That is the key abstraction test.

## 6. Ownership that makes reuse safe

```text
caller source -> verified semantic MLIR -> compiled program
                                             |
device offsets -> owned validated layout -----+-- prepare(values, output)
                                                  |
                                        bound execution + scratch
                                                  |
                                         run on its CUDA stream
```

The compiled program owns semantic facts and the admitted kernel family, independently of tensor addresses. The layout owns a validated int32 snapshot and row/segment bounds, independently of program-specific tasks. Preparation combines program, layout, feature width and target to choose tasks, then binds values/output storage and owns scratch and kernel leases. A layout may be reusable even when its task list is not. These are internal ownership boundaries; the first public experiment need not expose three separate objects:

```python
# Proposed interface, not implemented.
program.run(values=values, offsets=offsets, out=output)
execution = program.prepare(values=values, offsets=offsets, out=output)
execution.run()
```

The prepared execution owns its layout snapshot internally. A separate public layout handle becomes useful if callers need cross-program sharing.

Subsequent mutation of the caller's original offsets does not change the layout. Mutation concurrent with snapshotting remains an unordered input race. The kernel reads the exact snapshot that was validated, including when the original offsets were int32. Keeping only a version number cannot establish that property.

For the first experiment, bind execution to one stream and use separate scratch for concurrent bindings. Permit value-content changes ordered on that stream; refuse storage rebinding. Retain allocation lifetimes through completion and handle failure cleanup. Start with eager execution. Managed graph replay is a later qualified boundary because external graph replay does not run Python guards. A graph handle must own storage and execution state; merely capturing a launch is insufficient.

The proposal needs native selected-function facts now. A broad new version of every buffer contract is not automatically a prerequisite. Reuse the existing checked tensor path for the bounded one-input experiment. Extend descriptors only where output categories, multi-input operands or stage buffers cannot be represented safely, and update all producers/consumers together. Do not replace regex metadata with another parallel Python semantic model.

Arbitrary programs must reject gradient recording until their derivatives are implemented and tested. The existing sum backward is wrong for `sum(x*x)`. Existing fixed operators retain their gradient behavior. Custom-operator registration is a later integration option; it does not derive derivatives or remove validation synchronization. Registration checks and gradient checks answer different questions. [PyTorch custom-operator guidance](https://docs.pytorch.org/tutorials/advanced/custom_ops_landing_page.html), [autograd registration](https://docs.pytorch.org/tutorials/advanced/python_custom_ops_registrations.html).

## 7. Two choices that cannot be hidden by implementation

**Fresh device offsets and immediate errors.** The host cannot establish validity of arbitrary device data without observing completed device work. Moving validation to the GPU can reduce readback volume, but preserving an immediate exception before numerical execution still requires a host decision. The honest initial choices are checked one-shot calls or already validated owned layouts. An asynchronous status/result contract is a different public API requiring bounds-safe execution and a defined error-observation point. It is not a transparent optimization.

**Dependent stages and splitting.** Direct normalization keeps completed summaries within one block. A split softmax must make full-segment summaries available to later chunks. A straightforward design needs partial maxima, merged maxima, partial exponent sums using those maxima, merged denominators, and the final normalization. Chunk-local maxima followed by naive denominator addition are wrong. Summary buffers require segment identity, feature indexing, producer order, lifetime and bounded scratch. Keep that orchestration in MLIR or a fixed compiler-declared template, not a Python dependency evaluator. A five-kernel implementation can lose to the current direct kernel; add it only for a workload that repays the cost.

## 8. Experiments that should decide the design

The following are **proposed acceptance rules**, not measured outcomes or existing release gates. The six rules below are the set to freeze; the runtime review's 20%/5% stop rules were its working thresholds and do not form a second gate set. The owner may choose a different acceptable performance premium before any candidate is timed.

1. **Programmability:** energy/rational pooling, normalization and caller softmax compile through one semantic path. Two held-out bodies require no production edits. Unsupported side effects, captures, types and gradient use fail with source locations.
2. **Correctness:** compare exact-representable fixtures and independent float64/PyTorch references; cover empty ranges, partial tiles, long tails, cancellation, intermediate overflow, signed zero, NaNs/infinities and expression error. Shared CPU/GPU emission is not the sole oracle. Run racecheck and lifetime/error-path tests for any new asynchronous state.
3. **Kernel usefulness:** compare equivalent manually fused Triton programs, matching output ownership, dtype and numerical policy. A candidate criterion is warm geometric-mean Swage/Triton at most 1.25, with no cell above 2.0. That explicitly permits some cost for a simpler programming model; it does not claim faster kernels.
4. **Planning value:** compare the same Swage body with the same wrapper, allocation and timing boundary under direct and planned policies. Require at least 10% total improvement on a predeclared target family. An offline best schedule may evaluate the selector; it cannot choose the deployed winner for free.
5. **Reuse value:** measure compilation separately, then layout creation, binding and `K=1,2,4,8,16,64,256` uses. Compare `P_swage + K*E_swage` with `P_baseline + K*E_baseline`, including equivalent preallocation and teardown. Adopt reuse only when measured break-even is below actual application reuse and the application region improves meaningfully.
6. **Fresh-layout value:** time the actual producer, compiled segment program and consumer on their streams. Compare checked wrappers to isolate kernels and real public APIs to assess usability. A custom-op graph node that still reads offsets on the host does not eliminate that dependency.

Use memory-admitted synthetic strata for short, mixed and long-tail segments, scalar and feature rows, lengths around 32/4096/8192 and batch counts around device SM count. Reserve held-out seeds and include at least one actual application. Keep compilation, validation/transfer, classification/upload, scratch bytes, eager dispatch, device time and whole-region time separate. Use multiple processes and fixed or predeclared tuning policies. Another GPU is required before a portability claim. CUB transformed segmented reduction is a relevant additional baseline for one-stage expressions, subject to matching its numeric contract; it was not measured here.

## 9. Suggested sequence for the owner's judgment

First repair artifact geometry validation and the numerical/status documentation, with focused regression evidence. These are correctness and claim-maintenance issues independent of the chosen route.

Then make a private end-to-end experiment for caller-written programs: direct AST-to-MLIR, native program facts, a checked convenience call and an owned-layout preparation path. Keep the first execution slice bounded and reject unsupported gradients. Demonstrate the two held-out bodies and publish matched fused comparisons before stabilizing public spelling.

After that experiment, choose the next extension from observed need: aligned inputs for weighted pooling, a fresh-layout dispatch improvement, or split dependent stages. Do not make all three prerequisites. Do not invest further in a persistent queue until a profile identifies a problem the simpler legal schedules cannot address.

The unresolved product input is a concrete application and its layout-reuse pattern. That does not block the bounded experiment. It does prevent a defensible claim that reusable layouts, asynchronous validation or a particular split policy should dominate the final API.

## Evidence and review scope

This PR keeps the design report, three independent reviews, the interactive showcase and the small native feasibility check with its results. It changes no compiler/runtime behavior and adopts no proposed API. The complete benchmark records, failure logs, profiles, screenshots and browser checker remain in the [evidence commit](https://github.com/abhiksark/swage/tree/e43af738844a8034c0d3650d314b278d2788ae32/benchmarks/results), outside the design-review diff. The measured source is still `98b17f3`; archiving a measurement does not qualify a later compiler revision.

The numerical and performance conclusions above retain their original limits. Fresh and prepared timings are separate; the prepared campaign failed after writing data, and recovered summaries are not a successful campaign gate. Raw evidence is linked rather than deleted from history.

During the investigation, the CPU examples, four PTX compilations, expected split refusal and source-preservation assertions passed. The public-call diagnostic and artifact reproducer passed their assertions on NVIDIA RTX A6000, `sm_86`, driver 580.178.04. The original review also passed 1,686 frontend, 124 lit, 93 C++ unit and 5,301 native Python/CUDA tests, with the two opt-in racecheck tests passed separately. Those checks describe the existing implementation, not the proposed designs. Full suites and the complete campaign were not rerun to narrow this PR.

Re-run the retained native feasibility check from the repository root with matching build-tree bindings:

```sh
record=benchmarks/results/segment-design-20261104-98b17f3
PYTHONPATH=build/python_packages:python:python/tests/mlir python \
  "$record/compiler-native-check.py" --output-dir /tmp/swage-native-recheck
```

The output directory must not exist. Set `SWAGE_ORACLE_BUILD_DIR` when the native build is outside this checkout. This executes the CPU oracle and compiles `sm_86` PTX; it launches no GPU kernel. Its recorded results do not establish a public Python segment frontend.

Serve the repository with `python -m http.server 8000` and open this directory's `showcase.html` to use the comparison. GitHub displays HTML source, not the live interactions. Supporting reproducers and original verification artifacts are available in the linked evidence commit.

Follow-up work remains the owner's decision; the artifact-geometry defect is tracked in #99. The main unresolved risk is whether real application compositions justify the language and runtime cost. Implementation, arbitrary-program derivatives, proposed performance gains, whole-application behavior and other architectures remain unverified.
