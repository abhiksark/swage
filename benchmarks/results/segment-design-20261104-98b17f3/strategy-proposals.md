<!-- benchmarks/results/segment-design-20261104-98b17f3/strategy-proposals.md -->

# Strategy proposals for a programmable segment compiler

Swage should pursue a useful programmable segment compiler, with composition
as its first proof of value and scheduling as an optimization that must earn
its cost. The owner has selected that objective. The proposals below are
independent review recommendations, not an implementation decision. The owner
remains the final judge of scope, numerical guarantees, and acceptance gates.

The recommended direction is B below. It can use a direct execution path
from C and explicit reuse from A without turning either into the project's
purpose. A faster implementation of one more built-in reduction is a useful
control experiment, but it would not establish a programmable compiler.

**The evidence supports a compiler experiment, with a narrow claim.**
The [current local comparison](https://github.com/abhiksark/swage/blob/e43af738844a8034c0d3650d314b278d2788ae32/benchmarks/results/triton-engineering-a6000-sm86-98b17f3/README.md)
at source `98b17f3` reports losses to a fixed Triton configuration in all
17 fresh-offset cases, while the prepared mixed kernel wins four of six
cases against a fixed configuration and three of six against the best
configuration selected after a sweep. These are three-process local A6000
measurements with finite float32 inputs, not release qualification. The
prepared campaign needed recovery from a provenance-validation defect;
serialized records passed the existing validator. The measured comparison
uses PyTorch 2.12.0 and Triton 3.7.0.

These observations support two different statements: useful kernels exist,
and the current public execution path does not make them competitive on the
tested fresh layouts. They do not isolate validation, classification, Python
dispatch, allocation, copies, or kernel geometry as the cause. In particular,
subtracting a prepared CUDA-graph time from a fresh synchronized wall time
would not measure preparation cost. The earlier
[composition record](../../../benchmarks/results/composable-reductions-a6000-sm86.md)
shows fused transforms and reductions, but its competing PyTorch path uses
eager transforms and allocations. It explicitly does not compare against an
equally fused competing kernel.

The subsequent [local diagnostic profile](https://github.com/abhiksark/swage/blob/e43af738844a8034c0d3650d314b278d2788ae32/benchmarks/results/segment-design-20261104-98b17f3/public-call-profile.json) adds
evidence of repeated validation, task tensor creation, and launch-contract
binding alongside the offset copy. It profiles 100 calls on 8,192 segments
of length 16. Its cumulative spans overlap, include waits, and are perturbed
by cProfile. It supports investigating the whole orchestration path; it does
not justify assigning every lost microsecond to host readback or predicting
the speedup from moving a helper to C++.

The [persistent-queue experiment](../../../docs/adr/ADR-0018-private-persistent-task-queue.md)
is a useful warning against choosing machinery before a workload. After two
correctness defects invalidated earlier results, the corrected experiment
improved 1.06%, short of its frozen 5% gate. That establishes a failed
experiment on one workload, not that persistent execution can never help.
It gives no reason to make a queue a prerequisite for programmability.

**The comparison with other systems sets the ownership boundary.**
Triton already supports runtime loops and explicit device scheduling. Its
[grouped GEMM tutorial](https://triton-lang.org/main/getting-started/tutorials/08-grouped-gemm.html)
walks variable problem sizes inside a fixed launch. Its
[loop API](https://triton-lang.org/main/python-api/generated/triton.language.range.html)
and [autotuner](https://triton-lang.org/main/python-api/generated/triton.autotune.html)
let authors control loops and evaluate supplied configurations. Swage's
opportunity is to derive legal implementations from segment-local source,
reducing the amount of schedule code a caller writes. Variable-length loops
themselves are not the distinction.

Simple transformed reductions also have library alternatives. NVIDIA's
[DeviceSegmentedReduce implementation](https://github.com/NVIDIA/cccl/blob/main/cub/cub/device/device_segmented_reduce.cuh)
takes device offsets, input iterators, reduction functions, and temporary
storage. Its [transform iterator](https://nvidia.github.io/cccl/unstable/libcudacxx/api/classcuda_1_1transform__iterator.html)
applies an expression when an element is read. An iterator-based transformed
reduction is therefore a relevant candidate baseline where its API and
numeric behavior match. It is not a measured baseline in the current record,
and no dependency or implementation code should be imported into Swage to
make this comparison.

The research question also has substantial prior work beyond Triton.
[Incremental Flattening for Nested Data Parallelism](https://futhark-lang.org/publications/ppopp19.pdf)
generates alternative mappings of regular nested parallelism and chooses
between them using tuned predicates. The Futhark authors' July 2026
[full-flattening account](https://www.futhark-lang.org/blog/2026-07-31-full-flattening.html)
discusses segmented representations and nonuniform intermediate arrays; it
describes work approaching merge at that time and explicitly separates
expressibility from good performance. Neither source proves Swage redundant,
but they rule out a broad novelty claim about separating semantics from
hardware mapping or flattening irregular work.

Framework integration is another existing capability to reuse. PyTorch's
[custom-operator guidance](https://docs.pytorch.org/tutorials/advanced/custom_ops_landing_page.html)
provides registration for compiler and autograd integration. Its
[Triton integration recipe](https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html)
explains that a `triton_op` can be traced through, while a `custom_op` is
opaque to `torch.compile`. A Swage native custom operator could avoid the
current graph break, but registration alone neither removes host copies nor
lets Inductor fuse through an opaque native body. Current online documents
describe newer development versions; a proposal must verify the chosen API
on the project's supported PyTorch floor.

Swage should own segment semantics, program legality, dependence analysis,
fusion and decomposition choices, and the numerical contract of those
choices. Upstream MLIR/LLVM should continue to own ordinary arithmetic and
backend lowering. The existing CUDA runtime and compiler-generated launch
contracts should own invocation. Triton and CUB should be external comparison
implementations. Adding Triton as a production backend conflicts with the
current repository rule and is not necessary to test the compiler idea.

| Option | Workload assumption | What it would deliver | What it sacrifices | Reason to reject it |
|---|---|---|---|---|
| A. Reliable operators with explicit reusable plans | Offsets, geometry, and selected program repeat for enough calls to amortize preparation | Validated ownership of layout and scratch, repeated launches, explicit lifecycle | Fresh layouts still pay setup; fixed operators do not give callers programmable composition | Reject as the main direction because the owner's goal is programmability. Retain only if real applications reuse layouts. |
| B. Programmable segment-local composition | Callers need fused element work and dependencies between reductions and stores | One source program, compiler-derived legal execution, useful diagnostics, competitive compiled kernels | A bounded language, explicit unsupported programs, no promise of automatic differentiation or every schedule for every program | Reject the proposed surface if new examples still require named special cases throughout native code or only beat unfused baselines. |
| C. Direct device execution for fresh layouts | Layouts change frequently and the dominant useful programs can run directly | Runtime loops over segments with no mandatory host task classification; later selective decomposition | Direct execution may leave long tails; eliminating eager content checks changes the validation contract | Reject an asynchronous variant if eager validation before any compute enqueue is non-negotiable and inputs are arbitrary device tensors. |

**A would need ownership semantics before becoming public.**
The [private prepared path](../../../docs/internals/task-execution.md) currently
binds values, offsets, and output storage. Its version-counter check misses
raw-pointer and other untracked writes; graph replay runs no host check.
Those facts make merely exporting the helper an inadequate public design.
The smallest stronger contract would own a validated offsets snapshot and
fixed metadata, retain the existing bound tensor storage, and state whether
the handle serializes overlapping uses or requires one handle per concurrent
execution. Split scratch must not race across streams or graph replays.
Changing value contents would remain allowed; changing the layout would
require a new handle. General rebinding is a separate requirement.

Measure total cost for K uses of the same layout, including creation and
teardown. If `P` is preparation cost and `L` is per-use cost, the claim must
compare `P_swage + K*L_swage` with `P_baseline + K*L_baseline`. A break-even
reuse count exists only if Swage's repeated cost is lower. No value of K is
established by the current prepared-only result. Both implementations get
equivalent preallocation and reuse. A plan whose benefit depends on reuse
absent from the application is not a useful optimization for that application.

**B should begin by exposing existing composition, then extending one real
missing pattern.** The private compiler already has the important pieces:
isolated element regions, ordered captures of preceding reduction results,
lazy maps, and segment-local map stores. The
[admission code](../../../lib/Conversion/SwageToPlan/Admission.cpp), lines
64-117 and 491-508, bounds the operations and captures. Lines 674-688 show
that the split-planning subset is narrower: exactly one capture-free
reduction. A useful first compiler must tell users which program is legal
and which schedules are legal for it, without making every valid program
qualify for every schedule.

These are candidate programs, not proposed final syntax. Each formula acts
within one segment; `eps` is a positive compile-time constant.

| Candidate | Meaning | Existing foundation | What it discriminates |
|---|---|---|---|
| Control. Identity sum | `sum(x)` | Existing public operator and every static reduction schedule | Control for runtime and schedule effects; not evidence of programmability |
| P1. Rational pooling | `sum(x*x / (1 + x*x))` | Admitted multiplication, addition, division, and single reduction | Caller-written arithmetic must flow through every legal schedule without a kernel per expression |
| P2. Stable softmax | `m=max(x); z=sum(exp2((x-m)*log2(e))); y=exp2((x-m)*log2(e))/z` | Existing reduction captures and map-store path | A caller can express the same dependency structure as the fixed operator; no body-name dispatch |
| P3. L1 normalization | `a=sum(max(x, -x)); y=x/(eps+a)` | Existing arithmetic, captured reduction, and map store | An independent two-stage program should use the same machinery as softmax, without adding an algorithm-specific lowering |
| P4. Weighted pooling | `sum(w*x)` over two aligned rank-one segments | New aligned input binding and element access would be required | A useful first expressiveness extension beyond a single values buffer; equal offsets and lengths must be verified |

P1-P3 are anchored in current native capabilities; public source capture and
execution of those user-written bodies still need work. P4 is deliberately
marked new. It should wait for an application that needs it, rather than
becoming an excuse for a general tensor language. After implementation is
frozen, independently write two held-out bodies using the advertised subset,
such as `max(x*x/(1+x*x))` and a reduction of `x-max(x)`. Both must compile
without changes to production admission, scheduling, runtime dispatch, or
artifact tables. Diagnostics tests are allowed; new named-kernel cases fail
the abstraction test.

The comparison for P1 must include a manual fused Triton loop evaluating the
same expression. P2 and P3 need equivalent direct fused Triton programs, plus
multikernel variants if Swage decomposes a long segment. P4 needs an equally
fused two-input baseline. Eager PyTorch compositions remain useful references
and user baselines, but a win over extra memory passes alone does not
establish a scheduling advantage. A CUB iterator-based comparison is useful
for P1 if the installed toolchain supports the matching contract.

The first public programmable slice can be explicitly inference-only if that
is the smallest complete contract. It must refuse gradient recording rather
than silently cut it. Existing fixed operators retain their current gradient
behavior. Automatically differentiating arbitrary user bodies is a distinct
compiler problem; the existing hand-authored backward functions do not solve
it. A later training milestone needs a real composed workload and a matched
forward-plus-backward baseline.

The interface test is concrete: adding P3 after P2 should reuse the same
source emitter, region verification, capture analysis, store lowering, and
launch binding. Existing MLIR SSA already expresses dependence. A second
Python program graph, a `SoftmaxStage` operation, a generic plugin registry,
or an abstract cost interface that still has only one implementation would
not improve this test. Extend the existing analysis only where two actual
consumers need the same fact. The
[compiler-generated launch contract](../../../docs/adr/ADR-0025-compiler-generated-kernel-contracts.md)
already prevents each new signature from requiring another positional
runtime launcher. This is a place to reuse an abstraction that has an actual
second consumer.

**C changes where costs occur, and may require a user-visible contract
decision.** Direct execution can avoid host classification without changing
the semantic IR. It is an admitted plan containing direct work, not an
escape hatch around planning verification. This gives B a simple reference
implementation for every admitted program before complex decomposition is
attempted. Long dependent stages may still need multiple passes over a
segment; removing an intermediate allocation does not imply reading values
only once.

For arbitrary fresh offsets that exist only on the GPU, the host cannot
know that their contents are valid before launch without receiving a
completed device result. Moving validation to a device kernel therefore
does not preserve the present promise of raising on bad offsets before
enqueuing compute while also removing synchronization. Three honest choices
are: keep eager validation; accept an already validated owned layout; or
offer an explicitly different asynchronous error contract. The third choice
requires bounds-safe execution, defined output behavior on errors, and a
defined status observation point. An unchecked fast path is not a solution.
Registering a PyTorch custom operator does not resolve this decision.

This constraint means that a direct-kernel experiment can still be useful
under today's checked public API, but it must not promise graph capture or
zero host synchronization. For pipelined fresh layouts, measure the preceding
GPU producer and consumers on their actual stream; an idle-device
microbenchmark misses the serialization that a host read can impose.

**The experiments should separate three questions before selecting more
machinery.** The following gates are proposed judgment thresholds, not
measurements, estimates, or existing release requirements. Freeze them before
new candidate timings. The owner may choose a different performance premium
for simpler source, but that choice should precede the experiment.

| Question | Controlled comparison | Proposed acceptance gate | What failure means |
|---|---|---|---|
| Can users actually program it? | P1-P3 plus two held-out bodies from the documented subset | All compile from user source with one semantic path, correct source diagnostics, and no production changes for the held-out bodies | Rework the source/IR contract before adding schedules |
| Does the compiler produce useful kernels? | Equivalent fused Triton, same dtype/accuracy policy/output ownership; warm kernel and full-call times reported separately | Across the frozen target matrix, warm geometric-mean Swage/Triton ratio at most 1.25 and no cell above 2.0; every result passes correctness | Investigate layout, generated code, or overly broad scope. Do not credit unfused-baseline wins as proof |
| Does planning help? | Same Swage body, runtime checks, allocation policy, and launch boundary, direct versus planned | At least 10% lower total time on a preregistered target family; automatic choice no more than 5% slower than the better measured legal schedule on at least 90% of held-out cells, and never more than 20% slower | Keep direct execution or explicit preparation; do not add a more complex selector on the same evidence |
| Does reuse occur where needed? | Total creation plus K uses for K=1,2,4,8,16,64,256, matched baseline reuse | Measured break-even is below the median actual reuse count of the chosen application, with at least 10% application-region improvement | Do not promote reusable plans as the general execution story |
| Does framework integration help users? | Whole producer-program-consumer region, with a matched registered Triton operator | Correct fullgraph capture for the admitted contract and at least 10% lower region time than the current eager Swage route; report independent Triton ratio | Registration may improve compatibility without proving speed; do not hide a host copy behind a graph node |

The 1.25 and 2.0 limits deliberately permit a measured performance premium
for a simpler programming model. They are candidate usefulness criteria, not
a claim that Swage is faster. The automatic-schedule gate measures regret
against an offline oracle only for evaluation; deployment cannot choose the
oracle's winner for free. Cost of computing runtime selection features belongs
inside its measured call. A persistent-queue experiment should start only if
a profile identifies a launch/dependency tail that the direct and static
split choices cannot address; preserve its old failed gate as historical
evidence and preregister any genuinely different experiment.

Use explicit workload strata: reuse count; offsets originating on host or
device; scalar or feature-row values; sparse or full device occupancy;
mostly short or heavy-tail segment lengths; capture-free or dependent stages;
and inference or training. Initially freeze a memory-admitted subset of the
existing nine distributions, widths 1, 3, 32, and 64, and small and large
segment counts. Include lengths around 32, 4096, and 8192, plus a long-tail
case. Add shape-boundary cases around a device's SM count to expose the
current batch-dependent selector. Use seed 7 for development and held-out
seeds 101 and 202 for evaluation. The frozen graph-degree trace is an
additional length-distribution test; a list of degrees is not an end-to-end
graph-model workload. A second GPU is required before claiming portability.

For each cell retain correctness against an independent reference,
compilation time, layout validation and transfer time, classification and
task-upload time, scratch bytes, warm dispatch time, device time, and total
region time. Compare a common checked wrapper around both generated kernels
to isolate kernel/schedule quality, and also compare each actual public API
to answer the user-facing question. Those are different results. Include
immutable-layout prepared tests for both candidates. Do not compare a
graph-replayed Swage kernel to an uncaptured Python Triton call as a compiler
speed claim. Run multiple independent processes, rotate candidate order,
retain failed cases, and choose configurations before held-out evaluation.

Numerical tests need cancellation and intermediate overflow, signed zeros,
NaNs, infinities, empty segments, partial tiles, and values that make omitted
or repeated elements visible. The current finite-input campaign does not
settle those cases. Different legal summation trees need a defined tolerance
contract rather than forced bit equality; overflow cannot be covered by the
currently unconditional finite error bound. For composed programs, validate
the element expression and its reduction separately so a permissive sum
tolerance does not conceal an incorrect expression or a dropped element.

The strongest recommendation is therefore to make P1-P3 caller-written
programs on the existing semantic and direct lowering path, use equivalent
manual Triton programs to judge usefulness, and extend toward P4 only when
an application needs it. Keep planning optional in the architecture until
its end-to-end gate passes. This gives the compiler something meaningful to
own and a way to reject unproductive complexity.

**Independent objections to the other proposals remain important.**
The [compiler proposal](compiler-proposals.md) and
[runtime proposal](runtime-proposals.md) share a reasonable separation of
compiled semantics, owned layout, and execution resources. They should not
be read as evidence that users need three explicit objects plus a four-step
compile/layout/prepare/run ceremony for their first program. Lifetimes are
an internal correctness requirement; public ceremonies are an API choice.
For a caller who needs one fresh computation, requiring a prepared handle
can make the system harder to use than the short Triton kernel it replaces.
One source object with an ordinary checked call and an explicit optional
preparation path is a serious alternative. This does not collapse immutable
program identity into mutable scratch or hide preparation cost in benchmarks.

The strongest objection to my recommended P1-P3 slice is that it may prove
little user value. Rational pooling is a transformed reduction that a library
iterator or a small Triton body can express. Caller-written softmax duplicates
an existing fixed operator. L1 normalization is a better held-out composition
test, but a researcher can still write it directly with little scheduling
code. Inference-only, one-input support should therefore be presented as an
abstraction test, not as the full useful compiler. If the owner's real
workload immediately needs a second input, implementing only that slice
would satisfy an easier question than the one the owner needs answered.

There is consequently a defensible **weighted-pooling-first alternative**:
choose one real caller's `sum(w*x)` over aligned segments as the first
public program, implement the two input bindings and shape/alignment checks
it actually needs, and admit a small family of held-out two-input element
expressions through the same path. Keep rank one, a fixed output per segment,
and ordinary sum composition at first. Test an equally fused Triton loop and
a library iterator baseline where available. This places more risk in native
admission and buffer binding than P1-P3, but it tests a meaningful dataflow
extension rather than a frontend over canned modules. It should win the
choice if an identified application needs weighted pooling now. Do not
generalize it to arbitrary gathers, heterogeneous segment lengths, tuple
reductions, or multidimensional broadcasting without another program.

There is also a defensible **fresh-call-first alternative** for callers whose
layouts change once per operation: add caller-written source and a compiled
program identity, then run a direct admitted kernel through one checked
invocation without exposing reusable layout objects initially. Retain eager
content validation and be candid that this cannot be fully asynchronous.
Reuse native program facts so each call need not rediscover semantic shape
from text. This attacks usability and repeated orchestration, but may still
lose on latency. It should win over a preparation API if measured application
reuse is near one. A native dispatcher without caller-written programs would
not satisfy the owner's compiler objective.

On schema changes, native program facts are necessary now: element types,
argument roles, rank, result shape rule, coverage requirement, and admitted
schedule families must come from verified native IR rather than textual
regexes or names. A trustworthy common binding boundary must check them.
An expanded universal physical buffer-description language is not
automatically necessary for that first source-to-execution proof. If the
existing physical contract plus verified semantic facts can enforce the
bounded API, use them. Expand and version the physical schema only for a
concrete missing invariant shared by actual consumers, and update every
affected parser/artifact path together. An incomplete descriptor that claims
general safety while callers still restate dtype or extent rules is worse
than an explicitly narrow validated interface. Conversely, convenience is
not grounds to retain the currently identified module/tensor mismatch risk.

Finally, separating immutable layout from per-call tensor binding is sound,
but repeated dynamic binding must not recreate today's metadata work in
three new wrappers. Bound storage with separate execution handles is the
smaller initial option if callers can reuse buffers; per-call binding is
needed if the target application supplies new allocations. Scratch and
completion state belong to an invocation or an execution handle, not to a
shared immutable layout. Capture must not be promised until that ownership
model survives concurrent streams, storage rebinding, and failed enqueues.

The stopping criterion for all three entry routes is the same: if a second
real body needs fresh production special cases, if the chosen contract
cannot preserve validation/lifetime guarantees, or if the measured performance
premium exceeds the owner's predeclared allowance, stop expanding the surface
and revise that design. The threshold table offers one allowance; it does
not replace the owner's decision about which application matters.

The biggest unresolved question is which real application needs these
compositions, where its offsets originate, and how often it reuses them.
Without that information, choosing host preparation, asynchronous validation,
split stages, or persistent scheduling would optimize an invented workload.
The present evidence is enough to choose a bounded compiler experiment; it
is insufficient to choose all of those runtime commitments.

This review added only this file. It changed no compiler behavior
and ran no additional GPU workload. Source inspection and primary-source
research support the proposals; their performance gates remain untested.
