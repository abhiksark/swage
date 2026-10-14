<!-- benchmarks/results/fixed-dispatch-a6000-sm86-3d0fe92.md -->

# Fixed CUDA dispatch proposal qualification

The method and entry-slot shortcuts were rejected. Across 20 independent
baseline/candidate process pairs each, their paired median warm-host gains
were 1.80% and 0.90%, below the predeclared 5% retention threshold. The
production dispatch implementation and frozen benchmark gates are unchanged.
The characterization tests remain useful for preserving lookup, ownership,
error ordering, and output-version behavior.

## Baseline and independent candidates

The baseline is `3d0fe92c1046c7e58a0bd074f3cfa220077eb9dc`, the reviewed
segmented integration candidate. Every timed process used a clean Release
wheel installed outside the checkout, with `PYTHONPATH` unset. The environment
was regular-GIL CPython 3.13.13, PyTorch 2.12.0+cu130, LLVM/MLIR 22.1.8,
driver 580.178.04, and NVIDIA RTX A6000 / `sm_86`.

- Method candidate `0cc4813e1d76d74045a7136a8176d58a9b40911a` calls exact
  captured keyword descriptors directly and retains their argument ownership.
- Slot candidate `379f2a57f5b6957af2c1fd4a7a50937ea5b9a188` reads validated
  object slots with owned references and preserves deleted-slot exceptions.
- The candidates were measured separately. Their complete patches are retained
  in the adjacent data directory. Neither patch is production code.

The prepared bound-callable proposal was rejected before timing: a real CUDA
characterization showed that it bypasses a later native launcher class
`__call__` override. Ordinary dispatch raised before writing or advancing the
output version; the captured bound callable still wrote and advanced it once.
The capture-query hunk was omitted because the baseline already handles
unavailable symbols and query errors more conservatively.

## Criteria and results

The criteria were recorded before candidate timing: at least 5% paired median
warm-host improvement, a 95% paired-bootstrap confidence interval wholly above
zero, no material regression above 5% in p95, cold-start, throughput ratios or
RSS, and every record within the unchanged frozen gates. There are 20 pairs
per candidate, with baseline-first and candidate-first order alternating.
Bootstrap intervals use 20,000 paired resamples and seed 20261014.

| Candidate | Pairs | Paired median warm gain | 95% bootstrap interval | Frozen gates, baseline / candidate | Decision |
|---|---:|---:|---:|---:|---|
| Direct methods | 20 | 1.80% | 0.77% to 2.58% | 20/20 / 18/20 | Reject |
| Entry slots | 20 | 0.90% | 0.06% to 1.15% | 20/20 / 20/20 | Reject |

The two method-candidate failures were cold maxima of 534.77 ms in pair 5 and
431.09 ms in pair 12, above the frozen 400 ms maximum. Their cold medians were
35.23 ms and 35.57 ms. The cause of those outliers was not established. No run
was removed, replaced, or used to retune a gate. The warm-gain criterion alone
also rejects both shortcuts.

All 80 process records remain valid and hardware-qualified; those flags do
not imply a timing pass or release qualification. All correctness preflights
passed. Median paired p95, cold, ratio, and RSS changes had no material
regression under the declared rule. A gain here concerns only the frozen
float32 fixed vector-add workload, not other dtypes, operations or devices.

## Correctness and ownership evidence

Before production candidate edits, the current baseline passed eight focused
method/ownership cases and ten slot cases. The separate candidates passed
5,290 and 5,295 native tests, respectively, with two opt-in skips, plus
124 lit tests and 93 C++ unit tests each. Each clean installed candidate wheel
passed CUDA add and multiply for float32, float16 and both FP8 formats.

The retained characterization cases cover allocator class mutation during
three post-enqueue records, contiguity class/instance overrides, native stream
reference-cycle collection, eight deleted residency slots, descriptor
replacement, and deletion at a controlled pre-enqueue stream-query callback.
Ordinary Python cleanup conservatively advances the output version after an
attempted launch even when function lookup raises; native pre-enqueue
refusal leaves it unchanged. The tests preserve that distinction.

The candidate keyword signatures and all eight entry-slot owners, flags,
types, offsets and alignment were inspected on the actual qualified ABI.
Existing guards and broader lifetime tests passed. Instrumented Python-binding
sanitizer qualification was not run: both production shortcuts were rejected
before final acceptance. Hosted sanitizer jobs instrument compiler targets
with Python bindings disabled; they do not establish binding sanitizer proof.

## Retained data and reproduction

The adjacent [data directory](fixed-dispatch-a6000-sm86-3d0fe92/) holds every
compressed raw JSON record, manifests with uncompressed SHA-256 values,
criteria, decisions, the two rejected patches and the callable probe. The
records' environment paths identify the actual local artifacts; manifests
use relative record paths for relocation. Failed records remain included.

Run `analyze.py` in that directory to verify hashes and recompute decisions
from all records. To reproduce a candidate, start from the recorded baseline,
apply only its patch, commit that source locally, build a clean Release wheel
using the contribution instructions, and install into a separate environment.
Run the unchanged `python -m swage.bench vector-add --output PATH --enforce`
outside the checkout for 20 alternating pairs, preserving every result.
A new source commit or environment gets a new campaign; do not overwrite this
record. The callable probe also runs in a fresh installed-wheel process.

These are local engineering results, not release publication or qualification
of unmeasured GPUs and Python ABIs. The rejected prototype branches and the
original dirty proposals remain preserved locally.
