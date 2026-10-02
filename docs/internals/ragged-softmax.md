<!-- docs/internals/ragged-softmax.md -->

# Ragged Softmax

Stable ragged softmax executes all phases in one CTA per segment
with fused maps. This page records the exact internal contracts;
none of them is a public API. The public `swage.segment_softmax` launches
this path with the default 128-thread block and requires offsets that cover
every value.

*Qualified on NVIDIA RTX A6000 (`sm_86`); see
[Verification](verification.md) and
[ADR-0013](../adr/ADR-0013-fusion-and-map-store-abi.md).*

The softmax path retains the five-argument ABI defined in
[Segmented Reductions](segmented-reductions.md). It admits ordered f32
reduction captures,
single-consumer map chains, and exactly one scalar store or map-store
terminal. The stable softmax module performs maximum, shifted exponential
sum, and normalize/store phases. Maps are fused into their consumer.

The CPU path executes the phases sequentially. The GPU path executes all
phases in one CTA per segment. Exact-target GPU compilation requires `sm_80`
or newer because every Swage compile request must name a processor from the
admitted list in [Runtime and Environment](../reference/runtime-environment.md).
The floor is not specific to this path: native `exp2` on f32 is legal for
every processor in the pinned NVPTX backend. The backend has no f64 `exp2`,
so planning admission refuses the program over f64 values and
`swage.segment_softmax` refuses float64 values. The runner validates host-visible
offset metadata, capacity, and output disjointness. Empty segments perform no
map-store writes.

<div class="doc-figure" tabindex="0" markdown="1">

![Three block-stride passes per segment separated by uniform all-reduce operations](../assets/figures/ragged-softmax-phases.svg)

</div>

*The one-CTA softmax schedule, with all-reduce as broadcast and phase barrier. [Open the full-size figure](../assets/figures/ragged-softmax-phases.svg).*

## Accuracy

The kernel computes each exponential in f32 as `exp2((v - max) * log2e)`,
with `log2e` the f32 constant `1.44269502`, and lowers `exp2` to the device
instruction `ex2.approx.f32`. The subtraction, the multiplication, the sum,
and the division are IEEE-754 round-to-nearest operations with no
contraction, which a compile-only test checks in the PTX. `ex2.approx.f32`
is the only approximate operation.

The error of an output grows with the distance of its logit below the
segment maximum, `d = max - v`, because rounding the exponent by a relative
amount `t` changes `exp(-d)` by `d * t`. To first order the relative error
of an output is at most

```text
(1.12 * (d + dbar) + 2 * E + 1.5 + k / 2) * eps32
```

where:

- `eps32` is `2**-23`.
- `1.12 * d` covers the rounding of the subtraction and of the
  multiplication, half an `eps32` each, and the f32 constant, which is
  `0.11 * eps32` below `log2(e)`.
- `dbar` is the mean of `d` over the segment weighted by the softmax
  probabilities. It is the error that the normalizer inherits from its
  terms. It is about 1 when the logits are spread evenly, and it exceeds
  neither the spread nor the natural logarithm of the segment length.
- `E` is the relative error of `ex2.approx.f32` in units of `eps32`, counted
  once for the output and once for the normalizer. The bound uses 1.5.
- `k` is `ceil(n / 128) + 6`, the rounding additions of the 128-lane sum
  over a segment of `n` elements (see
  [Segmented Reductions](segmented-reductions.md#sum-rounding)).
- `1.5` covers the division and the higher-order terms.

Both `d` and `dbar` are at most the spread of the segment, the difference
between its largest and smallest logit. As a function of spread alone the
bound is therefore

```text
(2.24 * spread + 4.5 + k / 2) * eps32
```

for any distribution of the logits, which is `2.4e-05` at spread 80 for a
4096-element segment.

A test asserts the first form for every output against `torch.softmax` in
float64, at spreads 8, 20, 50, and 80. Its segments hold logits on a grid
of eighths, uniformly drawn logits in segments of up to 4096 elements, and
one maximum above a single far level. The largest errors it measures on the
RTX A6000 (`sm_86`) are:

| Spread | Largest relative error | In `eps32` | Fraction of the bound |
|---|---|---|---|
| 8 | `5.5e-07` | 4.6 | 0.25 |
| 20 | `1.0e-06` | 8.4 | 0.32 |
| 50 | `3.4e-06` | 28.2 | 0.45 |
| 80 | `3.7e-06` | 31.0 | 0.44 |

The same file measures `ex2.approx.f32` alone, through one-element sums of
`exp2`, over `2**20` evenly spaced arguments from -126 to 126. On the RTX
A6000 every result is within 2 ulp of the correctly rounded f32 and within
`1.22 * eps32` of the exact value, and every integer argument is exact. The
test asserts 2 ulp and `1.5 * eps32`. No other architecture has been
measured.

The bound applies to outputs that are normal f32 numbers. A logit more than
about 87 below its segment maximum has an output below the smallest normal
f32, which is subnormal or zero and carries no relative accuracy. The
comparison against float32 PyTorch in `test_segmented_runtime.py` keeps its
relative tolerance of `2e-06`, which is sized for spreads of at most 8.

Continue with the [SwagePlan Dialect](swage-plan-dialect.md) for the
planning IR, and then [Task Planning](planning.md) for how segments become
schedulable tasks.
