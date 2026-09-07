<!-- docs/internals/verification.md -->

# Verification

Status claims are grounded in executable tests and committed artifacts. This
matrix identifies the smallest evidence source for each current boundary and
does not turn private qualification into public API.

The v0.5.2 fixed-vector native-wheel implementation is **unreleased pending
gates**; the latest tagged release is v0.5.1. The release rows below describe
implemented checks. A local repaired cp313 candidate now passes the frozen
installed-wheel A6000 SLOs, but records `source_clean=false`. Official
qualification still requires a clean committed candidate, second-process
persistent-cache reuse, and trusted release/security gates. The broader
segmented compiler remains experimental.

That recorded FP32 SLO candidate predates the low-precision dtype extension.
Its raw benchmark is unchanged; new release artifacts must be rebuilt and
pass the expanded correctness smoke. It is not FP16 or FP8 performance
evidence.

<div class="doc-figure" tabindex="0" markdown="1">

![One semantic module executed on the GPU path, the CPU oracle, and PyTorch, feeding a differential comparison](../assets/figures/oracle-topology.svg)

</div>

*The comparison topology behind every correctness claim. [Open the full-size figure](../assets/figures/oracle-topology.svg).*

| Boundary | Status | Primary evidence | Applicable command |
|---|---|---|---|
| Frontend, public stubs, and reproducible CMake-free sdist | Implemented v0.5.2 gate; hosted CPython 3.10/3.11/3.12/3.13 | `tests/python`, `tests/typing`, `ci-python.yml` | `ruff check .`; `python -m pytest tests/python -q`; `python -m build --sdist`; `python -m twine check --strict dist/*.tar.gz` |
| Environment report, health checks, availability errors, and opt-in logging | Implemented v0.5.2 public contract | `tests/python/test_env.py`, `tests/python/test_runtime.py` | `python -m pytest tests/python/test_env.py tests/python/test_runtime.py -q`; installed `python -m swage.env --json --check cpu` or `--check cuda` |
| Installed native CPU wheel on every ABI | Implemented v0.5.2 gate; hosted release jobs | `publish-pypi.yml`, `scripts/smoke_installed_wheel.py` | CPU health and smoke outside checkout with `PYTHONPATH` unset; `torch==2.6.0+cpu` on cp310/cp311/cp312/cp313 |
| Current-PyTorch native CPU, lit, C++ units, and bindings | Implemented hosted gate | `ci-cpp.yml`, fixed native integration suites | `SWAGE_BUILD_TYPE=Release ./scripts/build_swage.sh`; `ninja -C build check-swage-unit check-swage-python`; installed cp313 CPU health/smoke with `torch==2.13.0+cpu` |
| Native ASan + UBSan | Implemented hosted gate; Swage and a separately cached, sanitizer-matched LLVM build at the unchanged pin | `ci-cpp.yml`, `SWAGE_LLVM_SANITIZERS`, `SWAGE_ENABLE_SANITIZERS` | `SWAGE_LLVM_SANITIZERS='Address;Undefined' ./scripts/build_llvm.sh`; separate bindings-off Swage build; `ninja -C <sanitizer-build> check-swage check-swage-unit` |
| CodeQL Python/C++ and dependency vulnerabilities/licenses | Implemented hosted security gates | `ci-python.yml`, `ci-cpp.yml`, `dependency-review.yml` | `CodeQL Python (3.13)`, `Native cp313 + CodeQL C/C++`, `dependency-review`; dependency review rejects newly introduced high/critical vulnerabilities and reports license changes |
| Self-contained repaired manylinux wheel, ABI, size, licenses, and provenance | Implemented v0.5.2 release gate; hosted artifact checks | `scripts/check_native_wheel.py`, `scripts/repair_native_wheel.py`, `publish-pypi.yml` | Shared repair/check scripts with `--expected-revision` and all source/build roots as `--forbid-prefix`; each wheel strictly below 95,000,000 bytes |
| Exact distribution set and source-tree/sdist wheel reproducibility | Implemented v0.5.2 release gate; hosted aggregation | `publish-pypi.yml`, repair JSON | Exactly four cp310–cp313 wheels plus one sdist; cp313 `--rebuild-sdist` SHA-256 equality; `SHA256SUMS` |
| SPDX SBOM, build provenance, and publication authorization | Implemented release gates; not yet a published attestation | `publish-pypi.yml` | One aggregate SPDX JSON SBOM; signed protected tag only for build/SBOM attestation and reviewed `pypi` OIDC publication; manual runs never attest or publish |
| Actual repaired cp313 wheel CUDA correctness, cache reuse, and fixed-runtime SLOs | Required trusted release gate; qualification pending | `publish-pypi.yml`, `scripts/smoke_installed_wheel.py`, `benchmarks/benchmark_fixed_runtime.py` | Installed fixed runtime pytest, CUDA health/smoke, second process `--require-persistent-hit`, and `--enforce --output fixed-runtime-slo.json` on NVIDIA RTX A6000 / `sm_86` |
| Restricted AST to verified native module | Public today, compile-only | `tests/python/test_frontend.py`, `python/tests/mlir/test_frontend.py` | `python -m pytest tests/python -q`; `ninja -C build check-swage-python` |
| Fixed vector add/multiply CPU/CUDA lowering and launch | Public today | fixed-block lit tests, `python/tests/mlir/test_cpu_runtime.py`, `python/tests/mlir/test_runtime.py` | `ninja -C build check-swage`; `ninja -C build check-swage-python`; trusted GPU workflow |
| FP16 and FP8 elementwise numerics and dtype isolation | Public source implementation; multiplication GPU qualification pending | `python/tests/mlir/test_low_precision_runtime.py`, fixed-block low-precision lit tests, `scripts/smoke_installed_wheel.py` | `ninja -C build check-swage-python`; installed CPU/CUDA smoke for both operations and all four dtypes; exhaustive FP8 checks remain operation-specific evidence |
| Compiler-generated physical launch contracts | Internal boundary | `unittests/KernelContractTest.cpp`, `python/tests/mlir/test_codegen.py`, `tests/python/test_abi.py` | `ninja -C build check-swage-unit`; `ninja -C build check-swage-python`; `python -m pytest tests/python/test_abi.py -q` |
| Segmented sum and max CPU/GPU parity | Private qualification | `test/Conversion/SwageToCPU`, `test/Conversion/SwageToGPU`, `python/tests/mlir/test_segmented_runtime.py` | `ninja -C build check-swage`; trusted GPU workflow |
| Stable ragged-softmax parity and edge cases | Private qualification | ragged-softmax lit files and `python/tests/mlir/test_segmented_runtime.py` | `ninja -C build check-swage`; trusted GPU workflow |
| Planning admission, limits, and descriptors | Private qualification | `test/Conversion/SwageToPlan`, `unittests/TaskClassifierTest.cpp` | `ninja -C build check-swage`; `ninja -C build check-swage-unit` |
| Pure and fused mixed identity-sum correctness | Private qualification | `python/tests/mlir/test_segmented_runtime.py` | trusted GPU workflow |
| Frozen mixed-policy performance gate | Private qualification | `benchmarks/results/mixed-sum-a6000-sm86.json`, `tests/python/test_benchmark_mixed_sum.py` | `python -m pytest tests/python/test_benchmark_mixed_sum.py -q` |
| Split coverage, ordering, failures, and f32 parity | Private qualification | `unittests/TaskClassifierTest.cpp`, `python/tests/mlir/test_segmented_runtime.py` | `ninja -C build check-swage-unit`; trusted GPU workflow |
| Persistent claims, fenced split completion, poisoned scratch, graph replay, randomized plans, and failure paths | Experimental; predeclared performance gate failed | `python/tests/mlir/test_segmented_codegen.py`, `python/tests/mlir/test_segmented_runtime.py`, `python/tests/mlir/test_persistent_runtime.py`, `benchmarks/results/persistent-sum-a6000-sm86.json` | `ninja -C build check-swage-python`; trusted GPU workflow after merge |
| Repeated-campaign archival isolation | Artifact contract | `tests/python/test_benchmark_triton_comparison.py` | `python -m pytest tests/python/test_benchmark_triton_comparison.py -q` |
| Bounded artifact/module lifecycle, compile coalescing, leases, and context-safe deferred unload | Internal boundary | fake-driver and concurrency cases in `tests/python/test_runtime.py`; graph replay in segmented runtime suites | `python -m pytest tests/python/test_runtime.py -q`; trusted GPU workflow |
| Fixed generic-launch host marshalling gate on RTX A6000 `sm_86` | Recorded evidence | `benchmarks/results/abi-launch-a6000-sm86.json`, `tests/python/test_benchmark_abi_launch.py` | `python -m pytest tests/python/test_benchmark_abi_launch.py -q` |
| Recorded RTX 5090 performance snapshot | Recorded evidence | `benchmarks/results/perf-5090-sm120.json` | Not re-executable in CI |
| Public segmented syntax and execution | Planned | No executable public contract | No passing gate yet |
| Public packed warps, reusable queues, and qualified persistent scheduling | Planned | No executable public contract | No passing public gate yet |

## Release evidence and trusted GPU separation

Hosted Python CI covers regular-GIL CPython 3.10–3.13 without LLVM, including
the typed public fixed-vector contract. The native hosted job uses exactly
LLVM/MLIR 22.1.8. The release wheel lanes use their active CPython ABI and
`X86;NVPTX` LLVM targets in a digest-pinned manylinux 2.28 x86-64 container;
a workstation-built `linux_x86_64` wheel does not establish manylinux
compatibility. The supported release platform is Linux x86-64 with glibc
2.28 or newer, and optional PyTorch >=2.6,<3.

Each repaired wheel must include `swage`, the self-contained private
`mlir_swage` extensions/runtime libraries, validated clean build identity,
public type metadata, and MIT/LLVM licenses. The artifact checker rejects
bytecode, private segmented Python modules, external MLIR runtime
dependencies, absolute or non-`$ORIGIN`-relative RPATH/RUNPATH, build-root
leaks, and `libcuda.so.1` as a linked dependency. The shared repair helper
is the only release repair path. The cp313 sdist rebuild uses the same
toolchain and commit-derived `SOURCE_DATE_EPOCH`; unequal repaired hashes
block release, with no waiver.

The required main checks are `test (3.10)`, `test (3.11)`, `test (3.12)`,
`test (3.13)`, `docs`, `CodeQL Python (3.13)`,
`Native cp313 + CodeQL C/C++`, `Native ASan + UBSan`, and
`dependency-review`. Operator configuration of those requirements, protected
`main` and `v*` tags, signed tags, and the `pypi` reviewer/trusted publisher
requires separate authorization; this page is not evidence of that setup.
On a tag run, the publication workflow checks GitHub's cryptographic tag
verification as well as tag identity and main protection/ancestry.

The scheduled/manual `ci-gpu` workflow runs only on trusted `main` through
the self-hosted `[linux, x64, swage-gpu]` runner. Its
`runtime-qualification` job retains build-tree/private research coverage;
its separate `fixed-runtime-slo` job installs a local native wheel and
retains raw SLO JSON even on failure. Neither replaces the release workflow's
gate on the **actual aggregated repaired cp313 artifact**, installed without
`PYTHONPATH` in an isolated environment reusing CUDA-enabled runner PyTorch.
That gate checks fixed native runtime behavior, boundary sizes
`0, 1, 127, 128, 129, 4097`, exact results, memory-cache reuse,
second-process persistent hits, and the SLOs below.

Only NVIDIA RTX A6000 / `sm_86` is release-qualification hardware.
Other admitted targets remain unqualified/best-effort; CUDA admission is
not a trusted performance result. Documentation in a branch can cite
committed evidence but cannot establish a new GPU result without executing
the trusted workflow or an equivalent recorded qualification.
Recorded evidence is a citation status, not a boundary status: historical
snapshots upgrade nothing, and their numbers are presented on
[Benchmarks](benchmarks.md). The private persistent performance gate remains
failed; fixed-vector release hardening neither completes it nor changes
the v0.6.0 mapping.

## Frozen fixed-runtime SLO gates

`benchmarks/benchmark_fixed_runtime.py` writes raw JSON before returning,
including failed gates. `--enforce` refuses hardware other than exactly
NVIDIA RTX A6000 / `sm_86`. Correctness runs before every timing section;
any mismatch invalidates the record regardless of timing.

| Gate | Frozen method | Passing threshold |
|---|---|---|
| Cold compile/load/first synchronized CUDA launch | Five fresh child processes, unique empty cache per process, `n=129`, `BLOCK=128` | Median <=250 ms and maximum <=400 ms |
| Warm host dispatch | `n=129`, `BLOCK=128`; 200 warmups, then 20 batches of 500 launches, synchronize before and after each batch | Median <=15 microseconds/call and p95 <=20 microseconds/call |
| Large-vector throughput | `n=2^18` and `2^20`, `BLOCK=256`; rotating interleaved CUDA-event batches of 32 launches, 25 warmups, 100 samples against `torch.add(out=...)` | Swage/PyTorch median ratio <=1.50 at each size |
| Native compiler memory | Linux `/proc/self/status` after PyTorch/tensor setup versus after first compile/launch | RSS increase <=512 MiB |

Retain raw samples, statistics, thresholds, correctness/pass fields,
package/build identity, Python/PyTorch/CUDA/driver versions, and hardware
facts. A manual `publish-pypi` dry run must retain four wheels, one sdist,
checksums, one SPDX JSON SBOM, repair/reproducibility/CPU evidence, and
trusted GPU evidence; it does not produce attestations or publish.
Signed-tag runs may attest and publish only after all gates pass.

### Local installed-wheel evidence

The 2026-09-05 repaired cp313 run on NVIDIA RTX A6000 / `sm_86`, CPython
3.13.13, and PyTorch `2.12.0+cu130` passes every frozen SLO and all 32 installed
fixed-runtime tests. The benchmark script is byte-identical to the
pre-remediation source distribution; neither timings nor thresholds were
redefined. [Raw samples and build identity](https://github.com/abhiksark/swage/blob/main/benchmarks/results/fixed-runtime-a6000-sm86.json)
record the following results:

| Measure | Observed |
|---|---|
| Cold compile/load/launch | 30.21 ms median, 31.12 ms maximum |
| Warm end-to-end host dispatch | 4.14 microseconds median, 4.92 microseconds p95 |
| Throughput ratio at `2^18` | 1.436 |
| Throughput ratio at `2^20` | 1.023 |
| Maximum native RSS increase | 78,839,808 bytes |

This is local engineering evidence, not official release qualification:
the candidate truthfully records an uncommitted source tree, so persistent
CUDA caching remains disabled. No signed tag, publication, or attestation is
implied by this passing run.

For historical planning and release mapping, continue with
[`ROADMAP.md`](https://github.com/abhiksark/swage/blob/main/ROADMAP.md). For
why the boundaries were chosen, continue with the [ADR Index](../decisions/index.md).
