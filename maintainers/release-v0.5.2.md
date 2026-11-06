<!-- maintainers/release-v0.5.2.md -->

# v0.5.2 release preparation

This document records the candidate scope, local verification, and
publication gates for the maintainer. Preparation began from local `dev`
at `760196c`; the package already reported version `0.5.2`. Publication
requires the gates below. Check the
[release page](https://github.com/abhiksark/swage/releases) for published
artifacts.

## Draft release notes

Swage v0.5.2 packages the native compiler and runtime with the Python API.
Binary wheels target Linux x86-64 with glibc 2.28 or newer and regular
CPython 3.10 through 3.13. The previous v0.5.1 wheel contains only Python.

- Fixed vector addition and multiplication run on the Native CPU or CUDA
  backend, with float32, float16, and both supported FP8 storage formats.
- Public segmented reductions provide sum, maximum, minimum, and mean over
  float32 or float64 scalars and rows of features. Segmented softmax uses
  float32. Both calls support gradients, including second derivatives.
- Ahead-of-time artifacts let the segmented calls run without loading the
  compiler. Artifact loading now rejects incompatible subgroup metadata
  before a launch instead of leaving segment outputs unwritten.
- Environment diagnostics, typed fixed-kernel interfaces, persistent kernel
  caching, and installed-wheel checks accompany the native distribution.
- Documentation states the limits of sum and mean accuracy after
  intermediate overflow. CI and documentation dependencies use the patched
  certifi and pytest releases.

Swage remains pre-alpha. The Python kernel language admits the documented
fixed vector bodies; caller-written segment programs remain planned.
Persistent scheduling remains private and experimental. The supported
release GPU qualification device is RTX A6000 (`sm_86`). See the
[support matrix](../docs/reference/support-matrix.md) and
[changelog](../CHANGELOG.md) for the full compatibility boundary.

## Preparation changes

- Require 32-thread NVIDIA subgroups in artifact metadata, with unit tests
  for incompatible widths and CUDA tests that check untouched outputs.
- Test finite cancellation that yields NaN after intermediate overflow, and
  qualify the corresponding sum and mean documentation.
- Regenerate the shared hash lock with `certifi==2026.10.18` and
  `pytest==9.0.3`; require `pytest>=9.0.3` for development installs.
  `docs/requirements.txt` already includes this lock.
- Connect `SWAGE_ENABLE_SANITIZERS` to LLVM's sanitizer configuration so
  Clang can link the instrumented shared runtime. The previous configuration
  combined ASan with LLVM's incompatible `-z,defs` linker option.

These dependency updates address the open
[certifi advisory](https://github.com/certifi/python-certifi/security/advisories/GHSA-6q48-6c7w-9gw2)
and [pytest advisory](https://github.com/advisories/GHSA-6w46-j5rx-g56g).
The GitHub alerts will remain open until the fixed lock reaches the default
branch and GitHub rescans it.

## Verification boundary

Local checks ran on 2026-11-07 with Python 3.13.13, the updated hash-locked
tools, LLVM 22.1.8, PyTorch 2.12.0+cu130, and an NVIDIA RTX A6000 (`sm_86`).

| Check | Result |
| --- | --- |
| Python suite | 1,694 passed |
| lit suite | 124 passed |
| C++ unit suite | 93 passed |
| Clang 18 ASan and UBSan build | Shared runtime linked; 124 lit tests and 93 C++ tests passed |
| Native Python and CUDA suite | 5,307 passed; two opt-in race checks skipped in this invocation |
| Opt-in Compute Sanitizer race checks | Both passed in a separate invocation |
| Installed-wheel segmented calls, gradients, and artifacts | 1,681 passed, no skips |
| Installed-wheel fixed add and multiply | CPU and CUDA passed for four dtypes and six sizes |
| Installed-wheel persistent CUDA cache | Eight persistent hits and no compilation in the second process |
| Documentation and lint | Strict site build, links, figures, diagrams, Ruff, and diff whitespace checks passed |
| Distribution metadata | Local sdist and CPython 3.13 Linux wheel built and passed strict Twine checks |

The installed checks run outside the checkout and verify that the Python
package and native bindings come from the wheel. The segmented tests also
use the local native build for their private CPU oracle. Host editable
installation hooks were excluded from the wheel test environment; the
installed-package guard was not relaxed.

The Clang sanitizer check reproduced the original shared-library link
failure before the CMake fix. After the fix, a valid runtime call passed
and ASan caught an intentional out-of-bounds read inside that library.
Reconfiguration with sanitizers off retained `-z,defs` and removed
instrumentation. This local build instruments Swage against the existing
Release LLVM; it disables LLVM's manual allocator poisoning as the ordinary
sanitizer CI job does. It does not replace the full instrumented-LLVM
workflow, which must run again on the patched commit. Local logs are under
`/tmp/swage-ci-clang18-*`.

Local distributions are in `/tmp/swage-release-prep-dist/`. Test logs and
JUnit reports are under `/tmp/swage-release-*`; installed segmented evidence
is in `/tmp/swage-release-installed-segments-isolated/`. These temporary
files are local review evidence, not retained release artifacts. The wheel
records base revision `760196c` and `source_clean=false`, because it includes
the preparation changes before they are committed.

Local source checks and locally built artifacts are engineering evidence.
They do not replace the release workflow's checks on the exact distributions
it will publish. In particular, a local CPython 3.13 Linux wheel does not
establish the manylinux policy, the other three ABIs, or reproducibility.

## Release integration and publication gates

The release request authorizes integration and publication after the required
checks. Remote `dev` was synchronized with `main` at `5bdd256` through merge
commit `52f7fe4`, preserving the earlier
[promotion PR #96](https://github.com/abhiksark/swage/pull/96) and the
test-directory cleanup. The release branch integrates the reviewed fixes
from `62b2c7a` and dates the changelog for 2026-11-07.

The operator separately approved protection for `v*` tags.
[Ruleset 26719329](https://github.com/abhiksark/swage/rules/26719329) is active
and blocks updates and deletion with no bypass actors. `main` has required
checks and the `pypi` environment has a required reviewer. GitHub signing-key
registration and the PyPI trusted-publisher mapping still need verification.

1. Run the required Python, docs, and native checks on the frozen release
   branch, then integrate it into `dev`.
2. Merge `dev` into `main` with a merge commit, then synchronize `dev`.
   The release source must be on `main`'s first-parent history, including
   for a manual workflow run. The full instrumented-LLVM sanitizer workflow
   must also pass with the CMake fix.
3. Run `gh workflow run publish-pypi.yml --ref main`. Require the Python and
   native security jobs, four CPython manylinux wheels and CPU checks,
   byte-identical CPython 3.13 rebuild, SBOM and artifact checks, and
   installed-wheel A6000 correctness and runtime SLO checks to pass.
   Retain the sdist, four wheels, and evidence from that exact source SHA.
4. Create and push a signed annotated
   `v0.5.2` tag for the verified `main` merge commit. GitHub must verify its
   signature. The tag workflow rebuilds and qualifies the distributions,
   then attests and publishes through the `pypi` environment approval.
5. Verify the published PyPI artifact set and installation, then publish the
   GitHub release notes for the same tag.

The fresh [A6000 run](https://github.com/abhiksark/swage/actions/runs/42275465830)
passed on current `main` at `5bdd256`: 5,301 native tests passed with the two
opt-in race checks skipped, all 1,675 installed segmented tests passed, and
the fixed-runtime SLO gates passed. This confirms the capture-test stream
ordering fix on `main`; that run does not include the pending preparation
changes or the sanitizer CMake fix.
No v0.5.2 release-workflow run was present when preparation began. The
[verification guide](../docs/internals/verification.md#release-evidence-and-trusted-gpu-separation)
defines the release gates; none are waived by this document.
