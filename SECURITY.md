# Security Policy

## Supported versions

Security fixes target the **latest released `0.x` line**, with development
fixes on `main`; older released lines are not maintained. The latest tagged
release is v0.5.1. The v0.5.2 fixed-vector native-wheel implementation is
unreleased pending its publication and qualification gates; its Beta
classifier is not a claim that those gates have passed. The broader
segmented compiler remains experimental.

## Threat model

Swage compiles user-provided Python kernel source into native code for an
explicitly selected backend. **Kernel source is treated as trusted input**:
compiling attacker-controlled kernels is outside the threat model, and
the compiler is **not a sandbox**. Do not use it to isolate untrusted code
or mutually distrustful tenants. Native LLVM/MLIR, the host JIT, CUDA driver,
and CUDA-enabled PyTorch execute within the application's trust boundary.
The v0.5.2 wheel bundles a self-contained private `mlir_swage` package and
pinned LLVM/MLIR runtime code (LLVM 22.1.8); an external `mlir` package is
neither required nor a way to patch that bundled copy. PyTorch and the
system NVIDIA driver are not bundled and need their own security updates.
Within that model, the project still commits to:

- Never executing arbitrary Python during compilation (the frontend parses
  a restricted AST; it does not `eval` kernel bodies).
- Restricting frontend call targets to the `swage.language` builtins.
- Validating tensor device, dtype, layout, and bounds metadata at the
  runtime boundary, including overflow-safe offset validation.
- Cache integrity: CUDA cache keys include validated compiler identity and
  target specialization; artifacts are not world-writable and cached PTX
  must pass metadata validation before loading. Treat the cache and its
  owning account as trusted, not as a signed executable store or sandbox.
  Do not share a writable cache across trust boundaries.
- Packaged `mlir_swage/_build_info.json` records schema/version, exact
  source revision, clean-tree state, LLVM pin, and Release build type.
  Validated clean identity enables persistent caching; malformed metadata
  disables persistence rather than inventing an identity. Missing packaged
  metadata retains the source-checkout identity fallback. Metadata alone is
  not a cryptographic attestation of a downloaded binary.
- CI secrets are not exposed to untrusted pull requests.

## Artifact provenance and release boundary

The implemented v0.5.2 release pipeline checks a clean source SHA, reproducible
sdist and cp313 source-tree/sdist wheel hashes, self-contained repaired
manylinux artifacts, CPU behavior on all four supported CPython ABIs,
native sanitizers, and security analysis. The actual repaired cp313 wheel
must also pass trusted NVIDIA RTX A6000 (`sm_86`) CUDA correctness, cache,
and SLO gates. Private research qualification does not qualify that release.

Publication is restricted to a protected signed annotated `v0.5.2` tag on
protected `main` ancestry. GitHub's tag API must cryptographically verify
the signature (`verified: true`, reason `valid`); an absent or indeterminate
verification fails closed. Successful tag runs attach build-provenance and
SPDX SBOM attestations before OIDC publication through the reviewed `pypi`
environment. Manual workflow runs never attest or publish. Repository
protection, signing policy, and publisher/reviewer configuration require
separately authorized operator setup; the workflow's existence does not
establish that administration or release gates are complete.

For a published release, set `RELEASE_RUN_ID` to the successful signed-tag
run and verify the downloaded distribution against the retained `SHA256SUMS`
and its detached build-provenance bundle:

```bash
gh run download "$RELEASE_RUN_ID" --repo abhiksark/swage \
  --name release-attestations --dir attestations
PROVENANCE_BUNDLE=attestations/build-provenance.json
sha256sum --check SHA256SUMS
gh attestation verify <downloaded-wheel-or-sdist> \
  --bundle "$PROVENANCE_BUNDLE" --repo abhiksark/swage
python -m swage.env --json --check native
```

Run the checksum command alongside the five distributions named in the
manifest. Match reported source revision to the verified release commit
after installation. Neither checksum agreement nor a valid attestation
guarantees that trusted source code is free of vulnerabilities. A defective
published v0.5.2 is yanked and replaced by v0.5.3 through the same gates,
never by overwriting files of an existing PyPI version.

## Reporting a vulnerability

Use GitHub's private vulnerability reporting on
`https://github.com/abhiksark/swage/security/advisories/new`, or email
`abhiksark@gmail.com` with subject `[swage security]`. Please include a
reproducer. You will get an acknowledgment within 7 days. Please do not
open public issues for suspected vulnerabilities.
