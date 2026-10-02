# Security Policy

## Supported versions

Swage is pre-alpha (`0.x`). Only the latest `main` receives fixes.

## Threat model

Swage compiles user-provided Python kernel source into GPU code. **Kernel
source is treated as trusted input**: compiling attacker-controlled kernels
is outside the current threat model, and the compiler is *not* sandboxed.
Within that model, the project still commits to:

- Never executing arbitrary Python during compilation (the frontend parses
  a restricted AST; it does not `eval` kernel bodies).
- Restricting frontend call targets to the `swage.language` builtins.
- Validating tensor device, dtype, layout, and bounds metadata at the
  runtime boundary, including overflow-safe offset validation.
- Cache integrity: cache keys include the SHA-256 digest of the frontend
  sources, the file name, size, and modification time of each native
  compiler library (the nanobind extension and `libSwagePythonCAPI`), and
  the target. The native part of the key is file metadata read from disk.
  It is not a digest of the library contents and not a record of the code
  the process loaded, so it does not detect a library replaced by one with
  the same name, size, and modification time. Where file times are
  normalized, as in a reproducibly built image, it reduces to the name and
  size. Cache entries must be owned by the current user and must not be
  world-writable or symlinks; PTX is not loaded from cache entries that
  fail metadata validation. The cache root is bounded, 1024 entries unless
  `SWAGE_CACHE_MAX_ENTRIES` sets another bound, and eviction removes only
  entry and staging directories that the current user owns. With
  `SWAGE_CACHE_READ_ONLY=1` a process reads verified entries and never
  creates, publishes, removes, or evicts anything under the cache root.
- CI secrets are not exposed to untrusted pull requests.

## Reporting a vulnerability

Use GitHub's private vulnerability reporting on
`https://github.com/abhiksark/swage/security/advisories/new`, or email
`abhiksark@gmail.com` with subject `[swage security]`. Please include a
reproducer. You will get an acknowledgment within 7 days. Please do not
open public issues for suspected vulnerabilities.
