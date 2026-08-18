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
- Cache integrity: cache keys include compiler revision and target; cache
  artifacts are not world-writable; PTX is not loaded from cache entries
  that fail metadata validation.
- CI secrets are not exposed to untrusted pull requests.

## Reporting a vulnerability

Use GitHub's private vulnerability reporting on
`https://github.com/abhiksark/swage/security/advisories/new`, or email
`abhiksark@gmail.com` with subject `[swage security]`. Please include a
reproducer. You will get an acknowledgment within 7 days. Please do not
open public issues for suspected vulnerabilities.
