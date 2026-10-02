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
  sources, a content identity of each native compiler library (the nanobind
  extension and `libSwagePythonCAPI`), and the target. The content identity
  is the ELF build id that the linker derived from the library, or the
  SHA-256 digest of the whole file for a library without one. File names
  are part of the key; file sizes and times are not. The identity is read
  from the files on disk and is not a record of the code the process
  loaded, so the cache is used only when no identified file changed after
  the process started. A build id identifies a linked library, not every
  byte of the file: a library altered after linking keeps its build id, and
  such a change is not detected. Cache entries must be owned by the current
  user and must not be
  world-writable or symlinks; PTX is not loaded from cache entries that
  fail metadata validation. The cache root is bounded, 1024 entries unless
  `SWAGE_CACHE_MAX_ENTRIES` sets another bound, and eviction removes only
  entry and staging directories that the current user owns. With
  `SWAGE_CACHE_READ_ONLY=1` a process reads verified entries and never
  creates, publishes, removes, or evicts anything under the cache root.
- Artifact directories: `SWAGE_ARTIFACT_DIR` names a directory of kernels
  that were compiled ahead of time, with a runtime library and a manifest.
  A process executes both: the PTX on the GPU and the library in the
  process. Naming the directory is the trust decision, as putting a
  directory on `PYTHONPATH` is, so the rule differs from the cache rule
  above. The owner of the directory is not compared with the current user,
  because an artifact is normally written by one account and read by
  another, and a read-only directory is admitted. The loader refuses a
  directory, a manifest, a kernel file, or a runtime library that has the
  group-write or the other-write permission bit. It follows symbolic links
  and applies that rule to what they lead to. It accepts only plain file
  names from the manifest, and it verifies the SHA-256 digest of every
  kernel and of the library against the manifest before it loads anything.
  These checks detect damage, a partial copy, and a file that an account
  other than its owner and root could change. They do not authenticate an
  artifact: the manifest is not signed, so whoever can write the directory
  or one of its parent directories can replace the artifact as a whole. The
  loader does not check the parent directories and does not read access
  control lists. Keep an artifact where only its owner and root can write.
- CI secrets are not exposed to untrusted pull requests.

## Reporting a vulnerability

Use GitHub's private vulnerability reporting on
`https://github.com/abhiksark/swage/security/advisories/new`, or email
`abhiksark@gmail.com` with subject `[swage security]`. Please include a
reproducer. You will get an acknowledgment within 7 days. Please do not
open public issues for suspected vulnerabilities.
