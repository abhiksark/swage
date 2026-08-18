# test/AGENTS.md

Rules for the lit/FileCheck suite (`ninja -C build check-swage`).

- One concept per test file: round trips in `roundtrip.mlir`, verifier
  rejections in `invalid.mlir`, one file per pass or conversion later.
- Every verifier diagnostic has a negative test using
  `--verify-diagnostics` with `--split-input-file`; match the stable core
  of the message, not incidental wording.
- CHECK lines assert semantic invariants (op structure, types, SSA
  relationships via capture variables), never unrelated IR details or
  value numbering; prefer `%[[NAME:.*]]` captures over hardcoded `%0`.
- Output must be deterministic; if a test needs a flag to be so, the RUN
  line carries it.
- Tests must pass against an install-tree LLVM: depend only on `swage-*`
  tools plus FileCheck/not/count from the LLVM install.
