# python/AGENTS.md

Rules for the Python package (`swage`, distributed as `swage-compiler`).

- The future kernel frontend parses a restricted subset via
  `inspect.getsource` → `ast.parse` and builds MLIR directly. Never execute
  user kernel bodies as normal Python; never widen the accepted subset
  without diagnostics tests.
- Preserve Python source locations into MLIR; diagnostics name the kernel
  function and line.
- Symbolic `swage.language` functions must fail clearly outside `@sw.jit`.
- Public API compatibility: anything importable from `swage` is public;
  removals or renames need a deprecation note in `CHANGELOG.md`.
- Style: Google Python style, 80 columns, Google-style docstrings; `ruff
  check .` must pass. Every file starts with its repo-relative path
  comment (see existing files).
- Tests live in `tests/python/`; every behavior change lands with a test.
  Frontend diagnostics get negative tests (bad kernels, clear errors).
