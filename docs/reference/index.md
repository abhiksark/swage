<!-- docs/reference/index.md -->

# API Reference

The API reference states the exact public contracts. Prose elsewhere in
this documentation links here rather than restating a rule; when pages
disagree, the reference is normative.

| Page | Scope |
|---|---|
| [swage](swage.md) | Package exports: `jit`, kernel objects, `emit_mlir`, `launch`, `segment_reduce`, `segment_softmax`, exceptions, `swage.env` |
| [swage.language](swage-language.md) | The symbolic kernel-language exports |
| [Kernel Language](kernel-language.md) | The accepted Python source grammar |
| [Runtime and Environment](runtime-environment.md) | Launch validation, segmented calls, targets, cache, streams, and diagnostics |
| [Support Matrix](support-matrix.md) | Tested, admitted, and unknown Python, PyTorch, driver, and GPU environments |

Compiler-facing references, including the MLIR dialects and registered
passes, live under [Internals](../internals/index.md); they are not public
API.

Continue with [swage](swage.md).
