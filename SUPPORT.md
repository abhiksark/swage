# Support

- **Questions and ideas** — open a GitHub Discussion (or an issue if
  Discussions are not enabled yet).
- **Bugs** — use the bug issue form; include `python -m swage.env --json`
  output, a minimal reproducer, the selected backend, and whether the
  package came from a wheel or a source build. For an unavailable component,
  include `python -m swage.env --json --check native`, `--check cpu`, or
  `--check cuda` as appropriate; a failing check still prints the report.
- **Build problems** — check `docs/getting-started/installation.md` and
  `docs/getting-started/troubleshooting.md`; the most common cause is an LLVM
  install not matching `cmake/llvm-version.txt`.
- **Security** — see `SECURITY.md`; do not open public issues.

Support targets the latest released `0.x` line, currently the v0.5.1 tag.
The v0.5.2 native-wheel implementation is **unreleased pending gates**, not
an already available or production-qualified PyPI release. Its intended
support boundary is Linux x86-64 / glibc >=2.28, regular-GIL CPython
3.10–3.13, and optional PyTorch >=2.6,<3. It supports only canonical
contiguous rank-1 vector add with matching `float32`, `float16`,
`float8_e4m3fn`, or `float8_e5m2` tensors and explicit CPU or CUDA selection
(CUDA by default), never automatic fallback. FP8 uses software conversion
inside the compiled kernel, not native FP8 arithmetic.

For v0.5.2, NVIDIA RTX A6000 (`sm_86`) is the only CUDA release-qualification
hardware, and qualification still requires the trusted installed-wheel
gates to pass. Other admitted targets (`sm_80`, `sm_87`, `sm_88`, `sm_89`,
`sm_90`, `sm_100`, `sm_101`, `sm_103`, `sm_110`, `sm_120`, `sm_121`) remain
unqualified/best-effort; admission or a historical benchmark is not a
production-support claim. Report GPU model, architecture, driver version,
and the CUDA version of PyTorch separately. Qualified `sm_86` requires the
greater of NVIDIA R455 (PTX 7.1) and the installed CUDA-enabled PyTorch
build's documented minimum driver. No CUDA toolkit compiler is needed at
runtime. Private segmented helpers remain unsupported.

Swage is a spare-time research project with a single maintainer; response
times are best-effort, with no uptime or normal-issue response-time SLA.
Security reports retain the separate seven-day acknowledgment policy in
[`SECURITY.md`](SECURITY.md).
