## Summary

<!-- What changed, in one or two sentences. -->

## Motivation

<!-- Why. Link the roadmap issue: Closes #NNN -->

## Semantic changes

<!-- Does any observable compiler/runtime behavior change? If IR changed,
show before/after snippets. Write "none" when honest. -->

## Tests

<!-- Commands you ran and their results. Distinguish passed / failed /
skipped / not run. -->

- [ ] `python -m pytest tests/python -q`
- [ ] `ruff check .`
- [ ] `ninja -C build check-swage` (or: not applicable because …)
- [ ] `ninja -C build check-swage-python` (or: native bindings not affected)

## Environment

<!-- CPU-only or GPU? If GPU: model, compute capability, driver. -->

## Performance impact

<!-- "none expected", or numbers with the exact reproduction command. -->

## Documentation impact

<!-- Docs updated, or why none were needed. Status tables (README/ROADMAP)
must not claim more than the tests in this PR show. -->

## Known limitations / follow-ups

<!-- What this PR deliberately does not do; follow-up issues filed. Native
bindings do not make the Python frontend complete; use Issue #4 where that
frontend remains deferred. -->
