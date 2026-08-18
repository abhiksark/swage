# Convenience wrappers around the real commands. Advanced users can run the
# underlying scripts and tools directly; see CONTRIBUTING.md.

.PHONY: setup build test test-cpu lint docs

setup:
	pip install -e ".[dev]"

build:
	./scripts/build_swage.sh

test: test-cpu

test-cpu:
	python -m pytest tests/python -q
	@if [ -d build ]; then ninja -C build check-swage; \
	else echo "note: MLIR build tree missing; run 'make build' to enable lit tests"; fi

lint:
	ruff check .

docs:
	mkdocs build --strict
