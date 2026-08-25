# Convenience wrappers around the real commands. Advanced users can run the
# underlying scripts and tools directly; see CONTRIBUTING.md.

.PHONY: setup build test test-cpu lint diagrams docs

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

diagrams:
	python3 scripts/render_docs_diagrams.py

docs:
	python3 scripts/render_docs_diagrams.py --check
	mkdocs build --strict
	python3 scripts/check_docs_site.py site
