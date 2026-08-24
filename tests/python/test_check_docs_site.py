# tests/python/test_check_docs_site.py
"""Tests for rendered documentation site integrity checks."""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / "scripts" / "check_docs_site.py"


def _load_checker():
    assert SCRIPT.is_file(), f"missing site checker: {SCRIPT}"
    spec = importlib.util.spec_from_file_location("check_docs_site", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, content: str = "asset") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_nested_pages_assets_fragments_and_external_urls_are_valid(tmp_path):
    """Accept browser-resolvable local links while skipping external URLs."""
    checker = _load_checker()
    site = tmp_path / "site"
    _write(site / "assets" / "app.css")
    _write(site / "assets" / "icon.svg")
    _write(
        site / "index.html",
        """<html><body id="home">
        <a href="">Top</a><a href="#">Top again</a>
        <a href="guide/?mode=short">Guide</a>
        <a href="nested/page/#details">Details</a>
        <link href="/assets/app.css?rev=1" rel="stylesheet">
        <img src="assets/icon.svg">
        <a href="https://example.com">Web</a>
        <a href="ftp://downloads.example.com/reference.pdf">Download</a>
        <a href="//cdn.example.com/a.js">CDN</a>
        <a href="mailto:docs@example.com">Email</a>
        <a href="tel:+10000000000">Telephone</a>
        <img src="data:image/png;base64,AA==">
        </body></html>""",
    )
    _write(
        site / "guide" / "index.html",
        '<a name="legacy"></a><a href="../#home">Home</a>',
    )
    _write(
        site / "nested" / "page" / "index.html",
        """<main id="details">
        <a href="#details">Current section</a>
        <a href="../../guide/index.html#legacy">Legacy section</a>
        <img src="../../assets/icon.svg?size=small">
        <a href="/">Root</a>
        </main>""",
    )

    assert checker.check_site(site) == []


def test_missing_targets_and_fragments_are_all_reported_in_stable_order(
    tmp_path,
):
    """Report every broken local reference instead of stopping at the first."""
    checker = _load_checker()
    site = tmp_path / "site"
    _write(
        site / "index.html",
        """<a href="guide/#absent">Missing fragment</a>
        <img src="assets/missing.svg">
        <a href="missing-page/#section">Missing page</a>""",
    )
    _write(site / "guide" / "index.html", '<h1 id="present">Guide</h1>')

    assert checker.check_site(site) == [
        "index.html: missing fragment: guide/#absent",
        "index.html: missing target: assets/missing.svg",
        "index.html: missing target: missing-page/#section",
    ]


def test_missing_site_and_missing_html_fail_clearly(tmp_path):
    """Reject absent or empty rendered-site inputs with direct diagnostics."""
    checker = _load_checker()
    missing = tmp_path / "missing"
    empty = tmp_path / "empty"
    empty.mkdir()

    assert checker.check_site(missing) == [
        f"site directory not found: {missing}"
    ]
    assert checker.check_site(empty) == [
        f"no HTML pages found in site directory: {empty}"
    ]
