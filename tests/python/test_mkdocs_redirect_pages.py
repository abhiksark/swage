# tests/python/test_mkdocs_redirect_pages.py
"""Validate the first-party MkDocs redirect-stub hook."""

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]
SCRIPT = REPO_ROOT / "scripts" / "mkdocs_redirect_pages.py"


def _load_hook():
    """Load the hook module from its script path."""
    spec = importlib.util.spec_from_file_location(
        "mkdocs_redirect_pages", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_redirects_cover_only_moved_pages_with_existing_targets():
    """Every redirect target resolves to a committed docs page."""
    module = _load_hook()
    assert module.REDIRECTS
    for old, new in module.REDIRECTS.items():
        assert old != new
        assert not (REPO_ROOT / "docs" / old).exists(), old
        assert (REPO_ROOT / "docs" / new).is_file(), new


def test_stub_urls_never_shadow_a_published_page():
    """No redirect stub may overwrite a real page's built output."""
    module = _load_hook()
    published = set()
    for path in (REPO_ROOT / "docs").rglob("*.md"):
        relative = path.relative_to(REPO_ROOT / "docs").as_posix()
        published.add(module.page_url(relative))
    for old in module.REDIRECTS:
        assert module.page_url(old) not in published, old


def test_page_urls_follow_directory_url_rules():
    """Markdown paths map to the directory URLs mkdocs publishes."""
    module = _load_hook()
    assert module.page_url("quickstart.md") == "quickstart/"
    assert module.page_url("internals/index.md") == "internals/"
    assert (
        module.page_url("getting-started/quickstart.md")
        == "getting-started/quickstart/"
    )


def test_relative_target_walks_between_directory_urls():
    """Stub hrefs are correct relative directory URLs."""
    module = _load_hook()
    assert (
        module.relative_target("quickstart.md", "getting-started/quickstart.md")
        == "../getting-started/quickstart/"
    )
    assert (
        module.relative_target(
            "qualification/private-m4-m8.md", "internals/index.md"
        )
        == "../../internals/"
    )
    assert (
        module.relative_target(
            "reference/public-python-api.md", "reference/swage.md"
        )
        == "../swage/"
    )


def test_post_build_writes_stub_pages_into_the_site(tmp_path):
    """The hook writes one linking stub per redirect after the build."""
    module = _load_hook()
    module.on_post_build({"site_dir": str(tmp_path)})
    for old, new in module.REDIRECTS.items():
        stub = tmp_path / module.page_url(old) / "index.html"
        text = stub.read_text()
        assert "http-equiv=\"refresh\"" in text
        assert module.relative_target(old, new) in text
        assert "<a href=" in text
