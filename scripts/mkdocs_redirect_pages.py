# scripts/mkdocs_redirect_pages.py
"""MkDocs hook that publishes redirect stubs for moved pages.

The 2026 documentation restructure moved and renamed pages. This hook
keeps every previously published URL alive by writing one small stub
page per old location after each build. Stubs carry a meta refresh, a
canonical link, and a plain anchor, so readers and crawlers land on
the new page and the site checker validates the target.

The hook is first party on purpose: the ecosystem redirect plugin
currently drags in a contested MkDocs fork, and this repository takes
no dependency on that dispute.
"""

import posixpath
from pathlib import Path

REDIRECTS = {
    "quickstart.md": "getting-started/quickstart.md",
    "concepts/swage-visual-guide.md": "user-guide/ragged-data.md",
    "concepts/segments-tiles-tasks.md": "user-guide/execution-model.md",
    "architecture/compiler-pipeline.md": "internals/compiler-pipeline.md",
    "reference/public-python-api.md": "reference/swage.md",
    "reference/swage-dialect.md": "internals/swage-dialect.md",
    "reference/swage-plan-dialect.md": "internals/swage-plan-dialect.md",
    "reference/compiler-tools.md": "internals/compiler-tools.md",
    "qualification/private-m4-m8.md": "internals/index.md",
    "qualification/evidence.md": "internals/verification.md",
    "qualification/performance.md": "internals/benchmarks.md",
}

_STUB = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="0; url={target}">
<link rel="canonical" href="{target}">
<title>Moved</title>
</head>
<body>
<p>This page moved to <a href="{target}">{target}</a>.</p>
</body>
</html>
"""


def page_url(markdown_path: str) -> str:
    """Return the published directory URL for one docs markdown path."""
    if markdown_path.endswith("/index.md"):
        return markdown_path[: -len("index.md")]
    if markdown_path == "index.md":
        return ""
    return markdown_path[: -len(".md")] + "/"


def relative_target(old: str, new: str) -> str:
    """Return the stub's relative href from the old URL to the new one."""
    source = page_url(old).rstrip("/") or "."
    target = page_url(new).rstrip("/") or "."
    walk = posixpath.relpath(target, source)
    return walk + "/"


def on_post_build(config) -> None:
    """Write one redirect stub per moved page into the built site."""
    site_dir = Path(config["site_dir"])
    for old, new in REDIRECTS.items():
        stub_dir = site_dir / page_url(old)
        stub_dir.mkdir(parents=True, exist_ok=True)
        target = relative_target(old, new)
        (stub_dir / "index.html").write_text(_STUB.format(target=target))
