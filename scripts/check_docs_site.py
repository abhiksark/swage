# scripts/check_docs_site.py
"""Validate local links and assets in the rendered documentation site."""

import argparse
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

DEFAULT_SITE_DIR = Path(__file__).parents[1] / "site"
EXTERNAL_SCHEMES = {"data", "http", "https", "mailto", "tel"}


class _PageParser(HTMLParser):
    """Collect local-reference inputs from one rendered HTML page."""

    def __init__(self) -> None:
        super().__init__()
        self.fragments = set()
        self.references = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        """Collect IDs, named anchors, links, and source references."""
        attributes = dict(attrs)
        if attributes.get("id") is not None:
            self.fragments.add(attributes["id"])
        if tag == "a" and attributes.get("name") is not None:
            self.fragments.add(attributes["name"])
        for name in ("href", "src"):
            if attributes.get(name) is not None:
                self.references.append(attributes[name])


def _parse_page(path: Path) -> _PageParser:
    parser = _PageParser()
    parser.feed(path.read_text(encoding="utf-8"))
    parser.close()
    return parser


def _resolve_target(site_dir: Path, page: Path, url_path: str) -> Path:
    if not url_path:
        return page
    if url_path.startswith("/"):
        target = site_dir / url_path.lstrip("/")
    else:
        target = page.parent / url_path
    target = target.resolve()
    if url_path.endswith("/") or target.is_dir():
        target /= "index.html"
    return target


def check_site(site_dir: Path = DEFAULT_SITE_DIR) -> list[str]:
    """Return stable diagnostics for broken local references in a site."""
    site_dir = site_dir.resolve()
    if not site_dir.is_dir():
        return [f"site directory not found: {site_dir}"]

    pages = sorted(site_dir.rglob("*.html"))
    if not pages:
        return [f"no HTML pages found in site directory: {site_dir}"]

    parsed_pages = {page.resolve(): _parse_page(page) for page in pages}
    errors = []
    for page in pages:
        parsed = parsed_pages[page.resolve()]
        page_name = page.relative_to(site_dir).as_posix()
        for reference in parsed.references:
            try:
                parts = urlsplit(reference)
            except ValueError:
                errors.append(
                    f"{page_name}: malformed reference: {reference}"
                )
                continue
            if (
                reference.startswith("//")
                or parts.netloc
                or parts.scheme.lower() in EXTERNAL_SCHEMES
            ):
                continue

            try:
                target = _resolve_target(
                    site_dir, page, unquote(parts.path)
                )
            except (OSError, ValueError):
                errors.append(
                    f"{page_name}: malformed reference: {reference}"
                )
                continue
            try:
                target.relative_to(site_dir)
            except ValueError:
                errors.append(f"{page_name}: missing target: {reference}")
                continue
            if not target.is_file():
                errors.append(f"{page_name}: missing target: {reference}")
                continue

            fragment = unquote(parts.fragment)
            if (
                fragment
                and target.suffix.lower() == ".html"
                and fragment not in parsed_pages[target].fragments
            ):
                errors.append(f"{page_name}: missing fragment: {reference}")

    return sorted(errors)


def main() -> int:
    """Check a rendered site and print every local-reference failure."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "site_dir", nargs="?", type=Path, default=DEFAULT_SITE_DIR
    )
    args = parser.parse_args()
    errors = check_site(args.site_dir)
    for error in errors:
        print(error)
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
