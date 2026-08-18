#!/usr/bin/env python3
"""
Download hotlinked images into docs/assets/ and rewrite the references.

The port inherited images pointing at ai.google.dev/static. That makes the site depend on
Google's CDN staying put, blocks offline builds, and leaves no way to add dark-mode variants.
This vendors them.

    python scripts/importers/vendor_images.py --dry-run    # show what would be downloaded
    python scripts/importers/vendor_images.py              # download and rewrite

Assets are grouped by the page's parent directory, matching the layout port_docs.py already
uses (docs/assets/paligemma/, docs/assets/diffusiongemma/, ...). Original filenames are kept
so an asset can always be traced back to its upstream URL.

A download failure leaves the reference untouched -- a broken local path is worse than a
working remote one. Re-run after fixing; the script is idempotent.

Dark-mode variants are deliberately out of scope; see PLAN.md Phase 1.
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lint_docs import load_pages, outside_code  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DOCS_DIR = REPO_ROOT / "docs"
ASSETS_DIR = DOCS_DIR / "assets"

REMOTE_IMAGE = re.compile(r"!\[([^\]]*)\]\((https?://[^)\s]+)\)")
HTML_IMAGE = re.compile(r"(<img[^>]+src=[\"'])(https?://[^\"']+)([\"'])")
UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


def asset_section(page_rel: str) -> str:
    """Directory under docs/assets/ for a page's images: its parent directory name."""
    parent = Path(page_rel).parent
    return parent.name or "shared"


def safe_filename(url: str) -> str:
    name = url.split("?")[0].rstrip("/").rsplit("/", 1)[-1]
    return UNSAFE.sub("-", name) or "image.png"


def download(url: str, target: Path) -> tuple[bool, str]:
    if target.exists():
        return True, "cached"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if not data:
        return False, "empty response"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return True, f"{len(data):,} bytes"


def collect(page) -> list[tuple[str, str]]:
    """Remote image URLs on a page, as (url, filename), outside code blocks, deduplicated."""
    found: dict[str, str] = {}
    for _, line in outside_code(page):
        for match in REMOTE_IMAGE.finditer(line):
            found.setdefault(match.group(2), safe_filename(match.group(2)))
        for match in HTML_IMAGE.finditer(line):
            found.setdefault(match.group(2), safe_filename(match.group(2)))
    return list(found.items())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    import os

    total = failed = 0
    for page in load_pages():
        images = collect(page)
        if not images:
            continue

        section = asset_section(page.rel)
        print(f"\n{page.repo_rel}  ->  docs/assets/{section}/")

        text = page.text
        rewritten = 0
        for url, filename in images:
            total += 1
            target = ASSETS_DIR / section / filename
            if args.dry_run:
                print(f"    would fetch {filename:<45} <- {url}")
                continue

            ok, detail = download(url, target)
            if not ok:
                failed += 1
                print(f"  !! FAILED   {filename:<45} {detail}")
                continue

            relative = os.path.relpath(target, page.path.parent).replace("\\", "/")
            before = text
            text = text.replace(f"]({url})", f"]({relative})")
            text = HTML_IMAGE.sub(
                lambda m: m.group(1) + relative + m.group(3) if m.group(2) == url else m.group(0),
                text,
            )
            if text != before:
                rewritten += 1
            print(f"     ok       {filename:<45} {detail}")

        if not args.dry_run and text != page.text:
            page.path.write_text(text, encoding="utf-8")
            print(f"     rewrote {rewritten} reference(s)")

    verb = "would vendor" if args.dry_run else "vendored"
    print(f"\n{verb} {total - failed}/{total} image(s)")
    if failed:
        print(f"{failed} failed -- references left pointing at the original URL; re-run to retry")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
