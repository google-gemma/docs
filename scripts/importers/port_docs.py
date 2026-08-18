#!/usr/bin/env python3
"""
Batch-port upstream pages listed in `scripts/importers/sources.toml`.

Only pages in the manifest are touched, and only those not marked `frozen = true`.
A page absent from the manifest, or frozen in it, is hand-maintained and must never be
re-converted -- see "Hand-maintained sections" in PLAN.md.

    python scripts/importers/port_docs.py --dry-run              # show what would change
    python scripts/importers/port_docs.py --filter models/       # port one section
    python scripts/importers/port_docs.py                        # port everything in the manifest

Every page is fidelity-checked after conversion. A page that drops prose is reported and, unless
--force is given, left unwritten -- a bad conversion never lands on disk.

Sources are cached under .sources/ so repeated runs don't re-fetch.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_fidelity import cache_name, compare, fetch  # noqa: E402
from convert_notebook import convert  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DOCS_DIR = REPO_ROOT / "docs"
ASSETS_DIR = DOCS_DIR / "assets"
MANIFEST = Path(__file__).resolve().parent / "sources.toml"
CACHE_DIR = REPO_ROOT / ".sources"


def load_source(url: str, name: str, refresh: bool) -> str:
    cached = CACHE_DIR / name
    if cached.exists() and not refresh:
        return cached.read_text(encoding="utf-8")
    text = fetch(url)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached.write_text(text, encoding="utf-8")
    return text


def relative_assets_url(assets_dir: Path, page_path: Path) -> str:
    import os
    return os.path.relpath(assets_dir, page_path.parent).replace("\\", "/")


def port(entry: dict, refresh: bool, dry_run: bool, force: bool) -> tuple[bool, str]:
    """Convert one page. Returns (ok, status)."""
    if entry.get("frozen"):
        # Hand-maintained since its port. The converter would discard those edits.
        return True, "frozen (hand-maintained, skipped)"

    page_path = DOCS_DIR / entry["path"]
    source_text = load_source(entry["source"], cache_name(entry["path"]), refresh)

    assets_dir = ASSETS_DIR / entry["assets"] if entry.get("assets") else None
    assets_url = relative_assets_url(assets_dir, page_path) if assets_dir else "../assets"

    # A dry run must not touch the filesystem, so extraction goes to a throwaway directory.
    # The URL still points at the real location, keeping the diff honest.
    with tempfile.TemporaryDirectory() as scratch:
        converted = convert(
            source_text,
            assets_dir=Path(scratch) if (dry_run and assets_dir) else assets_dir,
            assets_url=assets_url,
            assets_stem=page_path.stem,
        )

    dropped, _ = compare(source_text, converted)
    if dropped and not force:
        top = ", ".join(f"{w!r}x{c}" for w, c in dropped.most_common(5))
        return False, f"LOST {sum(dropped.values())} word(s): {top}"

    if dry_run:
        before = page_path.read_text(encoding="utf-8") if page_path.exists() else ""
        if before == converted:
            return True, "unchanged"
        delta = len(converted) - len(before)
        return True, f"would rewrite ({delta:+,} bytes)"

    page_path.parent.mkdir(parents=True, exist_ok=True)
    page_path.write_text(converted, encoding="utf-8")
    return True, "ported"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--filter", default="", help="only pages whose path starts with this")
    parser.add_argument("--dry-run", action="store_true", help="report changes without writing")
    parser.add_argument("--refresh", action="store_true", help="re-fetch sources, ignoring cache")
    parser.add_argument("--force", action="store_true", help="write even if prose was dropped")
    args = parser.parse_args()

    pages = tomllib.loads(MANIFEST.read_text(encoding="utf-8"))["page"]
    selected = [p for p in pages if p["path"].startswith(args.filter)]
    if not selected:
        print(f"No manifest pages match {args.filter!r}")
        return 2

    mode = "DRY RUN" if args.dry_run else "PORTING"
    print(f"{mode}: {len(selected)} page(s)\n")

    failures = []
    for entry in selected:
        ok, status = port(entry, args.refresh, args.dry_run, args.force)
        marker = "  " if ok else "!!"
        print(f"{marker} {entry['path']:<45} {status}")
        if not ok:
            failures.append(entry["path"])

    print(f"\n{len(selected) - len(failures)} ok, {len(failures)} failed")

    if not args.dry_run and len(selected) > len(failures):
        # Conversion restores the upstream image URLs, undoing any previous vendoring.
        # The linter's `images` check is at zero, so a missed run fails CI -- but say it here
        # rather than making someone discover it from a red build.
        print("\nNext: re-vendor images, which this run reverted to upstream URLs:")
        print("  uv run python scripts/importers/vendor_images.py")
    if failures:
        print("\nFidelity failures (not written):")
        for path in failures:
            print(f"  {path}")
        print("\nInvestigate before using --force; dropped prose is a converter bug.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
