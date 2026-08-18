#!/usr/bin/env python3
"""
Rewrite upstream ai.google.dev links to local page paths, per `scripts/importers/link_map.toml`.

    python scripts/importers/rewrite_links.py --dry-run
    python scripts/importers/rewrite_links.py
    python scripts/importers/rewrite_links.py --report      # unmapped targets only, no changes

Three behaviours, driven by the map:
  [ported]   -> relative link to an existing page
  [planned]  -> relative link to a page that does not exist yet, so the linter's `internal`
                check tracks it as known-pending instead of leaving a live upstream link
  [delink]   -> drop the link, keep the text (used for CodeGemma, which is out of scope)

Code blocks are never touched. URLs there are sample data or BibTeX citations that must stay
external -- the same rule check_links applies.

An unmapped target is reported and left alone. Silently dropping a link is worse than leaving
one to fix; add it to link_map.toml and re-run.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tomllib
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lint_docs import load_pages, outside_code  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DOCS_DIR = REPO_ROOT / "docs"
LINK_MAP = Path(__file__).resolve().parent / "link_map.toml"

BASE = re.compile(r"https?://ai\.google\.dev/gemma/docs")
# A markdown link whose target is an upstream docs URL.
MD_LINK = re.compile(r"\[([^\]]*)\]\((https?://ai\.google\.dev/gemma/docs[^)\s]*)\)")
# Anything with a file extension is an asset (sample image/audio), not a documentation page.
ASSET = re.compile(r"\.[a-z0-9]{2,4}$", re.IGNORECASE)


HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
LOCAL_LINK_WITH_ANCHOR = re.compile(r"(\]\()([^)\s#]*\.md)(#+)([^)\s]+)(\))")


def slugify(heading: str) -> str:
    """Mirror python-markdown's default toc slugify, which is what Zensical uses.

    Order matters. Markdown syntax is reduced to its visible text first, then python-markdown's
    own rule applies: drop everything that is not a word character, whitespace or hyphen, then
    collapse runs to single hyphens.

    Note what is *not* stripped: parentheses. "Google Kubernetes Engine (GKE)" becomes
    `google-kubernetes-engine-gke`, keeping the GKE. An earlier version removed parenthesised
    text wholesale and produced anchors that silently never matched.
    """
    value = re.sub(r"<[^>]+>", "", heading)              # inline html
    value = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", value)  # [text](url) -> text
    value = re.sub(r"[*`_]", "", value)                  # emphasis / code markers
    value = re.sub(r"\{:?[^}]*\}", "", value)            # attr_list, e.g. {: #id }
    value = re.sub(r"[^\w\s-]", "", value).strip().lower()
    return re.sub(r"[-\s]+", "-", value)


def page_anchors() -> dict[Path, set[str]]:
    """Heading slugs available on each page, keyed by resolved path."""
    anchors: dict[Path, set[str]] = {}
    for page in load_pages():
        slugs = set()
        in_code = False
        for line in page.body.splitlines():
            if line.lstrip().startswith("```"):
                in_code = not in_code
                continue
            if in_code:
                continue
            match = HEADING.match(line)
            if match:
                slugs.add(slugify(match.group(2)))
        anchors[page.path.resolve()] = slugs
    return anchors


def fix_anchors(dry_run: bool) -> int:
    """Repair fragments on local links: devsite used `#model_data`, Zensical emits `#model-data`.

    Also collapses `##`, which appears in at least one upstream URL verbatim.
    """
    anchors = page_anchors()
    overrides = tomllib.loads(LINK_MAP.read_text(encoding="utf-8")).get("anchors", {})
    fixed = unresolved = collapsed = 0

    for page in load_pages():
        if ".md#" not in page.text and ".md##" not in page.text:
            continue

        def replace(match: re.Match) -> str:
            nonlocal fixed, unresolved, collapsed
            opening, target, hashes, fragment, closing = match.groups()
            resolved = (page.path.parent / target).resolve()
            available = anchors.get(resolved)

            if len(hashes) > 1:
                collapsed += 1

            if available is None:
                return f"{opening}{target}#{fragment}{closing}"  # page doesn't exist yet
            if fragment in available:
                return f"{opening}{target}#{fragment}{closing}"

            candidate = fragment.replace("_", "-")
            if candidate in available:
                fixed += 1
                return f"{opening}{target}#{candidate}{closing}"

            # Explicit override: the section was renamed, so no heuristic can find it safely.
            try:
                key = f"{resolved.relative_to(DOCS_DIR).as_posix()}#{fragment}"
            except ValueError:
                key = ""
            if key in overrides:
                fixed += 1
                return f"{opening}{target}#{overrides[key]}{closing}"

            unresolved += 1
            print(f"  ?? {page.repo_rel}: #{fragment} not found in {target}")
            return f"{opening}{target}#{fragment}{closing}"

        updated = LOCAL_LINK_WITH_ANCHOR.sub(replace, page.text)
        if updated != page.text and not dry_run:
            page.path.write_text(updated, encoding="utf-8")

    verb = "would fix" if dry_run else "fixed"
    print(f"\n{verb} {fixed} anchor(s), collapsed {collapsed} '##', {unresolved} unresolved")
    return 0


def load_map() -> tuple[dict[str, str], dict[str, str], set[str], dict[str, dict[str, str]]]:
    data = tomllib.loads(LINK_MAP.read_text(encoding="utf-8"))
    return (
        data.get("ported", {}),
        data.get("planned", {}),
        set(data.get("delink", {}).get("paths", [])),
        data.get("page_overrides", {}),
    )


def split_target(url: str) -> tuple[str, str]:
    """Split an upstream URL into (path after /gemma/docs, '#fragment' or '')."""
    remainder = BASE.sub("", url)
    path, sep, fragment = remainder.partition("#")
    return path.rstrip("/"), (sep + fragment if sep else "")


def relative(target: str, page_path: Path) -> str:
    return os.path.relpath(DOCS_DIR / target, page_path.parent).replace("\\", "/")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", action="store_true", help="list unmapped targets and exit")
    parser.add_argument(
        "--fix-anchors", action="store_true",
        help="repair #fragments on local links (devsite underscores -> Zensical hyphens)",
    )
    args = parser.parse_args()

    if args.fix_anchors:
        return fix_anchors(args.dry_run)

    ported, planned, delink, page_overrides = load_map()
    mapping = {**ported, **planned}

    unmapped: Counter = Counter()
    stats = Counter()
    changed_pages = 0

    for page in load_pages():
        if "ai.google.dev/gemma/docs" not in page.text:
            continue

        # Match over the whole document, not line by line: devsite wraps long links across
        # lines (`[thinking\nmodes](url)`), which a per-line regex silently misses.
        # A per-character prose mask keeps code blocks exempt.
        prose_lines = {n for n, _ in outside_code(page)}
        mask = bytearray(len(page.text))
        position = 0
        for number, line in enumerate(page.text.splitlines(keepends=True), start=1):
            if number in prose_lines:
                mask[position : position + len(line)] = b"\x01" * len(line)
            position += len(line)

        # A page-specific mapping wins over the global one; see [page_overrides].
        overrides = page_overrides.get(page.rel, {})
        modified = False

        def replace(match: re.Match) -> str:
            nonlocal modified
            if not mask[match.start()]:
                return match.group(0)  # inside a code block
            text, url = match.group(1), match.group(2)
            path, fragment = split_target(url)

            if ASSET.search(path):
                stats["asset (left external)"] += 1
                return match.group(0)
            if path in delink:
                stats["delinked"] += 1
                modified = True
                return text
            if path in overrides:
                stats["page override"] += 1
                modified = True
                return f"[{text}]({relative(overrides[path], page.path)}{fragment})"
            if path in mapping:
                stats["planned" if path in planned else "ported"] += 1
                modified = True
                return f"[{text}]({relative(mapping[path], page.path)}{fragment})"

            unmapped[path] += 1
            return match.group(0)

        updated = MD_LINK.sub(replace, page.text)

        if modified:
            changed_pages += 1
            if not args.dry_run and not args.report:
                page.path.write_text(updated, encoding="utf-8")

    if args.report or unmapped:
        print("Unmapped upstream targets:" if unmapped else "All targets mapped.")
        for path, count in unmapped.most_common():
            print(f"  {count:3}  {path}")
        if unmapped:
            print("\nAdd these to scripts/importers/link_map.toml and re-run.")

    if not args.report:
        verb = "would rewrite" if args.dry_run else "rewrote"
        print(f"\n{verb} links on {changed_pages} page(s):")
        for kind, count in sorted(stats.items()):
            print(f"  {count:4}  {kind}")

    return 1 if unmapped else 0


if __name__ == "__main__":
    sys.exit(main())
