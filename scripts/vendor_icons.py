"""Copy every icon the docs reference into `overrides/.icons/`.

`main.py`'s `icon()` macro can read icons straight out of the installed Zensical package, which
is convenient while authoring but wrong to depend on:

  - The lint workflow installs only PyYAML and Jinja2, by design. Zensical is not there, so an
    `icon()` call raised `not found. Available sets:` and took the test suite with it.
  - `templates/.icons/` is Zensical's internal layout, with no stability guarantee.

So the icons in use are vendored and committed, the same way `llms.txt` and the notebooks are.
`main.py` looks in `overrides/.icons/` first and falls back to the package, so a new icon works
immediately while authoring; `lint_docs.py --check icons` then asks for it to be vendored.

    uv run python scripts/vendor_icons.py            # copy what pages reference
    uv run python scripts/vendor_icons.py --prune    # also delete ones nothing references
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import shutil
from pathlib import Path
from typing import Iterator

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
FRAMEWORKS = REPO_ROOT / "data" / "frameworks.yml"
VENDOR_DIR = REPO_ROOT / "overrides" / ".icons"

# An `icon(...)` call, captured to the end of the line. The name is not always the first thing
# inside it -- `icon(card.icon | default('lucide/external-link'))` puts it inside a filter -- so
# the call span is extracted first and every quoted name inside it taken.
ICON_CALL = re.compile(r"icon\((.*)")
ICON_NAME = re.compile(r"['\"]([a-z0-9]+/[a-z0-9-]+)['\"]")


def _yaml_icon_values(node) -> Iterator[str]:
    """Every value of an `icon:` key, at any depth.

    Covers data/frameworks.yml and page frontmatter alike, including the per-card `icon:` on
    a `cards:` list.
    """
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "icon" and isinstance(value, str) and "/" in value:
                yield value
            else:
                yield from _yaml_icon_values(value)
    elif isinstance(node, list):
        for item in node:
            yield from _yaml_icon_values(item)


def _frontmatter(text: str):
    if not text.startswith("---"):
        return None
    lines = text.splitlines()
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            try:
                return yaml.safe_load("\n".join(lines[1:index]))
            except yaml.YAMLError:
                return None
    return None


def referenced_icons() -> set[str]:
    """Every icon name the docs ask for, from pages and from data/frameworks.yml."""
    names: set[str] = set()
    for path in DOCS_DIR.rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        for call in ICON_CALL.findall(text):
            names |= set(ICON_NAME.findall(call))
        names |= set(_yaml_icon_values(_frontmatter(text)))
    names |= set(_yaml_icon_values(yaml.safe_load(FRAMEWORKS.read_text(encoding="utf-8"))))
    return names


def package_icon_root() -> Path | None:
    spec = importlib.util.find_spec("zensical")
    if not (spec and spec.origin):
        return None
    root = Path(spec.origin).parent / "templates" / ".icons"
    return root if root.is_dir() else None


def missing() -> set[str]:
    """Referenced icons that are not vendored. Answerable without Zensical installed."""
    return {n for n in referenced_icons() if not (VENDOR_DIR / f"{n}.svg").is_file()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prune", action="store_true", help="delete vendored icons nothing references")
    args = parser.parse_args()

    wanted = referenced_icons()
    source = package_icon_root()
    if source is None:
        print("zensical is not installed, so there is nothing to copy from.")
        gaps = missing()
        if gaps:
            print("Missing from overrides/.icons/: " + ", ".join(sorted(gaps)))
            return 1
        print(f"All {len(wanted)} referenced icons are already vendored.")
        return 0

    copied = 0
    for name in sorted(wanted):
        target = VENDOR_DIR / f"{name}.svg"
        origin = source / f"{name}.svg"
        if not origin.is_file():
            print(f"  NOT IN ZENSICAL: {name}")
            continue
        text = origin.read_text(encoding="utf-8")
        if not target.is_file() or target.read_text(encoding="utf-8") != text:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
            copied += 1

    removed = 0
    if args.prune and VENDOR_DIR.is_dir():
        for path in sorted(VENDOR_DIR.rglob("*.svg")):
            name = path.relative_to(VENDOR_DIR).with_suffix("").as_posix()
            if name not in wanted:
                path.unlink()
                removed += 1

    print(f"{len(wanted)} icons referenced, {copied} written, {removed} pruned")
    gaps = missing()
    if gaps:
        print("Still missing: " + ", ".join(sorted(gaps)))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
