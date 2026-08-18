"""List every external link that is pinned to a model generation.

When a new Gemma generation ships, some external URLs go stale — a Hugging Face collection,
an OpenRouter model page, a provider's model doc. They are scattered across pages and there
is no way to tell them apart from durable links by looking at the URL alone. So we mark them,
and this script is the readout:

    uv run python scripts/pinned_links.py                  # everything, grouped by generation
    uv run python scripts/pinned_links.py --model gemma-4  # just the ones to revisit

Two sources are scanned:

  data/frameworks.yml   entries carrying `pinned: <family-id>`
  docs/**/*.md          links annotated `[text](url){ data-model="<family-id>" }`

This is a report, not a gate. It never fails a build; it answers "what has to change for the
next generation?" in one command.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
FRAMEWORKS = REPO_ROOT / "data" / "frameworks.yml"
MODELS = REPO_ROOT / "data" / "models.yml"

# [label](url){ ... data-model="gemma-4" ... } -- attr_list, so it survives into the HTML as a
# real attribute and can be found in the built site too.
TAGGED_LINK = re.compile(
    r"\[(?P<label>[^\]]*)\]\((?P<url>[^)\s]+)[^)]*\)\{[^}]*data-model=\"(?P<model>[^\"]+)\"[^}]*\}"
)


def known_families() -> set[str]:
    data = yaml.safe_load(MODELS.read_text(encoding="utf-8"))
    return {str(f["id"]) for f in data.get("families", [])}


def from_frameworks() -> list[tuple[str, str, str, str]]:
    data = yaml.safe_load(FRAMEWORKS.read_text(encoding="utf-8"))
    rows = []
    for entry in data.get("frameworks", []):
        if "pinned" in entry:
            rows.append(
                (str(entry["pinned"]), "data/frameworks.yml", str(entry["name"]), str(entry["url"]))
            )
    return rows


def from_markdown() -> list[tuple[str, str, str, str]]:
    rows = []
    for path in sorted(DOCS_DIR.rglob("*.md")):
        rel = path.relative_to(REPO_ROOT).as_posix()
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for match in TAGGED_LINK.finditer(line):
                rows.append(
                    (match["model"], f"{rel}:{lineno}", match["label"], match["url"])
                )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", help="only show links pinned to this family id, e.g. gemma-4")
    args = parser.parse_args()

    rows = from_frameworks() + from_markdown()
    families = known_families()

    unknown = sorted({model for model, *_ in rows if model not in families})
    if args.model:
        rows = [row for row in rows if row[0] == args.model]

    if not rows:
        target = f" pinned to {args.model}" if args.model else ""
        print(f"No external links{target} are tagged.")
        return 0

    by_model: dict[str, list[tuple[str, str, str]]] = {}
    for model, where, label, url in rows:
        by_model.setdefault(model, []).append((where, label, url))

    total = 0
    for model in sorted(by_model):
        entries = sorted(by_model[model])
        total += len(entries)
        print(f"\n{model}  ({len(entries)} link{'s' if len(entries) != 1 else ''})")
        print("-" * 78)
        for where, label, url in entries:
            print(f"  {where}")
            print(f"      {label or '(no label)'} -> {url}")

    print(f"\n{total} pinned link(s).")
    if unknown:
        print(f"\nWarning: tagged with unknown family id(s): {', '.join(unknown)}")
        print(f"Known ids come from data/models.yml: {', '.join(sorted(families))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
