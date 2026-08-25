"""Render data/links.yml and data/glossary.yml into the snippets pages include.

Two files, injected by `pymdownx.snippets`:

  snippets/links.md      Markdown reference definitions. Auto-appended to every page, because
                         a definition renders as nothing at all -- it only resolves `[text][id]`
                         where one is used.
  snippets/glossary.md   Abbreviation definitions. NOT auto-appended: `abbr` marks up *every*
                         occurrence of a term, so pages opt in with `--8<-- "glossary.md"`.

Run directly to regenerate; `lint_docs.py --check generated` fails if either has gone stale.
"""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
LINKS_DATA = REPO_ROOT / "data" / "links.yml"
GLOSSARY_DATA = REPO_ROOT / "data" / "glossary.yml"
SNIPPETS_DIR = REPO_ROOT / "snippets"
LINKS_TARGET = SNIPPETS_DIR / "links.md"
GLOSSARY_TARGET = SNIPPETS_DIR / "glossary.md"

HEADER = "<!-- GENERATED from {source} by scripts/generate_snippets.py -- do not edit. -->\n\n"


def link_data() -> dict[str, dict]:
    return yaml.safe_load(LINKS_DATA.read_text(encoding="utf-8"))["links"]


def glossary_data() -> dict[str, str]:
    return yaml.safe_load(GLOSSARY_DATA.read_text(encoding="utf-8"))["glossary"]


def render_links() -> str:
    lines = [HEADER.format(source="data/links.yml")]
    for name, entry in link_data().items():
        lines.append(f"[{name}]: {entry['url']}\n")
    return "".join(lines)


def render_glossary() -> str:
    lines = [HEADER.format(source="data/glossary.yml")]
    for term, definition in glossary_data().items():
        lines.append(f"*[{term}]: {definition}\n")
    return "".join(lines)


def generated_outputs() -> tuple[tuple[Path, str], ...]:
    return ((LINKS_TARGET, render_links()), (GLOSSARY_TARGET, render_glossary()))


def main() -> None:
    SNIPPETS_DIR.mkdir(exist_ok=True)
    for target, content in generated_outputs():
        target.write_text(content, encoding="utf-8")
        print(f"wrote {target.relative_to(REPO_ROOT)} ({len(content.splitlines())} lines)")


if __name__ == "__main__":
    main()
