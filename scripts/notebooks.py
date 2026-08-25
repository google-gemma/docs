#!/usr/bin/env python3
"""Export explicitly marked Zensical Markdown code blocks to a Jupyter notebook.

This is intentionally a narrow exporter. Markdown remains canonical; only fences carrying the
`.notebook-cell` class become executable cells. Site-only regions can be wrapped in
`<!-- notebook:skip-start -->` and `<!-- notebook:skip-end -->` comments.
"""

from __future__ import annotations

import argparse
import json
import re
from functools import lru_cache
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = REPO_ROOT / "docs" / "get-started" / "capabilities" / "thinking.md"
DEFAULT_TARGET = REPO_ROOT / "docs" / "notebooks" / "thinking.ipynb"
NOTEBOOKS = {
    DEFAULT_SOURCE: DEFAULT_TARGET,
    REPO_ROOT / "docs" / "get-started" / "capabilities" / "text.md":
        REPO_ROOT / "docs" / "notebooks" / "text.ipynb",
    REPO_ROOT / "docs" / "get-started" / "capabilities" / "function-calling.md":
        REPO_ROOT / "docs" / "notebooks" / "function-calling.ipynb",
    REPO_ROOT / "docs" / "get-started" / "capabilities" / "image.md":
        REPO_ROOT / "docs" / "notebooks" / "image.ipynb",
    REPO_ROOT / "docs" / "get-started" / "capabilities" / "video.md":
        REPO_ROOT / "docs" / "notebooks" / "video.ipynb",
    REPO_ROOT / "docs" / "get-started" / "capabilities" / "audio.md":
        REPO_ROOT / "docs" / "notebooks" / "audio.ipynb",
    REPO_ROOT / "docs" / "get-started" / "capabilities" / "mtp.md":
        REPO_ROOT / "docs" / "notebooks" / "mtp.ipynb",
    REPO_ROOT / "docs" / "tuning" / "qlora.md":
        REPO_ROOT / "docs" / "notebooks" / "qlora.ipynb",
    REPO_ROOT / "docs" / "tuning" / "full-finetune.md":
        REPO_ROOT / "docs" / "notebooks" / "full-finetune.ipynb",
    REPO_ROOT / "docs" / "tuning" / "vision-qlora.md":
        REPO_ROOT / "docs" / "notebooks" / "vision-qlora.ipynb",
}

FENCE_OPEN = re.compile(r"^(?P<fence>`{3,})(?P<language>[\w+-]*)(?P<attributes>.*)$")
SKIP_REGION = re.compile(
    r"<!--\s*notebook:skip-start\s*-->.*?<!--\s*notebook:skip-end\s*-->",
    re.DOTALL,
)
ADMONITION = re.compile(
    r"^!!!\s+(?P<label>[A-Za-z]+)(?:\s+\"[^\"]*\")?\n(?P<body>(?:[ \t]+.*(?:\n|$))*)",
    re.MULTILINE,
)
# `attr_list` attaches classes and attributes to inline spans -- `<|think|>`{.token-think}, or a
# link followed by { data-model="gemma-4" }. Jupyter has no such extension and renders the braces
# as literal text, so they are stripped on the way out: the site keeps the styling, the notebook
# keeps the words. Anchored to the closing `, ) or ] so ordinary prose braces are left alone.
ATTR_LIST = re.compile(r"(?<=[`)\]])[ \t]*\{[^}\n]*\}")
# A card grid is site navigation: raw HTML, and `{{ icon(...) }}` calls that only the Jinja
# preprocessor resolves. Both render as literal text in Jupyter, and a notebook has nowhere to
# navigate to anyway, so the grid and the heading introducing it are dropped.
#
# `^</div>` anchors to a line that *starts* with the closing tag. Card titles are indented, so
# this stops at the end of the grid rather than at the first `</div>` inside it.
CARD_GRID = re.compile(
    r"(?:^#{1,6}[^\n]*\n+)?^<div class=\"card-grid\">.*?^</div>[ \t]*$\n?",
    re.DOTALL | re.MULTILINE,
)
# `--8<-- "glossary.md"` is a snippets include and `*[TERM]: ...` an abbreviation definition.
# Neither is standard Markdown, so Jupyter would show both as literal text.
SNIPPET_INCLUDE = re.compile(r'^-{2,}8<-{2,}[ \t]+"[^"]+"[ \t]*$\n?', re.MULTILINE)
ABBR_DEF = re.compile(r"^\*\[[^\]]+\]:.*$\n?", re.MULTILINE)
# Shared reference links resolve against snippets/links.md, which the site appends to every page
# and a notebook never sees. Resolve them to ordinary inline links on the way out, so the export
# carries real URLs instead of unresolved brackets.
REFERENCE_LINK = re.compile(r"\[([^\]]+)\]\[([a-z0-9][a-z0-9-]*)\]")


def split_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines()
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            metadata = yaml.safe_load("\n".join(lines[1:index])) or {}
            return metadata, "\n".join(lines[index + 1 :]).lstrip("\n")
    raise ValueError("unterminated frontmatter")


@lru_cache(maxsize=1)
def shared_links() -> dict[str, str]:
    data = yaml.safe_load((REPO_ROOT / "data" / "links.yml").read_text(encoding="utf-8"))
    return {name: entry["url"] for name, entry in (data.get("links") or {}).items()}


def notebook_markdown(text: str) -> str:
    text = SKIP_REGION.sub("", text)

    def resolve_reference(match: re.Match) -> str:
        url = shared_links().get(match.group(2))
        return f"[{match.group(1)}]({url})" if url else match.group(0)

    text = REFERENCE_LINK.sub(resolve_reference, text)

    def replace_admonition(match: re.Match) -> str:
        body = "\n".join(line.strip() for line in match.group("body").splitlines()).strip()
        quoted = "\n> ".join(body.splitlines())
        return f"> **{match.group('label').title()}:** {quoted}\n"

    text = CARD_GRID.sub("", text)
    text = SNIPPET_INCLUDE.sub("", text)
    text = ABBR_DEF.sub("", text)
    text = ADMONITION.sub(replace_admonition, text)
    return ATTR_LIST.sub("", text).strip()


def markdown_cell(source: str) -> dict | None:
    source = notebook_markdown(source)
    if not source:
        return None
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code_cell(language: str, source: str) -> dict:
    source = source.rstrip() + "\n"
    if language in {"bash", "sh", "shell"}:
        source = "%%bash\n" + source
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"language": language or "python"},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def export_cells(body: str) -> list[dict]:
    body = SKIP_REGION.sub("", body)
    lines = body.splitlines(keepends=True)
    cells: list[dict] = []
    prose: list[str] = []
    index = 0
    while index < len(lines):
        match = FENCE_OPEN.match(lines[index].rstrip("\r\n"))
        if not match:
            prose.append(lines[index])
            index += 1
            continue

        closing = index + 1
        while closing < len(lines) and lines[closing].strip() != match.group("fence"):
            closing += 1
        if closing == len(lines):
            raise ValueError(f"unclosed code fence at line {index + 1}")

        attributes = match.group("attributes")
        block = "".join(lines[index + 1 : closing])
        if ".notebook-cell" in attributes:
            pending = markdown_cell("".join(prose))
            if pending:
                cells.append(pending)
            prose = []
            cells.append(code_cell(match.group("language"), block))
        elif ".output" not in attributes:
            prose.extend(lines[index : closing + 1])
        index = closing + 1

    pending = markdown_cell("".join(prose))
    if pending:
        cells.append(pending)
    return cells


def render(source: Path = DEFAULT_SOURCE) -> str:
    metadata, body = split_frontmatter(source.read_text(encoding="utf-8"))
    try:
        canonical_source = source.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        canonical_source = source.name
    notebook = {
        "cells": export_cells(body),
        "metadata": {
            "colab": {"name": f"{source.stem}.ipynb", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
            "gemma_docs": {
                "canonical_source": canonical_source,
                "title": metadata.get("title", source.stem),
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"


def generated_outputs() -> tuple[tuple[Path, str], ...]:
    """Return every registered output and its canonical rendered contents."""
    return tuple((target, render(source)) for source, target in NOTEBOOKS.items())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if bool(args.source) != bool(args.output):
        parser.error("--source and --output must be provided together")
    outputs = (
        ((args.output, render(args.source)),)
        if args.source
        else generated_outputs()
    )
    if args.check:
        stale = [
            target for target, expected in outputs
            if not target.exists() or target.read_text(encoding="utf-8") != expected
        ]
        for target in stale:
            print(f"stale: {target}")
        return bool(stale)
    for target, expected in outputs:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(expected, encoding="utf-8")
        print(f"wrote {target.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
