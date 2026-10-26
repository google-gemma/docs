#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Export explicitly marked Zensical Markdown code blocks to a Jupyter notebook.

This is intentionally a narrow exporter. Markdown remains canonical; only fences carrying the
`.notebook-cell` class become executable cells. Site-only regions can be wrapped in
`<!-- notebook:skip-start -->` and `<!-- notebook:skip-end -->` comments.
Tutorials supporting model switching (.model-swap) export multiple notebooks, one per supported
Gemma 4 model size (e.g. thinking-gemma-4-e2b.ipynb).
"""

from __future__ import annotations

import argparse
import json
import re
from functools import lru_cache
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = REPO_ROOT / "docs" / "notebooks"
MODELS_DATA = REPO_ROOT / "data" / "models.yml"
SITE_URL = "https://gemma.ai"

DEFAULT_SOURCE = REPO_ROOT / "docs" / "get-started" / "capabilities" / "thinking.md"
DEFAULT_MODEL = "gemma-4/e2b"

SOURCES: tuple[Path | tuple[Path, str], ...] = (
    DEFAULT_SOURCE,
    REPO_ROOT / "docs" / "get-started" / "capabilities" / "text.md",
    REPO_ROOT / "docs" / "get-started" / "capabilities" / "function-calling.md",
    REPO_ROOT / "docs" / "get-started" / "capabilities" / "image.md",
    REPO_ROOT / "docs" / "get-started" / "capabilities" / "video.md",
    REPO_ROOT / "docs" / "get-started" / "capabilities" / "audio.md",
    REPO_ROOT / "docs" / "get-started" / "capabilities" / "mtp.md",
    (REPO_ROOT / "docs" / "models" / "diffusiongemma" / "inference.md", "diffusiongemma-inference"),
    (REPO_ROOT / "docs" / "models" / "functiongemma" / "inference.md", "functiongemma-inference"),
    (REPO_ROOT / "docs" / "models" / "functiongemma" / "full-sequence.md", "functiongemma-full-sequence"),
    (REPO_ROOT / "docs" / "models" / "functiongemma" / "fine-tuning.md", "functiongemma-fine-tuning"),
    (REPO_ROOT / "docs" / "models" / "embeddinggemma" / "inference.md", "embeddinggemma-inference"),
    (REPO_ROOT / "docs" / "models" / "embeddinggemma" / "multimodal.md", "embeddinggemma-multimodal"),
    (REPO_ROOT / "docs" / "models" / "embeddinggemma" / "fine-tuning.md", "embeddinggemma-fine-tuning"),
    (REPO_ROOT / "docs" / "models" / "embeddinggemma" / "fine-tuning-multimodal.md", "embeddinggemma-fine-tuning-multimodal"),
    (REPO_ROOT / "docs" / "models" / "paligemma" / "inference.md", "paligemma-inference"),
    (REPO_ROOT / "docs" / "models" / "paligemma" / "fine-tuning.md", "paligemma-fine-tuning"),
    (REPO_ROOT / "docs" / "models" / "recurrentgemma" / "inference.md", "recurrentgemma-inference"),
    (REPO_ROOT / "docs" / "models" / "recurrentgemma" / "fine-tuning.md", "recurrentgemma-fine-tuning"),
    (REPO_ROOT / "docs" / "deployment" / "huggingface.md", "run-gemma-with-hf"),
    (REPO_ROOT / "docs" / "deployment" / "keras.md", "run-gemma-with-keras"),
    REPO_ROOT / "docs" / "tuning" / "qlora.md",
    REPO_ROOT / "docs" / "tuning" / "audio-qlora.md",
    REPO_ROOT / "docs" / "tuning" / "vision-qlora.md",
    REPO_ROOT / "docs" / "tuning" / "full-finetune.md",
)

FENCE_OPEN = re.compile(r"^(?P<fence>`{3,})(?P<language>[\w+-]*)(?P<attributes>.*)$")
SKIP_REGION = re.compile(
    r"<!--\s*notebook:skip-start\s*-->.*?<!--\s*notebook:skip-end\s*-->",
    re.DOTALL,
)
REQUIRES_REGION = re.compile(
    r"<!--\s*notebook:requires-start\s+(?P<modality>[a-z0-9_-]+)\s*-->[ \t]*\n?"
    r"(?P<content>.*?)"
    r"<!--\s*notebook:requires-end\s*-->[ \t]*\n?",
    re.DOTALL,
)
REQUIRES_COMMENT = re.compile(
    r"<!--\s*notebook:requires-(?:start\s+[a-z0-9_-]+|end)\s*-->[ \t]*\n?"
)
FENCE_REQUIRES = re.compile(r"\b(?:data-)?requires=[\"'](?P<modality>[a-z0-9_-]+)[\"']")
ADMONITION = re.compile(
    r"^!!!\s+(?P<label>[A-Za-z]+)(?:\s+\"[^\"]*\")?\n(?P<body>(?:[ \t]+.*(?:\n|$))*)",
    re.MULTILINE,
)
# `attr_list` attaches classes and attributes to inline spans -- `<|think|>`{.token-think}, or a
# link followed by { data-model="gemma-4" }. Jupyter has no such extension and renders the braces
# as literal text, so they are stripped on the way out: the site keeps the styling, the notebook
# keeps the words. Inline code spans are matched whole so braces inside backticks are left alone.
_ATTR_BRACE = r"[ \t]*\{[ \t]*(?:[.#][A-Za-z0-9_-]+|[A-Za-z_:][A-Za-z0-9_.:-]*=)[^}\n]*\}"
ATTR_LIST = re.compile(
    rf"(?P<code>(?P<ticks>`+)[^`\n]+?(?P=ticks))(?:{_ATTR_BRACE})?|(?<=[)\]]){_ATTR_BRACE}"
)
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
OUTPUT_IMAGE = re.compile(r"!\[[^\]]*\]\((?:[^)\n]*/)?output_[^/)\n]*\)[ \t]*\n?")
OUTPUT_AUDIO = re.compile(
    r"<audio\b(?=[^>]*\bclass=(?:\"[^\"]*\boutput\b[^\"]*\"|'[^']*\boutput\b[^']*')"
    r"|(?:(?!</audio>).)*?\bsrc=(?:\"(?:[^\"\n]*/)?output_[^/\"\n]*\"|'(?:[^'\n]*/)?output_[^/'\n]*'))"
    r"[^>]*>.*?</audio>[ \t]*\n?",
    re.DOTALL | re.IGNORECASE,
)
OUTPUT_VIDEO = re.compile(
    r"<video\b(?=[^>]*\bclass=(?:\"[^\"]*\boutput\b[^\"]*\"|'[^']*\boutput\b[^']*')"
    r"|(?:(?!</video>).)*?\bsrc=(?:\"(?:[^\"\n]*/)?output_[^/\"\n]*\"|'(?:[^'\n]*/)?output_[^/'\n]*'))"
    r"[^>]*>.*?</video>[ \t]*\n?",
    re.DOTALL | re.IGNORECASE,
)
RELATIVE_ASSET = re.compile(
    r"(?P<prefix>\]\(|(?:\b(?:src|href)=[\"']))(?:\.\.?/|/)*(?P<path>assets/[^)\s\"']+)"
)
# In Markdown tables, Jupyter/Colab table parsers split cells on unescaped `|` even inside inline
# code spans (`<|turn>`). Escaping `|` as `\|` inside inline code spans on table rows keeps table
# cells intact.
INLINE_CODE = re.compile(r"(`+)([^`\n]+?)\1")


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


@lru_cache(maxsize=1)
def load_models_data() -> dict:
    return yaml.safe_load(MODELS_DATA.read_text(encoding="utf-8"))


def get_gemma4_models() -> dict[str, dict]:
    data = load_models_data()
    models = {}
    for size in data.get("gemma4", {}).get("sizes", []):
        mid = f"gemma-4/{size['id']}"
        models[mid] = {
            "name": size["name"],
            "fullName": size.get("full_name", size["name"]),
            "ids": size.get("ids", {}),
            "inputs": size.get("modalities", {}).get("input", []),
        }
    return models


@lru_cache(maxsize=1)
def get_model_index() -> list[tuple[str, str, str]]:
    """Return list of (identifier_text, engine, model_id) sorted by length descending."""
    models = get_gemma4_models()
    index = []
    for mid, model in models.items():
        for engine, value in (model.get("ids") or {}).items():
            index.append((str(value), str(engine), mid))
    index.sort(key=lambda x: len(x[0]), reverse=True)
    return index


def extract_swappable_blocks(body: str) -> list[str]:
    lines = body.splitlines()
    blocks = []
    index = 0
    while index < len(lines):
        match = FENCE_OPEN.match(lines[index].rstrip("\r\n"))
        if not match:
            index += 1
            continue
        closing = index + 1
        while closing < len(lines) and lines[closing].strip() != match.group("fence"):
            closing += 1
        if closing == len(lines):
            break
        attributes = match.group("attributes")
        if ".model-swap" in attributes:
            blocks.append("".join(lines[index + 1 : closing]))
        index = closing + 1
    return blocks


def detect_engines_in_use(body: str) -> set[str]:
    index = get_model_index()
    if not index:
        return set()
    text_to_engine = {text: engine for text, engine, _ in index}
    pattern = re.compile("|".join(re.escape(text) for text, _, _ in index))
    blocks = extract_swappable_blocks(body)
    return {
        text_to_engine[match.group(0)]
        for block in blocks
        for match in pattern.finditer(block)
    }


def candidate_models(source: Path) -> list[str]:
    metadata, body = split_frontmatter(source.read_text(encoding="utf-8"))
    engines = detect_engines_in_use(body)
    if not engines:
        return []
    models = get_gemma4_models()
    requires = metadata.get("requires", "")
    candidates = []
    for mid, model in models.items():
        if requires and requires not in (model.get("inputs") or []):
            continue
        if all(engine in model.get("ids", {}) for engine in engines):
            candidates.append(mid)
    return candidates


def rewrite_block(block: str, target_model: str) -> str:
    models = get_gemma4_models()
    target_ids = models.get(target_model, {}).get("ids", {})
    index = get_model_index()
    text_to_engine = {text: engine for text, engine, _ in index}
    pattern = re.compile("|".join(re.escape(text) for text, _, _ in index))

    def repl(match: re.Match) -> str:
        text = match.group(0)
        engine = text_to_engine.get(text)
        if engine and engine in target_ids:
            return target_ids[engine]
        return text

    return pattern.sub(repl, block)


def model_supports_modality(model: str | None, modality: str) -> bool:
    if not model:
        return True
    models = get_gemma4_models()
    if model not in models:
        return True
    return modality in (models[model].get("inputs") or [])


def filter_requires_regions(text: str, model: str | None = None) -> str:
    def repl(match: re.Match) -> str:
        if model_supports_modality(model, match.group("modality")):
            return match.group("content")
        return ""

    text = REQUIRES_REGION.sub(repl, text)
    return REQUIRES_COMMENT.sub("", text)


def notebook_markdown(text: str) -> str:
    text = SKIP_REGION.sub("", text)
    text = REQUIRES_COMMENT.sub("", text)

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
    text = ATTR_LIST.sub(lambda m: m.group("code") or "", text)
    text = OUTPUT_IMAGE.sub("", text)
    text = OUTPUT_AUDIO.sub("", text)
    text = OUTPUT_VIDEO.sub("", text)
    text = RELATIVE_ASSET.sub(rf"\g<prefix>{SITE_URL}/\g<path>", text)
    lines = []
    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith("|"):
            line = INLINE_CODE.sub(
                lambda m: m.group(1) + re.sub(r"(?<!\\)\|", r"\|", m.group(2)) + m.group(1),
                line,
            )
        lines.append(line)
    return "".join(lines).strip()


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


def export_cells(body: str, model: str | None = None) -> list[dict]:
    body = SKIP_REGION.sub("", body)
    body = filter_requires_regions(body, model=model)
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
        req_match = FENCE_REQUIRES.search(attributes)
        if req_match and not model_supports_modality(model, req_match.group("modality")):
            index = closing + 1
            continue
        block = "".join(lines[index + 1 : closing])
        if model and ".model-swap" in attributes:
            block = rewrite_block(block, model)

        if ".notebook-cell" in attributes:
            pending = markdown_cell("".join(prose))
            if pending:
                cells.append(pending)
            prose = []
            cells.append(code_cell(match.group("language"), block))
        elif ".output" not in attributes:
            if model and ".model-swap" in attributes:
                prose.append(lines[index])
                prose.append(block)
                prose.append(lines[closing])
            else:
                prose.extend(lines[index : closing + 1])
        index = closing + 1

    pending = markdown_cell("".join(prose))
    if pending:
        cells.append(pending)
    return cells


def default_stem(source: Path) -> str:
    if source.parent.parent.name == "models" or source.stem in {"inference", "fine-tuning", "index"}:
        return f"{source.parent.name}-{source.stem}"
    return source.stem


LICENSE_CELLS: tuple[dict, ...] = (
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "##### Copyright 2026 Google LLC."
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"cellView": "form"},
        "outputs": [],
        "source": [
            "#@title Licensed under the Apache License, Version 2.0 (the \"License\");\n",
            "# you may not use this file except in compliance with the License.\n",
            "# You may obtain a copy of the License at\n",
            "#\n",
            "# https://www.apache.org/licenses/LICENSE-2.0\n",
            "#\n",
            "# Unless required by applicable law or agreed to in writing, software\n",
            "# distributed under the License is distributed on an \"AS IS\" BASIS,\n",
            "# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n",
            "# See the License for the specific language governing permissions and\n",
            "# limitations under the License.",
        ],
    },
)


def render(
    source: Path = DEFAULT_SOURCE,
    model: str | None = None,
    target_name: str | None = None,
) -> str:
    metadata, body = split_frontmatter(source.read_text(encoding="utf-8"))
    try:
        canonical_source = source.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        canonical_source = source.name

    if target_name is None:
        stem = default_stem(source)
        if model:
            target_name = f"{stem}-{model.replace('/', '-')}-it.ipynb"
        else:
            target_name = f"{stem}.ipynb"

    notebook = {
        "cells": [*LICENSE_CELLS, *export_cells(body, model=model)],
        "metadata": {
            "colab": {"name": target_name, "provenance": []},
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


def infer_model(name: str) -> str | None:
    match = re.search(r"-gemma-4-([a-z0-9-]+?)(?:-it)?\.ipynb$", name)
    if match:
        return f"gemma-4/{match.group(1)}"
    return None


def build_notebook_map(
    sources: tuple[Path | tuple[Path, str], ...] = SOURCES,
) -> dict[tuple[Path, str | None], Path]:
    mapping: dict[tuple[Path, str | None], Path] = {}
    for entry in sources:
        if isinstance(entry, tuple):
            source, stem = entry
            stem = stem.removesuffix(".ipynb")
        else:
            source = entry
            stem = default_stem(source)
        candidates = candidate_models(source)
        if candidates:
            for mid in candidates:
                slug = f"{mid.replace('/', '-')}"
                target = NOTEBOOKS_DIR / f"{stem}-{slug}.ipynb"
                mapping[(source, mid)] = target
        else:
            target = NOTEBOOKS_DIR / f"{stem}.ipynb"
            mapping[(source, None)] = target
    return mapping


NOTEBOOKS = build_notebook_map()


def generated_outputs() -> tuple[tuple[Path, str], ...]:
    """Return every registered output and its canonical rendered contents."""
    return tuple(
        (target, render(source, model=model, target_name=target.name))
        for (source, model), target in NOTEBOOKS.items()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", type=str, help="target Gemma model (e.g. gemma-4/e2b)")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if bool(args.source) != bool(args.output):
        parser.error("--source and --output must be provided together")
    if args.source:
        model = args.model or infer_model(args.output.name)
        outputs = ((args.output, render(args.source, model=model, target_name=args.output.name)),)
    else:
        outputs = generated_outputs()
    if args.check:
        stale = [
            target
            for target, expected in outputs
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
