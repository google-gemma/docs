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

"""Generate agent-facing documentation artifacts from the canonical Markdown pages.

The committed `docs/llms.txt` works with the normal Zensical preview. After a site build,
`--publish site` also writes raw Markdown beside the
rendered HTML.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path
from typing import Iterable

import yaml
from jinja2 import Environment, StrictUndefined

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
CONFIG_FILE = REPO_ROOT / "zensical.toml"
MODELS_FILE = REPO_ROOT / "data" / "models.yml"
FRAMEWORKS_FILE = REPO_ROOT / "data" / "frameworks.yml"

# Pages call `icon()`; the macros extension provides it during a Zensical build, but these raw
# routes are rendered here, in a Jinja environment of our own. Same source, same output.
sys.path.insert(0, str(REPO_ROOT))
from main import icon as _icon  # noqa: E402

LLMS_TARGET = DOCS_DIR / "llms.txt"
SITE_URL = "https://gemma.ai"
HIDDEN_PREFIXES = ("contributing/",)

H1 = re.compile(r"^#\s+", re.MULTILINE)


def split_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines()
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            metadata = yaml.safe_load("\n".join(lines[1:index])) or {}
            if not isinstance(metadata, dict):
                raise ValueError("frontmatter must be a mapping")
            return metadata, "\n".join(lines[index + 1 :]).lstrip("\n")
    raise ValueError("unterminated frontmatter")


def is_hidden_page(relative: str) -> bool:
    return relative.startswith(HIDDEN_PREFIXES)


def source_pages() -> dict[str, tuple[Path, dict, str]]:
    pages = {}
    for path in sorted(DOCS_DIR.rglob("*.md")):
        relative = path.relative_to(DOCS_DIR).as_posix()
        if is_hidden_page(relative):
            continue
        metadata, body = split_frontmatter(path.read_text(encoding="utf-8"))
        pages[relative] = (path, metadata, body)
    return pages


def walk_paths(node) -> Iterable[str]:
    if isinstance(node, str) and node.endswith(".md"):
        yield node
    elif isinstance(node, list):
        for item in node:
            yield from walk_paths(item)
    elif isinstance(node, dict):
        for value in node.values():
            yield from walk_paths(value)


def navigation_sections() -> list[tuple[str, list[str]]]:
    config = tomllib.loads(CONFIG_FILE.read_text(encoding="utf-8"))["project"]
    sections = []
    for item in config["nav"]:
        if isinstance(item, str):
            sections.append(("Docs", list(walk_paths(item))))
            continue
        for label, children in item.items():
            sections.append((str(label), list(dict.fromkeys(walk_paths(children)))))
    return sections


def route_for(relative: str) -> str:
    path = Path(relative)
    if path.name == "index.md":
        return "/" if path.parent == Path(".") else f"/{path.parent.as_posix()}/"
    return f"/{path.with_suffix('').as_posix()}/"


def markdown_route_for(relative: str) -> str:
    route = route_for(relative)
    return "/index.md" if route == "/" else f"{route.rstrip('/')}.md"


def display_title(metadata: dict, family_names: dict[str, str]) -> str:
    title = str(metadata.get("title", "Untitled"))
    models = metadata.get("models") or []
    if title.lower() in {"model card", "overview"} and len(models) == 1:
        family = str(models[0]).split("/", 1)[0]
        if family in family_names:
            return f"{family_names[family]} — {title}"
    return title


def model_data() -> dict:
    return yaml.safe_load(MODELS_FILE.read_text(encoding="utf-8"))


def framework_data() -> dict:
    return yaml.safe_load(FRAMEWORKS_FILE.read_text(encoding="utf-8"))


def render_llms() -> str:
    pages = source_pages()
    data = model_data()
    family_names = {str(item["id"]): str(item["name"]) for item in data["families"]}
    lines = [
        "# Gemma Documentation",
        "",
        "Official documentation for building with the Gemma family of open models.",
        "Start with Gemma 4 unless a page or the user's request names another generation.",
        "Follow the Markdown links below to read canonical page content without cloning the repository.",
    ]
    seen: set[str] = set()
    for section, paths in navigation_sections():
        entries = [path for path in paths if path in pages and path not in seen]
        if not entries:
            continue
        lines.extend(("", f"## {section}"))
        for relative in entries:
            seen.add(relative)
            _path, metadata, _body = pages[relative]
            title = display_title(metadata, family_names)
            description = " ".join(str(metadata.get("description", "")).split())
            url = f"{SITE_URL}{markdown_route_for(relative)}"
            suffix = f": {description}" if description else ""
            lines.append(f"- [{title}]({url}){suffix}")
    lines.extend(
        (
            "",
            "## Instructions for language model agents",
            "",
            "- Prefer the model and framework identifiers shown in the selected page's examples.",
            "- Check capability pages before assuming that every Gemma model accepts the same modalities.",
            "- Use the Run section for engine-specific commands and Fine-tune for training workflows.",
            "",
        )
    )
    return "\n".join(lines)


def render_raw(relative: str, metadata: dict, body: str) -> str:
    environment = Environment(undefined=StrictUndefined, autoescape=False, keep_trailing_newline=True)
    context = dict(metadata)
    context["modeldata"] = model_data()
    context["frameworkdata"] = framework_data()
    context["icon"] = _icon
    rendered = environment.from_string(body).render(**context).strip() + "\n"
    if not H1.search(rendered):
        rendered = f"# {metadata['title']}\n\n{rendered}"
    return rendered


def raw_targets(site_dir: Path) -> Iterable[tuple[Path, str]]:
    for relative, (_path, metadata, body) in source_pages().items():
        content = render_raw(relative, metadata, body)
        source_shape = site_dir / relative
        yield source_shape, content
        clean_route = markdown_route_for(relative).lstrip("/")
        clean_target = site_dir / clean_route
        if clean_target != source_shape:
            yield clean_target, content
        yield Path(f"{clean_target}.txt"), content


def generated_outputs() -> tuple[tuple[Path, str], ...]:
    return ((LLMS_TARGET, render_llms()),)


def write_generated() -> None:
    for path, content in generated_outputs():
        path.write_text(content, encoding="utf-8")
        print(f"wrote {path.relative_to(REPO_ROOT)}")


def check_generated() -> int:
    stale = []
    for path, expected in generated_outputs():
        if not path.exists() or path.read_text(encoding="utf-8") != expected:
            stale.append(path)
    for path in stale:
        print(f"stale: {path.relative_to(REPO_ROOT)}")
    return 1 if stale else 0


def publish(site_dir: Path) -> None:
    resolved = site_dir.resolve()
    if not resolved.is_relative_to(REPO_ROOT) or resolved == REPO_ROOT:
        raise ValueError("site directory must be a child of the repository")
    for path, content in raw_targets(resolved):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    print(f"published raw Markdown to {resolved.relative_to(REPO_ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify committed generated files")
    parser.add_argument("--publish", type=Path, help="write raw Markdown into a built site")
    args = parser.parse_args()
    if args.check:
        return check_generated()
    write_generated()
    if args.publish:
        publish(args.publish)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
