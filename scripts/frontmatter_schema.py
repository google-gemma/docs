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

"""The frontmatter schema every page in docs/ must satisfy.

Single source of truth enforced by `lint_docs.py`.
Documented for humans in PLAN.md, "Frontmatter schema".

Two kinds of key live here, and the distinction is the whole design:

  AUTHORED   Facts only a person knows -- what the page is about, which models it applies
             to, what you should read first. Hand-written, validated here.

  COMPUTED   Facts derivable from the file itself -- token count and last edit date.
             Hand-writing these guarantees they go stale, so authoring one is
             an *error*. The build hook in PLAN.md Step 3.2 fills them in.

Vocabularies are read from the repository rather than hardcoded, so adding a model
directory or a size in data/models.yml extends the allowed values automatically.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
MODELS_DATA = REPO_ROOT / "data" / "models.yml"
FRAMEWORKS_DATA = REPO_ROOT / "data" / "frameworks.yml"
SOURCES = Path(__file__).resolve().parent / "importers" / "sources.toml"

# Descriptions feed llms.txt, search results, and page cards. Too short says nothing;
# too long gets truncated everywhere it is used.
DESCRIPTION_MIN = 40
DESCRIPTION_MAX = 200

# `models` separates a family from a size within it: `gemma-4` is the whole family,
# `gemma-4/12b` is one model in it. A bare `12b` would be ambiguous -- Gemma 3 and Gemma 4
# both ship a 12B. `/` rather than `:` so the value stays a plain string in every YAML
# parser; `[gemma-4:12b]` risks being read as a mapping in flow context.
MODEL_SEPARATOR = "/"

# Theme and authoring keys Zensical itself consumes. Allowed, not validated by us.
THEME_KEYS = frozenset({"template", "hide", "cards", "search", "icon", "subtitle", "image"})

# Reserved for the Step 3.2 build hook. Authoring one is an error: a hand-written token
# count is wrong the moment the page is edited.
COMPUTED_KEYS = frozenset({"estimated_tokens", "last_updated"})

URL = re.compile(r"^https?://\S+$")
SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


@dataclass(frozen=True)
class Field:
    name: str
    required: bool
    kind: str  # "str" | "list" | "date"
    doc: str


SCHEMA: tuple[Field, ...] = (
    Field("title", True, "str", "Page title. Rendered as the H1 when the page has none."),
    Field("description", True, "str", "One sentence. Feeds llms.txt, search, and cards."),
    Field("models", False, "list", "Models the page is about. Omit when it is about none."),
    Field("requires", False, "str", "Input modality the page needs, narrowing `models`."),
    Field("name", False, "str", "Stable kebab-case id for agents to reference."),
    Field("frameworks", False, "list", "Engines/libraries the page shows code for."),
    Field("prerequisites", False, "list", "Pages to read first, as docs-relative paths."),
    Field("source_url", False, "str", "Upstream page this was ported from."),
    Field("colab", False, "str", "Generated notebook URL for the page context strip."),
)

REQUIRED = tuple(f.name for f in SCHEMA if f.required)
KNOWN = frozenset(f.name for f in SCHEMA) | THEME_KEYS


@cache
def model_families() -> frozenset[str]:
    """Every family id. Derived from `docs/models/`, so adding a model directory is enough."""
    return frozenset(p.name for p in (DOCS_DIR / "models").iterdir() if p.is_dir())


@cache
def model_vocabulary() -> frozenset[str]:
    """Every accepted `models` value: each family, plus `<family>/<size>` where sizes exist.

    Gemma 4's sizes live under the top-level `gemma4` key; every other family carries its own
    `sizes` list. A family with no sizes recorded stays addressable at family level.
    """
    data = yaml.safe_load(MODELS_DATA.read_text(encoding="utf-8"))
    sizes = {f"gemma-4{MODEL_SEPARATOR}{size['id']}" for size in data["gemma4"]["sizes"]}
    for family in data.get("families", []):
        for size in family.get("sizes") or ():
            sizes.add(f"{family['id']}{MODEL_SEPARATOR}{size['id']}")
    return frozenset(model_families() | sizes)


@cache
def input_modalities() -> frozenset[str]:
    """Every input a Gemma 4 size accepts -- the vocabulary for `requires`."""
    data = yaml.safe_load(MODELS_DATA.read_text(encoding="utf-8"))
    return frozenset(
        modality
        for size in data["gemma4"]["sizes"]
        for modality in size["modalities"]["input"]
    )


@cache
def frameworks() -> dict[str, dict]:
    """id -> entry, from data/frameworks.yml."""
    data = yaml.safe_load(FRAMEWORKS_DATA.read_text(encoding="utf-8"))
    return {str(entry["id"]): entry for entry in data["frameworks"]}


@cache
def models_with_modality(modality: str) -> tuple[str, ...]:
    """Gemma 4 sizes accepting a given input, e.g. audio -- E2B, E4B and 12B, not 31B."""
    data = yaml.safe_load(MODELS_DATA.read_text(encoding="utf-8"))
    return tuple(
        f"gemma-4{MODEL_SEPARATOR}{size['id']}"
        for size in data["gemma4"]["sizes"]
        if modality in size["modalities"]["input"]
    )


@cache
def ported_sources() -> dict[str, str]:
    """docs-relative path -> upstream URL, from sources.toml."""
    manifest = tomllib.loads(SOURCES.read_text(encoding="utf-8"))
    return {entry["path"]: entry["source"] for entry in manifest["page"]}


@dataclass
class Problem:
    key: str
    message: str


def _as_list(value) -> list | None:
    return value if isinstance(value, list) else None


def validate(frontmatter: dict, *, page_paths: set[str] | None = None) -> list[Problem]:
    """Check one page's frontmatter. `page_paths` enables prerequisite resolution."""
    problems: list[Problem] = []

    for key in REQUIRED:
        if key not in frontmatter:
            problems.append(Problem(key, f"missing required key: {key}"))

    for key in sorted(frontmatter):
        if key in COMPUTED_KEYS:
            problems.append(
                Problem(key, f"`{key}` is computed at build time -- remove it from the page")
            )
        elif key not in KNOWN:
            problems.append(Problem(key, f"unknown key: {key}"))

    description = frontmatter.get("description")
    if isinstance(description, str):
        if not DESCRIPTION_MIN <= len(description) <= DESCRIPTION_MAX:
            problems.append(
                Problem(
                    "description",
                    f"description is {len(description)} chars; "
                    f"want {DESCRIPTION_MIN}-{DESCRIPTION_MAX}",
                )
            )
        if "](" in description or description.startswith("#"):
            problems.append(Problem("description", "description contains markdown syntax"))
    elif description is not None:
        problems.append(Problem("description", "description must be a string"))

    models = frontmatter.get("models")
    if models is not None:
        listed = _as_list(models)
        if listed is None:
            problems.append(Problem("models", "models must be a list"))
        elif not listed:
            problems.append(
                Problem("models", "models is empty; omit the key when a page is about no model")
            )
        else:
            vocabulary = model_vocabulary()
            for value in listed:
                text = str(value)
                if text in vocabulary:
                    continue
                family, _, size = text.partition(MODEL_SEPARATOR)
                if size and family in model_families():
                    hint = f"{family} has no size {size!r} in data/models.yml"
                elif size:
                    hint = f"no family {family!r} in docs/models/"
                elif any(v.endswith(f"{MODEL_SEPARATOR}{text}") for v in vocabulary):
                    # A bare size: the most common mistake, and the one worth naming exactly.
                    suggestion = next(v for v in sorted(vocabulary) if v.endswith(f"/{text}"))
                    hint = f"a size needs its family -- did you mean {suggestion!r}?"
                else:
                    hint = "not a known family; see docs/models/ and data/models.yml"
                problems.append(Problem("models", f"unknown model {text!r}: {hint}"))

    requires = frontmatter.get("requires")
    if requires is not None:
        if not isinstance(requires, str):
            problems.append(Problem("requires", "requires must be a single modality string"))
        elif requires not in input_modalities():
            known = ", ".join(sorted(input_modalities()))
            problems.append(Problem("requires", f"unknown modality {requires!r}; expected {known}"))
        elif not frontmatter.get("models"):
            problems.append(
                Problem("requires", "requires narrows `models`, so `models` must name a family")
            )

    listed_frameworks = frontmatter.get("frameworks")
    if listed_frameworks is not None:
        listed = _as_list(listed_frameworks)
        if listed is None:
            problems.append(Problem("frameworks", "frameworks must be a list"))
        else:
            known = frameworks()
            for value in listed:
                if str(value) not in known:
                    problems.append(
                        Problem(
                            "frameworks",
                            f"unknown framework: {value} (add it to data/frameworks.yml first)",
                        )
                    )

    prerequisites = frontmatter.get("prerequisites")
    if prerequisites is not None:
        listed = _as_list(prerequisites)
        if listed is None:
            problems.append(Problem("prerequisites", "prerequisites must be a list"))
        elif page_paths is not None:
            for value in listed:
                if str(value) not in page_paths:
                    problems.append(Problem("prerequisites", f"no such page: {value}"))

    name = frontmatter.get("name")
    if name is not None and not SLUG.match(str(name)):
        problems.append(Problem("name", f"name must be kebab-case: {name}"))

    source_url = frontmatter.get("source_url")
    if source_url is not None and not URL.match(str(source_url)):
        problems.append(Problem("source_url", f"source_url must be an http(s) URL: {source_url}"))

    colab = frontmatter.get("colab")
    if colab is not None and not URL.match(str(colab)):
        problems.append(Problem("colab", f"colab must be an http(s) URL: {colab}"))

    return problems
