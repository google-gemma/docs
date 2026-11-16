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

"""Generate the Jinja lookup the page context header uses for model display names.

Zensical exposes `page.meta` to templates but not the macros plugin's YAML data, and it has no
hook mechanism to inject any. So the id -> display-name map is generated from data/models.yml
into a template partial, and `lint_docs.py --check generated` fails if the two drift apart.

That keeps data/models.yml the single source of truth: the partial is build output that happens
to be committed, in the same spirit as .lintbaseline.json.

    uv run python scripts/generate_model_names.py            # write
    uv run python scripts/generate_model_names.py --check    # verify, write nothing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path, PurePosixPath

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DATA = REPO_ROOT / "data" / "models.yml"
FRAMEWORKS_DATA = REPO_ROOT / "data" / "frameworks.yml"
TARGET = REPO_ROOT / "overrides" / "partials" / "model-names.html"

# The same data, for the browser: the model switcher rewrites identifiers in opted-in code
# blocks and needs the per-engine ids that only data/models.yml has.
JS_TARGET = REPO_ROOT / "docs" / "javascripts" / "model-data.js"

HEADER = """{#-
  Copyright 2026 Google LLC

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
-#}
{#-
  GENERATED FILE -- DO NOT EDIT.

  Written by scripts/generate_model_names.py from data/models.yml.
  Verified by: uv run python scripts/lint_docs.py --check generated

  `gemma_model_names` maps every family, size and framework id to its display name.
  `gemma_framework_short_names` maps tools with a compact UI label to that label.
  `gemma_model_urls` maps a family id to its canonical local documentation page.
  `gemma_framework_urls` maps a tool id to its project home.

  No size lists here: the strip shows the page's subject, and the size selector is built in
  JavaScript from what the page's marked code blocks can actually be rewritten to.
-#}
"""


def _map(name: str, entries: list[tuple[str, str]]) -> str:
    width = max((len(key) for key, _ in entries), default=0) + 2
    lines = [f'  {(chr(34) + key + chr(34)):<{width}}: {value},' for key, value in entries]
    return "{% set " + name + " = {\n" + "\n".join(lines) + "\n} %}\n"


def _site_url(source: str) -> str:
    """Convert a docs-relative Markdown source path to its clean site URL."""
    path = PurePosixPath(source)
    target = path.parent if path.name == "index.md" else path.with_suffix("")
    return f"/{target.as_posix().strip('/')}/"


def render() -> str:
    data = yaml.safe_load(MODELS_DATA.read_text(encoding="utf-8"))
    frameworks = yaml.safe_load(FRAMEWORKS_DATA.read_text(encoding="utf-8"))
    sizes = data["gemma4"]["sizes"]

    names: list[tuple[str, str]] = [
        (str(family["id"]), f'"{family["name"]}"') for family in data["families"]
    ]
    names += [(f"gemma-4/{size['id']}", f'"{size["name"]}"') for size in sizes]
    names += [
        (str(entry["id"]), f'"{entry["name"]}"') for entry in frameworks["frameworks"]
    ]

    model_urls = [
        (str(family["id"]), f'"{_site_url(str(family["page"]))}"')
        for family in data["families"]
        if family.get("page")
    ]
    framework_urls = [
        (str(entry["id"]), f'"{entry["url"]}"')
        for entry in frameworks["frameworks"]
        if entry.get("url")
    ]
    framework_short_names = [
        (str(entry["id"]), f'"{entry["short_name"]}"')
        for entry in frameworks["frameworks"]
        if entry.get("short_name")
    ]

    return (
        HEADER
        + _map("gemma_model_names", names)
        + "\n"
        + _map("gemma_model_urls", model_urls)
        + "\n"
        + _map("gemma_framework_urls", framework_urls)
        + "\n"
        + _map("gemma_framework_short_names", framework_short_names)
    )


def render_js() -> str:
    """The switcher's data: per-engine identifiers, so it can rewrite a snippet's model.

    Also carries each size's parameter label, architecture, structural dimensions and published
    memory figures, which the tuning lab (docs/javascripts/tuning-lab.js) turns into its parameter
    and VRAM estimates -- so those numbers come from data/models.yml rather than JavaScript.
    """
    data = yaml.safe_load(MODELS_DATA.read_text(encoding="utf-8"))
    gemma4 = data["gemma4"]
    vocab_size = gemma4["vocab_size"]
    memory_overhead = gemma4["memory_overhead"]
    payload = {}
    for size in gemma4["sizes"]:
        entry: dict[str, object] = {
            "name": size["name"],
            "fullName": size["full_name"],
            "context": size["context"],
            "platform": size["platform_short"],
            "parameters": size["parameters"],
            "architecture": size.get("architecture", ""),
            "vocabSize": vocab_size,
            "memoryOverhead": memory_overhead,
            "hiddenSize": size["hidden_size"],
            "layers": size["layers"],
        }
        for yaml_key, js_key in (
            ("ple_dim", "pleDim"),
            ("experts", "experts"),
            ("expert_dim", "expertDim"),
            ("lora_rank1_params", "loraRank1Params"),
        ):
            if yaml_key in size:
                entry[js_key] = size[yaml_key]
        entry["memory"] = size.get("memory", {})
        entry["ids"] = size.get("ids", {})
        entry["inputs"] = size["modalities"]["input"]
        payload[f"gemma-4/{size['id']}"] = entry
    body = json.dumps(payload, indent=2, sort_keys=False)
    return (
        "/**\n"
        " * Copyright 2026 Google LLC\n"
        " *\n"
        ' * Licensed under the Apache License, Version 2.0 (the "License");\n'
        " * you may not use this file except in compliance with the License.\n"
        " * You may obtain a copy of the License at\n"
        " *\n"
        " *     http://www.apache.org/licenses/LICENSE-2.0\n"
        " *\n"
        " * Unless required by applicable law or agreed to in writing, software\n"
        ' * distributed under the License is distributed on an "AS IS" BASIS,\n'
        " * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n"
        " * See the License for the specific language governing permissions and\n"
        " * limitations under the License.\n"
        " */\n\n"
        "/* GENERATED FILE -- DO NOT EDIT.\n"
        "   Written by scripts/generate_model_names.py from data/models.yml.\n"
        "   Verified by: uv run python scripts/lint_docs.py --check generated\n\n"
        "   Consumed by docs/javascripts/model-switcher.js. A model with no id for an engine\n"
        "   has no published build there -- the switcher must leave such snippets alone\n"
        "   rather than invent a repository name. */\n"
        f"window.GEMMA_MODELS = {body};\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify without writing")
    args = parser.parse_args()

    outputs = [(TARGET, render()), (JS_TARGET, render_js())]

    if args.check:
        stale = [
            path
            for path, expected in outputs
            if not path.exists() or path.read_text(encoding="utf-8") != expected
        ]
        for path in stale:
            print(
                f"{path.relative_to(REPO_ROOT)} is stale -- "
                "run: uv run python scripts/generate_model_names.py",
                file=sys.stderr,
            )
        if stale:
            return 1
        print(f"{len(outputs)} generated file(s) up to date")
        return 0

    for path, expected in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(expected, encoding="utf-8")
        print(f"wrote {path.relative_to(REPO_ROOT)} ({expected.count(chr(10))} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
