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

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import agent_docs


class AgentDocsTests(unittest.TestCase):
    def test_routes_match_rendered_pages(self):
        self.assertEqual(agent_docs.markdown_route_for("index.md"), "/index.md")
        self.assertEqual(agent_docs.markdown_route_for("models/index.md"), "/models.md")
        # A section index collapses to the section route, so the quickstart is /get-started.md.
        self.assertEqual(
            agent_docs.markdown_route_for("get-started/index.md"),
            "/get-started.md",
        )
        # A non-index page inside a section keeps its own file name.
        self.assertEqual(
            agent_docs.markdown_route_for("get-started/best-practices.md"),
            "/get-started/best-practices.md",
        )

    def test_llms_uses_absolute_markdown_links(self):
        rendered = agent_docs.render_llms()
        self.assertTrue(rendered.startswith("# Gemma Documentation\n"))
        self.assertIn("https://gemma.ai/get-started.md", rendered)
        self.assertIn("- [Overview](https://gemma.ai/deployment.md):", rendered)
        self.assertNotIn("[Gemma 4 — Overview](https://gemma.ai/deployment.md)", rendered)
        self.assertIn(
            "- [DiffusionGemma — Generate output](https://gemma.ai/models/diffusiongemma/inference.md):",
            rendered,
        )
        self.assertIn(
            "- [RecurrentGemma — Fine-tuning using JAX and Flax](https://gemma.ai/models/recurrentgemma/fine-tuning.md):",
            rendered,
        )
        self.assertNotIn("llms-full", rendered)
        self.assertNotIn("/contributing/", rendered)

    def test_display_title_scopes_family_prefix_to_model_pages(self):
        family_names = {
            "gemma-4": "Gemma 4",
            "diffusiongemma": "DiffusionGemma",
            "recurrentgemma": "RecurrentGemma",
            "gemma-aps": "Gemma APS",
        }
        self.assertEqual(
            agent_docs.display_title(
                {"title": "Overview", "models": ["gemma-4"]},
                family_names,
                "deployment/index.md",
            ),
            "Overview",
        )
        self.assertEqual(
            agent_docs.display_title(
                {"title": "Model card", "models": ["gemma-4"]},
                family_names,
                "models/gemma-4/model-card.md",
            ),
            "Gemma 4 — Model card",
        )
        self.assertEqual(
            agent_docs.display_title(
                {"title": "Generate output", "models": ["diffusiongemma"]},
                family_names,
                "models/diffusiongemma/inference.md",
            ),
            "DiffusionGemma — Generate output",
        )
        self.assertEqual(
            agent_docs.display_title(
                {"title": "Fine-tuning using JAX and Flax", "models": ["recurrentgemma"]},
                family_names,
                "models/recurrentgemma/fine-tuning.md",
            ),
            "RecurrentGemma — Fine-tuning using JAX and Flax",
        )
        self.assertEqual(
            agent_docs.display_title(
                {
                    "title": "Gemma for abstractive proposition segmentation (APS)",
                    "models": ["gemma-aps"],
                },
                family_names,
                "models/gemma-aps/index.md",
            ),
            "Gemma for abstractive proposition segmentation (APS)",
        )

    def test_raw_markdown_renders_model_macros(self):
        pages = agent_docs.source_pages()
        _path, metadata, body = pages["get-started/index.md"]
        rendered = agent_docs.render_raw("get-started/index.md", metadata, body)
        self.assertIn("| **E2B** |", rendered)
        self.assertNotIn("{% for", rendered)
        self.assertNotIn("glossary.md", rendered)
        self.assertNotIn("## What's next", rendered)
        self.assertIn("https://gemma.ai/assets/gemma-mark.png", rendered)

    def test_raw_markdown_resolves_relative_asset_paths(self):
        rendered = agent_docs.render_raw(
            "models/embeddinggemma/fine-tuning-multimodal.md",
            {"title": "Fine-Tune Multimodal"},
            "![Shin Yunbok](../../assets/embeddinggemma/ink-wash-1.png)\n",
        )
        self.assertIn(
            "![Shin Yunbok](https://gemma.ai/assets/embeddinggemma/ink-wash-1.png)",
            rendered,
        )

    def test_raw_markdown_resolves_shared_reference_links(self):
        pages = agent_docs.source_pages()
        _path, metadata, body = pages["models/gemma-4/index.md"]
        rendered = agent_docs.render_raw("models/gemma-4/index.md", metadata, body)
        self.assertIn(
            "[Hugging Face](https://huggingface.co/collections/google/gemma-4)",
            rendered,
        )
        self.assertIn("[commercial use](/resources/terms)", rendered)
        self.assertNotIn("[hf-gemma4]", rendered)
        self.assertNotIn("[gemma-terms]", rendered)

        _path, metadata, body = pages["models/gemma-3/index.md"]
        rendered = agent_docs.render_raw("models/gemma-3/index.md", metadata, body)
        self.assertIn(
            "[Gemma Prohibited Use Policy](/resources/prohibited_use_policy)",
            rendered,
        )
        self.assertNotIn("[prohibited-use]", rendered)

    def test_raw_targets_include_inline_text_routes(self):
        site_dir = Path("/tmp/site")
        targets = {path.relative_to(site_dir).as_posix() for path, _ in agent_docs.raw_targets(site_dir)}
        self.assertIn("get-started.md", targets)
        self.assertIn("get-started.md.txt", targets)
        self.assertIn("get-started/best-practices.md.txt", targets)
        self.assertFalse(any(target.startswith("contributing/") for target in targets))


if __name__ == "__main__":
    unittest.main()
