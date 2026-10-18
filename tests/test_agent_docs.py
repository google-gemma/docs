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
        self.assertNotIn("llms-full", rendered)
        self.assertNotIn("/contributing/", rendered)

    def test_raw_markdown_renders_model_macros(self):
        pages = agent_docs.source_pages()
        _path, metadata, body = pages["get-started/index.md"]
        rendered = agent_docs.render_raw("get-started/index.md", metadata, body)
        self.assertIn("| **E2B** |", rendered)
        self.assertNotIn("{% for", rendered)
        self.assertNotIn("glossary.md", rendered)
        self.assertNotIn("## What's next", rendered)

    def test_raw_targets_include_inline_text_routes(self):
        site_dir = Path("/tmp/site")
        targets = {path.relative_to(site_dir).as_posix() for path, _ in agent_docs.raw_targets(site_dir)}
        self.assertIn("get-started.md", targets)
        self.assertIn("get-started.md.txt", targets)
        self.assertIn("get-started/best-practices.md.txt", targets)
        self.assertFalse(any(target.startswith("contributing/") for target in targets))


if __name__ == "__main__":
    unittest.main()
