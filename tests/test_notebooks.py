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

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import notebooks


class NotebookExportTests(unittest.TestCase):
    def test_only_marked_cells_are_executable(self):
        body = """# Guide

Intro.

```python { .notebook-cell .model-swap }
print("run")
```

```python
print("display only")
```

``` { .no-copy .output }
run
```
"""
        cells = notebooks.export_cells(body)
        code = [cell for cell in cells if cell["cell_type"] == "code"]
        self.assertEqual(len(code), 1)
        self.assertEqual("".join(code[0]["source"]), 'print("run")\n')
        markdown = "".join("".join(cell["source"]) for cell in cells if cell["cell_type"] == "markdown")
        self.assertIn('print("display only")', markdown)
        self.assertNotIn("\nrun\n", markdown)

    def test_site_only_region_is_removed(self):
        body = "Before\n<!-- notebook:skip-start -->\nSite UI\n<!-- notebook:skip-end -->\nAfter"
        cells = notebooks.export_cells(body)
        rendered = "".join(cells[0]["source"])
        self.assertIn("Before", rendered)
        self.assertIn("After", rendered)
        self.assertNotIn("Site UI", rendered)

    def test_nested_fence_remains_markdown(self):
        body = """````
Sample output:
```json
{"answer": 42}
```
````

```python { .notebook-cell }
print(42)
```
"""
        cells = notebooks.export_cells(body)
        self.assertEqual([cell["cell_type"] for cell in cells], ["markdown", "code"])
        self.assertIn("```json", "".join(cells[0]["source"]))

    def test_render_is_a_version_four_notebook(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "guide.md"
            source.write_text(
                "---\ntitle: Guide\n---\n# Guide\n\n```python { .notebook-cell }\nprint(1)\n```\n",
                encoding="utf-8",
            )
            payload = json.loads(notebooks.render(source))
            self.assertEqual(payload["nbformat"], 4)
            self.assertEqual(payload["cells"][:2], list(notebooks.LICENSE_CELLS))
            self.assertEqual(payload["cells"][3]["cell_type"], "code")

    def test_registered_notebooks_have_unique_outputs_and_code_cells(self):
        self.assertEqual(len(notebooks.SOURCES), 25)
        self.assertEqual(len(notebooks.NOTEBOOKS), 68)
        self.assertEqual(len(set(notebooks.NOTEBOOKS.values())), len(notebooks.NOTEBOOKS))
        for (source, model), target in notebooks.NOTEBOOKS.items():
            with self.subTest(target=target.name):
                payload = json.loads(notebooks.render(source, model=model, target_name=target.name))
                self.assertTrue(
                    any(cell["cell_type"] == "code" for cell in payload["cells"][2:]),
                    f"{target} exports no executable cells",
                )
                self.assertEqual(payload["metadata"]["colab"]["name"], target.name)

    def test_model_swap_rewrites_code_cells(self):
        body = """---
title: Guide
---
# Guide

```python { .notebook-cell .model-swap }
MODEL_ID = "google/gemma-4-E2B-it"
```

```python { .notebook-cell }
# Unswapped cell
print(MODEL_ID)
```
"""
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "guide.md"
            source.write_text(body, encoding="utf-8")
            payload = json.loads(notebooks.render(source, model="gemma-4/12b"))
            code_cells = [c for c in payload["cells"][2:] if c["cell_type"] == "code"]
            self.assertEqual(len(code_cells), 2)
            self.assertIn('MODEL_ID = "google/gemma-4-12B-it"', "".join(code_cells[0]["source"]))
            self.assertIn("# Unswapped cell", "".join(code_cells[1]["source"]))

    def test_audio_tutorial_filters_models(self):
        audio_source = notebooks.REPO_ROOT / "docs" / "get-started" / "capabilities" / "audio.md"
        candidates = notebooks.candidate_models(audio_source)
        self.assertEqual(candidates, ["gemma-4/e2b", "gemma-4/e4b", "gemma-4/12b"])
        self.assertNotIn("gemma-4/26b-a4b", candidates)
        self.assertNotIn("gemma-4/31b", candidates)

    def test_infer_model_from_output_filename(self):
        self.assertEqual(notebooks.infer_model("thinking-gemma-4-e2b-it.ipynb"), "gemma-4/e2b")
        self.assertEqual(notebooks.infer_model("thinking-gemma-4-26b-a4b-it.ipynb"), "gemma-4/26b-a4b")
        self.assertEqual(notebooks.infer_model("thinking-gemma-4-e2b.ipynb"), "gemma-4/e2b")
        self.assertEqual(notebooks.infer_model("mtp-gemma-4-e2b-it.ipynb"), "gemma-4/e2b")
        self.assertIsNone(notebooks.infer_model("full-finetune.ipynb"))

    def test_output_audio_and_video_tags_are_removed(self):
        body = """Before
<audio class="output" controls="controls" >
    <source src="data:audio/x-wav;base64,AAAA" />
    Your browser does not support the audio element.
</audio>
<audio controls>
  <source src="../../assets/embeddinggemma/output_roses-are.wav" type="audio/mpeg">
  Your browser does not support the audio element.
</audio>
<video controls>
  <source src="../../assets/embeddinggemma/output_cat_dozing.mp4" type="video/mp4">
  Your browser does not support the video element.
</video>
<audio controls>
    <source src="sample.wav" />
</audio>
<video controls>
    <source src="sample.mp4" />
</video>
After"""
        cells = notebooks.export_cells(body)
        rendered = "".join(cells[0]["source"])
        self.assertIn("Before", rendered)
        self.assertIn("After", rendered)
        self.assertNotIn('class="output"', rendered)
        self.assertNotIn("output_roses-are.wav", rendered)
        self.assertNotIn("output_cat_dozing.mp4", rendered)
        self.assertIn('<source src="sample.wav" />', rendered)
        self.assertIn('<source src="sample.mp4" />', rendered)

    def test_attr_list_stripped_without_eating_inline_code_braces(self):
        body = (
            'Pass an image as a dictionary `{"image": <source>}`, '
            'use `<|think|>`{.token-think}, and see [Kaggle](https://kaggle.com){ data-model="gemma-4" }.'
        )
        cells = notebooks.export_cells(body)
        rendered = "".join(cells[0]["source"])
        self.assertIn('`{"image": <source>}`', rendered)
        self.assertIn("`<|think|>`", rendered)
        self.assertNotIn(".token-think", rendered)
        self.assertNotIn("data-model", rendered)

    def test_relative_asset_paths_are_resolved(self):
        body = "![Shin Yunbok](../../assets/embeddinggemma/ink-wash-1.png)\n![](../assets/deployment/thali-indian-plate.jpg)"
        cells = notebooks.export_cells(body)
        rendered = "".join(cells[0]["source"])
        self.assertIn(
            "![Shin Yunbok](https://gemma.ai/assets/embeddinggemma/ink-wash-1.png)",
            rendered,
        )
        self.assertIn(
            "![](https://gemma.ai/assets/deployment/thali-indian-plate.jpg)",
            rendered,
        )


if __name__ == "__main__":
    unittest.main()
