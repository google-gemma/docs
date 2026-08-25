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
            self.assertEqual(payload["cells"][1]["cell_type"], "code")

    def test_registered_notebooks_have_unique_outputs_and_code_cells(self):
        self.assertEqual(len(notebooks.NOTEBOOKS), 10)
        self.assertEqual(len(set(notebooks.NOTEBOOKS.values())), len(notebooks.NOTEBOOKS))
        for source, target in notebooks.NOTEBOOKS.items():
            with self.subTest(target=target.name):
                payload = json.loads(notebooks.render(source))
                self.assertTrue(
                    any(cell["cell_type"] == "code" for cell in payload["cells"]),
                    f"{source} exports no executable cells",
                )


if __name__ == "__main__":
    unittest.main()
