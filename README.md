# Gemma documentation

The Google DeepMind Gemma documentation, ported to
[Zensical](https://github.com/zensical/zensical).

## Local development

The project uses [uv](https://docs.astral.sh/uv/) and pins the documentation
toolchain in `uv.lock`.

```bash
uv sync --frozen
uv run --frozen python scripts/agent_docs.py
uv run --frozen zensical serve
```

The preview is available at <http://localhost:8000/>.
Zensical's preview server may download raw `.md` routes because of its local MIME mapping;
GitHub Pages serves those static files with their Markdown content type.

## Verification

Run these checks before committing documentation or styling changes:

```bash
uv run --frozen python -m unittest discover -s tests -p "test_*.py"
uv run --frozen python scripts/lint_docs.py --baseline
uv run --frozen python scripts/build_docs.py
```

`build_docs.py` runs the normal Zensical build and then publishes raw Markdown routes such as
`/get-started/index.md`. The committed `docs/llms.txt` is regenerated from page
frontmatter and verified by the `generated` lint check.

See `PLAN.md` for the migration roadmap and `docs/contributing/components.md`
for the CSS tokens and reusable components.
