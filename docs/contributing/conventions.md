---
title: Conventions
description: How this documentation stays consistent — where facts live, what is generated, and what the linter enforces.
---

# Conventions

[Components](components.md) covers what pages look like. This page covers how they stay
correct: where a fact is allowed to live, what is generated from it, and which of these rules a
machine enforces rather than a reviewer.

The through-line is simple. **A fact belongs in exactly one place.** Everything that repeats it
is generated from that place, and a linter fails when the two drift apart.

## Where facts live

Four YAML files under `data/` own everything that would otherwise be repeated across pages.

| File | Owns | Read by |
|---|---|---|
| `models.yml` | Every family, its sizes, lineage, context windows, memory, and per-engine checkpoint ids | Model tables, the family tree, the model selector, frontmatter validation |
| `frameworks.yml` | Every engine: display name, home page, our page, blurb, logo — plus `tuning_paths`, the routes shown on the fine-tuning overview | The engine grids on Run and Fine-tune, the page context strip, frontmatter validation |
| `links.yml` | Any URL used on more than one page | Reference links on every page, the pinned-link sweep |
| `glossary.yml` | Acronym definitions | Abbreviation tooltips on pages that opt in |

**Hand-editing a model table, an engine list or a repeated URL in a page is a bug**, not a
shortcut. Change the data file; the page follows.

Two rules keep the data honest:

- **Never invent a value to fill a gap.** KerasHub publishes no 12B preset and LiteRT-LM ships
  only E2B and E4B. Those absences are recorded by omission, and the selector simply offers
  fewer sizes on those pages. An invented id produces a command that 404s.
- **Cite anything inferred.** Every `based_on` in `models.yml` carries a comment naming the
  claim on that model's own page. A family whose page states no Gemma base has no `based_on`.

## What is generated

Nothing in this list should ever be edited by hand. Each has a generator, and
`lint_docs.py --check generated` fails when a committed file no longer matches its source.

| Output | Generator | From |
|---|---|---|
| `overrides/partials/model-names.html` | `generate_model_names.py` | `models.yml` |
| `docs/javascripts/model-data.js` | `generate_model_names.py` | `models.yml`, `frameworks.yml` |
| `snippets/links.md`, `snippets/glossary.md` | `generate_snippets.py` | `links.yml`, `glossary.yml` |
| `docs/llms.txt` and the raw `.md` routes | `agent_docs.py` | Page frontmatter and the nav |
| `docs/notebooks/*.ipynb` | `notebooks.py` | The ten registered tutorial pages |
| `overrides/.icons/**.svg` | `vendor_icons.py` | The icons pages actually reference |

Icons are the one entry checked by `--check icons` rather than `--check generated`, because
they are copied out of Zensical rather than rendered from our own data. They have to be
committed: the lint workflow installs only PyYAML and Jinja2, so an icon that exists solely
inside the Zensical package is missing there and takes the whole test suite down with it.
Referencing a new icon works immediately while authoring, then fails lint until it is vendored.

## Imported pages are frozen

Most pages were ported from the upstream documentation. A port changes formatting, never prose
or code.

- Every imported page records `source_url` in frontmatter.
- Every entry in `scripts/importers/sources.toml` is `frozen = true`. Frozen pages are never
  reconverted in bulk.
- A new page is added to the manifest unfrozen, imported, reviewed **once**, then frozen before
  any documentation-specific editing begins.
- Converter improvements affect future imports only. Repair existing pages by hand.

When a page is retired or merged, the manifest keeps a comment saying where it went, so the
upstream source is still findable.

## Frontmatter

`scripts/frontmatter_schema.py` is the single source of truth, and the linter enforces it. The
distinction it draws matters:

- **Authored** keys are facts only a person knows — `title`, `description`, `models`,
  `frameworks`, `prerequisites`.
- **Computed** keys are derivable from the file. Writing one by hand is an *error*, because a
  hand-written token count is wrong the moment the page is edited.

Vocabularies come from the data files rather than a hardcoded list, so adding a size to
`models.yml` extends what `models:` accepts automatically.

## Links that name a model

Some external URLs embed a generation and go stale when the next one ships. They are tagged so
they can be found later, in whichever form the context allows:

```yaml
# data/links.yml — preferred, declares it once beside the URL
hf-gemma4:
  url: https://huggingface.co/collections/google/gemma-4
  model: gemma-4
```

```markdown
[Kaggle](https://www.kaggle.com/models/google/gemma-4){ data-model="gemma-4" }
```

Inside a card grid, which is raw HTML, `attr_list` does not apply — put `data-model` on the
anchor as a plain attribute instead.

```bash
uv run python scripts/pinned_links.py --model gemma-4
```

lists every one of them. Tag only genuinely version-pinned URLs: a project's home page is not
one, and marking it up buries the links that do need revisiting.

## Markdown is the source, notebooks are output

Ten pages are exported to Jupyter. Markdown owns every subsequent edit; the `.ipynb` is a build
artifact.

Only fences marked `.notebook-cell` become executable cells. Everything else in the page is
site chrome, and `notebooks.py` strips what Jupyter cannot render — `attr_list` annotations,
card grids, snippet includes and abbreviation definitions. **Anything added to the authoring
vocabulary needs a matching strip**, or it lands in the notebook as literal text.

## Verification

Three commands, and CI runs all three:

```bash
uv run --frozen python -m unittest discover -s tests -p "test_*.py"
uv run --frozen python scripts/lint_docs.py --baseline
uv run --frozen python scripts/build_docs.py
```

`lint_docs.py --list` shows every check. The baseline in `.lintbaseline.json` is **empty**, and
should stay that way: it exists to let a known problem be carried temporarily, not to hide one.

Two extra tools answer questions rather than gate anything:

```bash
uv run python scripts/pinned_links.py      # what breaks when a new generation ships
uv run python scripts/check_contrast.py    # whether a new brand colour is legible
```

Visual work is not finished until it has been measured in a browser. Source inspection does not
tell you that a card's hover state puts a dark icon on a dark fill, or that a heading's
specificity loses to the theme's.
