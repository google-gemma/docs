---
title: Components
models: [gemma-4]
frameworks: [ollama]
description: The reusable building blocks these docs are made from, with live examples and the markup to copy.
---

# Components

Every visual element on this site comes from this catalogue. Build pages from these rather than
writing new CSS — if something you need isn't here, add it to
`docs/stylesheets/components.css` and document it on this page.

!!! note
    Colours, spacing, radii and shadows are **tokens**, defined once in
    `docs/stylesheets/gemma.css`. It is the only stylesheet allowed to contain a raw hex value,
    and `scripts/lint_docs.py --check css` enforces that.

## Colour tokens

Brand blue is a single ramp — one hue (212°), one saturation, lightness only:

| Step | Hex | Contrast |
|---|---|---|
| `--gemma-blue-50` / `-100` | `#E6EFF9` / `#C5D9F2` | Tints and hover washes. |
| `--gemma-blue-300` | `#71A4DF` | 6.19:1 on ink — dark-scheme text. |
| `--gemma-blue-500` | `#2C72C4` | 4.87:1 under white — decorative fills, the hero. |
| `--gemma-blue-600` | `#2561A7` | 6.28:1 under white — dark-scheme fills. |
| `--gemma-blue-700` | `#20538F` | 7.79:1 both ways. AAA. The primary. |

Use the semantic roles, not the ramp directly:

| Token | Use |
|---|---|
| `--gemma-accent` | Brand colour that is **text-safe** in the current scheme. Prefer this. |
| `--gemma-accent-fill` | A fill that **carries white text**. Never point it at a light step. |
| `--gemma-white`, `--gemma-ink` | Surfaces. |
| `--gemma-blue`, `-deep`, `-light` | Legacy aliases into the ramp. Prefer the roles above. |

!!! warning "Contrast is symmetric"
    The palette this replaced was built on `#3186FF` at 3.51:1. That number fails AA as text on
    white *and* as a fill behind white text — the same ratio, so a colour too light to read on
    can never be dark enough to read against. Fills that carry white text come from 600 or 700.

Scales: `--gemma-space-1…7`, `--gemma-radius-sm|md|lg|pill`, `--gemma-shadow-sm|md|lg`.

## Boxes

Every bordered container on the site — cards, engine tiles, tree nodes, tables — uses one
recipe. Only padding changes.

```css
background: var(--gemma-surface);
border: 1px solid var(--gemma-border);
border-radius: var(--gemma-radius-md);
```

```css
/* Interactive boxes, on hover and focus */
border-color: var(--gemma-accent);
box-shadow: var(--gemma-shadow-sm);
```

| Density | Padding | Used by |
|---|---|---|
| Compact | `--gemma-space-2` / `-3` | Engine tiles, tree nodes |
| Default | `--gemma-space-4` / `-5` | Cards |
| Panel | `--gemma-space-3` / `-4` | Admonitions, on `--gemma-surface-sunken` with a rail |

Do not reach for Material's `--md-default-fg-color--lighter` for a border. It is an 0.32-alpha
black that draws a visibly heavier line than `--gemma-border`; cards used it for a long time,
which is why they read as a different design language from everything around them.

## Cards

A grid of links. Used on the cookbook, model overviews and ecosystem pages.

<div class="card-grid">
<a class="card" href="#cards">
  <div class="card-title">Card title <span class="card-ext">↗</span></div>
  <p class="card-desc">One or two lines describing where this card leads.</p>
</a>
<a class="card" href="#cards">
  <div class="card-title">Second card</div>
  <p class="card-desc">The grid reflows automatically and collapses to one column on mobile.</p>
</a>
</div>

```html
<div class="card-grid">
<a class="card" href="…">
  <div class="card-title">Card title <span class="card-ext">↗</span></div>
  <p class="card-desc">One or two lines describing where this card leads.</p>
</a>
</div>
```

`.cookbook-grid` / `.cookbook-card` are aliases kept for converter output and existing pages.
Use `.card-grid` / `.card` in anything new.

## Badges

Inline chips for model attributes. Drive them from `data/models.yml` so a badge can never
disagree with the tables.

<p>
  <span class="badge">12B</span>
  <span class="badge">256K context</span>
  <span class="badge badge--accent">Recommended</span>
  <span class="badge badge--muted">Audio</span>
</p>

```html
<span class="badge">12B</span>
<span class="badge badge--accent">Recommended</span>
<span class="badge badge--muted">Audio</span>   <!-- capability this model lacks -->
```

`--muted` is how the model switcher (Step 4.2) will show a capability the selected model does
not have — struck through rather than hidden, so the reader learns the limitation.

## Page context strip

The quiet band under every H1. You never write this markup; it is rendered from frontmatter by
`overrides/partials/page-context.html`. The strip on *this* page is the live example.

```yaml
models: [gemma-4]        # what the page is about
frameworks: [ollama]     # primary tools the page teaches, linked to their home pages
requires: audio          # narrows the selector to models that accept audio
```

Two kinds of item sit on the line, and keeping them apart is the design:

| Item | Comes from | Meaning |
|---|---|---|
| `Models` | `models` frontmatter | What the page is **about**. Links to the local model page. |
| `Uses` | `frameworks` frontmatter | Primary tools the page teaches. Links out to each project. |
| `Examples` | the page's own code | What **you** want the commands to target. A preference. |

Pairs are separated by a 1px rule, values all share one weight, and labels are muted — 16px of
space alone was not enough to stop `Frameworks Sentence Transformers` reading as a single run
of grey text.

Keep `frameworks` editorial rather than exhaustive. Name the user-facing stack needed to follow
the page, not every imported or transitive dependency. For example, a Transformers tutorial that
imports PyTorch tensors should say `frameworks: [transformers]`; add `pytorch` only when the page
actually teaches its APIs or workflow.

`Examples` is deliberately *not* derived from `models:`. Those answer different questions:
the Ollama page is about Gemma 4, but its commands could target anything Ollama publishes. So
the selector is built from what the page's marked blocks can actually be rewritten to — which
means it disappears on pages with nothing switchable rather than claiming a size the page never
mentioned.

### Swappable code blocks

Add `{ .model-swap }` to a fence and the selector rewrites its model identifiers:

````markdown
```bash { .model-swap }
ollama pull gemma4:12b
```
````

Substitution is **opt-in per block** because a page's code was validated as written. Unmarked
blocks are never touched — which is what keeps the tag table on `deployment/ollama.md` listing
every tag while the `pull` command above it follows the selector.

A model is only offered if it has a published id for **every** engine appearing in the page's
marked blocks, so switching can never rewrite some snippets and silently leave others behind.
If an intentionally separate example lacks one size—such as a QAT checkpoint—mark it
`{ .model-fixed }` and explain the limitation beside it instead of unnecessarily narrowing the
page's primary examples.

Identifier variants live under each size's `ids` mapping in `data/models.yml`. Use a distinct
flavour when syntax has different semantics—for example `hf_base` for training, `hf_qat` for a
quantization-aware Transformers checkpoint, and `gguf_qat` for its llama.cpp equivalent. The
generator exposes those variants to the switcher; do not encode size substitutions by hand in
JavaScript.

The choice is remembered across the site and reflected in the page URL. If a remembered model
is unavailable, the examples remain at their authored default and the picker explains why;
there is no silent substitution. Use `{ .model-fixed }` to acknowledge an intentionally static
block when the same page has swappable examples for that engine.

### Exportable notebook cells

Markdown remains the source of truth for tutorials. Add `.notebook-cell` only to fences that
should become executable cells in the generated notebook:

````markdown
```python { .notebook-cell .model-swap }
print("This becomes a code cell; the authored model is retained.")
```
````

Unmarked fences remain part of a Markdown cell, and `{ .output }` blocks are omitted because a
notebook recreates them when it runs. Add the generated notebook's Colab URL to frontmatter to
show a quiet, accessible Colab icon in the page context strip:

```yaml
colab: https://colab.research.google.com/github/google-gemma/docs/blob/main/docs/notebooks/tutorial.ipynb
```

Wrap any other site-only controls in `<!-- notebook:skip-start -->` and
`<!-- notebook:skip-end -->`. Register the source and output in `scripts/notebooks.py`, then
generate every registered notebook with `uv run python scripts/notebooks.py`; use `--check`
during verification. Do not edit the generated `.ipynb`
directly.

## Engine grid

The "run it your way" tiles on the quickstart. A compact relative of `.card` for when the reader
is picking one of a dozen tools rather than reading about one.

Generated from `data/frameworks.yml` — never hand-write the tiles. An engine appears once it has
`page`, `blurb` and `mark` keys; the page chooses which ids to feature and in what order, since
that is editorial rather than a fact about the framework.

```jinja
{% raw %}{% set featured = ["ollama", "llamacpp"] %}
<ul class="engine-grid">
{% for id in featured %}{% for fw in frameworkdata.frameworks if fw.id == id %}
  <li><a class="engine" href="../{{ fw.page }}" title="{{ fw.blurb }}">
    <span class="engine-mark" aria-hidden="true">{{ fw.mark }}</span>
    <span class="engine-name">{{ fw.short_name | default(fw.name) }}</span>
  </a></li>
{% endfor %}{% endfor %}
</ul>{% endraw %}
```

`mark` is a two- or three-character monogram standing in for a real vendor logo. Replacing
`.engine-mark`'s text with an `<img>` needs no other markup change. Add `.engine--all` to a
trailing tile to make it read as an action rather than another engine.

## Model family tree

The lineage visual on the models overview. Generated from `data/models.yml` `families`:
`generation` orders the mainline spine, and `based_on` attaches each variant to the generation
its own model page names as its base.

Interaction is pure CSS — hovering or keyboard-focusing one generation dims the others. The
markup underneath is an ordered list of generations each containing a list of descendants, so it
still reads correctly with no CSS at all.

```html
<div class="model-tree">
<ol class="tree-spine">
  <li class="tree-gen">
    <a class="tree-node tree-node--gen" href="…">
      <span class="tree-gen-name">Gemma 4</span>
      <span class="tree-gen-note">One line about the generation.</span>
    </a>
    <ul class="tree-branch">
      <li><a class="tree-node" href="…">
        <span class="tree-name">DiffusionGemma</span>
        <span class="tree-tagline">What it is for.</span>
      </a></li>
    </ul>
  </li>
</ol>
<ul class="tree-aside">…</ul>   <!-- families with no Gemma generation as a base -->
</div>
```

Never add a `based_on` a model's own page does not state, and never invent a release date to
order the tree — the spine already orders it. A family whose page describes a non-Gemma
architecture takes `based_on: null` and renders below the dashed rule.

!!! warning "List containers need extra specificity"
    The theme styles lists with `[dir="ltr"] .md-typeset ol li` (0,2,2) and
    `.md-typeset ul:not([hidden])` (0,2,1). A plain `.md-typeset .my-grid` is only (0,2,0) and
    silently loses, so `display: grid` never applies and items keep their `1.25em` indent.
    Qualify the tag — `.md-typeset ul.my-grid:not([hidden])` — for any list-based component.

## Steps

A numbered walkthrough. Built on an ordered list, so it still reads correctly without CSS.

<ol class="steps" markdown>
<li markdown>
**Install the dependencies**

Anything can go inside a step, including code blocks.
</li>
<li markdown>
**Run the model**

The connecting rail is drawn automatically between steps.
</li>
</ol>

```html
<ol class="steps" markdown>
<li markdown>
**Step title**

Body content, including code blocks.
</li>
</ol>
```

The `markdown` attribute is required — it tells `md_in_html` to keep parsing Markdown inside
the tag.

## Figures

A framed diagram with a caption.

```html
<figure class="figure" markdown>
![Alt text](../assets/example.png)
<figcaption>What the diagram shows.</figcaption>
</figure>
```

For a diagram that needs different artwork per theme, mark the pair and let CSS choose:

```html
![Alt text](../assets/diagram-light.png){ .only-light }
![Alt text](../assets/diagram-dark.png){ .only-dark }
```

## Video

Emitted automatically by `scripts/importers/convert_notebook.py` from a standalone `[Video](youtube…)`
link. Full content width, 16:9, lazy-loaded, privacy-mode host.

```html
<div class="video-embed">
  <iframe src="https://www.youtube-nocookie.com/embed/VIDEO_ID"
          title="Video" loading="lazy" allowfullscreen></iframe>
</div>
```

## Model-card header

Centres the banner, resource links and License/Authors line on a model card. Also emitted by
the converter — you shouldn't need to write it by hand.

```html
<div class="model-card-header" markdown>
![Banner](../assets/banner.png)

[Hugging Face](…) \| [GitHub](…)

**License** : [Apache 2.0](…)
</div>
```

## Admonitions

Standard Zensical admonitions, used throughout. The converter produces these from upstream's
`> NOTE:` blockquotes and `> [!NOTE]` callouts.

!!! note
    Neutral supporting information.

!!! tip
    A recommendation or shortcut.

!!! warning
    Something that will cost the reader time or money if ignored.

## Motion

Shared component transitions live in `docs/stylesheets/motion.css`, including a blanket
`prefers-reduced-motion` opt-out — anything added later inherits it. Motion is feedback
(something is interactive, something changed), never decoration.

The isolated homepage animation also checks that preference before it starts.

To check the reduced-motion path, enable *Reduce motion* in your OS accessibility settings and
reload; hover effects should become instant.

## Stylesheet layout

| File | Owns |
|---|---|
| `gemma.css` | Tokens. The only file with raw hex. |
| `layout.css` | Page grid, sidebars, typography scale |
| `nav.css` | Header, tabs, sidebars, table of contents |
| `content.css` | Tables, admonitions, code, content tabs |
| `components.css` | Cards, badges, figures, steps, embeds |
| `engine-grid.css` | The quickstart's engine tiles |
| `model-tree.css` | The model family lineage tree |
| `page-context.css` | The applicability bar above every H1 |
| `tokens.css` | Chat-token badges (`<start_of_turn>` etc.) |
| `home.css` | Landing page: chrome, backdrop, hero |
| `motion.css` | Transitions and the reduced-motion opt-out |

Each stays under 250 lines so it can be read end-to-end before editing. Adding a stylesheet
for a new concern is the right move; growing one into a catch-all is not.
