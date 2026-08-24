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

Brand blue has **one knob**. `--gemma-blue-seed` in `docs/stylesheets/gemma.css` is the only
colour to change; every step, both schemes and every component follow from it:

```css
--gemma-blue-seed: #0B57D0;
```

The steps are the seed mixed with white (black at the dark end) in sRGB, so they stay on the
seed's own hue. With the current seed:

| Step | Hex | Contrast | Role |
|---|---|---|---|
| `--gemma-blue-50` / `-100` | `#EBF2FB` / `#CEDDF6` | — | Tints, hover washes. |
| `--gemma-blue-300` | `#80A8E7` | 6.65:1 on ink | Dark-scheme text. |
| `--gemma-blue-500` | `#417CDA` | 4.10:1 under white | **Decorative only** — gradients, 2px rules. |
| `--gemma-blue-600` | `#2368D5` | 5.23:1 under white | Dark-scheme fills. |
| `--gemma-blue-700` | `#0B57D0` | 6.39:1 both ways | The primary. |
| `--gemma-blue-800` | `#0944A2` | 8.90:1 under white | Pressed / strong. |

Before committing a new seed, check it:

```bash
uv run python scripts/check_contrast.py --seed "#RRGGBB"
```

It prints the derived ramp and exits non-zero if any step misses the threshold its role needs,
so a colour change never silently breaks legibility.

Use the semantic roles in components, not the ramp directly:

| Token | Use |
|---|---|
| `--gemma-accent` | Brand colour that is **text-safe** in the current scheme. Prefer this. |
| `--gemma-accent-fill` | A fill that **carries white text**. Never point it at a light step. |
| `--gemma-white`, `--gemma-ink` | Surfaces. |
| `--gemma-blue`, `-deep`, `-light` | Legacy aliases into the ramp. Prefer the roles above. |

!!! warning "Contrast is symmetric"
    A colour too light to read *on* white is also too light to sit *behind* white text — it is
    the same ratio. The palette this replaced was built on `#3186FF` at 3.51:1, which failed
    both ways; that is what made it read neon. Fills carrying white text come from 600 or 700.

Scales: `--gemma-space-1…7`, `--gemma-radius-sm|md|lg|pill`, `--gemma-shadow-sm|md|lg`.

## Icons

Five icon sets ship with Zensical: **lucide** (line icons, the default here), **material**,
**fontawesome**, **octicons**, and **simple** (brand logos). Project-specific icons go in
`overrides/.icons/`.

**In prose**, use the shortcode:

```markdown
Run it locally :lucide-rocket: and you are done.
```

**In markup** — any card, grid or generated block — use the `icon()` macro instead:

```jinja
{% raw %}<span class="card-icon">{{ icon('lucide/rocket') }}</span>
{{ icon('simple/ollama', 'engine-logo') }}{% endraw %}
```

!!! warning "The shortcode does not work inside HTML"
    `:lucide-rocket:` is an *inline Markdown* construct. Inside a raw HTML block — which every
    card grid is — it stays literal text. Adding `markdown` to the block does render it, but
    `md_in_html` then wraps each card in a `<p>` and the grid layout breaks. The `icon()` macro
    runs in the Jinja preprocessor, before Markdown, so the SVG is already in place. It also
    reaches the grids generated from `data/*.yml`, where a shortcode cannot go at all.

Icons inherit `currentColor`, so set the colour on the parent and never on the SVG. Size comes
from CSS in `icons.css`; the macro strips the file's own `width`/`height` so the sets cannot
disagree.

!!! danger "Do not use emoji shortcodes"
    `:rocket:` renders as `<img src="https://cdn.jsdelivr.net/...twemoji...">` — an external CDN
    request per emoji, which breaks offline viewing and adds a third-party dependency. Raw
    Unicode emoji avoid the request but render in the system font, so they differ on every
    platform and cannot be styled. Use an icon set.

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

## Ending a page

Two headings, and only these two. Never `Summary and next steps`, never `Next steps`.

**`## Summary`** is a prose recap of what the reader just did. Optional -- use it on a tutorial
that walked through something, skip it on a reference page. No cards.

**`## What's next`** is always a `.card-grid`, always with an icon, and always last. Two to
four cards, and **every one must point forward** -- never back to a section the reader already
passed through. Linking across sections is fine; a Get started page may point at Learn.

```markdown
## What's next

<div class="card-grid">
<a class="card" href="../tuning/index.md">
  <span class="card-icon">{% raw %}{{ icon('lucide/graduation-cap') }}{% endraw %}</span>
  <div class="card-title">Fine-tune Gemma</div>
  <p class="card-desc">QLoRA, full fine-tunes and distributed training on your own data.</p>
</a>
</div>
```

A card pointing off-site takes the `↗` in its title and **no icon**, so the arrow stays the
signal that you are leaving:

```markdown
<a class="card" href="https://…" target="_blank" rel="noopener" data-model="gemma-4">
  <div class="card-title">Gemma on Vertex AI <span class="card-ext">↗</span></div>
  <p class="card-desc">Use Gemma open models on Google Cloud.</p>
</a>
```

Note `data-model` sits on the anchor as a plain attribute. `attr_list` syntax does not work
inside an HTML block, and `scripts/pinned_links.py` reads both forms.

Which pages get what:

| Section | Ends with |
|---|---|
| Get started, Learn | `What's next` on every page |
| Run, Fine-tune | `What's next` |
| Models | `Summary` where a recap helps; `What's next` only where the page genuinely leads on. Model cards get neither -- a spec sheet ends. |

!!! warning "Card grids are stripped from notebooks"
    A grid is site navigation: raw HTML plus {% raw %}`{{ icon() }}`{% endraw %} calls that only the Jinja
    preprocessor resolves, both of which render as literal text in Jupyter. `scripts/notebooks.py`
    drops any `card-grid` block and the heading above it, so a page registered as a notebook
    source needs no special handling.

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

## Prompt tokens

Gemma's control tokens are chips, not code. Colour carries the category, so a reader scanning
`get-started/prompt-formatting.md` can tell a turn marker from a tool call without reading it.

<p>
  <code class="token-turn">&lt;|turn|&gt;</code>
  <code class="token-role">system</code>
  <code class="token-multi">&lt;|image|&gt;</code>
  <code class="token-tool">&lt;|tool_call|&gt;</code>
  <code class="token-think">&lt;|think|&gt;</code>
</p>

| Class | Category | Covers |
|---|---|---|
| `.token-turn` | Turn structure | <code class="token-turn">&lt;&#124;turn&#124;&gt;</code> <code class="token-turn">&lt;turn&#124;&gt;</code> <code class="token-turn">&lt;start_of_turn&gt;</code> <code class="token-turn">&lt;end_of_turn&gt;</code> |
| `.token-role` | Roles | <code class="token-role">system</code> <code class="token-role">user</code> <code class="token-role">model</code> <code class="token-role">assistant</code> |
| `.token-multi` | Modality placeholders | <code class="token-multi">&lt;&#124;image&#124;&gt;</code> <code class="token-multi">&lt;&#124;audio&#124;&gt;</code> |
| `.token-tool` | Tool use | <code class="token-tool">&lt;&#124;tool&#124;&gt;</code> <code class="token-tool">&lt;&#124;tool_call&#124;&gt;</code> <code class="token-tool">&lt;&#124;tool_response&#124;&gt;</code> <code class="token-tool">&lt;&#124;"&#124;&gt;</code> |
| `.token-think` | Reasoning | <code class="token-think">&lt;&#124;think&#124;&gt;</code> <code class="token-think">&lt;&#124;channel&gt;thought</code> <code class="token-think">&lt;channel&#124;&gt;</code> |

**Inline**, in prose or a table cell, use `attr_list` on inline code:

```markdown
Token to indicate a system instruction: `system`{.token-role}
Enable reasoning with `<|think|>`{.token-think}
```

**In an example dialogue**, use a `<pre><code>` block with spans, so the surrounding prompt text
stays unstyled and line breaks are preserved. Escape the angle brackets:

```html
<pre><code><span class="token-turn">&lt;|turn&gt;</span><span class="token-role">user</span>
Describe this image: <span class="token-multi">&lt;|image|&gt;</span><span class="token-turn">&lt;turn|&gt;</span></code></pre>
```

Which renders as:

<pre><code><span class="token-turn">&lt;|turn&gt;</span><span class="token-role">user</span>
Describe this image: <span class="token-multi">&lt;|image|&gt;</span><span class="token-turn">&lt;turn|&gt;</span></code></pre>

Keep a token in one chip when the model treats it as one unit —
<code class="token-think">&lt;|channel&gt;thought</code> is a single chip, not a chip followed by
bare text. Do not use a fenced code block for a dialogue
that needs tokens: syntax highlighting and these classes cannot both apply.

Colours live in `gemma.css` as `--gemma-cat-*` triplets and are categorical on purpose — they
stay distinct from each other rather than being derived from the brand blue.

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

Add `.engine-grid--detailed` to show each blurb on the card instead of behind a hover, and wrap
the name and blurb in `.engine-body`. Use it where the grid *is* the page and the reader is
deciding rather than recognising — the Run overview does — and the compact form where the grid
is a shortcut past a list of names, as on the quickstart.

An entry with no `page` is a hosted endpoint: link straight to its `url`, and mark it up as an
external link.

### Links pinned to a model generation

Some external URLs name a generation — a Hugging Face collection, a provider's model page — and
go stale when the next one ships. Tag them so they can be found later:

```markdown
[Hugging Face](https://huggingface.co/collections/google/gemma-4){ data-model="gemma-4" }
```

In `data/frameworks.yml` the equivalent is a `pinned:` key on the entry. Either way:

```bash
uv run python scripts/pinned_links.py --model gemma-4
```

lists every one of them. Tag only genuinely version-pinned URLs — a project's home page is not
one, and marking it up would bury the links that do need revisiting.

Two things the name alone will mislead you about. `paligemma-2` and `shieldgemma-2` are
PaliGemma 2 and ShieldGemma 2, not Gemma 2; tag them `paligemma` and `shieldgemma`. And pages
registered in `scripts/notebooks.py` are exported to Jupyter, which renders `attr_list` as
literal text — leave those untagged.

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

!!! warning "Content tabs do not work inside a step"
    `pymdownx.tabbed` is a block-level extension and does not run inside an `md_in_html` block,
    so `=== "Label"` inside a `<li markdown>` renders as literal text above an un-tabbed code
    block. Use labelled fences inside the step, or move the tabs outside the list.

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
