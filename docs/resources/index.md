---
title: Resources
description: Demos, tools, community projects and the external repositories and collections worth knowing about.
cards:
  - title: "Gemma Cookbook"
    desc: "Official Jupyter notebooks covering inference, fine-tuning, RAG and function calling."
    url: "https://github.com/google-gemma/gemma-cookbook"
  - title: "Gemma Skills"
    desc: "Agent skills for building with Gemma and for training it, installable with one command."
    url: "https://github.com/google-gemma/gemma-skills"
  - title: "google-gemma on GitHub"
    desc: "The organization behind these docs: every repository, tool and community project."
    url: "https://github.com/google-gemma"
  - title: "Gemma 4 on Hugging Face"
    desc: "Every Gemma 4 checkpoint, including the QAT and GGUF collections."
    url: "https://huggingface.co/collections/google/gemma-4"
    model: gemma-4
---

# Resources

Everything that sits alongside the documentation: things you can try in the browser, projects
the community has built, and the repositories worth bookmarking.

## Demos and tools

<div class="card-grid">
<a class="card" href="vram-calculator.md">
  <div class="card-title">(V)RAM calculator</div>
  <p class="card-desc">What each Gemma 4 size needs at 16-, 8- and 4-bit, before context.</p>
</a>
<a class="card" href="vision-token-budget.md">
  <div class="card-title">Vision token budget</div>
  <p class="card-desc">How Gemma 4 allocates image tokens across resolutions and aspect ratios.</p>
</a>
</div>

## Community

<div class="card-grid">
<a class="card" href="../community/index.md">
  <div class="card-title">Gemmaverse</div>
  <p class="card-desc">Projects and showcases built on Gemma by the community.</p>
</a>
<a class="card" href="../community/gemma-skills.md">
  <div class="card-title">Gemma Skills</div>
  <p class="card-desc">Curated agent skills, system prompts and tool definitions.</p>
</a>
<a class="card" href="../community/discord.md">
  <div class="card-title">Discord</div>
  <p class="card-desc">Where Gemma developers ask questions and share what they are building.</p>
</a>
</div>

## Elsewhere

<div class="card-grid">
{% for card in cards %}
<a class="card" href="{{ card.url }}" target="_blank" rel="noopener">
  <div class="card-title">{{ card.title }} <span class="card-ext">↗</span></div>
  <p class="card-desc">{{ card.desc }}</p>
</a>
{% endfor %}
</div>

!!! tip "Adding to this page"
    The **Elsewhere** cards come from the `cards:` list in this page's frontmatter — add a
    title, description and URL and it appears. Demos and tools get their own page under
    `docs/resources/` and a card in the grid above.

## Contribute

The [component catalogue](../contributing/components.md) documents every building block these
pages are made from, the design tokens behind them, and the conventions to follow when adding
a page.
