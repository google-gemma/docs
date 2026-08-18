---
title: Gemma Model Family
description: What Gemma is, how the generations relate, and which variant was built from which base model.
models: [gemma-4]
---

# The Gemma model family

Gemma is Google DeepMind's family of **open models**: the weights are published, so you download
them and run them on your own hardware — a laptop, a phone, or a cluster you control. Nothing is
sent to a server unless you choose to send it.

The family has two halves. A **mainline generation** — Gemma 1 through Gemma 4 — is the general
purpose model most people want. Around each generation sits a set of **variants**: the same
architecture retrained or restructured for one job, like embeddings, safety classification or
vision-language tasks. A variant always descends from a specific generation, which is what the
tree below shows.

## How the family fits together

{% set generations = modeldata.families | selectattr('generation', 'defined') | sort(attribute='generation') %}
{% set independent = modeldata.families | selectattr('based_on', 'defined')
                                        | selectattr('based_on', 'none') | list %}

<!-- Generated from data/models.yml -- edit that file, not this tree. -->
<div class="model-tree">
<ol class="tree-spine">
{% for gen in generations %}
{% set kids = modeldata.families | selectattr('based_on', 'defined')
                                 | rejectattr('based_on', 'none')
                                 | selectattr('based_on', 'equalto', gen.id) | list %}
  <li class="tree-gen">
    <a class="tree-node tree-node--gen" href="../{{ gen.page }}">
      <span class="tree-gen-name">{{ gen.name }}</span>
      <span class="tree-gen-note">{{ gen.tagline }}</span>
    </a>
    {% if kids %}
    <ul class="tree-branch">
    {% for kid in kids %}
      <li><a class="tree-node" href="../{{ kid.page }}">
        <span class="tree-name">{{ kid.name }}</span>
        <span class="tree-tagline">{{ kid.tagline }}</span>
      </a></li>
    {% endfor %}
    </ul>
    {% else %}
    <p class="tree-none">No variants built from this generation.</p>
    {% endif %}
  </li>
{% endfor %}
</ol>

{% if independent %}
<ul class="tree-aside">
{% for fam in independent %}
  <li><a class="tree-node" href="../{{ fam.page }}">
    <span class="tree-name">{{ fam.name }}</span>
    <span class="tree-tagline">{{ fam.tagline }}</span>
  </a></li>
{% endfor %}
</ul>
{% endif %}
</div>

<p class="tree-legend">
Each variant sits under the generation its own model page names as its base. Models below the
dashed line are Gemma models whose pages describe a different architecture rather than a Gemma
generation.
</p>

## Gemma 4

The current generation, and where to start unless you need something a variant does.

<!-- Generated from data/models.yml -- edit that file, not this table. -->

| Variant | Parameters | Input | Context | Intended Platform |
|---|---|---|---|---|
{% for size in modeldata.gemma4.sizes -%}
| **{{ size.name }}** | {{ size.parameters }} | {{ size.modalities.input | map('capitalize') | join(', ') }} | {{ size.context }} | {{ size.platform_short }} |
{% endfor %}
!!! note "Audio and video support"
    Audio input is native to **E2B, E4B and 12B**. The **26B A4B** and **31B** models accept
    video as individual frames — that is, as images — but do not process audio.

[Run one now](../get-started/index.md) or read the [Gemma 4 model card](gemma-4/model-card.md)
for the full technical detail.

## Choosing between them

- **Just starting?** Take the current generation, instruction-tuned, at the smallest size that
  fits your hardware. It answers a wide range of prompts with no further training.
- **Need a specific job done?** Check whether a variant already does it — embeddings
  ([EmbeddingGemma](embeddinggemma/index.md)), safety labelling
  ([ShieldGemma](shieldgemma/index.md)), tool calling
  ([FunctionGemma](functiongemma/index.md)) — before fine-tuning a core model yourself.
- **On a phone or edge device?** [Gemma 3n](gemma-3n/index.md) and the Gemma 4 E2B and E4B sizes
  are built for it.

For the full dated history of every release, see [Releases](releases.md).
