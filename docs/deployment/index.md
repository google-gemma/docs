---
title: Overview
description: Choose an execution framework and a Gemma variant for your hardware, then run your first generation request.
models: [gemma-4]
source_url: "https://ai.google.dev/gemma/docs/run.md.txt"
---
There are two key decisions to make when you want to run a Gemma model:
1) what Gemma variant you want to run, and 2) what AI execution framework you
are going to use to run it? A key issue in making both these decisions has to do
with what hardware you and your users have available to run the model.

This overview helps you navigate these decisions and start working with Gemma
models. The general steps for running a Gemma model are as follows:

<ol class="steps" markdown>
<li markdown>
**[Choose a framework](index.md#1-choose-a-framework)**

Pick the runtime that matches your hardware — a hosted endpoint, a desktop app, a Python
library or an inference server.
</li>
<li markdown>
**[Select a Gemma variant](index.md#2-select-a-gemma-variant)**

Decide on a family, a size and a quantization level. Start with the smallest
instruction-tuned core model that could work.
</li>
<li markdown>
**[Run generation and inference requests](index.md#3-run-generation-and-inference-requests)**

Follow your framework's guide, and format prompts the way the variant expects.
</li>
</ol>

## 1. Choose a framework

Gemma runs on a wide range of ecosystem tools. Which one suits you depends on your hardware —
cloud GPUs or a laptop — and whether you want to write Python or open an application. Every
option below runs the same weights, so you can move between them later.

<!-- Generated from data/frameworks.yml -- edit that file, not these grids. -->
{% macro engines(ids) %}
<ul class="engine-grid engine-grid--detailed">
{% for id in ids %}{% for fw in frameworkdata.frameworks if fw.id == id %}
{% set page = fw.get('page') %}
  <li><a class="engine" href="{% if page %}../{{ page }}{% else %}{{ fw.url }}{% endif %}"
         {% if not page %}target="_blank" rel="noopener"{% endif %}>
    <span class="engine-mark" aria-hidden="true">
      {%- if fw.get('icon') %}{{ icon(fw.icon, 'engine-logo') }}{% else %}{{ fw.mark }}{% endif -%}
    </span>
    <span class="engine-body">
      <span class="engine-name">{{ fw.short_name | default(fw.name) }}{% if not page %} ↗{% endif %}</span>
      <span class="engine-blurb">{{ fw.blurb }}</span>
    </span>
  </a></li>
{% endfor %}{% endfor %}
</ul>
{% endmacro %}

### Try it without installing anything

Hosted endpoints. No download, no GPU — useful for evaluating a model before committing
hardware to it.

{{ engines(["ai-studio", "openrouter"]) }}

### Run it on your own machine

Desktop applications and local runtimes. These use quantized formats such as GGUF, so a laptop
is usually enough.

{{ engines(["lmstudio", "ollama", "llamacpp", "mlx", "litert"]) }}

### Build and fine-tune in Python

The libraries you reach for when Gemma is part of an application or a training run.

{{ engines(["transformers", "unsloth", "keras"]) }}

Also see [Tunix](https://tunix.readthedocs.io/en/latest/) for tuning in JAX.

### Serve it at scale

Inference servers and managed platforms for production traffic.

{{ engines(["vllm", "sglang", "google-cloud", "gemini-api"]) }}

For enterprise deployments, Google Cloud also offers the
[Gemini Enterprise Agent Platform](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/open-models/use-gemma).

Make sure your intended model format — Keras built-in, Safetensors, or GGUF — is supported by
the framework you pick.

## 2. Select a Gemma variant

Gemma models are available in several variants and sizes, including the
foundation or [core](../models/gemma-4/index.md) Gemma models, and more
specialized model variants such as
[PaliGemma](../models/paligemma/index.md) and
[DataGemma](../models/datagemma/index.md), and many variants
created by the AI developer community on sites such as
[Kaggle](https://www.kaggle.com/models?query=gemma) and
[Hugging Face](https://huggingface.co/models?search=gemma). If you are unsure
about what variant you should start with, select the latest Gemma
[core](../models/gemma-4/index.md) instruction-tuned (IT) model with
the lowest number of parameters. This type of Gemma model has low compute
requirements and be able to respond to a wide variety of prompts without
requiring additional development.

Consider the following factors when choosing a Gemma variant:

- **Gemma core, and other variant families such as PaliGemma and EmbeddingGemma** : *Recommend Gemma (core).* Gemma variants beyond the core version have the same architecture as the core model, and are trained to perform better at specific tasks. Unless your application or goals align with the specialization of a specific Gemma variant, it is best to start with a Gemma core, or base, model.
- **Instruction-tuned (IT), pre-trained (PT), fine-tuned (FT), mixed
  (mix)** : *Recommend IT.*
    - *Instruction-tuned* (IT) Gemma variants are models that have been trained to respond to a variety of instructions or requests in human language. These model variants are the best place to start because they can respond to prompts without further model training.
    - *Pre-trained* (PT) Gemma variants are models that have been trained to make inferences about language or other data, but have not been trained to follow human instructions. These models require additional training or tuning to be able to perform tasks effectively, and are meant for researchers or developers who want to study or develop the capabilities of the model and its architecture.
    - *Fine-tuned* (FT) Gemma variants can be considered IT variants, but are typically trained to perform a specific task, or perform well on a specific generative AI benchmark. The PaliGemma variant family includes a number of FT variants.
    - *Mixed* (mix) Gemma variants are versions of PaliGemma models that have been instruction tuned with a variety of instructions and are suitable for general use.
- **Parameters** : *Recommend smallest number available*. In general, the more parameters a model has, the more capable it is. However, running larger models requires larger and more complex compute resources, and generally slows down development of an AI application. Unless you have already determined that a smaller Gemma model cannot meet your needs, choose a one with a small number of parameters.
- **Quantization levels:** *Recommend half precision (16-bit), except for
  tuning*. Quantization is a complex topic that boils down to what size and precision of data, and consequently how much memory a generative AI model uses for calculations and generating responses. After a model is trained with high-precision data, which is typically 32-bit floating point data, models like Gemma can be modified to use lower precision data such as 16, 8 or 4-bit sizes. These quantized Gemma models can still perform well, depending on the complexity of the tasks, while using significantly less compute and memory resources. However, tools for tuning quantized models are limited and may not be available within your chosen AI development framework. Typically, you must fine-tune a model like Gemma at full precision, then quantize the resulting model.

For a list of key, Google-published Gemma models, see the
[Getting started with Gemma models](../get-started/index.md), Gemma model list.

## 3. Run generation and inference requests

After you have selected an AI execution framework and a Gemma variant, you can
start running the model, and prompting it to generate content or complete tasks.
For more information on how to run Gemma with a specific framework, see the
guides linked in the [Choose a framework](index.md#1-choose-a-framework) section.

### Prompt formatting

All instruction-tuned Gemma variants have specific prompt formatting
requirements. Some of these formatting requirements are handled automatically by
the framework you use to run Gemma models, but when you are sending prompt data
directly to a tokenizer, you must add specific tags, and the tagging
requirements can change depending on the Gemma variant you are using. See the
following guides for information on Gemma variant prompt formatting and system
instructions:

- [Gemma prompt and system instructions](../get-started/prompt-formatting.md)
- [PaliGemma prompt and system instructions](../models/paligemma/prompting.md)
- [FunctionGemma formatting and best practices](../models/functiongemma/formatting.md)

--8<-- "glossary.md"
