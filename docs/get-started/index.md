---
title: Quickstart
description: Run your first Gemma 4 generation in one command, then pick the engine, model size and checkpoint that fit your hardware.
models: [gemma-4]
frameworks: [transformers, llamacpp, ollama, mlx, vllm, lmstudio]
---

# Quickstart

<img class="page-mark" src="../assets/gemma-mark.png"
     alt="" width="132" height="132">

Gemma is a family of lightweight, state-of-the-art open models built from the same research and technology used to create the [Gemini](https://deepmind.google/models/gemini/) models. Developed by Google DeepMind and other teams across Google, Gemma is named after the Latin *gemma*, meaning precious stone. The Gemma model weights are supported by developer tools that promote innovation, collaboration, and the responsible use of artificial intelligence (AI). 

The latest in the Gemma family is [Gemma 4](../models/gemma-4/index.md) featuring multimodal (text, image, audio) inputs.

!!! tip "New to Gemma?"
    To learn more about the Gemma family of models, visit the [model family](../models/index.md) page for a tour.

## Choose a model

Gemma 4 ships in five sizes. They share a tokenizer and prompt format, so you can start small
and move up without rewriting anything.

<!-- Generated from data/models.yml -- edit that file, not this table. -->

| Size | Input ➔ Output | Context | Intended Platform |
|---|---|---|---|
{% for size in modeldata.gemma4.sizes -%}
| **{{ size.name }}** | {{ size.modalities.input | map('capitalize') | join(', ') }} ➔ {{ size.modalities.output | map('capitalize') | join(', ') }} | {{ size.context }} | {{ size.platform }} |
{% endfor %}

## Run your first inference

=== "Hugging Face Transformers"
    Install [`transformers`](https://huggingface.co/docs/transformers/installation), then run:

    ```python { .model-swap }
    from transformers import pipeline

    pipe = pipeline(
        "text-generation",
        model="google/gemma-4-E2B-it",
        device_map="auto"
    )
    output = pipe("Explain quantum computing simply:", max_new_tokens=150)
    print(output[0]["generated_text"])
    ```

=== "llama.cpp"
    Install [`llama.cpp`](https://llama.app/), then run:

    ```bash { .model-swap }
    llama cli -hf google/gemma-4-E2B-it-qat-q4_0-gguf --prompt "Write a poem about the Kraken."
    ```

=== "Ollama"
    Install [`ollama`](https://ollama.com/download), then run:

    ```bash { .model-swap }
    ollama run gemma4:e2b "Explain quantum computing simply:"
    ```

=== "Apple MLX"

    Install [`MLX LM`](https://github.com/ml-explore/mlx-lm), then run:

    ```bash { .model-swap }
    mlx_lm.generate --model mlx-community/gemma-4-E2B-it-qat-4bit \
        --prompt "Explain quantum computing simply:"
    ```

=== "vLLM"

    Install [`vLLM`](https://docs.vllm.ai/en/latest/getting_started/installation/), then run:

    ```bash { .model-swap }
    vllm serve google/gemma-4-E2B-it
    ```

=== "LMStudio"
    Install [`LMStudio`](https://lmstudio.ai/), then run:

    ```bash { .model-swap }
    lms chat google/gemma-4-e2b
    ```

!!! example "Using a coding agent?"

    Install the gemma-dev skill so your agent stays current with the latest models:

    ```bash
    npx skills add google-gemma/gemma-skills --skill gemma-dev --global
    ```

## Providers

If you want to run Gemma 4 on a different stack, you can choose from the following:

<!-- Generated from data/frameworks.yml -- edit that file, not this list. -->
{% set featured = ["ollama", "llamacpp", "lmstudio", "mlx", "litert", "transformers",
                   "keras", "unsloth", "vllm", "sglang", "gemini-api", "google-cloud"] %}
<ul class="engine-grid">
{% for id in featured %}{% for fw in frameworkdata.frameworks if fw.id == id %}
  <li><a class="engine" href="../{{ fw.page }}" title="{{ fw.blurb }}">
    <span class="engine-mark" aria-hidden="true">
      {%- if fw.get('icon') %}{{ icon(fw.icon, 'engine-logo') }}{% else %}{{ fw.mark }}{% endif -%}
    </span>
    <span class="engine-name">{{ fw.short_name | default(fw.name) }}</span>
  </a></li>
{% endfor %}{% endfor %}
  <li><a class="engine engine--all" href="../deployment/index.md">All ways to run →</a></li>
</ul>


## Get the weights

Every size ships full-precision weights plus [Quantization-Aware Training](qat.md) (QAT) checkpoints,
which hold up far better at 4-bit than a generic post-training quant.

| Where | What's there |
|---|---|
| [Hugging Face][hf-gemma4] | Safetensors, GGUF and QAT collections |
| [Kaggle][kaggle-gemma4] | The same checkpoints, mirrored |
| [Ollama][ollama-gemma4] | Prebuilt tags, pulled on first run |

Checkpoint names follow `google/gemma-4-{SIZE}-it{SUFFIX}` — pick the suffix your engine wants:

| Running on | Suffix |
|---|---|
| llama.cpp, LM Studio, Ollama | `-qat-q4_0-gguf` |
| vLLM, SGLang | `-qat-w4a16-ct` |
| Mobile | `-qat-mobile-transformers`, `-qat-mobile-ct` |


## What's next

<div class="card-grid">
<a class="card" href="best-practices.md">
  <span class="card-icon">{{ icon('lucide/sliders-horizontal') }}</span>
  <div class="card-title">Best practices</div>
  <p class="card-desc">Sampling parameters, thinking-mode tokens, modality order and image token budgets.</p>
</a>
<a class="card" href="capabilities/text.md">
  <span class="card-icon">{{ icon('lucide/type') }}</span>
  <div class="card-title">Capabilities</div>
  <p class="card-desc">Text, reasoning, tool calling, vision, video and audio — each with runnable code.</p>
</a>
<a class="card" href="../models/index.md">
  <span class="card-icon">{{ icon('lucide/boxes') }}</span>
  <div class="card-title">The model family</div>
  <p class="card-desc">Every generation and variant, and which base model each one came from.</p>
</a>
</div>

--8<-- "glossary.md"
