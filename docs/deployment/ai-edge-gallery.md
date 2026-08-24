---
title: AI Edge Gallery
description: Run Gemma 4 on an Android or iOS phone with no code — chat, vision, audio transcription and benchmarking, fully offline.
models: [gemma-4]
frameworks: [litert, android, ios]
---

# Try Gemma on your phone with AI Edge Gallery

[AI Edge Gallery](https://github.com/google-ai-edge/gallery) is Google AI Edge's app for running
open models directly on a phone or laptop. Every inference happens on the device — no network,
no account, nothing leaves the hardware — which makes it the fastest way to see what Gemma 4
actually feels like on the device you carry.

It is the no-code counterpart to [LiteRT-LM](litert.md): the same runtime underneath, wrapped in
an interface you can hand to someone who does not write Python.

## Install

| Platform | Requirement |
|---|---|
| [Google Play](https://play.google.com/store/apps/details?id=com.google.aiedge.gallery) | Android 12 and up |
| [App Store](https://apps.apple.com/app/google-ai-edge-gallery/id6737687734) | iOS 17 and up |
| macOS | Download from the [releases page](https://github.com/google-ai-edge/gallery/releases) |

Without Google Play access, install the APK from the
[latest release](https://github.com/google-ai-edge/gallery/releases).

## What you can do with it

- **AI Chat with thinking mode** — multi-turn conversation, with a toggle that shows the model's
  step-by-step reasoning. Thinking mode works with the Gemma 4 family.
- **Ask Image** — point the camera or pick a photo and ask about it.
- **Audio Scribe** — transcribe and translate voice recordings in real time.
- **Prompt Lab** — a workspace for single-turn prompts with direct control over temperature and
  top-k, useful for tuning before you commit settings to code.
- **Agent Skills** — augment the model with tools such as Wikipedia grounding, maps and visual
  summary cards. Skills can be loaded from a URL.
- **Mobile Actions** and **Tiny Garden** — offline device control and a natural-language garden
  game, both powered by a fine-tune of [FunctionGemma](../models/functiongemma/index.md) 270M.
- **Model management and benchmarking** — download models, side-load your own, and benchmark
  each one on your actual hardware rather than trusting someone else's numbers.

!!! tip "Bring your own weights"
    Gallery loads custom models, so a Gemma you fine-tuned and exported with
    [LiteRT-LM](litert.md) can be tested on-device without building an app around it.

## How it fits together

| Layer | What it does |
|---|---|
| [Google AI Edge](https://ai.google.dev/edge) | Core on-device ML APIs and tooling |
| [LiteRT](litert.md) | The runtime that executes the model |
| [Hugging Face](https://huggingface.co/litert-community){ data-model="gemma-4" } | Where the `.litertlm` weights come from |

## What's next

<div class="card-grid">
<a class="card" href="litert.md">
  <span class="card-icon">{{ icon('lucide/terminal') }}</span>
  <div class="card-title">LiteRT-LM</div>
  <p class="card-desc">Run the same models from a terminal or your own app.</p>
</a>
<a class="card" href="../tuning/index.md">
  <span class="card-icon">{{ icon('lucide/graduation-cap') }}</span>
  <div class="card-title">Fine-tune Gemma</div>
  <p class="card-desc">Adapt a model first, then export it.</p>
</a>
<a class="card" href="https://github.com/google-ai-edge/gallery/wiki" target="_blank" rel="noopener">
  <div class="card-title">Project wiki <span class="card-ext">↗</span></div>
  <p class="card-desc">Detailed setup, including corporate-managed devices.</p>
</a>
</div>
