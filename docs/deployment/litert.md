---
title: LiteRT-LM
description: Run Gemma 4 on-device with LiteRT-LM — CLI install, prebuilt .litertlm weights, and exporting your own fine-tune.
models: [gemma-4]
frameworks: [litert]
source_url: "https://developers.google.com/edge/litert-lm/models/gemma-4"
---

# Run Gemma on-device with LiteRT-LM

[LiteRT-LM](https://developers.google.com/edge/litert-lm) is Google AI Edge's runtime for
running language models directly on a device — Android, iOS, macOS, Windows, Linux and
Raspberry Pi — with CPU, GPU and NPU acceleration and no network round trip.

!!! note "Sizes supported today"
    LiteRT-LM ships **E2B** and **E4B**, with larger models coming. The selector above offers
    the two sizes with published `.litertlm` weights.

## Install

```bash
uv tool install litert-lm
```

## Run

Weights are pulled straight from the Hugging Face LiteRT Community:

```bash { .model-swap }
litert-lm run \
  --from-huggingface-repo=litert-community/gemma-4-E2B-it-litert-lm \
  gemma-4-E2B-it.litertlm \
  --prompt="What is the capital of France?"
```

## Multi-Token Prediction

LiteRT-LM supports [MTP](../get-started/capabilities/mtp.md), which speeds up decoding with no
quality loss: up to **2.2x** on mobile GPUs and **1.5x** on mobile CPUs.

Google recommends MTP for every task on GPU backends, and for E4B on CPU. On E2B on CPU it
helps most with rewriting, summarizing and coding, but can slow down freeform generation
slightly — so enable it selectively there.

## Deploy your own fine-tune

Export Safetensors — including a model you have fine-tuned — to the `.litertlm` format:

```bash { .model-swap }
uv tool install litert-torch-nightly

litert-torch export_hf \
  --model=google/gemma-4-E2B-it \
  --output_dir=/tmp/gemma4 \
  --externalize_embedder \
  --jinja_chat_template_override=litert-community/gemma-4-E2B-it-litert-lm
```

Then run the exported bundle the same way:

```bash
litert-lm run /tmp/gemma4/model.litertlm --prompt="What is the capital of France?"
```

## Model sizes

| Size | `.litertlm` bundle |
|---|---|
| E2B | 2.58 GB |
| E4B | 3.65 GB |

Google publishes measured prefill, decode and time-to-first-token figures per device and
backend on the [Gemma 4 LiteRT-LM page](https://developers.google.com/edge/litert-lm/models/gemma-4){ data-model="gemma-4" }.

## Platform guides

LiteRT-LM has APIs for the CLI, Python, Android (Kotlin), iOS (Swift) and C++. See the
[LiteRT-LM documentation](https://developers.google.com/edge/litert-lm) for each.

To try Gemma on a phone without writing code, use the
[AI Edge Gallery](ai-edge-gallery.md).
