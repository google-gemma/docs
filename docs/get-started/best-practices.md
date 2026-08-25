---
title: Best practices
description: Sampling parameters, thinking-mode tokens, modality order and image token budgets that get the best results out of Gemma 4.
models: [gemma-4]
frameworks: [transformers]
---

# Best practices

For the best performance when using [Gemma 4](../models/gemma-4/index.md), use these configurations and best practices.

## 1. Sampling parameters

Use the following standardized sampling configuration across all use cases:

| Parameter | Value |
|---|---|
| `temperature` | `1.0` |
| `top_p` | `0.95` |
| `top_k` | `64` |

## 2. Thinking mode configuration

Compared to Gemma 3, the models use standard `system`{.token-role}, `assistant`{.token-role},
and `user`{.token-role} roles. To properly manage the thinking process, use the following
control tokens:

- **Trigger thinking:** Thinking is enabled by including the `<|think|>`{.token-think} token at
  the start of the system prompt. To disable thinking, remove the token.
- **Standard generation:** When thinking is enabled, the model will output its internal
  reasoning followed by the final answer using this structure:

<pre><code><span class="token-think">&lt;|channel&gt;thought</span>
<strong>[Internal reasoning]</strong><span class="token-think">&lt;channel|&gt;</span></code></pre>

- **Disabled thinking behavior:** For all models except for the E2B and E4B variants, if
  thinking is disabled, the model will still generate the tags but with an empty thought block:

<pre><code><span class="token-think">&lt;|channel&gt;thought</span>
<span class="token-think">&lt;channel|&gt;</span><strong>[Final answer]</strong></code></pre>

!!! note
    Many libraries like Transformers and llama.cpp handle the complexities of the chat template
    for you.

## 3. Multi-turn conversations

**No thinking content in history**: In multi-turn conversations, the historical model output
should only include the final response. Thoughts from previous model turns must *not be added*
before the next user turn begins, with the exception of tool call turns where thinking content
should be preserved.

## 4. Modality order

For optimal performance with multimodal inputs, place:

- Image content **before** the text in your prompt.
- Audio content **after** the text in your prompt.

## 5. Variable image resolution

Aside from variable aspect ratios, Gemma 4 supports variable image resolution through a
configurable visual token budget, which controls how many tokens are used to represent an image.
A higher token budget preserves more visual detail at the cost of additional compute, while a
lower budget enables faster inference for tasks that don't require fine-grained understanding.

The supported token budgets are **70**, **140**, **280**, **560**, and **1120**.

- Use *lower budgets* for classification, captioning, or video understanding, where faster
  inference and processing many frames outweigh fine-grained detail.
- Use *higher budgets* for tasks like OCR, document parsing, or reading small text.

!!! tip
    The [vision token budget demo](../resources/vision-token-budget.md) shows the quality and speed
    tradeoff at each budget.

## 6. Audio

Use the following prompt structures for audio processing.

**Audio speech recognition (ASR)**

```text
Transcribe the following speech segment in {LANGUAGE} into {LANGUAGE} text.

Follow these specific instructions for formatting the answer:
* Only output the transcription, with no newlines.
* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three.
```

**Automatic speech translation (AST)**

```text
Transcribe the following speech segment in {SOURCE_LANGUAGE}, then translate it into {TARGET_LANGUAGE}.
When formatting the answer, first output the transcription in {SOURCE_LANGUAGE}, then one newline, then output the string '{TARGET_LANGUAGE}: ', then the translation in {TARGET_LANGUAGE}.
```

## 7. Audio and video length

All models support image inputs and can process videos as frames whereas the E2B, E4B, and 12B
models also support audio inputs. Audio supports a maximum length of 30 seconds. Video supports
a maximum of 60 seconds assuming the images are processed at one frame per second.

## What's next

<div class="card-grid">
<a class="card" href="capabilities/thinking.md">
  <span class="card-icon">{{ icon('lucide/brain') }}</span>
  <div class="card-title">Reasoning</div>
  <p class="card-desc">Let Gemma reason before it answers, and parse the two apart.</p>
</a>
<a class="card" href="prompt-formatting.md">
  <span class="card-icon">{{ icon('lucide/braces') }}</span>
  <div class="card-title">Prompt formatting</div>
  <p class="card-desc">The control tokens that structure a Gemma prompt.</p>
</a>
<a class="card" href="capabilities/vision.md">
  <span class="card-icon">{{ icon('lucide/eye') }}</span>
  <div class="card-title">Vision understanding</div>
  <p class="card-desc">Detection, OCR, visual question answering and captioning.</p>
</a>
</div>

--8<-- "glossary.md"
