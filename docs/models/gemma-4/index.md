---
title: Gemma 4
description: Gemma is a family of generative artificial intelligence models and you can use them in a wide variety of generation tasks, including question answering, summarization, and reasoning.
models: [gemma-4]
source_url: "https://ai.google.dev/gemma/docs/core.md.txt"
---
# Gemma 4 {: #hidden-title style="display: none;" }

<div class="video-embed">
  <iframe src="https://www.youtube.com/embed/jZVBoFOJK-Q"
          title="Video" loading="lazy" allowfullscreen
          allow="accelerometer; encrypted-media; gyroscope; picture-in-picture"
          referrerpolicy="strict-origin-when-cross-origin"></iframe>
</div>

Gemma is a family of generative artificial intelligence models and you can
use them in a wide variety of generation tasks, including question answering,
summarization, and reasoning. Gemma models are provided with open weights and
permit responsible
[commercial use][gemma-terms],
allowing you to tune and deploy them in your own projects and applications.

Gemma 4 model family spans four distinct architectures tailored for specific
hardware requirements:

- **Small Sizes:** 2B and 4B effective parameter models built for ultra-mobile, edge, and browser deployment (e.g., Pixel, Chrome).
- **Dense:** A powerful 31B parameter dense model that bridges the gap between server-grade performance and local execution.
- **Mixture-of-Experts:** A highly efficient 26B MoE model designed for high-throughput, advanced reasoning.
- **Unified:** A 12B parameter encoder free model for multimodal tasks, replaced vision and audio encoders with direct linear projections of the input.

You can download Gemma 4 models from
[Kaggle](https://www.kaggle.com/models?query=gemma-4&publisher=google){ data-model="gemma-4" } and
[Hugging Face][hf-gemma4].
For more technical details on Gemma 4, see the
[Model Card](model-card.md) and
[Technical Report](https://goo.gle/Gemma4Report){ data-model="gemma-4" }.
Earlier versions of Gemma core models are also available for download. For more
information, see [Previous Gemma models](index.md#previous-gemma-models).

<div class="cookbook-grid">
<a class="cookbook-card" href="https://www.kaggle.com/models?query=gemma-4&publisher=google" target="_blank" rel="noopener">
  <div class="cookbook-card-title">Kaggle <span class="cookbook-card-ext">↗</span></div>
  <p class="cookbook-card-desc">Get it on Kaggle</p>
</a>
<a class="cookbook-card" href="https://huggingface.co/collections/google/gemma-4" target="_blank" rel="noopener">
  <div class="cookbook-card-title">Hugging Face <span class="cookbook-card-ext">↗</span></div>
  <p class="cookbook-card-desc">Get it on Hugging Face</p>
</a>
</div>


## Capabilities

- **Reasoning:** All models in the family are designed as highly capable reasoners, with configurable [thinking
  modes](../../get-started/capabilities/thinking.md).
- **Extended Multimodalities:** Processes Text, [Image](../../get-started/capabilities/image.md) with variable aspect ratio and resolution support (all models), [Video](../../get-started/capabilities/video.md), and [Audio](../../get-started/capabilities/audio.md) (featured natively on the E2B, E4B and 12B models).
- **Increased Context Window:** Small models feature a 128K context window, while the medium models support 256K.
- **Enhanced Coding & Agentic Capabilities:** Achieves notable improvements in coding benchmarks alongside built-in [function-calling
  support](../../get-started/capabilities/function-calling.md), powering highly capable autonomous agents.
- **Native System Prompt Support:** Gemma 4 introduces built-in support for the system role, enabling more structured and controllable conversations.
- **[Multi-Token Prediction](../../get-started/capabilities/mtp.md):** All Gemma 4 models (E2B, E4B, 12B, 31B, and 26B A4B) include a dedicated draft model for speculative decoding, enabling significantly faster inference with no quality loss.

## Parameter sizes and quantization

Gemma 4 models are available in 5 parameter sizes: E2B, E4B, 12B, 31B and 26B
A4B. The models can be used with their default precision (16-bit) or with a
lower precision using quantization. The different sizes and precisions represent
a set of trade-offs for your AI application. Models with higher parameters and
bit counts (higher precision) are generally more capable, but are more expensive
to run in terms of processing cycles, memory cost and power consumption. Models
with lower parameters and bit counts (lower precision) have less capabilities,
but may be sufficient for your AI task.

### Gemma 4 Inference Memory Requirements

The following table details the approximate GPU or TPU memory requirements for
running inference with each size of the Gemma 4 model versions.

!!! Note
    These numbers may change based on your specific inference tool and environment.


| Parameters | BF16 (16-bit) | SFP8 (8-bit) | Q4_0 (4-bit) | Mobile | Mobile (Text-only) |
|---|---|---|---|---|---|
| Gemma 4 E2B | 11.4 GB | 5.7 GB | 2.9 GB | 1.1 GB | 0.84 GB |
| Gemma 4 E4B | 17.9 GB | 8.9 GB | 4.5 GB | 2.5 GB | 2.2 GB |
| Gemma 4 12B | 26.7 GB | 13.4 GB | 6.7 GB | - | - |
| Gemma 4 26B A4B | 57.7 GB | 28.8 GB | 14.4 GB | - | - |
| Gemma 4 31B | 69.9 GB | 34.9 GB | 17.5 GB | - | - |

**Table 1.** Approximate GPU or TPU memory required to load Gemma 4 models based
on parameter count, quantization level and 20% overhead of loading additional
things. Mobile versions use LiteRT-LM.

### Key Considerations for Memory Planning

- **Efficient Architecture (E2B and E4B):** The "E" stands for "effective" parameters. The smaller models incorporate Per-Layer Embeddings (PLE) to maximize parameter efficiency in on-device deployments. Rather than adding more layers to the model, PLE gives each decoder layer its own small embedding for every token. These embedding tables are large but only used for quick lookups, which is why the total memory required to load static weights is higher than the effective parameter count suggests.
- **The MoE Architecture (26B A4B):** The 26B is a Mixture of Experts model. While it only activates 4 billion parameters per token during generation, **all 26 billion parameters** must be loaded into memory to maintain fast routing and inference speeds. This is why its baseline memory requirement is much closer to a dense 26B model than a 4B model.
- **Base Weights Only:** The estimates in the preceding table *only* account for the memory required to load the static model weights. They don't include the additional VRAM needed for supporting software or the context window.
- **Context Window (KV Cache):** Memory consumption will increase dynamically based on the total number of tokens in your prompt and the generated response. Larger context windows require significantly more VRAM on top of the base model weights.
- **Fine-Tuning Overhead:** Memory requirements for *fine-tuning* Gemma models are drastically higher than for standard inference. Your exact footprint will depend heavily on the development framework, batch size, and whether you are using full-precision tuning versus a Parameter-Efficient Fine-Tuning (PEFT) method like Low-Rank Adaptation (LoRA).

### Quantization-Aware Training (QAT)

For deployments requiring maximum efficiency with minimal quality compromise,
Gemma offers official **Quantization-Aware Training (QAT)** models.

Unlike standard Post-Training Quantization (PTQ), which compresses a fully
trained model and can lead to quality degradation, QAT integrates quantization
simulation into the training process itself. This allows the model to learn to
compensate for the precision loss, resulting in smaller models that perform
nearly identically to their high-precision baselines.

#### Quick Routing Table

| Target Deployment Engine | Download Suffix | Primary Use Case |
|---|---|---|
| llama.cpp / LM Studio (Local) | `{model-name}-qat-q4_0-gguf` | Zero-setup local deployment on CPU, Apple Silicon, or consumer GPUs. |
| vLLM / SGLang | SERVER: `{model-name}-qat-w4a16-ct` MOBILE: `{model-name}-qat-mobile-ct` | High-throughput inference utilizing 4-bit weights with 16-bit activations. |
| Speculative Decoding | MODEL: `{model-name}-qat-q4_0-unquantized` DRAFTER: `{model-name}-qat-q4_0-unquantized-assistant` | Running a primary model alongside its matching MTP draft model to drastically accelerate token generation. The model must be quantized. |
| Other formats | `{model-name}-qat-q4_0-unquantized` | Unquantized weights for converting to other formats (e.g. MLX) |
| Mobile Deployment (Transformers) | `{model-name}-qat-mobile-transformers` | Edge weights optimized for mobile use cases. They serve as reference for other formats. |

Official QAT collections on Hugging Face

- **[collections/google/gemma-4-qat-q4-0][hf-gemma4-qat-q40]**
    - **Unquantized QAT Checkpoints (`-unquantized` / `-assistant`):** Half-precision weights extracted directly from the QAT pipeline. These are ideal for custom downstream compilation, research, or running speculative decoding using the assistant draft models. *Available for Gemma 4 E2B, E4B, 12B, 26B A4B, and 31B.*
    - **GGUF (`-gguf`):** Checkpoints available for immediate drop-in compatibility across the local LLM ecosystem. *Available for Gemma 4 E2B, E4B, 12B, 26B A4B, and 31B.*
    - **Compressed Tensors (`-w4a16-ct`):** Serialized natively in the `compressed-tensors` standard for optimized, high-concurrency cloud serving. *Available for Gemma 4 E2B, E4B, 12B, and 31B.*
- **[collections/google/gemma-4-qat-mobile][hf-gemma4-qat-mobile]**
    - **Mobile-Optimized (`-mobile-transformers` / `-mobile-ct`):** Built on a custom `wNa8o8` schema engineered specifically for mobile hardware limits. It utilizes targeted 2-bit decoding layers, optimized KV caches, and static activations to maximize on-device RAM savings without choking edge processors. *Available for Gemma 4 E2B and E4B.*

All official Gemma 4 QAT checkpoints can also be accessed directly from
[Kaggle](https://www.kaggle.com/models/google/gemma-4/transformers){ data-model="gemma-4" }.

## Previous Gemma models

You can work with previous generations of Gemma models, which are also
available from [Kaggle](https://www.kaggle.com/models?query=gemma) and
[Hugging Face](https://huggingface.co/google/collections).
For more technical details about previous Gemma models, see the following
model card pages:

- Gemma 3 [Model Card](../gemma-3/index.md)
- Gemma 2 [Model Card](../gemma-2/index.md)
- Gemma 1 [Model Card](../gemma-1/index.md)

Ready to start building?
[Get started](../../get-started/index.md)
with Gemma models!

--8<-- "glossary.md"
