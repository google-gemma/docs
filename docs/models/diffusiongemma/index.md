---
title: DiffusionGemma model overview
description: DiffusionGemma is an experimental open model that explores text diffusion, an exceptionally fast approach to text generation.
models: [diffusiongemma]
source_url: "https://ai.google.dev/gemma/docs/diffusiongemma.md.txt"
---
# DiffusionGemma model overview

DiffusionGemma is an experimental open model that explores text diffusion, an
exceptionally fast approach to text generation. Based on the 26B (4B active)
Mixture-of-Experts (MoE) Gemma 4 architecture, DiffusionGemma generates tokens
using discrete diffusion. This open-weights model is multimodal, handling text,
image, and video inputs to generate text output.

Built on a MoE foundation, DiffusionGemma is designed to improve generation
speed (tokens per second) while remaining deployable across various hardware
environments. DiffusionGemma builds upon the architectural and capability
advancements of Gemma 4, introducing several core features:

- **Discrete Text Diffusion:** Shifts away from traditional causal token generation to block-autoregressive multi-canvas sampling. The model generates text by iteratively denoising blocks of tokens (a "canvas") in parallel to dramatically boost decoding speeds.
- **Multimodal Processing:** Natively accepts text, images (with variable aspect ratio and resolution support), and video inputs. (Note: Audio input is not supported).
- **Encoder-Decoder Architecture:** Utilizes an autoregressive encoder to process and cache prompt context, paired with denoising that applies bi-directional attention over the generation canvas.
- **Mixture-of-Experts (MoE) Efficiency:** Leverages a sparse MoE design based on the 26B (4B active) MoE variant, offering deep reasoning capabilities with minimal overhead. When quantized, it fits within the 18GB VRAM limits of consumer GPUs, ideal for local execution.
- **Thinking Mode:** Built-in configurable reasoning channels allow the model to think step-by-step before emitting a final answer.

## Tradeoff with traditional models

While traditional language models are highly efficient for large-scale cloud
deployments because they can batch thousands of requests, running them locally
for a single user leaves hardware underutilized. DiffusionGemma solves this by
generating an entire 256-token block simultaneously rather than one token at a
time, maximizing local hardware performance.

However, this approach is strictly aimed at consumer-facing, low-concurrency
local use; because its parallel decoding offers diminishing returns under
high-QPS cloud workloads, the throughput advantage is strongest at low-to-medium
batch sizes on a single accelerator.

## Recommended Serving Configuration

For optimal latency and quality, we recommend deploying with the following
default parameters for the Diffusion Sampling Settings:

| Parameter | Recommended Value | Function | Rationale |
|---|---|---|---|
| Maximum Number of Denoising Steps | 48 | Upper bound on number of denoising steps per canvas. | A safe limit on the number of denoising steps. Denoising will stop in fewer steps when adaptive stopping is enabled, typically 12-16 steps depending on the task. |
| Temperature Schedule | Linear 0.8 -\> 0.4 | Temperature scaling schedule that starts high and reduces as a function of denoising steps. | High temperature (0.8) encourages early exploration; low temperature (0.4) locks in final tokens. |
| Adaptive Early Stopping | Entropy threshold: 0.005 | Halts execution early if A) the average model entropy over the canvas is below the threshold, and B) if two consecutive denoiser predictions remain identical. | Simpler prompts and structured tasks like code require fewer denoising steps, enabling dynamic tokens-per-second speeds based on task complexity. |
| Token selection | Entropy bound: 0.1 | At each step, the sampler selects the lowest-entropy tokens such that their mutual information bound stays below entropy bound. The sampler fully renoises the non-selected tokens. | Ensures only tokens that the model is relatively certain about are selected to refine the canvas, leaving other tokens to be refined in later denoising steps. |

<div class="cookbook-grid">
<a class="cookbook-card" href="https://huggingface.co/collections/google/diffusiongemma" target="_blank" rel="noopener">
  <div class="cookbook-card-title">Hugging Face <span class="cookbook-card-ext">↗</span></div>
  <p class="cookbook-card-desc">Get it on Hugging Face</p>
</a>
<a class="cookbook-card" href="https://www.kaggle.com/models/google/diffusiongemma" target="_blank" rel="noopener">
  <div class="cookbook-card-title">Kaggle <span class="cookbook-card-ext">↗</span></div>
  <p class="cookbook-card-desc">Get it on Kaggle</p>
</a>
</div>

[Access it on Vertex](https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/diffusiongemma){ data-model="diffusiongemma" }

Access the experimental model weights (released under the Apache 2.0 license),
allowing you to deploy it in your own projects and applications.

[Learn more about DiffusionGemma architecture](explained.md)
[Try DiffusionGemma](inference.md)

[Fine-tune DiffusionGemma](https://github.com/google/hackable_diffusion)
[Deploy DiffusionGemma](https://docs.cloud.google.com/kubernetes-engine/docs/tutorials/serve-gemma-gpu-vllm)
