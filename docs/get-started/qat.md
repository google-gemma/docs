---
title: Quantization & QAT
description: Which Gemma 4 quantized weights to download for each engine, and why QAT beats post-training quantization at 4-bit.
models: [gemma-4]
frameworks: [llamacpp, vllm]
---

# Quantization and QAT

Google publishes **quantization-aware training** (QAT) checkpoints alongside the full-precision
weights. QAT simulates quantization *during* training, so the model learns to compensate for the
precision loss — unlike post-training quantization, which compresses a finished model and loses
quality doing it.

**At 4-bit, prefer a QAT build over a generic Q4 of the same size.**

## Which one do I download?

| Running on | Suffix |
|---|---|
| llama.cpp, LM Studio, Ollama | `-qat-q4_0-gguf` |
| vLLM, SGLang | `-qat-w4a16-ct` |
| Mobile (Transformers) | `-qat-mobile-transformers` |
| Mobile (compressed tensors) | `-qat-mobile-ct` |
| Speculative decoding | `-qat-q4_0-unquantized` + `-qat-q4_0-unquantized-assistant` |
| Converting to another format (e.g. MLX) | `-qat-q4_0-unquantized` |

Names follow `google/gemma-4-{SIZE}-it{SUFFIX}`, for example:

```bash { .model-swap }
# Local, via llama.cpp
llama-cli -hf google/gemma-4-12B-it-qat-q4_0-gguf

# Served, via vLLM
vllm serve google/gemma-4-12B-it-qat-w4a16-ct
```

## Availability

| Variant | Sizes |
|---|---|
| `-gguf`, `-unquantized`, `-assistant` | E2B, E4B, 12B, 26B A4B, 31B |
| `-w4a16-ct` | E2B, E4B, 12B, 31B |
| `-mobile-transformers`, `-mobile-ct` | E2B, E4B |

Collections:
[gemma-4-qat-q4-0](https://huggingface.co/collections/google/gemma-4-qat-q4-0) ·
[gemma-4-qat-mobile](https://huggingface.co/collections/google/gemma-4-qat-mobile).
All QAT checkpoints are also on
[Kaggle](https://www.kaggle.com/models/google/gemma-4/transformers).

!!! note "Mobile builds use a different scheme"
    `-mobile-*` weights are built on a custom `wNa8o8` schema with 2-bit decoding layers and
    optimized KV caches, tuned for on-device RAM limits rather than raw quality.

## Memory

Approximate footprint for the static weights, from the
[Gemma 4 overview](../models/gemma-4/index.md):

| Size | BF16 | SFP8 | Q4_0 |
|---|---|---|---|
| E2B | 11.4 GB | 5.7 GB | 2.9 GB |
| E4B | 17.9 GB | 8.9 GB | 4.5 GB |
| 12B | 26.7 GB | 13.4 GB | 6.7 GB |
| 26B A4B | 57.7 GB | 28.8 GB | 14.4 GB |
| 31B | 69.9 GB | 34.9 GB | 17.5 GB |

Context (KV cache) is on top of this. Size it with the
[VRAM calculator](../tools/vram-calculator.md).
