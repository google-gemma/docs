---
title: (V)RAM Calculator
description: Estimate memory requirements for running Gemma models.
---
# (V)RAM Calculator

Estimate memory requirements for running Gemma models.

## Gemma 4 Memory Requirements

| Model | BF16 (16-bit) | SFP8 (8-bit) | Q4_0 (4-bit) |
|---|---|---|---|
| E2B | 11.4 GB | 5.7 GB | 2.9 GB |
| E4B | 17.9 GB | 8.9 GB | 4.5 GB |
| 12B | 26.7 GB | 13.4 GB | 6.7 GB |
| 26B A4B | 57.7 GB | 28.8 GB | 14.4 GB |
| 31B | 69.9 GB | 34.9 GB | 17.5 GB |

!!! note
    These numbers only account for static model weights with ~20% overhead. Context window (KV cache) will increase memory dynamically based on prompt and generation length.

## Key Considerations

- **E2B and E4B:** The "E" stands for "effective" parameters. These models use Per-Layer Embeddings (PLE), so total memory is higher than the effective parameter count suggests.
- **26B A4B (MoE):** While only 4B parameters are active per token, all 26B parameters must be loaded into memory.
- **Fine-tuning** requires drastically more memory than inference.
