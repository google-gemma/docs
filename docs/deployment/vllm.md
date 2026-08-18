---
title: vLLM
description: Serve Gemma 4 with vLLM — install, vllm serve, model repositories, and the OpenAI-compatible API.
models: [gemma-4]
frameworks: [vllm]
---

# Serve Gemma with vLLM

[vLLM](https://docs.vllm.ai) is a high-throughput serving engine. Use it when you have more than
one concurrent user; for single-user local work, [Ollama](ollama.md) or
[llama.cpp](llamacpp.md) are simpler.

## Install

```bash
uv pip install vllm --torch-backend=auto
```

## Serve

```bash { .model-swap }
vllm serve google/gemma-4-12B-it
```

Serves an OpenAI-compatible API on `http://localhost:8000`.

## Model repositories

| Size | Repository |
|---|---|
| E2B | `google/gemma-4-E2B-it` |
| E4B | `google/gemma-4-E4B-it` |
| 12B | `google/gemma-4-12B-it` |
| 26B A4B | `google/gemma-4-26B-A4B-it` |
| 31B | `google/gemma-4-31B-it` |

Drop `-it` for base weights, which you want only for further training.

For 4-bit serving, use the [QAT](../get-started/qat.md) compressed-tensors builds — 4-bit weights with 16-bit
activations:

```bash { .model-swap }
vllm serve google/gemma-4-12B-it-qat-w4a16-ct
```

Published as `google/gemma-4-{E2B,E4B,12B,31B}-it-qat-w4a16-ct`. No 26B A4B compressed-tensors
checkpoint exists, so this page's selector offers the other four sizes; serve 26B A4B from the
full-precision repository above.

## Call it

```python { .model-swap }
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

response = client.chat.completions.create(
    model="google/gemma-4-12B-it",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke."},
    ],
)
print(response.choices[0].message.content)
```
