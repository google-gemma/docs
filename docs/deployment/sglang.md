---
title: SGLang
description: Serve Gemma 4 with SGLang — install, launch_server, model repositories, and the OpenAI-compatible API.
models: [gemma-4]
frameworks: [sglang]
---

# Serve Gemma with SGLang

[SGLang](https://docs.sglang.io) is a serving engine built for high throughput and structured
generation. Like [vLLM](vllm.md), it targets multi-user production rather than local
single-user work.

## Install

```bash
uv pip install --prerelease=allow sglang
```

## Serve

```bash { .model-swap }
python3 -m sglang.launch_server \
  --model-path google/gemma-4-12B-it \
  --host 0.0.0.0 \
  --port 30000
```

## Model repositories

| Size | Repository |
|---|---|
| E2B | `google/gemma-4-E2B-it` |
| E4B | `google/gemma-4-E4B-it` |
| 12B | `google/gemma-4-12B-it` |
| 26B A4B | `google/gemma-4-26B-A4B-it` |
| 31B | `google/gemma-4-31B-it` |

For 4-bit serving, use the [QAT](../get-started/qat.md) compressed-tensors builds,
`google/gemma-4-{E2B,E4B,12B,31B}-it-qat-w4a16-ct`.

## Call it

```python { .model-swap }
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:30000/v1")

response = client.chat.completions.create(
    model="google/gemma-4-12B-it",
    messages=[{"role": "user", "content": "Say this is a test"}],
)
print(response.choices[0].message.content)
```
