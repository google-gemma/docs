---
title: LM Studio
description: Run Gemma 4 with LM Studio — desktop app, lms CLI, and the OpenAI-compatible local server.
models: [gemma-4]
frameworks: [lmstudio]
---

# Run Gemma with LM Studio

[LM Studio](https://lmstudio.ai) is a desktop app for running local models, with a GUI, a CLI
and a local server. It supports Gemma in both GGUF (llama.cpp) and MLX formats.

## Install

Download for macOS, Windows or Linux from
[lmstudio.ai/download](https://lmstudio.ai/download).

## Get a model

In the app, press ++cmd+shift+m++ (macOS) or ++ctrl+shift+m++ (Windows/Linux), search for
"Gemma", and LM Studio suggests the variant that fits your hardware.

From the terminal:

```bash { .model-swap }
lms get google/gemma-4-12b
```

List what you have, then load one:

```bash
lms ls
lms load <model_key>
```

`lms load` takes `--gpu=max|auto|0.0-1.0` and `--context-length=1-N`.

## Server

```bash
lms server start
```

OpenAI-compatible API at `http://localhost:1234/v1`, serving `/v1/chat/completions`,
`/v1/completions`, `/v1/embeddings`, `/v1/models` and `/v1/responses`.

```python { .model-swap }
from openai import OpenAI

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

response = client.chat.completions.create(
    model="google/gemma-4-12b",
    messages=[{"role": "user", "content": "Say this is a test"}],
)
print(response.choices[0].message.content)
```

## Your own GGUF

```bash
lms import <path/to/model.gguf>
```

The model appears under **My Models**. See
[Import Models](https://lmstudio.ai/docs/app/advanced/import-model) if it isn't detected.
