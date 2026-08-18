---
title: Ollama
description: Run Gemma 4 locally with Ollama — model tags, image input, and the OpenAI-compatible server.
models: [gemma-4]
frameworks: [ollama]
---

# Run Gemma with Ollama

Ollama downloads quantized weights for you and exposes a CLI plus a local HTTP API. No GPU
required, though one helps.

## Install

Download from [ollama.com/download](https://ollama.com/download), then check it worked:

```bash
ollama --version
```

## Run

```bash { .model-swap }
ollama run gemma4:12b "Explain quantum computing simply"
```

Downloads the model on first use. To fetch without running:

```bash { .model-swap }
ollama pull gemma4:12b
```

## Model tags

| Tag | Download | Context |
|---|---|---|
| `gemma4:e2b` | 7.2 GB | 128K |
| `gemma4:e4b` | 9.6 GB | 128K |
| `gemma4:12b` | 7.6 GB | 256K |
| `gemma4:26b` | 18 GB | 256K |
| `gemma4:31b` | 20 GB | 256K |

`gemma4:latest` is an alias for `e4b`. On Apple Silicon, the `-mlx` tags
(`gemma4:12b-mlx`) use the MLX runtime. Full list:
[ollama.com/library/gemma4/tags](https://ollama.com/library/gemma4/tags){ data-model="gemma-4" }.

## Images

```bash { .model-swap }
ollama run gemma4:12b ./image.png "What is in this image?"
```

!!! note
    Ollama publishes Gemma 4 as **text and image only**. For audio input, use
    [Transformers](huggingface.md) or [llama.cpp](llamacpp.md).

## Server

Ollama serves on port `11434`.

=== "OpenAI-compatible"

    ```python { .model-swap }
    from openai import OpenAI

    client = OpenAI(
        base_url="http://localhost:11434/v1/",
        api_key="ollama",  # required but ignored
    )

    response = client.chat.completions.create(
        model="gemma4:12b",
        messages=[{"role": "user", "content": "Say this is a test"}],
    )
    print(response.choices[0].message.content)
    ```

=== "Native API"

    ```bash { .model-swap }
    curl http://localhost:11434/api/generate -d '{
      "model": "gemma4:12b",
      "prompt": "roses are red"
    }'
    ```

=== "Image via API"

    ```bash { .model-swap }
    IMG=$(base64 < test.jpg | tr -d '\n')

    curl -X POST http://localhost:11434/api/chat \
      -H "Content-Type: application/json" \
      -d '{
        "model": "gemma4:12b",
        "messages": [{
          "role": "user",
          "content": "What is in this image?",
          "images": ["'"$IMG"'"]
        }],
        "stream": false
      }'
    ```

!!! tip "Context defaults to 4K under 24 GiB VRAM"
    Ollama sizes context from available VRAM, not from the model. Raise it when you need it:
    `OLLAMA_CONTEXT_LENGTH=64000 ollama serve`, or per request with
    `"options": {"num_ctx": 64000}`. Agents and coding tools want at least 64K —
    [Ollama: context length](https://docs.ollama.com/context-length).
