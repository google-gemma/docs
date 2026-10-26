---
title: llama.cpp
description: Run Gemma 4 with llama.cpp — GGUF repos for every size, image/audio/video input, and the OpenAI-compatible server.
models: [gemma-4]
frameworks: [llamacpp]
---

# Run Gemma with llama.cpp

[llama.cpp](https://github.com/ggml-org/llama.cpp) is the engine underneath most local LLM
tooling. Using it directly gives full control over quantization and GPU offload, and it is the
most capable local runtime for Gemma 4 — **image, audio and video input all work**.

## Install

=== "curl"

    ```bash
    curl -LsSf https://llama.app/install.sh | sh
    ```

=== "powershell"

    ```powershell
    irm https://llama.app/install.ps1 | iex
    ```

## Run

```bash { .model-swap }
llama cli -hf ggml-org/gemma-4-E2B-it-GGUF --prompt "Write a poem about the Kraken."
```

`-hf` downloads the weights on first use, including the multimodal projector. With a system
prompt:

```bash { .model-swap }
llama cli -hf ggml-org/gemma-4-E2B-it-GGUF -sys "You are Hong Gildong." -p "Who are you?"
```

## Model repositories

| Size | Repository |
|---|---|
| E2B | `ggml-org/gemma-4-E2B-it-GGUF` |
| E4B | `ggml-org/gemma-4-E4B-it-GGUF` |
| 12B | `ggml-org/gemma-4-12B-it-GGUF` |
| 26B A4B | `ggml-org/gemma-4-26B-A4B-it-GGUF` |
| 31B | `ggml-org/gemma-4-31B-it-GGUF` |

For 4-bit, prefer Google's [QAT](../get-started/qat.md) weights — trained with quantization simulated rather
than compressed afterwards:

```bash { .model-swap }
llama cli -hf google/gemma-4-E2B-it-qat-q4_0-gguf -p "Explain gravity in one paragraph"
```

Available as `google/gemma-4-{E2B,E4B,12B,26B-A4B,31B}-it-qat-q4_0-gguf`.

## Multimodal

```bash { .model-swap }
llama cli -hf ggml-org/gemma-4-E2B-it-GGUF --image "image.png" -p "Describe this image."
```

With local files, pass the projector explicitly:

```bash { .model-swap }
llama cli -m gemma-4-E2B-it-Q4_0.gguf --mm mmproj-gemma-4-E2B-it-Q8_0.gguf
```

!!! note
    Audio works on E2B, E4B and 12B. The 26B A4B and 31B models take text and images, and read
    video as frames — they have no audio encoder.

## Server

```bash { .model-swap }
llama server -hf ggml-org/gemma-4-E2B-it-GGUF
```

Browser UI at `http://localhost:8080`, OpenAI-compatible API at `http://localhost:8080/v1`.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="gemma-4",  # llama-server serves whichever model it was started with
    messages=[{"role": "user", "content": "Say this is a test"}],
)
print(response.choices[0].message.content)
```

Multimodal input consumes context quickly; set it with `-c`, for example `-c 32768`.
