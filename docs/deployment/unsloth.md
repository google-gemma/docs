---
title: Unsloth
description: Run Gemma 4 with Unsloth — the desktop Studio app, dynamic-quant GGUFs for llama.cpp, and 2x faster fine-tuning.
models: [gemma-4]
frameworks: [unsloth, llamacpp]
---

# Run Gemma with Unsloth

[Unsloth](https://unsloth.ai/docs) publishes dynamic-quantized GGUF builds of Gemma 4 and ships
a desktop application, Unsloth Studio, that downloads and runs them locally. It is also the
fastest route to fine-tuning Gemma on a single consumer GPU.

There are two ways to run Gemma 4 here: the Studio app, or Unsloth's GGUFs directly in
[llama.cpp](llamacpp.md).

## Hardware requirements

Unsloth's own recommended totals — RAM + VRAM combined, or unified memory:

| Size | 4-bit | 8-bit | BF16 / FP16 |
|---|---:|---:|---:|
| E2B | 4 GB | 5–8 GB | 10 GB |
| E4B | 5.5–6 GB | 9–12 GB | 16 GB |
| 12B | 7–8 GB | 13–14 GB | 25 GB |
| 26B A4B | 16–18 GB | 28–30 GB | 52 GB |
| 31B | 17–20 GB | 34–38 GB | 62 GB |

!!! note
    These are Unsloth's figures for their GGUF builds and include runtime overhead, so they run
    slightly above the static weight sizes in the
    [Gemma 4 overview](../models/gemma-4/index.md). Context (KV cache) is on top of both.

Unsloth recommends **dynamic 4-bit** (`UD-Q4_K_XL`) for 12B, 26B A4B and 31B, and **8-bit**
(`Q8_0`) for E2B and E4B.

## Recommended settings

Use Google's default Gemma 4 sampling parameters — the same ones described in
[Best practices](../get-started/best-practices.md):

```text
temperature = 1.0
top_p = 0.95
top_k = 64
```

## Unsloth Studio

<ol class="steps" markdown>
<li markdown>
**Install**

Download the [Unsloth app](https://unsloth.ai/download) for macOS, Windows or Linux, or install
it manually.

macOS, Linux, WSL:

```bash
curl -fsSL https://unsloth.ai/install.sh | sh
```

Windows PowerShell:

```powershell
irm https://unsloth.ai/install.ps1 | iex
```
</li>
<li markdown>
**Launch**

```bash
unsloth studio -H 0.0.0.0 -p 8888
```

Then open `http://127.0.0.1:8888`. To serve it over HTTPS through a free Cloudflare tunnel
instead, use `unsloth studio --secure`.
</li>
<li markdown>
**Download and run Gemma 4**

Search for Gemma 4 in the Chat tab and pick a quant. Inference parameters are set automatically,
and both GGUF and MLX files are supported.
</li>
</ol>

## llama.cpp

Build `llama.cpp` first — set `-DGGML_CUDA=OFF` for CPU-only machines and for Apple silicon,
where Metal is on by default:

```bash
git clone https://github.com/ggml-org/llama.cpp
cmake llama.cpp -B llama.cpp/build \
    -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON
cmake --build llama.cpp/build --config Release -j --clean-first \
    --target llama-cli llama-mtmd-cli llama-server llama-gguf-split
cp llama.cpp/build/bin/llama-* llama.cpp
```

Then pull and run a model straight from Hugging Face. There is no need to set a context length —
llama.cpp uses exactly what is required.

=== "12B"

    ```bash
    export LLAMA_CACHE="unsloth/gemma-4-12B-it-GGUF"
    ./llama.cpp/llama-cli \
        -hf unsloth/gemma-4-12b-it-GGUF:UD-Q4_K_XL \
        --temp 1.0 --top-p 0.95 --top-k 64
    ```

=== "26B A4B"

    ```bash
    export LLAMA_CACHE="unsloth/gemma-4-26B-A4B-it-GGUF"
    ./llama.cpp/llama-cli \
        -hf unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_XL \
        --temp 1.0 --top-p 0.95 --top-k 64
    ```

=== "31B"

    ```bash
    export LLAMA_CACHE="unsloth/gemma-4-31B-it-GGUF"
    ./llama.cpp/llama-cli \
        -hf unsloth/gemma-4-31B-it-GGUF:UD-Q4_K_XL \
        --temp 1.0 --top-p 0.95 --top-k 64
    ```

=== "E4B"

    ```bash
    export LLAMA_CACHE="unsloth/gemma-4-E4B-it-GGUF"
    ./llama.cpp/llama-cli \
        -hf unsloth/gemma-4-E4B-it-GGUF:Q8_0 \
        --temp 1.0 --top-p 0.95 --top-k 64
    ```

=== "E2B"

    ```bash
    export LLAMA_CACHE="unsloth/gemma-4-E2B-it-GGUF"
    ./llama.cpp/llama-cli \
        -hf unsloth/gemma-4-E2B-it-GGUF:Q8_0 \
        --temp 1.0 --top-p 0.95 --top-k 64
    ```

### Vision

Download the weights together with the multimodal projector, then pass both:

```bash
hf download unsloth/gemma-4-26B-A4B-it-GGUF \
    --local-dir unsloth/gemma-4-26B-A4B-it-GGUF \
    --include "*mmproj-BF16*" \
    --include "*UD-Q4_K_XL*"
```

```bash
./llama.cpp/llama-cli \
    --model unsloth/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf \
    --mmproj unsloth/gemma-4-26B-A4B-it-GGUF/mmproj-BF16.gguf \
    --temp 1.0 --top-p 0.95 --top-k 64
```

### Server

```bash
./llama.cpp/llama-server \
    --model unsloth/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf \
    --mmproj unsloth/gemma-4-26B-A4B-it-GGUF/mmproj-BF16.gguf \
    --temp 1.0 --top-p 0.95 --top-k 64 \
    --alias "unsloth/gemma-4-26B-A4B-it-GGUF" \
    --port 8001 \
    --chat-template-kwargs '{"enable_thinking":true}'
```

## Thinking mode

Thinking is enabled by putting `<|think|>` at the start of the system prompt. To turn it off
through llama.cpp instead, pass:

=== "macOS, Linux"

    ```bash
    --chat-template-kwargs '{"enable_thinking":false}'
    ```

=== "Windows PowerShell"

    ```powershell
    --chat-template-kwargs "{\"enable_thinking\":false}"
    ```

See [Best practices](../get-started/best-practices.md) for how thinking output is structured and
what to keep in conversation history.

## Fine-tuning

Unsloth trains Gemma with substantially less memory than a stock Transformers loop, which is
what makes single-GPU and free-Colab fine-tuning practical. See Unsloth's
[Gemma 4 fine-tuning guide](https://unsloth.ai/docs/models/gemma-4){ data-model="gemma-4" }, and our own
[fine-tuning overview](../tuning/index.md) for the QLoRA and full fine-tune paths.

## Resources

- [Gemma 4 GGUF collection](https://huggingface.co/collections/unsloth/gemma-4){ data-model="gemma-4" }
- [Unsloth documentation](https://unsloth.ai/docs/models/gemma-4){ data-model="gemma-4" }
