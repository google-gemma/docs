---
title: EmbeddingGemma model overview
description: EmbeddingGemma 2 is a lightweight, open-source 740M parameter model designed for unified multimodal embeddings.
models: [embeddinggemma-2, embeddinggemma]
source_url: "https://ai.google.dev/gemma/docs/embeddinggemma.md.txt"
---
# EmbeddingGemma model overview {: #hidden-title style="display: none;" }

<div class="video-embed">
  <iframe src="https://www.youtube.com/embed/anPsS6huQk0"
          title="Video" loading="lazy" allowfullscreen
          allow="accelerometer; encrypted-media; gyroscope; picture-in-picture"
          referrerpolicy="strict-origin-when-cross-origin"></iframe>
</div>

**EmbeddingGemma 2** is a lightweight, open-source 740M parameter model designed
for unified multimodal embeddings. Built on the Gemma 4 decoder architecture, it
maps **text, images, audio, and video** into a unified 768-dimensional vector
space. You can run demanding cross-modal workflows (e.g. multimodal
Retrieval-Augmented Generation (RAG), semantic search, and zero-shot
classification) locally and offline on consumer GPUs or CPUs.

<div class="cookbook-grid">
<a class="cookbook-card" href="https://huggingface.co/collections/google/embeddinggemma-2" target="_blank" rel="noopener">
  <div class="cookbook-card-title">Hugging Face <span class="cookbook-card-ext">↗</span></div>
  <p class="cookbook-card-desc">Get it on Hugging Face</p>
</a>
</div>

EmbeddingGemma 2 is provided with open weights and licensed under the [Apache
2.0 License](../../resources/apache-2.md), allowing you to fine-tune and deploy it in your
own projects and applications.

[Try EmbeddingGemma 2](inference.md)

[Fine-tune EmbeddingGemma 2](fine-tuning.md)

## Key features

EmbeddingGemma 2 introduces major improvements over EmbeddingGemma:

- **Native Multimodality**: Unlocks zero-shot text, code, image, video, and audio search directly in a single unified vector space.
- **Superior quality across text and code**: Outperforms EmbeddingGemma 1 on Code MTEB (Massive Text Embedding Benchmark).
- **Smaller base footprint** : Reduces the base text and code model size from 300M parameters down to **270M parameters**.
- **High storage efficiency** : Matryoshka Representation Learning (MRL) enables dynamic output truncation across **128, 256, 512, and 768
  dimensions** , delivering up to a **6x reduction in vector database storage
  costs** with minimal quality degradation.
- **Extended context length** : Supports up to **8,192 tokens** for text and source code across over 100 languages.

!!! tip
    **Tip:** Deploy EmbeddingGemma 2 with Gemma 4 E2B to build contextually relevant mobile-first Retrieval Augmented Generation (RAG) pipelines and chatbots. See our [quickstart RAG notebook](https://github.com/google-gemma/cookbook/blob/main/tutorials/RAG_with_EmbeddingGemma.ipynb){ data-model="embeddinggemma-2" } to get started.

## Previous Versions

### EmbeddingGemma

EmbeddingGemma is a 308M parameter multilingual text embedding model based on
Gemma 3. It is optimized for use in everyday devices, such as phones, laptops,
and tablets. The model produces numerical representations of text to be used
for downstream tasks like information retrieval, semantic similarity
search, classification, and clustering.

EmbeddingGemma includes the following key features:

- **Multilingual support**: Wide linguistic data understanding, trained in over 100 languages.
- **Flexible output dimensions**: Customize your output dimensions from 768 to 128 for speed and storage tradeoffs using Matryoshka Representation Learning (MRL).
- **2K token context**: Substantial input context for processing text data and documents directly on your hardware.
- **Storage efficient**: Run it on less than 200MB of RAM with quantization.
- **Low latency**: Generative embeddings in less than 22ms on EdgeTPU for fast and fluid applications.
- **Offline and secure**: Generate embeddings of documents directly on your hardware, works without internet connection to keep sensitive data secure.

--8<-- "glossary.md"
