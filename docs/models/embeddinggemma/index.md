---
title: EmbeddingGemma model overview
description: EmbeddingGemma is a 308M parameter multilingual text embedding model based on Gemma 3.
models: [embeddinggemma]
source_url: "https://ai.google.dev/gemma/docs/embeddinggemma.md.txt"
---
# EmbeddingGemma model overview {: #hidden-title style="display: none;" }

<div class="video-embed">
  <iframe src="https://www.youtube-nocookie.com/embed/Xu1X-J-r5Xk"
          title="Video" loading="lazy" allowfullscreen
          allow="accelerometer; encrypted-media; gyroscope; picture-in-picture"
          referrerpolicy="strict-origin-when-cross-origin"></iframe>
</div>

EmbeddingGemma is a 308M parameter multilingual text embedding model based on
Gemma 3. It is optimized for use in everyday devices, such as phones, laptops,
and tablets. The model produces numerical representations of text to be used
for downstream tasks like information retrieval, semantic similarity
search, classification, and clustering.

EmbeddingGemma includes the following key features:

- **Multilingual support**: Wide linguistic data understanding, trained in over 100 languages.
- **Flexible output dimensions**: Customize your output dimensions from 768 to 128 for speed and storage tradeoffs using Matryoshka Representation Learning (MRL).
- **2K token context**: Substantial input context for processing text data and documents directly on your hardware.
- **Storage efficient**: Run it on less than 200MB of RAM with quantization
- **Low latency**: Generative embeddings in less than 22ms on EdgeTPU for fast and fluid applications.
- **Offline and secure**: Generate embeddings of documents directly on your hardware, works without internet connection to keep sensitive data secure.

!!! Tip
    Deploy EmbeddingGemma with Gemma 3n to build contextually relevant mobile-first Retrieval Augmented Generation (RAG) pipelines and chatbots. See our [quickstart RAG notebook](https://github.com/google-gemma/cookbook/blob/main/tutorials/RAG_with_EmbeddingGemma.ipynb){ data-model="embeddinggemma" } to get started.


<div class="cookbook-grid">
<a class="cookbook-card" href="https://huggingface.co/collections/google/embeddinggemma-68b9ae3a72a82f0562a80dc4" target="_blank" rel="noopener">
  <div class="cookbook-card-title">Hugging Face <span class="cookbook-card-ext">↗</span></div>
  <p class="cookbook-card-desc">Get it on Hugging Face</p>
</a>
<a class="cookbook-card" href="https://www.kaggle.com/models/google/embeddinggemma" target="_blank" rel="noopener">
  <div class="cookbook-card-title">Kaggle <span class="cookbook-card-ext">↗</span></div>
  <p class="cookbook-card-desc">Get it on Kaggle</p>
</a>
</div>

[Access it on Vertex](https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/embeddinggemma){ data-model="embeddinggemma" }

As with other Gemma models, EmbeddingGemma is provided with open weights and
licensed for responsible [commercial use](https://ai.google.dev/gemma/terms), allowing you to
fine tune and deploy it in your own projects and applications.

[Try EmbeddingGemma](inference.md)
[Fine-tune EmbeddingGemma](fine-tuning.md)
