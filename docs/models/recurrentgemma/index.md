---
title: RecurrentGemma
description: RecurrentGemma is an open model based on Griffin, a hybrid model architecture that mixes gated linear recurrences with local sliding window attention.
models: [recurrentgemma]
source_url: "https://ai.google.dev/gemma/docs/recurrentgemma.md.txt"
---
# RecurrentGemma

RecurrentGemma is an open model based on
[Griffin](https://arxiv.org/abs/2402.19427), a hybrid model architecture that
mixes gated linear recurrences with local sliding window attention.


Like Gemma, RecurrentGemma is well-suited for a
variety of text generation tasks, including question answering, summarization, and
reasoning. However, RecurrentGemma's unique architecture offers the following additional
advantages:

- **Reduced memory usage** — Lower memory requirements allow for the generation of longer samples on devices with limited memory, like single GPUs or CPUs.
- **Higher throughput** — RecurrentGemma can perform inference at significantly higher batch sizes, meaning it can generate substantially more tokens per second --- especially when generating long sequences.
- **High performance** — RecurrentGemma matches Gemma's performance while requiring less memory and achieving faster inference.

## More resources

### [View the model card](model-card.md)

RecurrentGemma's model card contains detailed information about the model, implementation information, evaluation information, model usage and limitations, and more.

### [View on Kaggle](https://www.kaggle.com/models/google/recurrentgemma){ data-model="recurrentgemma" }

View more code, Colab notebooks, information, and discussions about RecurrentGemma on Kaggle.

### [Run on GitHub](https://github.com/google-deepmind/recurrentgemma/blob/main/colabs/){ data-model="recurrentgemma" }

Run example Colab notebooks for JAX and PyTorch on GitHub.
