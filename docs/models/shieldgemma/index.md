---
title: ShieldGemma
description: ShieldGemma is a set of instruction tuned models for evaluating the safety of text and images against a set of defined safety policies.
models: [shieldgemma]
source_url: "https://ai.google.dev/gemma/docs/shieldgemma.md.txt"
---
# ShieldGemma {: #hidden-title style="display: none;" }

<div class="video-embed">
  <iframe src="https://www.youtube.com/embed/BgWO2iVPlyw"
          title="Video" loading="lazy" allowfullscreen
          allow="accelerometer; encrypted-media; gyroscope; picture-in-picture"
          referrerpolicy="strict-origin-when-cross-origin"></iframe>
</div>


ShieldGemma is a set of instruction tuned models for evaluating the safety of text and images against a set of defined safety policies. You can use
this model as part of a larger implementation of a generative AI application to help
evaluate and prevent generative AI applications from violating safety policies. The ShieldGemma family of models is provided with open weights to allow you to fine-tune it
for your specific use case.


ShieldGemma 2 is a 4B parameter model built to label images for safety.
ShieldGemma 1 is built on [Gemma 2](../gemma-2/index.md) in 2B, 9B, and 27B parameter sizes.


- **Content safety evaluation** — Evaluate the safety of prompt input and output responses against a set of defined safety policies.
- **[Tuneable, open models](https://huggingface.co/collections/google/shieldgemma-67d130ef8da6af884072a789){ data-model="shieldgemma" }** — ShieldGemma models are provided with open weights and can be fine-tuned for your specific use case.

## ShieldGemma 2

### [View the model card](model-card-v2.md)

ShieldGemma's 2 model card contains detailed information about the model implementation, evaluations, model usage and limitations, and more.

### [View on Kaggle](https://www.kaggle.com/models/google/shieldgemma-2){ data-model="shieldgemma" }

View more code, Colab notebooks, information, and discussions about ShieldGemma on Kaggle.

### [View on HuggingFace](https://huggingface.co/google/shieldgemma-2-4b-it){ data-model="shieldgemma" }

Run a working example for using ShieldGemma to evaluate images.

## ShieldGemma 1

### [View the model card](model-card-v1.md)

ShieldGemma's model card contains detailed information about the model implementation, evaluations, model usage and limitations, and more.

### [View on Kaggle](https://www.kaggle.com/models/google/shieldgemma){ data-model="shieldgemma" }

View more code, Colab notebooks, information, and discussions about ShieldGemma on Kaggle.

### [Run in Colab](https://colab.research.google.com/github/google-gemini/gemma-cookbook/blob/main/responsible/shieldgemma_on_keras.ipynb){ data-model="shieldgemma" }

Run a working example for using ShieldGemma to evaluate text prompt input and output.
