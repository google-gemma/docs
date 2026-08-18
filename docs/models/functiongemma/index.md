---
title: FunctionGemma model overview
description: FunctionGemma is a specialized version of our Gemma 3 270M model tuned for function calling.
models: [functiongemma]
source_url: "https://ai.google.dev/gemma/docs/functiongemma.md.txt"
---
# FunctionGemma model overview {: #hidden-title style="display: none;" }

<div class="video-embed">
  <iframe src="https://www.youtube-nocookie.com/embed/-Tgc_9uYJLI"
          title="Video" loading="lazy" allowfullscreen
          allow="accelerometer; encrypted-media; gyroscope; picture-in-picture"
          referrerpolicy="strict-origin-when-cross-origin"></iframe>
</div>

FunctionGemma is a specialized version of our Gemma 3 270M model tuned for
function calling. It is designed as a strong base for further training into
custom, fast, private, local agents that translate natural language into
executable API actions.

## When to choose FunctionGemma

FunctionGemma is the bridge between natural language and software execution. It
is the right tool if:

- **You have a defined API surface:** Your application has a defined set of actions (e.g., smart home, media, navigation).
- **You are ready to fine-tune:** You need the consistent, deterministic behavior that comes from fine-tuning on specific data, rather than the variability of zero-shot prompting.
- **You prioritize local-first deployment:** Your application requires near-instant latency and total data privacy, running efficiently within the compute and battery limits of edge devices.
- **You are building compound systems:** You need a lightweight edge model to handle local actions, allowing your system to process common commands on-device and only query larger models (like Gemma 3 27B) for more complex tasks.

<div class="cookbook-grid">
<a class="cookbook-card" href="https://huggingface.co/google/functiongemma-270m-it" target="_blank" rel="noopener">
  <div class="cookbook-card-title">Hugging Face <span class="cookbook-card-ext">↗</span></div>
  <p class="cookbook-card-desc">Get it on Hugging Face</p>
</a>
<a class="cookbook-card" href="https://www.kaggle.com/models/google/functiongemma" target="_blank" rel="noopener">
  <div class="cookbook-card-title">Kaggle <span class="cookbook-card-ext">↗</span></div>
  <p class="cookbook-card-desc">Get it on Kaggle</p>
</a>
</div>

[Access it on Vertex](https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/functiongemma){ data-model="functiongemma" }

As with other Gemma models, FunctionGemma is provided with open weights and
licensed for responsible [commercial use](https://ai.google.dev/gemma/terms), allowing you to
fine tune and deploy it in your own projects and applications.

[Formatting and best practices](formatting.md)
[Try FunctionGemma](inference.md)

[Fine-tune FunctionGemma](fine-tuning.md)
[Fine-tune the Mobile Actions demo](../../tuning/mobile-actions.md)
