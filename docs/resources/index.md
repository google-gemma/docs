---
title: Cookbook
description: Interactive demos, tools, notebooks, and community resources for the Gemma model family.
cards:
  - title: "Vision Token Budget"
    desc: "Visualize how Gemma 4 allocates image tokens across resolutions and aspect ratios."
    url: "https://huggingface.co/spaces/google/gemma4_vision_token_budget"
  - title: "Gemma Skills"
    desc: "A collection of task-specific skills and prompts for Gemma models, from code generation to analysis."
    url: "https://github.com/google-gemma/gemma-skills"
  - title: "Gemma Cookbook"
    desc: "Official Jupyter notebooks and guides covering inference, fine-tuning, RAG, function calling, and more."
    url: "https://github.com/google-gemma/gemma-cookbook"
  - title: "Google AI Studio"
    desc: "Experiment with Gemma models in the browser. Prototype prompts, test multimodal inputs, and export code."
    url: "https://aistudio.google.com/"
  - title: "Hugging Face Collection"
    desc: "Browse all Gemma 4 model weights, quantized variants, and community fine-tunes."
    url: "https://huggingface.co/collections/google/gemma-4-67ab89fa1f8b53e3d1da28d9"
  - title: "Kaggle Models"
    desc: "Download Gemma model weights from Kaggle with starter notebooks and benchmarks."
    url: "https://www.kaggle.com/models/google/gemma-4"
  - title: "Ollama Library"
    desc: "Run Gemma 4 locally with a single command. Pre-quantized weights for CPU and GPU inference."
    url: "https://ollama.com/library/gemma4"
  - title: "LM Studio"
    desc: "Run Gemma models with a native desktop GUI. Chat, completions, and a local OpenAI-compatible server."
    url: "https://lmstudio.ai/"
  - title: "Gemma GitHub"
    desc: "The official google-gemma GitHub organization with all repositories, tools, and community projects."
    url: "https://github.com/google-gemma"
hide:
  - navigation
  - toc
---

# Cookbook

Interactive demos, tools, notebooks, and community resources for the Gemma model family.

<div class="cookbook-grid">
{% for card in cards %}
<a class="cookbook-card" href="{{ card.url }}" target="_blank" rel="noopener">
  <div class="cookbook-card-title">{{ card.title }} <span class="cookbook-card-ext">↗</span></div>
  <p class="cookbook-card-desc">{{ card.desc }}</p>
</a>
{% endfor %}
</div>
