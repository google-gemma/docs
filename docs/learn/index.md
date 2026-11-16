---
title: Learn
description: Illustrated walkthroughs of how the Gemma models work, from the architecture down to individual mechanisms.
---

# Learn

This section explains how the Gemma models actually work. Where the rest of the documentation
shows you what to run, these pages take an architecture apart and draw it — what the layers do,
why a design choice was made, and where a model's behaviour comes from.

Read them in any order. Nothing here is a prerequisite for using Gemma.

## Interactive guide

<div class="card-grid">
<a class="card" href="how-llms-work.md">
  <span class="card-icon">{{ icon('lucide/play') }}</span>
  <div class="card-title">How LLMs work</div>
  <p class="card-desc">Tokenize text, explore embedding space, inspect attention
, tune Temperature, Top-K and Top-P, and stream a chat with thinking and tool ca
lls.</p>
</a>
</div>

## What's next

<div class="card-grid">
<a class="card" href="../tutorials/visual-guides/gemma-4.md">
  <span class="card-icon">{{ icon('lucide/book-open') }}</span>
  <div class="card-title">A Visual Guide to Gemma 4</div>
  <p class="card-desc">How the architecture works, drawn out step by step.</p>
</a>
<a class="card" href="../tutorials/visual-guides/gemma-4-12b.md">
  <span class="card-icon">{{ icon('lucide/book-open') }}</span>
  <div class="card-title">A Visual Guide to Gemma 4 12B</div>
  <p class="card-desc">What the unified 12B model does differently.</p>
</a>
<a class="card" href="../tutorials/visual-guides/mtp.md">
  <span class="card-icon">{{ icon('lucide/book-open') }}</span>
  <div class="card-title">A Visual Guide to MTP</div>
  <p class="card-desc">Drafter models and speculative decoding, illustrated.</p>
</a>
<a class="card" href="../tutorials/visual-guides/diffusiongemma.md">
  <span class="card-icon">{{ icon('lucide/book-open') }}</span>
  <div class="card-title">A Visual Guide to DiffusionGemma</div>
  <p class="card-desc">How denoising produces text instead of pixels.</p>
</a>
</div>

## Previous Gemma Explained Series

<div class="card-grid">
<a class="card" href="../tutorials/gemma-explained/overview.md">
  <span class="card-icon">{{ icon('lucide/book-open') }}</span>
  <div class="card-title">Overview of Gemma Architectures</div>
  <p class="card-desc">Text-to-text models and specialization for coding.</p>
</a>
<a class="card" href="../tutorials/gemma-explained/gemma-2.md">
  <span class="card-icon">{{ icon('lucide/book-open') }}</span>
  <div class="card-title">What’s New in Gemma 2</div>
  <p class="card-desc">Updated architecture with knowledge distillation for 2B and 9B models.</p>
</a>
<a class="card" href="../tutorials/gemma-explained/recurrentgemma.md">
  <span class="card-icon">{{ icon('lucide/book-open') }}</span>
  <div class="card-title">RecurrentGemma Architecture</div>
  <p class="card-desc">The Griffin architecture mixing local attention and linear recurrences for fast long-sequence inference.</p>
</a>
<a class="card" href="../tutorials/gemma-explained/paligemma.md">
  <span class="card-icon">{{ icon('lucide/book-open') }}</span>
  <div class="card-title">PaliGemma Architecture</div>
  <p class="card-desc">Vision-language model that takes text and images as input to generate text outputs.</p>
</a>
<a class="card" href="../tutorials/gemma-explained/gemma-3.md">
  <span class="card-icon">{{ icon('lucide/book-open') }}</span>
  <div class="card-title">What’s New in Gemma 3</div>
  <p class="card-desc">Updated vision-language model with 5-to-1 interleaved attention and 128k context handling.</p>
</a>
<a class="card" href="../tutorials/gemma-explained/embeddinggemma.md">
  <span class="card-icon">{{ icon('lucide/book-open') }}</span>
  <div class="card-title">EmbeddingGemma Architecture</div>
  <p class="card-desc">Text embedding model for search and retrieval.</p>
</a>
</div>

--8<-- "glossary.md"

