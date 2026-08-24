---
title: Vision understanding
description: Prepare and pass visual data to Gemma 4 for object detection, OCR, visual question answering, captioning, and multi-image reasoning.
models: [gemma-4, gemma-3, gemma-3n, paligemma]
source_url: "https://ai.google.dev/gemma/docs/capabilities/vision.md.txt"
---
# Vision understanding

Gemma 4, the latest model from the Gemma family, can perform a wide range of
vision-language tasks such as object detection, Optical Character Recognition
(OCR), visual question answering, image captioning, and reasoning across
multiple images. It also supports variable resolution processing, allowing you
to balance inference speed and output accuracy.

This section explores how to effectively prepare and use visual data in your
prompts.

## Visual data

Visual data can come in many formats and resolution. The specific file formats
supported (such as JPEG and PNG) depend on the framework you choose to convert
your visual data into tensors.

Here are the key considerations when preparing visual data for Gemma:

- **Token cost:** Each image typically uses 256 tokens, though PaliGemma image token costs vary depending on the specific model selected.
- **Resolution:** The interpreted resolution---meaning the number of pixels encoded into tokens and processed by the model---depends on the Gemma version you are using:
    - **Gemma 4:** Variable resolution based on token budget. You can decide between budget sizes of 70, 140, 280, 560, or 1120 tokens, which determines how much the input image is resized and processed.
    - **Gemma 3:** (4B and higher) 896x896 resolution, with pan-and-scan options for larger images.
    - **Gemma 3n:** 256x256, 512x512, or 768x768 resolution
    - **PaliGemma 2:** 224x224, 448x448, or 896x896 resolution

Lower-resolution images process faster but capture fewer visual details. To
optimize inference speed, you should aim to provide visual data matching one of
the built-in interpreted resolutions of your chosen Gemma model.

## Variable resolution and token budgets

Gemma 4 models introduce the ability to process images at varying resolutions,
allowing you to tailor the visual input to your specific task. For example, you
might opt for a high resolution to pinpoint small details in object detection,
whereas a lower resolution might be preferable for analyzing individual video
frames to speed up processing. Ultimately, this feature lets you balance
inference speed against the accuracy of the visual representation.

You manage this tradeoff using a **token budget**. This budget sets a hard limit
on the number of visual tokens (also known as visual token embeddings) the model
can generate for a single image.

You can choose a budget of 70, 140, 280, 560, or 1120 tokens:

- **High budgets (e.g., 1120 tokens):** Preserve a higher image resolution. This generates more patches for the model to process, making it ideal for capturing fine, intricate details.
- **Low budgets (e.g., 70 tokens):** Downscale the image, resulting in fewer patches. This significantly accelerates inference times.

**How the budget works** The token budget directly controls how much an image is
resized by dictating the maximum number of initial image patches. The system
generates nine times as many patches as your selected budget. For example, a
budget of 280 tokens yields up to 2,520 patches (280 × 9).

The multiplier of 9 exists because of how the patches are compressed: during
processing, the model takes every 3x3 grid of adjacent patches and averages them
together to create a single embedding. These consolidated embeddings become your
final visual tokens. Consequently, a higher token budget yields more final
embeddings, allowing the model to extract richer, more granular information from
your visual data.

## Do's

Here are some best practices to follow when prompting Gemma with visual data.

- **Be specific**: If you have any specific tasks, provide sufficient context
  and guidance. Instead of "describe this image", try "describe the scene in this
  image, focusing on the relationship between the people and the objects."

- **Provide constraints**: To achieve a particular style or tone, be sure to
  specify it in your prompt. For example, instead of a general story request, ask
  Gemma to "Write a short story about this image in the style of a film noir."

- **Iterative Refinement**: Getting the intended output often requires
  experimentation and refining the prompts. Begin with a basic prompt and
  gradually add complexity.

## Don'ts

Here are some things to avoid when prompting Gemma with visual data.

- **Expect Exact Counts for Extremely Dense Objects**: While Gemma 4 excels at
  object detection and OCR, it may still provide approximations rather than
  exact counts for extremely dense or tiny objects (such as counting
  individual blades of grass). To achieve the best accuracy for visual tasks,
  use a higher token budget.

- **Vague or Ambiguous Prompts**: Instead of general prompts like "Generate
  something based on this image", provide specific instructions to achieve
  intended outputs. Clearly define what "something" is. For example, a poem,
  recipe, or code snippet.
## What's next

<div class="card-grid">
<a class="card" href="image.md">
  <span class="card-icon">{{ icon('lucide/image') }}</span>
  <div class="card-title">Image understanding</div>
  <p class="card-desc">Analyse and describe the contents of an image.</p>
</a>
<a class="card" href="video.md">
  <span class="card-icon">{{ icon('lucide/film') }}</span>
  <div class="card-title">Video understanding</div>
  <p class="card-desc">Pass video as frames and reason over what happens.</p>
</a>
<a class="card" href="../prompt-formatting.md">
  <span class="card-icon">{{ icon('lucide/braces') }}</span>
  <div class="card-title">Prompt formatting</div>
  <p class="card-desc">The control tokens that structure a Gemma prompt.</p>
</a>
</div>
