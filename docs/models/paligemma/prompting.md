---
title: PaliGemma prompt and system instructions
description: This page describes prompt formatting and system instructions for PaliGemma models.
models: [paligemma-2, paligemma]
source_url: "https://ai.google.dev/gemma/docs/paligemma/prompt-system-instructions.md.txt"
---
# PaliGemma prompt and system instructions

This page describes prompt formatting and system instructions for PaliGemma
models. These Gemma model variants use the same general formatting as Gemma
foundation models and support a special syntax for specific image-related
tasks.

## Prompt format

PaliGemma models use the same prompt formatting as the Gemma foundation models
they are based on. However, PaliGemma models also support a special
[task syntax](prompting.md#prompt-task-syntax), which is described in the next section. For
more information on Gemma prompt formatting, see
[Gemma prompt and system instructions](../../get-started/prompt-formatting.md).

!!! Note
    although the pre-trained checkpoints support this prompt structure, they require fine-tuning to your specific use cases.


### Image and text data order

When prompting PaliGemma models with text and image data, the image data must
always be provided *first*, and then the text prompting data after it. Reversing
the order of image and text prompt data, or mixing image and text data will
typically generate unusable responses.

### Prompt task syntax

The PaliGemma models are trained with specific prompt patterns and syntax for
tasks such as object identification and image captioning. You can use this
prompt task syntax to request specific behavior from the PaliGemma models as
follows:

- **`"cap {lang}\n"`:** Very raw short caption (only supported by PT)
- **`"caption {lang}\n"`:** Short captions
- **`"describe {lang}\n"`:** Somewhat longer, more descriptive captions (only supported by PT)
- **`"ocr"`:** Optical character recognition (only supported by PT)
- **`"answer {lang} {question}\n"`:** Question answering about the image contents
- **`"question {lang} {answer}\n"`:** Question generation for a given answer (only supported by PT)
- **`"detect {object} ; {object}\n"`:** Locate listed objects in an image and return the bounding boxes for those objects
- **`"segment {object} ; {object}\n"`:** Locate the area occupied by the listed objects in an image to create an image segmentation for that object

!!! Note
    the "object" for detection and segmentation can also be an expression, such as "the penguin on the left".


The `{lang}` options are for language codes. PaliGemma supports language
recognition for 34 different languages for task prompts with this option. You
can find the list of supported languages on
[GitHub](https://github.com/google/crossmodal-3600/blob/main/web-data/README.md).

!!! Important
    The [PaliGemma FT](index.md#model-categories) category of models are trained for specific tasks and *only* support the syntax for the target task.


For a detailed code examples showing how to use this syntax, see the
[Generate PaliGemma output with Keras](inference.md)
tutorial.

### Prompting with natural language

Although the syntax in the previous section is recommended, the mix models also
support natural language for many of the tasks. For example, "describe this
image briefly" or "what is this text" will still work even if not prompted with
the exact syntax.

### Batched prompt commands

You can provide more than one prompt command within a single prompt as a batch
of instructions. Each prompt command must end with a `\n` character. The
following example demonstrates how to structure your prompt text to provide
multiple instructions.

```python
prompts = [
    'answer en where is the cow standing?\n',
    'answer en what color is the cow?\n',
    'describe en\n',
    'detect cow\n',
    'segment cow\n',
]
images = [cow_image, cow_image, cow_image, cow_image, cow_image]
outputs = paligemma.generate(
    inputs={
        "images": images,
        "prompts": prompts,
    }
)
for output in outputs:
    print(output)
```

## System instructions

The PaliGemma models don't support any additional system instructions beyond
the [Gemma system instructions](../../get-started/prompt-formatting.md#system-instructions)
from the foundation models they are based on.
