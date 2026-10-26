---
title: Run Gemma with Hugging Face Transformers
description: Generating text, summarizing, and analysing content are just some of the tasks you can accomplish with Gemma open models.
models: [gemma-4, gemma-3, gemma-3n]
frameworks: [transformers]
source_url: "https://ai.google.dev/gemma/docs/core/huggingface_inference.md.txt"
colab: https://colab.research.google.com/github/google-gemma/docs/blob/main/docs/notebooks/run-gemma-with-hf-gemma-4-e2b.ipynb
---
# Run Gemma with Hugging Face Transformers


Generating text, summarizing, and analysing content are just some of the tasks you can accomplish with Gemma open models. This tutorial shows you how to get started running Gemma using Hugging Face Transformers using both text and image input to generate text content. The Transformers Python library provides a API for accessing pre-trained generative AI models, including Gemma. For more information, see the [Transformers](https://huggingface.co/docs/transformers/en/index) documentation.

!!! Note
    Gemma 3 and later models support input of both images and text. Earlier versions of Gemma only support text input, except for some variants, including [PaliGemma](../get-started/index.md).


### Install Python packages

Install the Hugging Face libraries required for running the Gemma model and making requests.

```python { .notebook-cell }
# Install Pytorch
%pip install torch

# Install a transformers
%pip install "transformers>=5.10.1"
```

## Generate text from text

Prompting a Gemma model with text to get a text response is the simplest way to use Gemma and works with nearly all Gemma variants. This section shows how to use the Hugging Face Transformers library load and configure a Gemma model for text to text generation.

### Load model

Use the `torch` and `transformers` libraries to create an instance of a model execution `pipeline` class with Gemma. When using a model for generating output or following directions, select an instruction tuned (IT) model, which typically has `it` in the model ID string. Using the `pipeline` object, you specify the Gemma variant you want to use, the type of task you want to perform, specifically `"any-to-any"` for multimodal generation, as shown in the following code example:

```python { .notebook-cell .model-swap }
from transformers import pipeline

MODEL_ID = "google/gemma-4-E2B-it"

pipe = pipeline(
    task="any-to-any",
    model=MODEL_ID,
    device_map="auto",
    dtype="auto"
)
```

Gemma supports only a few `task` settings for generation. For more information on the available `task` settings, see the Hugging Face Pipelines [task()](https://huggingface.co/docs/transformers/main/en/main_classes/pipelines#transformers.pipeline.task) documentation. For more information about using the Pipeline class, see the Hugging Face [Pipelines](https://huggingface.co/docs/transformers/main/en/main_classes/pipelines) documentation.

### Run text generation

Once you have the Gemma model loaded and configured in a `pipeline` object, you can send prompts to the model. The following example code shows a basic request using the `text` parameter:

```python { .notebook-cell }
pipe(text="<|turn>user\nroses are red<turn|>\n<|turn>model\n")
```

``` { .no-copy .output }
[{'input_text': '<|turn>user\nroses are red<turn|>\n<|turn>model\n',
  'generated_text': "<|turn>user\nroses are red<turn|>\n<|turn>model\nThat's a classic rhyme! It immediately brings up the image of **roses and the color red**.\n\nIs there anything else you'd like to talk about, or perhaps another rhyme you'd like to try? 😊<turn|>"}]
```

### Use a prompt template

When generating content with more complex prompting, use a prompt template to structure your request. A prompt template allows you to specify input from specific roles, such as `user`{.token-role} or `model`{.token-role}, and is a required format for managing multi-turn chat interactions with Gemma models. The following example code shows how to constuct a prompt template for Gemma:

```python { .notebook-cell }
from transformers import GenerationConfig
config = GenerationConfig.from_pretrained(MODEL_ID)
config.max_new_tokens = 512
gen_kwargs = dict(generation_config=config)

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}]
    },
    {
        "role": "user",
        "content": [{"type": "text", "text": "Roses are red..."}]
    },
]

pipe(messages, return_full_text=False, generate_kwargs=gen_kwargs)
```

``` { .no-copy .output }
[{'input_text': [{'role': 'system',
    'content': [{'type': 'text', 'text': 'You are a helpful assistant.'}]},
   {'role': 'user',
    'content': [{'type': 'text', 'text': 'Roses are red...'}]}],
  'generated_text': 'Roses are red,\nViolets are blue,\nHow do you do?<turn|>'}]
```

## Generate text from image data

Starting with Gemma 3, for model sizes 4B and higher, you can use image data as part of your prompt. This section shows how to use the Transformers library to load and configure a Gemma model to use image data and text input to generate text output.

### Use a prompt template

When generating content with more complex prompting, use a prompt template to structure your request. A prompt template allows you to specify input from specific roles, such as `user`{.token-role} or `model`{.token-role}, and is a required format for managing multi-turn chat interactions with Gemma models. The following example code shows how to constuct a prompt template for Gemma:

```python { .notebook-cell }
from transformers import GenerationConfig
config = GenerationConfig.from_pretrained(MODEL_ID)
config.max_new_tokens = 512
gen_kwargs = dict(generation_config=config)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://ai.google.dev/static/gemma/docs/images/thali-indian-plate.jpg"},
            {"type": "text", "text": "What is shown in this image?"},
        ]
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "This image shows"},
        ],
    },
]

pipe(text=messages, return_full_text=False, generate_kwargs=gen_kwargs)
```

``` { .no-copy .output }
[{'input_text': [{'role': 'user',
    'content': [{'type': 'image',
      'url': 'https://ai.google.dev/static/gemma/docs/images/thali-indian-plate.jpg'},
     {'type': 'text', 'text': 'What is shown in this image?'}]},
   {'role': 'assistant',
    'content': [{'type': 'text', 'text': 'This image shows'}]}],
  'generated_text': " a platter of Indian food, likely a meal or a spread featuring various side dishes and bread.\n\nHere's a breakdown of what appears to be present:\n\n* **Flatbread:** There is a large, folded flatbread in the center, which looks like a type of roti, chapati, or perhaps a paratha.\n* **Dips/Sauces/Condiments (in small bowls):**\n    * A bowl containing a yellow substance (possibly a sauce, chutney, or butter/ghee).\n    * A bowl containing a creamy white sauce (like yogurt dip or raita).\n    * A bowl containing a reddish-brown, somewhat thick sauce or curry.\n    * A bowl containing a bright yellow/orange sauce.\n    * A bowl containing a green sauce or chutney.\n* **Rice:** A portion of white, fluffy rice is visible on the left side.\n* **Bread/Crackers:** In the upper right corner, there is a piece of what looks like a dry, textured bread or cracker.\n* **Other Items:** There are some reddish, textured pieces mixed in with the sauces on the right side, which might be fried items or components of a curry.\n\nOverall, it looks like a traditional Indian meal setup with bread, rice, and various accompanying sauces or curries.<turn|>"}]
```

You can include multiple images in your prompt by including additional `"type": "image",` entries in the `content` list.
!!! Note
    Do not use `<|image|>`{.token-multi}, `<start_of_image>`{.token-multi} or `<image_soft_token>`{.token-multi} tokens in text portion of a prompt template as this approach creates redundant tokens and processing errors.
    

<!-- notebook:requires-start audio -->
## Generate text from audio data

With [Gemma 4](../models/gemma-4/index.md) and [Gemma 3n](../models/gemma-3n/index.md), you can use audio data as part of your prompt. This section shows how to use the Transformers library to load and configure a Gemma model to use audio data and text input to generate text output.

### Use a prompt template

When generating content with audio, use a prompt template to structure your request. A prompt template allows you to specify input from specific roles, such as `user`{.token-role} or `model`{.token-role}, and is a required format for managing multi-turn chat interactions with Gemma models. The following example code shows how to constuct a prompt template for Gemma with audio data input:

```python { .notebook-cell data-requires="audio" }
from transformers import GenerationConfig
config = GenerationConfig.from_pretrained(MODEL_ID)
config.max_new_tokens = 512
gen_kwargs = dict(generation_config=config)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Transcribe the following speech segment in its original language. Follow these specific instructions for formatting the answer:\n* Only output the transcription, with no newlines.\n* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three."},
            {"type": "audio", "audio": "https://ai.google.dev/gemma/docs/audio/roses-are.wav"},
        ]
    }
]

pipe(text=messages, return_full_text=False, generate_kwargs=gen_kwargs)
```

``` { .no-copy .output }
[{'input_text': [{'role': 'user',
    'content': [{'type': 'text',
      'text': 'Transcribe the following speech segment in its original language. Follow these specific instructions for formatting the answer:\n* Only output the transcription, with no newlines.\n* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three.'},
     {'type': 'audio',
      'audio': 'https://ai.google.dev/gemma/docs/audio/roses-are.wav'}]}],
  'generated_text': 'Roses are red, violets are blue.<turn|>'}]
```

You can include multiple audio files in your prompt by including additional `"type": "audio",` entries in the `content` list.
!!! Note
    Do not use `<|audio|>`{.token-multi} or `<audio_soft_token>`{.token-multi} tokens in text portion of a prompt template as this approach creates redundant tokens and processing errors.
    
<!-- notebook:requires-end -->

## Next steps

Build and explore more with Gemma models:

- [Fine-tune Gemma for text tasks using Hugging Face Transformers](../tuning/qlora.md)
- [Fine-tune Gemma for vision tasks using Hugging Face Transformers](../tuning/vision-qlora.md)
- [Use Gemma open models with Vertex AI][vertex-use-gemma]
- [Fine-tune Gemma using Keras and deploy to Vertex AI](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/model_garden/model_garden_gemma_kerasnlp_to_vertexai.ipynb)
