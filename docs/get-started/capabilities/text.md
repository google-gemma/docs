---
title: Inference
description: "Basic text inference with Gemma 4 in Hugging Face Transformers: single-turn prompts, multi-turn conversations, and system instructions."
models: [gemma-4]
frameworks: [transformers]
colab: https://colab.research.google.com/github/google-gemma/docs/blob/main/docs/notebooks/text.ipynb
---
# Inference

Gemma is a family of lightweight, state-of-the-art open models built from the same research and technology used to create the [Gemini](https://deepmind.google/technologies/gemini/#introduction) models. Gemma 4 is designed to be the world's most efficient open-weight model family.

This document provides a guide to performing basic text inference with Gemma 4 using the Hugging Face `transformers` library. It covers environment setup, model loading, and various text generation scenarios including single-turn prompts, structured multi-turn conversations, and applying system instructions.

## Install Python packages

Install the Hugging Face libraries required for running the Gemma model and making requests.

```bash { .notebook-cell }
# Install PyTorch & other libraries
pip install torch accelerate

# Install the transformers library
pip install "transformers>=5.10.1"
```

[Dialog](https://github.com/google-deepmind/dialog) is a library to manipulate and display conversations.

```bash { .notebook-cell }
pip install dialog
```

## Load Model

Use `transformers` library to load the pipeline

```python { .notebook-cell .model-swap }
MODEL_ID = "google/gemma-4-E2B-it"

from transformers import pipeline

txt_pipe = pipeline(
    task="text-generation",
    model=MODEL_ID,
    device_map="auto",
    dtype="auto"
)
```


## Run text generation

Once you have the Gemma model loaded and configured in a `pipeline` object, you can send prompts to the model. The following example code shows a basic request using the `text_inputs` parameter:

```python { .notebook-cell }
output = txt_pipe(text_inputs="<|turn>user\nRoses are..<turn|>\n<|turn>model\n")
print(output[0]['generated_text'].removesuffix("<turn|>"))
```

``` { .no-copy .output }
[transformers] Both `max_new_tokens` (=256) and `max_length`(=20) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
<|turn>user
Roses are..<turn|>
<|turn>model
Here are a few ways to complete the phrase "Roses are...":

**Focusing on the color:**

* **Roses are red.** (A classic, though slightly contradictory!)
* **Roses are beautiful.**
* **Roses are pink.**

**Focusing on the feeling/meaning:**

* **Roses are lovely.**
* **Roses are sweet.**
* **Roses are a sign of affection.**

**A slightly more poetic answer:**

* **Roses are a memory.**

**Which one feels right to you? 😊**
```

## Use Dialog library

```python { .notebook-cell }
import dialog
from transformers import GenerationConfig
config = GenerationConfig.from_pretrained(MODEL_ID)
config.max_new_tokens = 512

conv = dialog.Conversation(
    dialog.User("Roses are...")
)
output = txt_pipe(text_inputs=conv.as_text(), return_full_text=False, generation_config=config)
conv += dialog.Model(output[0]['generated_text'].removesuffix("<turn|>"))

print(conv.as_text())
conv.show()
```

``` { .no-copy .output }
<|turn>user
Roses are...<turn|>
<|turn>model
Here are a few ways to complete the phrase "Roses are...":

**Focusing on the scent:**

* **...fragrant.**
* **...scented.**

**Focusing on the visual:**

* **...beautiful.**
* **...vibrant.**
* **...red.**

**Focusing on the emotion (the most classic completion):**

* **...a symbol of love.**
* **...a declaration.**
* **...perfect.**

**If you want a simple, classic answer, I recommend:**

**"Roses are beautiful."** or **"Roses are a symbol of love."**
<dialog._src.widget.Conversation object at 0x7957faa35ac0>
```

## Use a prompt template

When generating content with more complex prompting, use a prompt template to structure your request. A prompt template allows you to specify input from specific roles, such as `user`{.token-role} or `model`{.token-role}, and is a required format for managing multi-turn chat interactions with Gemma models. The following example code shows how to construct a prompt template for Gemma:

```python { .notebook-cell }
from transformers import GenerationConfig
config = GenerationConfig.from_pretrained(MODEL_ID)
config.max_new_tokens = 512

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Write a short poem about the Kraken."},
        ]
    }
]

output = txt_pipe(messages, return_full_text=False, generation_config=config)
print(output[0]['generated_text'].removesuffix("<turn|>"))
```

``` { .no-copy .output }
Beneath the waves, where sunlight dies,
A shadow stirs, with ancient sighs.
The Kraken wakes, a monstrous might,
With tentacles of endless night.

A crushing grip, a salty dread,
Where ships are lost and hope is dead.
A legend spun of ink and brine,
A primal fear, a dark design.
```

## Multi-turn conversation

In a multi-turn setup, the conversation history is preserved as a sequence of alternating `user`{.token-role} and `model`{.token-role} roles. This cumulative list serves as the model's memory, ensuring that each new output is informed by the preceding dialogue.

```python { .notebook-cell }
import dialog
from transformers import GenerationConfig
config = GenerationConfig.from_pretrained(MODEL_ID)
config.max_new_tokens = 512

# User turn #1
conv = dialog.Conversation(
    dialog.User("Write a short poem about the Kraken.")
)

# Model response #1
output = txt_pipe(text_inputs=conv.as_text(), return_full_text=False, generation_config=config)
conv += dialog.Model(output[0]['generated_text'].removesuffix("<turn|>"))

# User turn #2
conv += dialog.User("Now with the Siren.")

# Model response #2
output = txt_pipe(text_inputs=conv.as_text(), return_full_text=False, generation_config=config)
conv += dialog.Model(output[0]['generated_text'].removesuffix("<turn|>"))

print(conv.as_text())
conv.show()
```

``` { .no-copy .output }
[transformers] You seem to be using the pipelines sequentially on GPU. In order to maximize efficiency please use a dataset
<|turn>user
Write a short poem about the Kraken.<turn|>
<|turn>model
In depths where sunlight cease,
A shadow vast and deep,
The Kraken wakes with might,
A terror to the sleep.
With tentacles of ink,
It pulls the ocean's brink.<turn|>
<|turn>user
Now with the Siren.<turn|>
<|turn>model
Where coral sleeps in silent grace,
A melody floats from the sea,
The Siren calls with silver thread,
A siren song for all to see.
With eyes of emerald, deep and wide,
She lures the sailor to the tide.
A siren's kiss, a deadly art,
That breaks the sailor's guarded heart.
<dialog._src.widget.Conversation object at 0x7955701cc0e0>
```

And here's the conversation exported as text.

```python { .notebook-cell }
chat_history = conv.as_text(training=True)
print(chat_history)
print("-"*80)

# display as Conversation widget
chat_history
```

``` { .no-copy .output }
<|turn>user
Write a short poem about the Kraken.<turn|>
<|turn>model
In depths where sunlight cease,
A shadow vast and deep,
The Kraken wakes with might,
A terror to the sleep.
With tentacles of ink,
It pulls the ocean's brink.<turn|>
<|turn>user
Now with the Siren.<turn|>
<|turn>model
Where coral sleeps in silent grace,
A melody floats from the sea,
The Siren calls with silver thread,
A siren song for all to see.
With eyes of emerald, deep and wide,
She lures the sailor to the tide.
A siren's kiss, a deadly art,
That breaks the sailor's guarded heart.<turn|>
---
<dialog._src.widget.ConversationStr object at 0x7957e64d30e0>
```


!!! Note
    If you set `training=True`, the conversation is assumed to be the full complete example. Always ends with `<turn|>`


## System instructions

Use the `system`{.token-role} role to provide the system-level instructions.

```python { .notebook-cell }
import dialog
from transformers import GenerationConfig
config = GenerationConfig.from_pretrained(MODEL_ID)
config.max_new_tokens = 512

conv = dialog.Conversation(
    dialog.System("Speak like a pirate."),
    dialog.User("Why is the sky blue?")
)

output = txt_pipe(text_inputs=conv.as_text(), return_full_text=False, generation_config=config)
conv += dialog.Model(output[0]['generated_text'].removesuffix("<turn|>"))

print(conv.as_text())
conv.show()
```

``` { .no-copy .output }
<|turn>system
Speak like a pirate.<turn|>
<|turn>user
Why is the sky blue?<turn|>
<|turn>model
Ahoy there! Why is the sky blue, ye ask? It be down to the **sunlight** and the **air** itself!

Imagine the sunlight be a big crew of tiny, invisible particles—like a whole fleet of little pirates! When the sunlight be crew of tiny particles, these particles go through the air and bump into the gas molecules that make up our sky (mostly nitrogen and oxygen).

When the sunlight hits these molecules, something magical happens! The light gets **scattered** in all directions, just like when a beam of light hits a big, dusty mirror and gets scattered everywhere!

Of all the colors in the sunlight—red, orange, yellow, green, blue, indigo—the **blue light gets scattered the most**! It gets bounced and spread out across the entire sky, making our beautiful daytime sky appear blue to our eyes!

So, next time ye look up, ye can tell the secret: it be the **sunlight** bein' **scattered** by the **air**!

**Shiver me timbers!** That's the pirate explanation!
<dialog._src.widget.Conversation object at 0x7957ec8ed2b0>
```

## Summary
In this guide, you learned how to perform basic text inference with Gemma 4 using the Hugging Face `transformers` library. You covered:

1. **Environment**: Setting up the environment and installing dependencies.
2. **Loading the model**: Using the `pipeline` abstraction.
3. **Inference**: Running basic text generation.
4. **Conversation**: Using the [`dialog`](https://github.com/google-deepmind/dialog) library for conversation tracking.
5. **Multi-turn conversations**: Multi-turn conversations and applying system instructions.

<!-- notebook:skip-start -->
## What's next

<div class="card-grid">
<a class="card" href="../prompt-formatting.md">
  <span class="card-icon">{{ icon('lucide/braces') }}</span>
  <div class="card-title">Prompt formatting</div>
  <p class="card-desc">The control tokens that structure a Gemma prompt.</p>
</a>
<a class="card" href="function-calling.md">
  <span class="card-icon">{{ icon('lucide/wrench') }}</span>
  <div class="card-title">Tool calling</div>
  <p class="card-desc">Give Gemma functions to call and feed the results back.</p>
</a>
<a class="card" href="../../deployment/index.md">
  <span class="card-icon">{{ icon('lucide/play') }}</span>
  <div class="card-title">Run Gemma</div>
  <p class="card-desc">Every engine that runs Gemma, from a laptop to a cluster.</p>
</a>
</div>
<!-- notebook:skip-end -->
