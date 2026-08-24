---
title: Function calling with Hugging Face Transformers
description: Call functions with FunctionGemma in Hugging Face Transformers, passing tools as manual JSON schemas or as raw Python functions.
models: [functiongemma]
frameworks: [transformers]
source_url: "https://ai.google.dev/gemma/docs/functiongemma/function-calling-with-hf.md.txt"
---
# Function calling with Hugging Face Transformers


FunctionGemma is a specialized version of the Gemma 3 270M model, trained specifically for function calling improvements. It has the same architecture as Gemma, but uses a different chat format and tokenizer.

This guide shows the process of using FunctionGemma within the Hugging Face ecosystem. It covers essential setup steps, such as installing the `torch` and `transformers` libraries and loading the model using `AutoProcessor` and `AutoModelForCausalLM`. Additionally, the guide explains how to pass tools to the model using either manual JSON schemas or raw Python functions and advises on when to use manual schemas to handle complex custom objects effectively.

## Setup

Before starting this tutorial, complete the following steps:

- Get access to FunctionGemma by logging into [Hugging Face](https://huggingface.co/google/functiongemma-270m-it){ data-model="functiongemma" } and selecting **Acknowledge license** for a FunctionGemma model.
- Generate a Hugging Face [Access Token](https://huggingface.co/docs/hub/en/security-tokens#how-to-manage-user-access-token) and add it to your Colab environment.


## Install Python packages

Install the Hugging Face libraries required for running the FunctionGemma model and making requests.

```bash
# Install PyTorch & other libraries
pip install torch

# Install the transformers library
pip install transformers
```

After you have accepted the license, you need a valid Hugging Face Token to access the model.

```python
# Login into Hugging Face Hub
from huggingface_hub import login
login()
```

## Load Model

Use the `torch` and `transformers` libraries to create an instance of a `processor` and `model` using the `AutoProcessor` and `AutoModelForCausalLM` classes as shown in the following code example:

```python
from transformers import AutoProcessor, AutoModelForCausalLM

GEMMA_MODEL_ID = "google/functiongemma-270m-it"

processor = AutoProcessor.from_pretrained(GEMMA_MODEL_ID, device_map="auto")
model = AutoModelForCausalLM.from_pretrained(GEMMA_MODEL_ID, dtype="auto", device_map="auto")
```

## Passing tools

You can pass tools to the model using the `apply_chat_template()` function via the `tools` argument. There are two methods for defining these tools:

- **JSON schema**: You can manually construct a JSON dictionary defining the function name, description, and parameters (including types and required fields).
- **Raw Python Functions** : You can pass actual Python functions. The system automatically generates the required JSON schema by parsing the function's type hints, arguments, and docstrings. For best results, docstrings should adhere to the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).

Below is the example with the JSON schema.

```python
weather_function_schema = {
    "type": "function",
    "function": {
        "name": "get_current_temperature",
        "description": "Gets the current temperature for a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city name, e.g. San Francisco",
                },
            },
            "required": ["location"],
        },
    }
}

message = [
    # ESSENTIAL SYSTEM PROMPT:
    # This line activates the model's function calling logic.
    {
        "role": "developer", "content": "You are a model that can do function calling with the following functions"
    },
    {
        "role": "user", "content": "What's the temperature in London?"
    }
]

inputs = processor.apply_chat_template(message, tools=[weather_function_schema], add_generation_prompt=True, return_dict=True, return_tensors="pt")

out = model.generate(**inputs.to(model.device), pad_token_id=processor.eos_token_id, max_new_tokens=128)
output = processor.decode(out[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)

print(f"Output: {output}")
```

``` { .no-copy .output }
Output: <start_function_call>call:get_current_temperature{location:<escape>London<escape>}<end_function_call>
```
!!! Note
    To ensure FunctionGemma correctly interprets the available tools and generates a structured call instead of plain text, the **developer** message is essential. This specific system prompt instructs the model that it has permission and capability to perform function calling.
    

```python
message = [
        # ESSENTIAL SYSTEM PROMPT:
        # This line activates the model's function calling logic.
        {"role": "developer", "content": "You are a model that can do function calling with the following functions"},
        {"role": "user", "content": prompt},
]
```

And the same example with the raw Python function.

```python
def get_current_temperature(location: str):
    """
    Gets the current temperature for a given location.

    Args:
        location: The city name, e.g. San Francisco
    """
    return "15°C"

message = [
    {
        "role": "user", "content": "What's the temperature in London?"
    }
]

inputs = processor.apply_chat_template(message, tools=[weather_function_schema], add_generation_prompt=True, return_dict=True, return_tensors="pt")

out = model.generate(**inputs.to(model.device), pad_token_id=processor.eos_token_id, max_new_tokens=128)
output = processor.decode(out[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)

print(f"Output: {output}")
```

``` { .no-copy .output }
Output: <start_function_call>call:get_current_temperature{location:<escape>London<escape>}<end_function_call>
```

## Important Caveat: Automatic vs. Manual Schemas

When relying on automatic conversion from Python functions to JSON schema, the generated output may not always meet specific expectations regarding complex parameters.

If a function uses a custom object (like a Config class) as an argument, the automatic converter may describe it simply as a generic "object" without detailing its internal properties.

In these cases, manually defining the JSON schema is preferred to ensure nested properties (such as theme or font_size within a config object) are explicitly defined for the model.

```python
import json
from transformers.utils import get_json_schema

class Config:
    def __init__(self):
        self.theme = "light"
        self.font_size = 14

def update_config(config: Config):
    """
    Updates the configuration of the system.

    Args:
        config: A Config object

    Returns:
        True if the configuration was successfully updated, False otherwise.
    """

update_config_schema = {
    "type": "function",
    "function": {
        "name": "update_config",
        "description": "Updates the configuration of the system.",
        "parameters": {
            "type": "object",
            "properties": {
                "config": {
                    "type": "object",
                    "description": "A Config object",
                    "properties": {"theme": {"type": "string"}, "font_size": {"type": "number"} },
                    },
                },
            "required": ["config"],
            },
        },
    }

print(f"--- [Automatic] ---")
print(json.dumps(get_json_schema(update_config), indent=2))

print(f"\n--- [Manual Schemas] ---")
print(json.dumps(update_config_schema, indent=2))
```

``` { .no-copy .output }
--- [Automatic] ---
{
  "type": "function",
  "function": {
    "name": "update_config",
    "description": "Updates the configuration of the system.",
    "parameters": {
      "type": "object",
      "properties": {
        "config": {
          "type": "object",
          "description": "A Config object"
        }
      },
      "required": [
        "config"
      ]
    }
  }
}

--- [Manual Schemas] ---
{
  "type": "function",
  "function": {
    "name": "update_config",
    "description": "Updates the configuration of the system.",
    "parameters": {
      "type": "object",
      "properties": {
        "config": {
          "type": "object",
          "description": "A Config object",
          "properties": {
            "theme": {
              "type": "string"
            },
            "font_size": {
              "type": "number"
            }
          }
        }
      },
      "required": [
        "config"
      ]
    }
  }
}
```

## Summary

Now you understand function calling with FunctionGemma. Key takeaways from this include:

- **Defining Tools**: You can define tools using two methods: creating a manual JSON schema or passing raw Python functions, where the system parses type hints and docstrings.
- **Schema Caveats**: While automatic conversion works for simple types, it struggles with complex custom objects. In these cases, manual JSON schema definition is required to ensure nested properties are visible to the model.

## What's next

<div class="card-grid">
<a class="card" href="full-sequence.md">
  <span class="card-icon">{{ icon('lucide/split') }}</span>
  <div class="card-title">Full calling sequence</div>
  <p class="card-desc">The complete cycle, from the model's call to the final answer.</p>
</a>
<a class="card" href="fine-tuning.md">
  <span class="card-icon">{{ icon('lucide/graduation-cap') }}</span>
  <div class="card-title">Fine-tune FunctionGemma</div>
  <p class="card-desc">Teach it to pick the right tool for your policies.</p>
</a>
</div>
