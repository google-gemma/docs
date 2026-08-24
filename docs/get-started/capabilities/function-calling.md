---
title: Tool calling
description: When using a generative artificial intelligence (AI) model such as Gemma, you may want to use the model to operate programming interfaces in order to complete tasks or answer questions.
models: [gemma-4]
frameworks: [transformers]
colab: https://colab.research.google.com/github/google-gemma/docs/blob/main/docs/notebooks/function-calling.ipynb
---
<!-- TODO: Rename this to "tool calling"? -->
# Tool calling

When using a generative artificial intelligence (AI) model such as Gemma, you
may want to use the model to operate programming interfaces in order to complete
tasks or answer questions. Instructing a model by defining a programming
interface and then making a request that uses that interface is called *function
calling* or *tool calling*.

!!! Tip "Important"
    *A Gemma model cannot execute code on its own.* When you generate code with function calling, you must run the generated code yourself or run it as part of your application. Always put safeguards in place to validate any generated code before executing it.

This guide shows the process of using Gemma 4 within the Hugging Face ecosystem.


## Install Python packages

Install the Hugging Face libraries required for running the Gemma model and making requests.

```bash { .notebook-cell }
# Install PyTorch & other libraries
pip install torch accelerate

# Install the transformers library
pip install "transformers>=5.10.1"
```

## Load Model

Use the `transformers` libraries to create an instance of a `processor` and `model` using the `AutoProcessor` and `AutoModelForImageTextToText` classes as shown in the following code example:

```python { .notebook-cell .model-swap }
MODEL_ID = "google/gemma-4-E2B-it"

from transformers import AutoProcessor, AutoModelForMultimodalLM

model = AutoModelForMultimodalLM.from_pretrained(MODEL_ID, dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained(MODEL_ID)
```

``` { .no-copy .output }
Loading weights:   0%|          | 0/1951 [00:00<?, ?it/s]
```

## Passing Tools

You can pass tools to the model using the `apply_chat_template()` function via the `tools` argument. There are two methods for defining these tools:

- **JSON schema**: You can manually construct a JSON dictionary defining the function name, description, and parameters (including types and required fields).
- **Raw Python Functions** : You can pass actual Python functions. The system automatically generates the required JSON schema by parsing the function's type hints, arguments, and docstrings. For best results, docstrings should adhere to the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).

Below is the example with the JSON schema.

```python { .notebook-cell }
from transformers import TextStreamer

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
    {
        "role": "system", "content": "You are a helpful assistant."
    },
    {
        "role": "user", "content": "What's the temperature in London?"
    }
]

text = processor.apply_chat_template(message, tools=[weather_function_schema], tokenize=False, add_generation_prompt=True)
inputs = processor(text=text, return_tensors="pt").to(model.device)
streamer = TextStreamer(processor)
outputs = model.generate(**inputs, streamer=streamer, max_new_tokens=64)
```

``` { .no-copy .output }
<bos><|turn>system
You are a helpful assistant.<|tool>declaration:get_current_temperature{description:<|"|>Gets the current temperature for a given location.<|"|>,parameters:{properties:{location:{description:<|"|>The city name, e.g. San Francisco<|"|>,type:<|"|>STRING<|"|>} },required:[<|"|>location<|"|>],type:<|"|>OBJECT<|"|>} }<tool|><turn|>
<|turn>user
What's the temperature in London?<turn|>
<|turn>model
<|tool_call>call:get_current_temperature{location:<|"|>London<|"|>}<tool_call|><|tool_response>
```

And the same example with the raw Python function.

```python { .notebook-cell }
from transformers.utils import get_json_schema

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

text = processor.apply_chat_template(message, tools=[get_json_schema(get_current_temperature)], tokenize=False, add_generation_prompt=True)
inputs = processor(text=text, return_tensors="pt").to(model.device)
streamer = TextStreamer(processor)
outputs = model.generate(**inputs, streamer=streamer, max_new_tokens=256)
```

``` { .no-copy .output }
<bos><|turn>system
<|tool>declaration:get_current_temperature{description:<|"|>Gets the current temperature for a given location.<|"|>,parameters:{properties:{location:{description:<|"|>The city name, e.g. San Francisco<|"|>,type:<|"|>STRING<|"|>} },required:[<|"|>location<|"|>],type:<|"|>OBJECT<|"|>} }<tool|><turn|>
<|turn>user
What's the temperature in London?<turn|>
<|turn>model
<|tool_call>call:get_current_temperature{location:<|"|>London<|"|>}<tool_call|><|tool_response>
```

## Full function calling sequence

This section demonstrates a three-stage cycle for connecting the model to external tools: the **Model's Turn** to generate function call objects, the **Developer's Turn** to parse and execute code (such as a weather API), and the **Final Response** where the model uses the tool's output to answer the user.

### Model's Turn

Here's the user prompt `"Hey, what's the weather in Tokyo right now?"`, and the tool `[get_current_weather]`. Gemma generates a function call object as follows.

```python { .notebook-cell }
# Define a function that our model can use.
def get_current_weather(location: str, unit: str = "celsius"):
    """
    Gets the current weather in a given location.

    Args:
        location: The city and state, e.g. "San Francisco, CA" or "Tokyo, JP"
        unit: The unit to return the temperature in. (choices: ["celsius", "fahrenheit"])

    Returns:
        temperature: The current temperature in the given location
        weather: The current weather in the given location
    """
    return {"temperature": 15, "weather": "sunny"}

prompt = "Hey, what's the weather in Tokyo right now?"
tools = [get_current_weather]

message = [
    {
        "role": "system", "content": "You are a helpful assistant."
    },
    {
        "role": "user", "content": prompt
    },
]

text = processor.apply_chat_template(message, tools=tools, tokenize=False, add_generation_prompt=True)
inputs = processor(text=text, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=128)
generated_tokens = out[0][len(inputs["input_ids"][0]):]
output = processor.decode(generated_tokens, skip_special_tokens=False)

print(f"Prompt: {prompt}")
print(f"Tools: {tools}")
print(f"Output: {output}")
```

``` { .no-copy .output }
Prompt: Hey, what's the weather in Tokyo right now?
Tools: [<function get_current_weather at 0x7bbe91d18180>]
Output: <|tool_call>call:get_current_weather{location:<|"|>Tokyo, JP<|"|>}<tool_call|><|tool_response>
```

### Developer's Turn

Your application should parse the model's response to extract the function name and argments, and append `tool_calls`{.token-tool} and `tool_responses`{.token-tool} with the `assistant`{.token-role} role.
!!! Note
    Always validate function names and arguments before execution.

```python { .notebook-cell }
import re
import json

def extract_tool_calls(text):
    def cast(v):
        try: return int(v)
        except:
            try: return float(v)
            except: return {'true': True, 'false': False}.get(v.lower(), v.strip("'\""))

    return [{
        "name": name,
        "arguments": {
            k: cast((v1 or v2).strip())
            for k, v1, v2 in re.findall(r'(\w+):(?:<\|"\|>(.*?)<\|"\|>|([^,}]*))', args)
        }
    } for name, args in re.findall(r"<\|tool_call>call:(\w+)\{(.*?)\}<tool_call\|>", text, re.DOTALL)]

calls = extract_tool_calls(output)
if calls:
    # Call the function and get the result
    #####################################
    # WARNING: This is a demonstration. #
    #####################################
    # Using globals() to call functions dynamically can be dangerous in
    # production. In a real application, you should implement a secure way to
    # map function names to actual function calls, such as a predefined
    # dictionary of allowed tools and their implementations.
    results = [
        {"name": c['name'], "response": globals()[c['name']](**c['arguments'])}
        for c in calls
    ]

    message.append({
        "role": "assistant",
        "tool_calls": [
            {"function": call} for call in calls
        ],
        "tool_responses": results
    })
    print(json.dumps(message[-1], indent=2))
```

``` { .no-copy .output }
{
  "role": "assistant",
  "tool_calls": [
    {
      "function": {
        "name": "get_current_weather",
        "arguments": {
          "location": "Tokyo, JP"
        }
      }
    }
  ],
  "tool_responses": [
    {
      "name": "get_current_weather",
      "response": {
        "temperature": 15,
        "weather": "sunny"
      }
    }
  ]
}
```
!!! Note
    For optimal results, append the tool execution result to your message history using the specific format below. This ensures the chat template correctly generates the required token structure (e.g., `response:get_current_weather{temperature:15,weather:<|"|>sunny<|"|>}`).

```python { .notebook-cell }
"tool_responses": [
  {
    "name": function_name,
    "response": function_response
  }
]
```

In case of multiple independent requests:

```python { .notebook-cell }
"tool_responses": [
  {
    "name": function_name_1,
    "response": function_response_1
  },
  {
    "name": function_name_2,
    "response": function_response_2
  }
]
```

### Final Response

Finally, Gemma reads the tool response and reply to the user.

```python { .notebook-cell }
text = processor.apply_chat_template(message, tools=tools, tokenize=False, add_generation_prompt=True)
inputs = processor(text=text, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=128)
generated_tokens = out[0][len(inputs["input_ids"][0]):]
output = processor.decode(generated_tokens, skip_special_tokens=True)
print(f"Output: {output}")
message[-1]["content"] = output
```

``` { .no-copy .output }
Output: The current weather in Tokyo is 15 degrees Celsius and sunny.
```

You can see the full chat history below.

```python { .notebook-cell }
# full history
print(json.dumps(message, indent=2))

print("-"*80)
output = processor.decode(out[0], skip_special_tokens=False)
print(f"Output: {output}")
```

``` { .no-copy .output }
[
  {
    "role": "system",
    "content": "You are a helpful assistant."
  },
  {
    "role": "user",
    "content": "Hey, what's the weather in Tokyo right now?"
  },
  {
    "role": "assistant",
    "tool_calls": [
      {
        "function": {
          "name": "get_current_weather",
          "arguments": {
            "location": "Tokyo, JP"
          }
        }
      }
    ],
    "tool_responses": [
      {
        "name": "get_current_weather",
        "response": {
          "temperature": 15,
          "weather": "sunny"
        }
      }
    ],
    "content": "The current weather in Tokyo is 15 degrees Celsius and sunny."
  }
]
---
Output: <bos><|turn>system
You are a helpful assistant.<|tool>declaration:get_current_weather{description:<|"|>Gets the current weather in a given location.<|"|>,parameters:{properties:{location:{description:<|"|>The city and state, e.g. "San Francisco, CA" or "Tokyo, JP"<|"|>,type:<|"|>STRING<|"|>},unit:{description:<|"|>The unit to return the temperature in.<|"|>,enum:[<|"|>celsius<|"|>,<|"|>fahrenheit<|"|>],type:<|"|>STRING<|"|>} },required:[<|"|>location<|"|>],type:<|"|>OBJECT<|"|>} }<tool|><turn|>
<|turn>user
Hey, what's the weather in Tokyo right now?<turn|>
<|turn>model
<|tool_call>call:get_current_weather{location:<|"|>Tokyo, JP<|"|>}<tool_call|><|tool_response>response:get_current_weather{temperature:15,weather:<|"|>sunny<|"|>}<tool_response|>The current weather in Tokyo is 15 degrees Celsius and sunny.<turn|>
```

### Function calling with Thinking

By utilizing an internal reasoning process, the model significantly enhances its function-calling accuracy. This allows for more precise decision-making regarding when to trigger a tool and how to define its parameters.

```python { .notebook-cell }
prompt = "Hey, I'm in Seoul. Is it good for running now?"
message = [
    {
        "role": "system", "content": "You are a helpful assistant."
    },
    {
        "role": "user", "content": prompt
    },
]

text = processor.apply_chat_template(message, tools=tools, tokenize=False, add_generation_prompt=True, enable_thinking=True)
inputs = processor(text=text, return_tensors="pt").to(model.device)
input_len = inputs["input_ids"].shape[-1]

out = model.generate(**inputs, max_new_tokens=1024)
output = processor.decode(out[0][input_len:], skip_special_tokens=False)
result = processor.parse_response(output)

for key, value in result.items():
  if key == "role":
    print(f"Role: {value}")
  elif key == "thinking":
    print(f"\n=== Thoughts ===\n{value}")
  elif key == "content":
    print(f"\n=== Answer ===\n{value}")
  elif key == "tool_calls":
    print(f"\n=== Tool Calls ===\n{value}")
  else:
    print(f"\n{key}: {value}...\n")
```

``` { .no-copy .output }
Role: assistant

=== Thoughts ===

1. **Analyze the Request:** The user is asking if it's good for running in Seoul right now.

2. **Identify Necessary Information:** To answer this question, I need current weather information for Seoul.

3. **Examine Available Tools:** The only tool available is `get_current_weather(location: str, unit: str = None)`.

4. **Determine Tool Usage:**
    * The request specifies the location: "Seoul".
    * The request implies needing current weather conditions to assess if it's suitable for running.
    * The `get_current_weather` tool is appropriate for this.

5. **Construct the Tool Call:**
    * `location` should be "Seoul".
    * `unit` is optional, but it's good practice to decide if a specific unit is needed or if the default is fine. Since the user didn't specify a unit, I can omit it or choose a default (though the tool definition doesn't specify a default, just that it's optional). Let's just call it with the location.

6. **Formulate the Response (Tool Call):** Call `get_current_weather` with `location="Seoul"`.

7. **Self-Correction/Refinement:** The tool call is the necessary next step to gather the information needed to answer the user's question. I cannot answer the question without the weather data.

8. **Final Output Generation:** Generate the JSON tool call.

=== Answer ===
<eos>

=== Tool Calls ===
[{'type': 'function', 'function': {'name': 'get_current_weather', 'arguments': {'location': 'Seoul'} } }]
```

Process the tool call and get the final answer.

```python { .notebook-cell }
calls = extract_tool_calls(output)
if calls:
    # Call the function and get the result
    #####################################
    # WARNING: This is a demonstration. #
    #####################################
    # Using globals() to call functions dynamically can be dangerous in
    # production. In a real application, you should implement a secure way to
    # map function names to actual function calls, such as a predefined
    # dictionary of allowed tools and their implementations.
    results = [
        {"name": c['name'], "response": globals()[c['name']](**c['arguments'])}
        for c in calls
    ]

    message.append({
        "role": "assistant",
        "tool_calls": [
            {"function": call} for call in calls
        ],
        "tool_responses": results
    })

text = processor.apply_chat_template(message, tools=tools, tokenize=False, add_generation_prompt=True)
inputs = processor(text=text, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=128)
generated_tokens = out[0][len(inputs["input_ids"][0]):]
output = processor.decode(generated_tokens, skip_special_tokens=True)
print(f"Output: {output}")
message[-1]["content"] = output

print("-"*80)
print("Full History")
print("-"*80)
print(json.dumps(message, indent=2))
```

``` { .no-copy .output }
Output: The current weather in Seoul is 15 degrees Celsius and sunny. That sounds like great weather for running!
---
Full History
---
[
  {
    "role": "system",
    "content": "You are a helpful assistant."
  },
  {
    "role": "user",
    "content": "Hey, I'm in Seoul. Is it good for running now?"
  },
  {
    "role": "assistant",
    "tool_calls": [
      {
        "function": {
          "name": "get_current_weather",
          "arguments": {
            "location": "Seoul"
          }
        }
      }
    ],
    "tool_responses": [
      {
        "name": "get_current_weather",
        "response": {
          "temperature": 15,
          "weather": "sunny"
        }
      }
    ],
    "content": "The current weather in Seoul is 15 degrees Celsius and sunny. That sounds like great weather for running!"
  }
]
```

## Important Caveat: Automatic vs. Manual Schemas

When relying on automatic conversion from Python functions to JSON schema, the generated output may not always meet specific expectations regarding complex parameters.

If a function uses a custom object (like a Config class) as an argument, the automatic converter may describe it simply as a generic "object" without detailing its internal properties.

In these cases, manually defining the JSON schema is preferred to ensure nested properties (such as theme or font_size within a config object) are explicitly defined for the model.

```python { .notebook-cell }
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

You have established how to build an application that can call functions with Gemma 4. The workflow is established through a four-stage cycle:

1. **Define Tools**: Create the functions your model can use, specifying arguments and descriptions (e.g., a weather lookup function).
2. **Model's Turn**: The model receives the user's prompt and a list of available tools, returning a structured function call object instead of plain text.
3. **Developer's Turn**: The developer parses this output using regular expressions to extract function names and arguments, executes the actual Python code, and appends the results to the chat history using the specific tool role.
4. **Final Response**: The model processes the tool's execution result to generate a final, natural language answer for the user.

<!-- notebook:skip-start -->
## What's next

<div class="card-grid">
<a class="card" href="../../deployment/index.md">
  <span class="card-icon">{{ icon('lucide/play') }}</span>
  <div class="card-title">Run Gemma</div>
  <p class="card-desc">Every engine that runs Gemma, from a laptop to a cluster.</p>
</a>
<a class="card" href="vision.md">
  <span class="card-icon">{{ icon('lucide/eye') }}</span>
  <div class="card-title">Vision understanding</div>
  <p class="card-desc">Detection, OCR, visual question answering and captioning.</p>
</a>
<a class="card" href="audio.md">
  <span class="card-icon">{{ icon('lucide/mic') }}</span>
  <div class="card-title">Audio understanding</div>
  <p class="card-desc">Transcribe, translate and reason over speech.</p>
</a>
<a class="card" href="thinking.md">
  <span class="card-icon">{{ icon('lucide/brain') }}</span>
  <div class="card-title">Reasoning</div>
  <p class="card-desc">Let Gemma reason before it answers, and parse the two apart.</p>
</a>
</div>
<!-- notebook:skip-end -->
