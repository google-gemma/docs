---
title: Full function calling sequence with FunctionGemma
description: "The complete FunctionGemma tool-calling cycle: the model's call, the developer's execution, and the final response built from the tool's output."
models: [functiongemma]
frameworks: [transformers]
source_url: "https://ai.google.dev/gemma/docs/functiongemma/full-function-calling-sequence-with-functiongemma.md.txt"
---
# Full function calling sequence with FunctionGemma


FunctionGemma is a specialized version of the Gemma 3 270M model, trained specifically for function calling improvements. It has the same architecture as Gemma, but uses a different chat format and tokenizer.

This guide shows the complete workflow for using FunctionGemma within the Hugging Face ecosystem. It covers the essential setup steps, including installing necessary Python packages like `torch` and `transformers`, and loading the model via the Hugging Face Hub. The core of the tutorial demonstrates a three-stage cycle for connecting the model to external tools: the **Model's Turn** to generate function call objects, the **Developer's Turn** to parse and execute code (such as a weather API), and the **Final Response** where the model uses the tool's output to answer the user.

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

## Example Use Cases

Function calling connects the generative capabilities of Gemma and the external data and services. Here are some common applications:

- **Answering Questions with Real-Time Data:** Use a search engine or weather API to answer questions like "What's the weather in Tokyo?" or "Who won the latest F1 race?"
- **Controlling External Systems:** Connect Gemma to other applications to perform actions, such as sending emails ("Send a reminder to the team about the meeting"), managing a calendar, or controlling smart home devices.
- **Creating Complex Workflows**: Chain multiple tool calls together to accomplish multi-step tasks, like planning a trip by finding flights, booking a hotel, and creating a calendar event.

## Using Tools

The core of function calling involves a four-step process:

1. **Define Tools**: Create the functions your model can use, specifying arguments and descriptions (e.g., a weather lookup function).
2. **Model's Turn**: FunctionGemma receives the user's prompt and a list of available tools. It generates a special object indicating which function to call and with what arguments instead of a plain text response.
3. **Developer's Turn**: Your code receives this object, executes the specified function with the provided arguments, and formats the result to be sent back to the model.
4. **Final Response**: FunctionGemma uses the function's output to generate a final, user-facing response.

Let's simulate this process.

```python
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
```

### Model's Turn

Here's the user prompt `"Hey, what's the weather in Tokyo right now?"`, and the tool `[get_current_weather]`. FunctionGemma generates a function call object as follows.

```python
prompt = "Hey, what's the weather in Tokyo right now?"
tools = [get_current_weather]

message = [
        # ESSENTIAL SYSTEM PROMPT:
        # This line activates the model's function calling logic.
        {"role": "developer", "content": "You are a model that can do function calling with the following functions"},
        {"role": "user", "content": prompt},
]

inputs = processor.apply_chat_template(message, tools=tools, add_generation_prompt=True, return_dict=True, return_tensors="pt")
output = processor.decode(inputs["input_ids"][0], skip_special_tokens=False)

out = model.generate(**inputs.to(model.device), pad_token_id=processor.eos_token_id, max_new_tokens=128)
generated_tokens = out[0][len(inputs["input_ids"][0]):]
output = processor.decode(generated_tokens, skip_special_tokens=True)

print(f"Prompt: {prompt}")
print(f"Tools: {tools}")
print(f"Output: {output}")
```

``` { .no-copy .output }
Prompt: Hey, what's the weather in Tokyo right now?
Tools: [<function get_current_weather at 0x79b7e0f52e80>]
Output: <start_function_call>call:get_current_weather{location:<escape>Tokyo, Japan<escape>}<end_function_call>
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

### Developer's Turn

Your application should parse the model's response to extract function name and argments, and append function call result with the `tool` role.
!!! Note
    Always validate function names and arguments before execution.
    

```python
import re

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
            for k, v1, v2 in re.findall(r"(\w+):(?:<escape>(.*?)<escape>|([^,}]*))", args)
        }
    } for name, args in re.findall(r"<start_function_call>call:(\w+)\{(.*?)\}<end_function_call>", text, re.DOTALL)]

calls = extract_tool_calls(output)
if calls:
    message.append({
        "role": "assistant",
        "tool_calls": [{"type": "function", "function": call} for call in calls]
    })
    print(message[-1])

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
        "role": "tool",
        "content": results
    })
    print(message[-1])
```

``` { .no-copy .output }
{'role': 'assistant', 'tool_calls': [{'type': 'function', 'function': {'name': 'get_current_weather', 'arguments': {'location': 'Tokyo, Japan'} } }]}
{'role': 'tool', 'content': [{'name': 'get_current_weather', 'response': {'temperature': 15, 'weather': 'sunny'} }]}
```
!!! Note
    For optimal results, append the tool execution result to your message history using the specific format below. This ensures the chat template correctly generates the required token structure (e.g., `response:get_current_weather{temperature:15,weather:<escape>sunny<escape>}`).
    

```python
message.append({
    "role": "tool",
    "content": {
        "name": function_name,
        "response": function_response
    }
})
```

In case of multiple independent requests:

```python
message.append({
    "role": "tool",
    "content": [
        {
            "name": function_name_1,
            "response": function_response_1
        },
        {
            "name": function_name_2,
            "response": function_response_2
        }
    ]
})
```

### Final Response

Finally, FunctionGemma reads the tool response and reply to the user.

```python
inputs = processor.apply_chat_template(message, tools=tools, add_generation_prompt=True, return_dict=True, return_tensors="pt")
out = model.generate(**inputs.to(model.device), pad_token_id=processor.eos_token_id, max_new_tokens=128)
generated_tokens = out[0][len(inputs["input_ids"][0]):]
output = processor.decode(generated_tokens, skip_special_tokens=True)
print(f"Output: {output}")
message.append({"role": "assistant", "content": output})
```

``` { .no-copy .output }
Output: The current weather in Tokyo is sunny with a temperature of 15 degrees Celsius.
```

You can see the full chat history below.

```python
# full history
for item in message:
  print(item)

print("-"*80)
output = processor.decode(out[0], skip_special_tokens=False)
print(f"Output: {output}")
```

``` { .no-copy .output }
{'role': 'developer', 'content': 'You are a model that can do function calling with the following functions'}
{'role': 'user', 'content': "Hey, what's the weather in Tokyo right now?"}
{'role': 'assistant', 'tool_calls': [{'type': 'function', 'function': {'name': 'get_current_weather', 'arguments': {'location': 'Tokyo, Japan'} } }]}
{'role': 'tool', 'content': [{'name': 'get_current_weather', 'response': {'temperature': 15, 'weather': 'sunny'} }]}
{'role': 'assistant', 'content': 'The current weather in Tokyo is sunny with a temperature of 15 degrees Celsius.'}
---
Output: <bos><start_of_turn>developer
You are a model that can do function calling with the following functions<start_function_declaration>declaration:get_current_weather{description:<escape>Gets the current weather in a given location.<escape>,parameters:{properties:{location:{description:<escape>The city and state, e.g. "San Francisco, CA" or "Tokyo, JP"<escape>,type:<escape>STRING<escape>},unit:{description:<escape>The unit to return the temperature in.<escape>,enum:[<escape>celsius<escape>,<escape>fahrenheit<escape>],type:<escape>STRING<escape>} },required:[<escape>location<escape>],type:<escape>OBJECT<escape>} }<end_function_declaration><end_of_turn>
<start_of_turn>user
Hey, what's the weather in Tokyo right now?<end_of_turn>
<start_of_turn>model
<start_function_call>call:get_current_weather{location:<escape>Tokyo, Japan<escape>}<end_function_call><start_function_response>response:get_current_weather{temperature:15,weather:<escape>sunny<escape>}<end_function_response>The current weather in Tokyo is sunny with a temperature of 15 degrees Celsius.<end_of_turn>
```

## Summary

You have established how to build an application that can calls functions with FunctionGemma. The workflow is established through a four-stage cycle:

1. **Define Tools**: Create the functions your model can use, specifying arguments and descriptions (e.g., a weather lookup function).
2. **Model's Turn**: The model receives the user's prompt and a list of available tools, returning a structured function call object instead of plain text.
3. **Developer's Turn**: The developer parses this output using regular expressions to extract function names and arguments, executes the actual Python code, and appends the results to the chat history using the specific tool role.
4. **Final Response**: The model processes the tool's execution result to generate a final, natural language answer for the user.

## What's next

<div class="card-grid">
<a class="card" href="fine-tuning.md">
  <span class="card-icon">{{ icon('lucide/graduation-cap') }}</span>
  <div class="card-title">Fine-tune FunctionGemma</div>
  <p class="card-desc">Teach it to pick the right tool for your policies.</p>
</a>
</div>
