---
title: Reasoning
description: Use Gemma 4's thinking mode to reason before answering, for text and image prompts, and parse the reasoning apart from the answer.
models: [gemma-4]
frameworks: [transformers]
colab: https://colab.research.google.com/github/google-gemma/docs/blob/main/docs/notebooks/thinking-gemma-4-e2b.ipynb
---
# Reasoning


Gemma is a family of lightweight, state-of-the-art open models built from the same research and technology used to create the [Gemini](https://deepmind.google/technologies/gemini/#introduction) models. Gemma 4 is designed to be the world's most efficient open-weight model family.

This document demonstrates how to use the thinking capabilities of Gemma 4 to generate reasoning processes before providing a final answer. You will learn how to enable thinking mode for both text-only and multimodal (image-text) tasks using the Hugging Face [`transformers`](https://github.com/huggingface/transformers) library, and how to parse the output to separate thinking from the answer.

## Install Python packages

Install the Hugging Face libraries required for running the Gemma model and making requests.

```bash { .notebook-cell }
# Install PyTorch & other libraries
pip install torch accelerate

# Install the transformers library
pip install "transformers>=5.10.1"
```

## Load Model

Use the [`transformers`](https://github.com/huggingface/transformers) libraries to create an instance of a `processor` and `model` using the `AutoProcessor` and `AutoModelForImageTextToText` classes as shown in the following code example:

```python { .model-swap .notebook-cell }
MODEL_ID = "google/gemma-4-E2B-it"

from transformers import AutoProcessor, AutoModelForMultimodalLM

model = AutoModelForMultimodalLM.from_pretrained(MODEL_ID, dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained(MODEL_ID)
```

## A single text inference with Thinking

To generate a response using the model's thinking capabilities, pass `enable_thinking=True`, the processor will insert the correct thinking tokens into the prompt, instructing the model to think before responding.
!!! Note
    We've added an empty thinking token to the chat template for `gemma-4-12B-it`, `gemma-4-26B-A4B-it` and `gemma-4-31B-it`. This stabilizes model output by suppressing "ghost" thought channels that may appear even when thinking is deactivated.

| Model Size | Thinking State | Template Structure / Output |
|---|---|---|
| **E2B/E4B** | **OFF** | `<|turn>user\n[Prompt]<turn|>\n<|turn>model` |
| **E2B/E4B** | **ON** | `<|turn>system\n<|think|><turn|>\n<|turn>user\n[Prompt]<turn|>\n<|turn>model` |
| **12B/26B/31B** | **OFF** | ⚠️ `<|turn>user\n[Prompt]<turn|>\n<|turn>model\n<|channel>thought\n<channel|>` |
| **12B/26B/31B** | **ON** | `<|turn>system\n<|think|><turn|>\n<|turn>user\n[Prompt]<turn|>\n<|turn>model` |

```python { .notebook-cell }
from transformers import TextStreamer

message = [
    {
        "role": "user", "content": "What is the water formula?"
    }
]

text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=True)
inputs = processor(text=text, return_tensors="pt").to(model.device)
input_len = inputs["input_ids"].shape[-1]

streamer = TextStreamer(processor)
outputs = model.generate(**inputs, streamer=streamer, max_new_tokens=1024)
```

``` { .no-copy .output }
<bos><|turn>system
<|think|>
<turn|>
<|turn>user
What is the water formula?<turn|>
<|turn>model
<|channel>thought
Thinking Process:

1.  **Analyze the Request:** The user is asking, "What is the water formula?"
2.  **Interpret Ambiguity:** This phrase is highly ambiguous. "Water formula" could refer to several things:
    *   **Chemical Formula:** The molecular formula (\\(\text{H}_2\text{O}\\)).
    *   **Hydration/Biological Formula:** How water interacts with cells, molecules, or systems.
    *   **Chemistry/Solubility Formula:** Equations related to water (e.g., \\(\text{H}_2\text{O} \leftrightarrow \text{H}^+ + \text{OH}^-\\)).
    *   **Engineering/Hydrology Formula:** Equations describing fluid dynamics, flow, or phase changes.
    *   **A specific context (e.g., a game, a specific scientific model, or a metaphor).**
3.  **Determine the Most Likely Interpretation:** In a general knowledge context, when someone asks for a "formula," they usually mean the chemical formula or a fundamental concept/equation. The chemical formula is the most direct answer.
4.  **Formulate the Answer (Addressing different levels of specificity):**
    *   *Start with the literal chemical formula.* (\\(\text{H}_2\text{O}\\))
    *   *Explain what it is.* (Two hydrogen atoms, one oxygen atom.)
    *   *Address potential deeper meanings.* (Mentioning its role in chemistry, bonding, or the \\(\text{H}_2\text{O}\\) molecular structure.)
5.  **Refine the Output:** Present the answer clearly, acknowledging the different interpretations to ensure the user gets the information they need, even if they didn't specify the context. (Self-Correction: Don't just give \\(\text{H}_2\text{O}\\); provide context.)

6.  **Final Output Generation.** (Proceed to generate the final response.)<channel|>The term "water formula" can refer to a few different things, depending on the context you are interested in.

Here are the most common interpretations:

### 1. The Chemical Formula (The most common answer)

The chemical formula for water is:


$$\text{H}_2\text{O}$$


**What this means:**

*   It represents one molecule of water.
*   It consists of **two** atoms of Hydrogen (\\(\text{H}\\)).
*   It consists of **one** atom of Oxygen (\\(\text{O}\\)).

---

### 2. The Molecular Structure (How it is bonded)

The "formula" also describes how these atoms are arranged:

*   **Polarity:** Water is a **polar molecule**. The oxygen atom is much more electronegative than the hydrogen atoms, meaning it pulls the shared electrons closer, giving the oxygen end a partial negative charge (\\(\delta-\\)) and the hydrogen ends partial positive charges (\\(\delta+\\)).
*   **Shape:** The molecule has a **bent** or **V-shape**, with a bond angle of about 104.5 degrees. This bent shape is crucial for water's ability to form hydrogen bonds with other water molecules, which gives water its unique properties (like high surface tension and high heat capacity).

---

### 3. The Chemical Equation (Reactions)

If you are referring to water in a chemical reaction, you might be thinking of the basic relationship between water, hydrogen ions, and hydroxide ions:


$$\text{H}_2\text{O} \rightleftharpoons \text{H}^+ + \text{OH}^-$$


This equation shows that water can dissociate (break apart) into hydrogen ions (\\(\text{H}^+\\)) and hydroxide ions (\\(\text{OH}^-\\)).

***

**If you were looking for a specific type of formula (e.g., in biochemistry, fluid dynamics, or physics), please provide more context, and I can give you a more precise answer!**<turn|>
```

Once the text is generated, the response will contain both the reasoning blocks and the final answer bounded by special tokens. You can use the `parse_response` utility to easily extract them into a dictionary containing `thinking` and `answer`.
!!! Note
    `skip_special_tokens=False` is mandatory to make the reasoning parser to work correctly.

```python { .notebook-cell }
response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
result = processor.parse_response(response, prefix="")

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
Thinking Process:

1.  **Analyze the Request:** The user is asking, "What is the water formula?"
2.  **Interpret Ambiguity:** This phrase is highly ambiguous. "Water formula" could refer to several things:
    *   **Chemical Formula:** The molecular formula (\\(\text{H}_2\text{O}\\)).
    *   **Hydration/Biological Formula:** How water interacts with cells, molecules, or systems.
    *   **Chemistry/Solubility Formula:** Equations related to water (e.g., \\(\text{H}_2\text{O} \leftrightarrow \text{H}^+ + \text{OH}^-\\)).
    *   **Engineering/Hydrology Formula:** Equations describing fluid dynamics, flow, or phase changes.
    *   **A specific context (e.g., a game, a specific scientific model, or a metaphor).**
3.  **Determine the Most Likely Interpretation:** In a general knowledge context, when someone asks for a "formula," they usually mean the chemical formula or a fundamental concept/equation. The chemical formula is the most direct answer.
4.  **Formulate the Answer (Addressing different levels of specificity):**
    *   *Start with the literal chemical formula.* (\\(\text{H}_2\text{O}\\))
    *   *Explain what it is.* (Two hydrogen atoms, one oxygen atom.)
    *   *Address potential deeper meanings.* (Mentioning its role in chemistry, bonding, or the \\(\text{H}_2\text{O}\\) molecular structure.)
5.  **Refine the Output:** Present the answer clearly, acknowledging the different interpretations to ensure the user gets the information they need, even if they didn't specify the context. (Self-Correction: Don't just give \\(\text{H}_2\text{O}\\); provide context.)

6.  **Final Output Generation.** (Proceed to generate the final response.)

=== Answer ===
The term "water formula" can refer to a few different things, depending on the context you are interested in.

Here are the most common interpretations:

### 1. The Chemical Formula (The most common answer)

The chemical formula for water is:


$$\text{H}_2\text{O}$$


**What this means:**

*   It represents one molecule of water.
*   It consists of **two** atoms of Hydrogen (\\(\text{H}\\)).
*   It consists of **one** atom of Oxygen (\\(\text{O}\\)).

---

### 2. The Molecular Structure (How it is bonded)

The "formula" also describes how these atoms are arranged:

*   **Polarity:** Water is a **polar molecule**. The oxygen atom is much more electronegative than the hydrogen atoms, meaning it pulls the shared electrons closer, giving the oxygen end a partial negative charge (\\(\delta-\\)) and the hydrogen ends partial positive charges (\\(\delta+\\)).
*   **Shape:** The molecule has a **bent** or **V-shape**, with a bond angle of about 104.5 degrees. This bent shape is crucial for water's ability to form hydrogen bonds with other water molecules, which gives water its unique properties (like high surface tension and high heat capacity).

---

### 3. The Chemical Equation (Reactions)

If you are referring to water in a chemical reaction, you might be thinking of the basic relationship between water, hydrogen ions, and hydroxide ions:


$$\text{H}_2\text{O} \rightleftharpoons \text{H}^+ + \text{OH}^-$$


This equation shows that water can dissociate (break apart) into hydrogen ions (\\(\text{H}^+\\)) and hydroxide ions (\\(\text{OH}^-\\)).

***

**If you were looking for a specific type of formula (e.g., in biochemistry, fluid dynamics, or physics), please provide more context, and I can give you a more precise answer!**
```

## Multi-Turn Example with Thought Stripping

Properly managing the model's generated thoughts is critical for maintaining performance across multi-turn conversations.

- **Standard Multi-Turn Conversations:** You must remove (strip) the model's generated thoughts from the previous turn before passing the conversation history back to the model for the next turn. If you want to disable thinking mode mid-conversation, you can remove the `<|think|>`{.token-think} token when you strip the previous thoughts.
- **Function Calling (Exception):** If a single model turn involves function or tool calls, thoughts must **NOT** be removed between the function calls.
- **Maintaining Conversation History:** The historical model output must only include the final response. Ensure that no generated thoughts from previous turns remain in the context window before the next user turn begins.

```python { .notebook-cell }
from transformers import TextStreamer

# Append the clean response to the message history
message.append({
    "role": "assistant",
    "content": result["content"]
})

# ==========================================
# TURN 2
# ==========================================
print("\n--- Turn 2 ---")
# Add the next user query to the history
message.append({
    "role": "user",
    "content": "What is its boiling point in Celsius?"
})

text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=True)
inputs = processor(text=text, return_tensors="pt").to(model.device)
input_len = inputs["input_ids"].shape[-1]

streamer = TextStreamer(processor)
outputs = model.generate(**inputs, streamer=streamer, max_new_tokens=1024)
```

``` { .no-copy .output }
--- Turn 2 ---
<bos><|turn>system
<|think|>
<turn|>
<|turn>user
What is the water formula?<turn|>
<|turn>model
The term "water formula" can refer to a few different things, depending on the context you are interested in.

Here are the most common interpretations:

### 1. The Chemical Formula (The most common answer)

The chemical formula for water is:


$$\text{H}_2\text{O}$$


**What this means:**

*   It represents one molecule of water.
*   It consists of **two** atoms of Hydrogen (\\(\text{H}\\)).
*   It consists of **one** atom of Oxygen (\\(\text{O}\\)).

---

### 2. The Molecular Structure (How it is bonded)

The "formula" also describes how these atoms are arranged:

*   **Polarity:** Water is a **polar molecule**. The oxygen atom is much more electronegative than the hydrogen atoms, meaning it pulls the shared electrons closer, giving the oxygen end a partial negative charge (\\(\delta-\\)) and the hydrogen ends partial positive charges (\\(\delta+\\)).
*   **Shape:** The molecule has a **bent** or **V-shape**, with a bond angle of about 104.5 degrees. This bent shape is crucial for water's ability to form hydrogen bonds with other water molecules, which gives water its unique properties (like high surface tension and high heat capacity).

---

### 3. The Chemical Equation (Reactions)

If you are referring to water in a chemical reaction, you might be thinking of the basic relationship between water, hydrogen ions, and hydroxide ions:


$$\text{H}_2\text{O} \rightleftharpoons \text{H}^+ + \text{OH}^-$$


This equation shows that water can dissociate (break apart) into hydrogen ions (\\(\text{H}^+\\)) and hydroxide ions (\\(\text{OH}^-\\)).

***

**If you were looking for a specific type of formula (e.g., in biochemistry, fluid dynamics, or physics), please provide more context, and I can give you a more precise answer!**<turn|>
<|turn>user
What is its boiling point in Celsius?<turn|>
<|turn>model
<|channel>thought
Thinking Process:

1.  **Identify the core question:** The user wants to know the boiling point of water in Celsius.
2.  **Recall/Verify the fact:** What is the standard boiling point of pure water at standard atmospheric pressure?
3.  **State the answer:** The boiling point of water is \\(100^\circ \text{C}\\).
4.  **Add necessary context/nuance (Crucial step):** Boiling point is dependent on pressure. Standard atmospheric pressure is the reference.
5.  **Determine the required output:** Clear, direct, and accurate. (Self-correction: Ensure the Celsius unit is explicitly stated.)

*Drafting the response.*<channel|>The boiling point of pure water at standard atmospheric pressure (1 atmosphere or 101.325 kPa) is:


$$\mathbf{100^\circ\text{C} }$$


**Note:** If the atmospheric pressure is lower (e.g., at high altitudes), the boiling point will be slightly lower than \\(100^\circ\text{C}\\). If the pressure is higher, the boiling point will be higher.<turn|>
```

```python { .notebook-cell }
response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
result = processor.parse_response(response, prefix="")

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
Thinking Process:

1.  **Identify the core question:** The user wants to know the boiling point of water in Celsius.
2.  **Recall/Verify the fact:** What is the standard boiling point of pure water at standard atmospheric pressure?
3.  **State the answer:** The boiling point of water is \\(100^\circ \text{C}\\).
4.  **Add necessary context/nuance (Crucial step):** Boiling point is dependent on pressure. Standard atmospheric pressure is the reference.
5.  **Determine the required output:** Clear, direct, and accurate. (Self-correction: Ensure the Celsius unit is explicitly stated.)

*Drafting the response.*

=== Answer ===
The boiling point of pure water at standard atmospheric pressure (1 atmosphere or 101.325 kPa) is:


$$\mathbf{100^\circ\text{C} }$$


**Note:** If the atmospheric pressure is lower (e.g., at high altitudes), the boiling point will be slightly lower than \\(100^\circ\text{C}\\). If the pressure is higher, the boiling point will be higher.
```

## A single image inference

The procedure for using the thinking model with visual data is very similar. You can provide an image as part of the `messages` array. Just ensure you pass the image to the processor along with the formatted text, and the model will reason about the visual input before responding.

```python { .notebook-cell }
from PIL import Image
import matplotlib.pyplot as plt

prompt = "What is shown in this image?"
image_url = "https://raw.githubusercontent.com/google-gemma/cookbook/refs/heads/main/apps/sample-data/GoldenGate.png"

# download image
!wget -q {image_url} -O image.png
image = Image.open("image.png")

# Display all images
print("=== Downloaded image ===")
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.imshow(image)
ax.set_title("Image 1")
ax.axis("off")
plt.tight_layout()
plt.show()

message = [
    {
        "role": "user", "content": [
          {"type": "image"},
          {"type": "text", "text": prompt}
        ]
    }
]

text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=True)
inputs = processor(text=text, images=image, return_tensors="pt").to(model.device)
input_len = inputs["input_ids"].shape[-1]

outputs = model.generate(**inputs, max_new_tokens=1024)
response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

result = processor.parse_response(response, prefix="")

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
=== Downloaded image ===
```

![png](../../assets/capabilities/output_0f334997_1.png)

``` { .no-copy .output }
Role: assistant

=== Thoughts ===
Here's a thinking process to arrive at the suggested answer:

1.  **Analyze the Image:**
    *   **Dominant Feature:** A massive red suspension bridge spanning a body of water.
    *   **Bridge Identification:** The structure is highly recognizable. It has distinctive towers and suspension cables. This is clearly the Golden Gate Bridge.
    *   **Foreground/Midground:**
        *   Water (ocean/bay).
        *   A rocky outcrop/island in the immediate foreground.
        *   A breakwater or shoreline where the land meets the water.
    *   **Background:**
        *   Hills/landmasses in the distance.
        *   A clear, pale blue sky.
    *   **Other Details:**
        *   A large brick/stone building near the shoreline (part of the structure/area).
        *   Vehicles on the shoreline (suggesting it's an accessible area).

2.  **Identify the Location (Verification):**
    *   The Golden Gate Bridge is iconic and located in San Francisco, California, spanning the Golden Gate strait. The landscape, water color, and overall feel match this location.

3.  **Determine the Subject of the Request:** The user asked, "What is shown in this image?"

4.  **Draft the Description (Focusing on Key Elements):**
    *   *Start with the main subject.* (The bridge).
    *   *Add context.* (Location/setting).
    *   *Include foreground details.* (Water, rocks).

5.  **Refine the Description (Making it comprehensive and clear):**
    *   *Initial Draft thought:* It's the Golden Gate Bridge over water with rocks.
    *   *Improved Draft:* The image shows the iconic Golden Gate Bridge spanning a strait. In the foreground is water and rocks.
    *   *Final Polish (Adding visual detail and clarity):* Ensure the description captures the scale and color of the bridge.

6.  **Final Output Generation.** (This matches the structured answer provided below.)

=== Answer ===
This image shows the **Golden Gate Bridge** in San Francisco, California.

Key elements visible in the photograph include:

*   **The Golden Gate Bridge:** The massive red suspension bridge is the dominant feature, stretching across the water.
*   **Water:** A large expanse of blue water (likely the Golden Gate Strait or the Pacific Ocean) is visible in the foreground.
*   **Foreground Rocks:** A dark, rocky outcrop or small island is prominent in the immediate foreground.
*   **Shoreline/Land:** Parts of the coastline and a large brick structure are visible near the water's edge.
*   **Sky:** A clear, pale blue sky dominates the upper portion of the image.

The image captures the iconic scale and grandeur of the bridge against the backdrop of the water and landscape.
```

## Summary

In this guide, you learned how to use the thinking capabilities of Gemma 4 models to generate reasoning processes before final answers. You covered:

1. **Enabling thinking mode**: Using `enable_thinking=True` in `apply_chat_template`.
2. **Text streamer**: Using `TextStreamer` to observe the thinking process in real-time.
3. **Parsing the combined output**: Using `parse_response` to separate `thinking` and `answer` blocks.
4. **Multimodal tasks**: Applying thinking capabilities to multimodal tasks (image + text).

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
