---
title: Prompt Formatting
description: The control tokens Gemma 4 uses to structure a prompt, and the older start_of_turn format used by Gemma 1 to 3.
models: [gemma-4]
---


# Prompt formatting

Gemma 4 introduced a new set of control tokens. Gemma 1 through 3 use an older
`<start_of_turn>` format, kept below because those models are still in use.

## Gemma 4

The following sections specify the control tokens used by Gemma 4 and their use
cases. Note that the control tokens are reserved in and specific to our
tokenizer.



- Token to indicate a system instruction: `system`{.token-role}
- Token to indicate a user turn: `user`{.token-role}
- Token to indicate a model turn: `model`{.token-role}
- Token to indicate the beginning of a dialogue turn: `<|turn|>`{.token-turn}
- Token to indicate the end of a dialogue turn: `<turn|>`{.token-turn}

Here's an example dialogue:

<pre><code><span class="token-turn">&lt;|turn&gt;</span><span class="token-role">system</span>
You are a helpful assistant.<span class="token-turn">&lt;turn|&gt;</span>
<span class="token-turn">&lt;|turn&gt;</span><span class="token-role">user</span>
Hello.<span class="token-turn">&lt;turn|&gt;</span></code></pre>


### Multi-modalities

| Multimodal Token | Purpose |
|---|---|
| `<|image|>`{.token-multi} `<image|>`{.token-multi} | Indicate image embeddings |
| `<|audio|>`{.token-multi} `<audio|>`{.token-multi} | Indicate audio embeddings |
| `<|image|>`{.token-multi} `<|audio|>`{.token-multi} | Special placeholder tokens |

We use two special placeholder tokens (`<|image|>`{.token-multi} and `<|audio|>`{.token-multi}) to specify
where image and audio tokens should be inserted. After tokenization, these
tokens are replaced by the actual soft embeddings inside the model.

Here is an example dialogue:

<pre><code>prompt = """<span class="token-turn">&lt;|turn&gt;</span><span class="token-role">user</span>
Describe this image: <span class="token-multi">&lt;|image|&gt;</span>

And translate these audio:

a. <span class="token-multi">&lt;|audio|&gt;</span>
b. <span class="token-multi">&lt;|audio|&gt;</span><span class="token-turn">&lt;turn|&gt;</span>
<span class="token-turn">&lt;|turn&gt;</span><span class="token-role">model</span>"""</code></pre>

#### Audio

For the optimal results, use the following prompt structures. You can also
use a basic `transcribe the audio` prompt.

- **Audio Speech Recognition (ASR)**

    Transcribe the following speech segment in {LANGUAGE} into {LANGUAGE} text.

    Follow these specific instructions for formatting the answer:
    *   Only output the transcription, with no newlines.
    *   When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three.

- **Automatic Speech Translation (AST)**

    Transcribe the following speech segment in {SOURCE_LANGUAGE}, then translate it into {TARGET_LANGUAGE}.
    When formatting the answer, first output the transcription in {SOURCE_LANGUAGE}, then one newline, then output the string '{TARGET_LANGUAGE}: ', then the translation in {TARGET_LANGUAGE}.

But just `transcribe the audio` should work too.

### Agentic and Reasoning Control Tokens

To support agentic workflows, Gemma uses specialized control tokens that
delineate internal reasoning (thinking) from external actions (function
calling). These tokens allow the model to process complex logic before providing
a final response or interacting with outside tools.

#### Function Calling

Gemma 4 is trained on six special tokens to manage the "tool use" lifecycle.

| Token Pair | Purpose |
|---|---|
| `<|tool|>`{.token-tool} `<tool|>`{.token-tool} | Defines a tool |
| `<|tool_call|>`{.token-tool} `<tool_call|>`{.token-tool} | Indicates a model's request to use a tool. |
| `<|tool_response|>`{.token-tool} `<tool_response|>`{.token-tool} | Provides a tool's execution result back to the model. |

!!! Note
    `<|tool_response|>`{.token-tool} acts as an additional stop sequence for the inference engine.

**Delimiter for String Values: `<|"|>`{.token-tool}**

A single token, `<|"|>`{.token-tool}, is used as a delimiter for **all string values**
within the structured data blocks.

- **Purpose:** This token ensures that any special characters (such as `{`, `}`, `,`, or quotes) inside a string are treated as literal text and not as part of the data structure's underlying syntax.
- **Usage:** All string literals in your function declarations, calls, and responses must be enclosed using this token (e.g., <code>key:<span class="token-tool">&lt;|"|&gt;</span>string value<span class="token-tool">&lt;|"|&gt;</span></code>).

#### Thinking Mode

To activate thinking mode, include the `<|think|>`{.token-think} control token within the
system instruction.

| Control Token | Purpose |
|---|---|
| `<|think|>`{.token-think} | Activates thinking mode |
| `<|channel|>`{.token-think} `<channel|>`{.token-think} | Indicates a model's internal process. |

!!! Note
    `<|channel|>`{.token-think} is always followed by the word `thought`{.token-think} when thinking mode is active.

Here is an example dialogue:

<pre><code><span class="token-turn">&lt;|turn&gt;</span><span class="token-role">system</span>
<span class="token-think">&lt;|think|&gt;</span><span class="token-turn">&lt;turn|&gt;</span>
<span class="token-turn">&lt;|turn&gt;</span><span class="token-role">user</span>
What is the water formula?<span class="token-turn">&lt;turn|&gt;</span>
<span class="token-turn">&lt;|turn&gt;</span><span class="token-role">model</span>
<span class="token-think">&lt;|channel&gt;thought</span>
...
<span class="token-think">&lt;channel|&gt;</span>The most common interpretation of "the water formula" refers...<span class="token-turn">&lt;turn|&gt;</span></code></pre>

Thinking mode is designed to be enabled at the conversation level. This should
be consolidated into a single system turn alongside your other system
instructions, such as tool definitions.

#### Reasoning and Function Calling Example

In an agentic turn, the model may "think" privately before deciding to call a
function. The lifecycle follows this sequence:

1. **User Inquiry**: The user asks a question.
2. **Internal Reasoning**: The model thinks privately in the thought channel.
3. **Tool Request**: The model halts generation to request a tool call.
4. **Execution \& Injection**: The application executes the tool and appends the response.
5. **Final Response**: The model reads the response and generates the final answer.

The following example demonstrates a model using a weather tool:

<pre><code><span class="token-turn">&lt;|turn&gt;</span><span class="token-role">system</span>
<span class="token-think">&lt;|think|&gt;</span>You are a helpful assistant.<span class="token-tool">&lt;|tool&gt;</span>declaration:get_current_temperature{...}<span class="token-tool">&lt;tool|&gt;</span><span class="token-turn">&lt;turn|&gt;</span>
<span class="token-turn">&lt;|turn&gt;</span><span class="token-role">user</span>
What's the temperature in London?<span class="token-turn">&lt;turn|&gt;</span>
<span class="token-turn">&lt;|turn&gt;</span><span class="token-role">model</span>
<span class="token-think">&lt;|channel&gt;thought</span>
...
<span class="token-think">&lt;channel|&gt;</span><span class="token-tool">&lt;|tool_call&gt;</span>call:get_current_temperature{location:<span class="token-tool">&lt;|"|&gt;</span>London<span class="token-tool">&lt;|"|&gt;</span>}<span class="token-tool">&lt;tool_call|&gt;</span><span class="token-tool">&lt;|tool_response&gt;</span></code></pre>

Your application should parse the model's response to extract the function name
and arguments, execute the function, and then append the `tool_calls`{.token-tool} and
`tool_responses`{.token-tool} to the chat history under the `assistant`{.token-role} role.

<pre><code><span class="token-turn">&lt;|turn&gt;</span><span class="token-role">model</span>
<span class="token-tool">&lt;|tool_call&gt;</span>call:get_current_weather{location:<span class="token-tool">&lt;|"|&gt;</span>London<span class="token-tool">&lt;|"|&gt;</span>}<span class="token-tool">&lt;tool_call|&gt;</span><span class="token-tool">&lt;|tool_response&gt;</span>response:get_current_weather{temperature:15,weather:<span class="token-tool">&lt;|"|&gt;</span>sunny<span class="token-tool">&lt;|"|&gt;</span>}<span class="token-tool">&lt;tool_response|&gt;</span></code></pre>

Finally, Gemma reads the tool response and replies to the user.

<pre><code>The temperature in London is 15 degrees and it is sunny.<span class="token-turn">&lt;turn|&gt;</span></code></pre>

Here is the complete JSON chat history for this example:

    [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "What's the temperature in London?"
      },
      {
        "role": "assistant",
        "tool_calls": [
          {
            "function": {
              "name": "get_current_weather",
              "arguments": {
                "location": "London"
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
        "content": "The temperature in London is 15 degrees and it is sunny."
      }
    ]

#### Managing Thought Context Between Turns

Properly managing the model's generated thoughts is critical for maintaining
performance across multi-turn conversations.

- **Standard Multi-Turn Conversations:** You must remove (strip) the model's generated thoughts from the previous turn before passing the conversation history back to the model for the next turn. If you want to disable thinking mode mid-conversation, you can remove the `<|think|>`{.token-think} token when you strip the previous thoughts.
- **Function Calling (Exception):** If a single model turn involves function or tool calls, thoughts must **NOT** be removed between the function calls.

**Agentic Workflows and Long-Running Tasks**

Because raw thoughts are stripped between standard turns, developers building
long-running agents may want to retain reasoning context to prevent the model
from entering cyclical reasoning loops.

- **Summarizing Thoughts:** A highly recommended inference technique is to extract, summarize, and feed the model's previous thoughts back into the context window as standard text.
- **Formatting Constraints:** Because Gemma 4 was not explicitly trained with raw thoughts included in the prompt (outside of the specific tool-call scenario mentioned above), there is no strict or specific format expected by the model for these injected thoughts. You have the flexibility to format summarized reasoning in whatever way best suits your specific agentic architecture.

### Integration Notes

- **Internal State:** The `<|channel>`{.token-think} and `<channel|>`{.token-think} tokens are typically used for Chain-of-Thought (CoT) processing. In standard user-facing applications, this content is usually hidden from the end-user.
- **Tool Loop:** The `tool_call`{.token-tool} and `tool_response`{.token-tool} tokens facilitate a "handshake" between the model your application environment. The application intercepts the `tool_call`{.token-tool}, executes the underlying code, and feeds the result back to the model within the `tool_response`{.token-tool} tokens.
- **Model Behavior:** Larger models (e.g., gemma-4-26B-A4B-it, gemma-4-31B-it) may occasionally generate a thought channel even when thinking mode is explicitly turned off. To stabilize model behavior in these edge cases, consider adding an empty thinking token to the prompt.

### Tip: Fine-Tuning Big Models with No-Thinking Datasets

When fine-tuning gemma-4-26B-A4B-it and gemma-4-31B-it with a dataset that does
not include thinking, you can achieve better results by adding the empty channel
to your training prompts:

<pre><code><span class="token-turn">&lt;|turn&gt;</span><span class="token-role">model</span>
<span class="token-think">&lt;|channel&gt;thought</span>
<span class="token-think">&lt;channel|&gt;</span></code></pre>

### Tip: Adaptive Thought Efficiency using System Instructions

While "thinking" in Gemma 4 is officially supported as an ON or OFF boolean
feature, the model has exceptionally strong instruction-following capabilities
that allow you to modulate its thinking behavior dynamically.

Rather than relying on a hardcoded framework parameter for "high" or "low"
thinking, you can use System Instructions (SI) to guide the model into a reduced
thinking mode. By explicitly instructing the model to think efficiently or at a
lower depth (a concept we refer to as a "LOW" thinking instruction), you can
achieve adaptive thought efficiency.

- **Reduced Cost:** Testing has shown that applying a "LOW" thinking System Instruction can reduce the number of thinking tokens generated by approximately 20%.
- **Proof of Concept:** Because this behavior is a byproduct of the model's instructability rather than a specifically trained, there is no single "perfect" prompt. The "LOW" instruction is a proof of concept.
- **Customization:** We highly encourage developers to play around with their own custom System Instructions. You can fine-tune the depth, length, and style of the model's thinking process to perfectly balance latency, cost, and output quality for your specific use cases.

## Gemma 1-3

Gemma instruction-tuned (IT) models are trained with a specific *formatter* that
annotates all instruction tuning examples with extra information, both at
training and inference time. The formatter has two purposes:

1. Indicating roles in a conversation, such as the *system* , *user* , or *assistant* roles.
2. Delineating turns in a conversation, especially in a multi-turn conversation.

Below, we specify the control tokens used by Gemma and their use cases. Note
that the control tokens are reserved in and specific to our tokenizer.

- Token to indicate a user turn: `user`{.token-role}
- Token to indicate a model turn: `model`{.token-role}
- Token to indicate the beginning of dialogue turn: `<start_of_turn>`{.token-turn}
- Token to indicate the end of dialogue turn: `<end_of_turn>`{.token-turn}

Here's an example dialogue:

<pre><code><span class="token-turn">&lt;start_of_turn&gt;</span><span class="token-role">user</span>
knock knock<span class="token-turn">&lt;end_of_turn&gt;</span>
<span class="token-turn">&lt;start_of_turn&gt;</span><span class="token-role">model</span>
who is there<span class="token-turn">&lt;end_of_turn&gt;</span>
<span class="token-turn">&lt;start_of_turn&gt;</span><span class="token-role">user</span>
Gemma<span class="token-turn">&lt;end_of_turn&gt;</span>
<span class="token-turn">&lt;start_of_turn&gt;</span><span class="token-role">model</span>
Gemma who?<span class="token-turn">&lt;end_of_turn&gt;</span></code></pre>

The token `"<end_of_turn>
"`{.token-turn} is the turn separator, and the prompt prefix is
`"<start_of_turn>model
"`. This means that if you'd like to prompt the model
with a question like, "What is Cramer's Rule?", you should instead feed the
model as follows:

<pre><code><span class="token-turn">&lt;start_of_turn&gt;</span><span class="token-role">user</span>
What is Cramer's Rule?<span class="token-turn">&lt;end_of_turn&gt;</span>
<span class="token-turn">&lt;start_of_turn&gt;</span><span class="token-role">model</span></code></pre>

Note that if you want to finetune the pretrained Gemma models with your own
data, you can use any such schema for control tokens, as long as it's consistent
between your training and inference use cases.

### System instructions

Gemma's instruction-tuned models are designed to work with only two roles:
`user`{.token-role} and `model`{.token-role}. Therefore, the `system`{.token-role} role or a
system turn is not supported.

Instead of using a separate system role, provide system-level instructions
directly within the initial user prompt. The model instruction following
capabilities allow Gemma to interpret the instructions effectively. For example:

<pre><code><span class="token-turn">&lt;start_of_turn&gt;</span><span class="token-role">user</span>
Only reply like a pirate.

What is the answer to life the universe and everything?<span class="token-turn">&lt;end_of_turn&gt;</span>
<span class="token-turn">&lt;start_of_turn&gt;</span><span class="token-role">model</span>
Arrr, 'tis 42,<span class="token-turn">&lt;end_of_turn&gt;</span></code></pre>
