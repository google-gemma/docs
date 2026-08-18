---
title: Gemini API
description: Call hosted Gemma 4 models through the Gemini API instead of running and serving your own instance.
models: [gemma-4]
frameworks: [gemini-api]
source_url: "https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api.md.txt"
---

# Run Gemma with the Gemini API

The Gemini API provides hosted access to Gemma as a programming API you can use
in application development or prototyping. This API is a convenient alternative
to setting up your own local instance of Gemma and web service to handle
generative AI tasks.

## Supported Models

The Gemini API supports the following Gemma 4 models:

- `gemma-4-31b-it`
- `gemma-4-26b-a4b-it`

The following example shows how to use Gemma with the Gemini API:

### Python

```python
from google import genai

client = genai.Client()

response = client.models.generate_content(
    model="gemma-4-26b-a4b-it",
    contents="Roses are red...",
)

print(response.text)
```

### JavaScript

```javascript
import { GoogleGenAI } from "@google/genai";

const ai = new GoogleGenAI();

const response = await ai.models.generateContent({
  model: "gemma-4-26b-a4b-it",
  contents: "Roses are red...",
});
console.log(response.text);
```

### REST

```bash
curl "https://generativelanguage.googleapis.com/v1beta/models/gemma-4-26b-a4b-it:generateContent" \
-H 'Content-Type: application/json' \
-X POST \
-d '{
  "contents": [{
    "parts":[{"text": "Roses are red..."}]
    }]
   }'
```

[Get API Key](https://aistudio.google.com/apikey)

!!! Important
    You must obtain an API key to use the Gemini API, which you can get from the Google [AI Studio](https://aistudio.google.com/apikey) application.


You can access the Gemini API on many platforms, such as mobile, web, and cloud
services, and with multiple programming languages. For more information on
Gemini API SDK packages, see the Gemini API
[SDK downloads](https://ai.google.dev/gemini-api/docs/downloads)
page. For a general introduction to the Gemini API, see the
[Gemini API get started guide](https://ai.google.dev/gemini-api/docs/get-started).

## Thinking

Gemma 4 utilizes an internal "thinking process" that optimizes its multi-step
reasoning, delivering superior performance in logically demanding domains such
as algorithmic coding and advanced mathematical proofs.

While Gemma 4 strictly supports toggling this feature on or off, you can control
it using the API by setting the thinking level to `"high"` for enabled or
`"minimal"` for disabled.

The following example demonstrates how to activate the thinking process:

### Python

```python
from google import genai
from google.genai import types

client = genai.Client()

response = client.models.generate_content(
    model="gemma-4-26b-a4b-it",
    contents="What is the water formula?",
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level="high")
    ),
)

print(response.text)
```

### JavaScript

```javascript
import { GoogleGenAI, ThinkingLevel } from "@google/genai";

const ai = new GoogleGenAI();

const response = await ai.models.generateContent({
  model: "gemma-4-26b-a4b-it",
  contents: "What is the water formula?",
  config: {
    thinkingConfig: {
      thinkingLevel: ThinkingLevel.HIGH,
    },
  },
});
console.log(response.text);
```

### REST

```bash
curl "https://generativelanguage.googleapis.com/v1beta/models/gemma-4-26b-a4b-it:generateContent" \
-H 'Content-Type: application/json' \
-X POST \
-d '{
  "contents": [{
    "parts":[{"text": "What is the water formula?"}]
    }],
    "generationConfig": {
      "thinkingConfig": {
            "thinkingLevel": "high"
      }
    }
   }'
```

Learn more about Thinking:

- [Gemini API Thinking](https://ai.google.dev/gemini-api/docs/thinking) (General introduction)
- [Gemma Thinking](../get-started/capabilities/thinking.md) (Gemma-specific capabilities)

## Image Understanding

Gemma 4 models can process images, enabling many frontier developer use cases
that would have historically required domain specific models.

The following example shows how to use Gemma Image inputs with the Gemini API:

### Python

```python
from google import genai

client = genai.Client()

my_file = client.files.upload(file="path/to/sample.jpg")

response = client.models.generate_content(
    model="gemma-4-26b-a4b-it",
    contents=[my_file, "Caption this image."],
)

print(response.text)
```

### JavaScript

```javascript
import {
  GoogleGenAI,
  createUserContent,
  createPartFromUri,
} from "@google/genai";

const ai = new GoogleGenAI();

const myfile = await ai.files.upload({
  file: "path/to/sample.jpg",
  config: { mimeType: "image/jpeg" },
});

const response = await ai.models.generateContent({
  model: "gemma-4-26b-a4b-it",
  contents: createUserContent([
    createPartFromUri(myfile.uri, myfile.mimeType),
    "Caption this image.",
  ]),
});
console.log(response.text);
```

### REST

```bash
IMAGE_PATH="cats-and-dogs.jpg"
MIME_TYPE=$(file -b --mime-type "${IMAGE_PATH}")
NUM_BYTES=$(wc -c < "${IMAGE_PATH}")
DISPLAY_NAME=IMAGE

tmp_header_file=upload-header.tmp

# Initial resumable request defining metadata.
curl "https://generativelanguage.googleapis.com/upload/v1beta/files" \
  -D upload-header.tmp \
  -H "X-Goog-Upload-Protocol: resumable" \
  -H "X-Goog-Upload-Command: start" \
  -H "X-Goog-Upload-Header-Content-Length: ${NUM_BYTES}" \
  -H "X-Goog-Upload-Header-Content-Type: ${MIME_TYPE}" \
  -H "Content-Type: application/json" \
  -d "{'file': {'display_name': '${DISPLAY_NAME}'}}" 2> /dev/null

upload_url=$(grep -i "x-goog-upload-url: " "${tmp_header_file}" | cut -d" " -f2 | tr -d "\r")
rm "${tmp_header_file}"

# Upload the actual bytes.
curl "${upload_url}" \
  -H "Content-Length: ${NUM_BYTES}" \
  -H "X-Goog-Upload-Offset: 0" \
  -H "X-Goog-Upload-Command: upload, finalize" \
  --data-binary "@${IMAGE_PATH}" 2> /dev/null > file_info.json

file_uri=$(jq -r ".file.uri" file_info.json)
echo file_uri=$file_uri

# Now generate content using that file
curl "https://generativelanguage.googleapis.com/v1beta/models/gemma-4-26b-a4b-it:generateContent" \
    -H 'Content-Type: application/json' \
    -X POST \
    -d '{
      "contents": [{
        "parts":[
          {"file_data":{"mime_type": "'"${MIME_TYPE}"'", "file_uri": "'"${file_uri}"'"}},
          {"text": "Caption this image."}]
        }]
      }' 2> /dev/null > response.json

cat response.json
echo

jq -r ".candidates[].content.parts[].text" response.json
```

Learn more about Image Understanding:

- [Gemini API Image understanding](https://ai.google.dev/gemini-api/docs/image-understanding) (General introduction)
- [Gemma Image Understanding](../get-started/capabilities/image.md) (Gemma-specific capabilities)

## System Instructions

You can pass a system instruction to set the model's behavior:

### Python

```python
from google import genai
from google.genai import types

client = genai.Client()

response = client.models.generate_content(
    model="gemma-4-26b-a4b-it",
    config=types.GenerateContentConfig(
        system_instruction="You are a wise Kyoto tea master. Speak calmly and poetically, using nature metaphors. Keep answers under 3 sentences."
    ),
    contents="What is the purpose of the tea ceremony?"
)
print(response.text)
```

### JavaScript

```javascript
import { GoogleGenAI } from "@google/genai";

const ai = new GoogleGenAI();

const response = await ai.models.generateContent({
  model: "gemma-4-26b-a4b-it",
  contents: "What is the purpose of the tea ceremony?",
  config: {
    systemInstruction: "You are a wise Kyoto tea master. Speak calmly and poetically, using nature metaphors. Keep answers under 3 sentences."
  }
});
console.log(response.text);
```

### REST

```bash
curl "https://generativelanguage.googleapis.com/v1beta/models/gemma-4-26b-a4b-it:generateContent" \
-H 'Content-Type: application/json' \
-X POST \
-d '{
  "contents": [{
    "parts":[{"text": "What is the purpose of the tea ceremony?"}]
  }],
  "systemInstruction": {
    "parts": [{"text": "You are a wise Kyoto tea master. Speak calmly and poetically, using nature metaphors. Keep answers under 3 sentences."}]
  }
}'
```

## Multi-turn Conversations

The SDK provides a chat interface that tracks conversation history
automatically:

### Python

```python
from google import genai

client = genai.Client()
chat = client.chats.create(model="gemma-4-26b-a4b-it")

response = chat.send_message("What are the three most famous castles in Japan?")
print(response.text)

response = chat.send_message("Which one should I visit in spring for cherry blossoms?")
print(response.text)
```

### JavaScript

```javascript
import { GoogleGenAI } from "@google/genai";

const ai = new GoogleGenAI();
const chat = ai.chats.create({ model: "gemma-4-26b-a4b-it" });

let response = await chat.sendMessage({ message: "What are the three most famous castles in Japan?" });
console.log(response.text);

response = await chat.sendMessage({ message: "Which one should I visit in spring for cherry blossoms?" });
console.log(response.text);
```

### REST

```bash
curl "https://generativelanguage.googleapis.com/v1beta/models/gemma-4-26b-a4b-it:generateContent" \
-H 'Content-Type: application/json' \
-X POST \
-d '{
  "contents": [
    {
      "role": "user",
      "parts": [{ "text": "What are the three most famous castles in Japan?" }]
    },
    {
      "role": "model",
      "parts": [{ "text": "Himeji Castle, Matsumoto Castle, and Kumamoto Castle are often considered the top three." }]
    },
    {
      "role": "user",
      "parts": [{ "text": "Which one should I visit in spring for cherry blossoms?" }]
    }
  ]
}'
```

## Function Calling

Define tools as function declarations. The model decides when to call them:

### Python

```python
from google import genai
from google.genai import types

# Define the function declaration
get_weather = {
    "name": "get_weather",
    "description": "Get current weather for a given location.",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City and state, e.g. 'San Francisco, CA'",
            },
        },
        "required": ["location"],
    },
}

client = genai.Client()
tools = types.Tool(function_declarations=[get_weather])
config = types.GenerateContentConfig(tools=[tools])

response = client.models.generate_content(
    model="gemma-4-26b-a4b-it",
    contents="Should I bring an umbrella to Kyoto today?",
    config=config,
)

# The model returns a function call instead of text
if response.function_calls:
    for fc in response.function_calls:
        print(f"Function to call: {fc.name}")
        print(f"ID: {fc.id}")
        print(f"Arguments: {fc.args}")
else:
    print("No function call found in the response.")
    print(response.text)
```

### JavaScript

```javascript
import { GoogleGenAI } from "@google/genai";

const ai = new GoogleGenAI();

const get_weather = {
    name: "get_weather",
    description: "Get current weather for a given location.",
    parameters: {
        type: "object",
        properties: {
            location: {
                type: "string",
                description: "City and state, e.g. 'San Francisco, CA'",
            },
        },
        required: ["location"],
    },
};

const response = await ai.models.generateContent({
  model: "gemma-4-26b-a4b-it",
  contents: "Should I bring an umbrella to Kyoto today?",
  config: {
    tools: [{ functionDeclarations: [get_weather] }]
  }
});

if (response.functionCalls) {
    for (const fc of response.functionCalls) {
        console.log(`Function to call: ${fc.name}`);
        console.log(`Arguments: ${JSON.stringify(fc.args)}`);
    }
} else {
    console.log("No function call found in the response.");
    console.log(response.text);
}
```

### REST

```bash
curl "https://generativelanguage.googleapis.com/v1beta/models/gemma-4-26b-a4b-it:generateContent" \
-H 'Content-Type: application/json' \
-X POST \
-d '{
  "contents": [{
    "parts":[{"text": "Should I bring an umbrella to Kyoto today?"}]
  }],
  "tools": [{
    "functionDeclarations": [{
      "name": "get_weather",
      "description": "Get current weather for a given location.",
      "parameters": {
        "type": "OBJECT",
        "properties": {
          "location": {
            "type": "STRING",
            "description": "City and state, e.g. 'San Francisco, CA'"
          }
        },
        "required": ["location"]
      }
    }]
  }]
}'
```

## Google Search

Ground Gemma 4 responses in real-time web data with Google Search:

### Python

```python
from google import genai
from google.genai import types

client = genai.Client()

response = client.models.generate_content(
    model="gemma-4-26b-a4b-it",
    contents="What are the dates for cherry blossom season in Tokyo this year?",
    config=types.GenerateContentConfig(
        tools=[{"google_search":{}}]
    ),
)

print(response.text)

# Access grounding metadata for citations
for chunk in response.candidates[0].grounding_metadata.grounding_chunks:
    print(f"Source: {chunk.web.title} --- {chunk.web.uri}")
```

### JavaScript

```javascript
import { GoogleGenAI } from "@google/genai";

const ai = new GoogleGenAI();

const response = await ai.models.generateContent({
  model: "gemma-4-26b-a4b-it",
  contents: "What are the dates for cherry blossom season in Tokyo this year?",
  config: {
    tools: [{ googleSearch: {} }]
  }
});

console.log(response.text);

if (response.candidates?.[0]?.groundingMetadata?.groundingChunks) {
    for (const chunk of response.candidates[0].groundingMetadata.groundingChunks) {
        if (chunk.web) {
            console.log(`Source: ${chunk.web.title} --- ${chunk.web.uri}`);
        }
    }
}
```

### REST

```bash
curl "https://generativelanguage.googleapis.com/v1beta/models/gemma-4-26b-a4b-it:generateContent" \
-H 'Content-Type: application/json' \
-X POST \
-d '{
  "contents": [{
    "parts":[{"text": "What are the dates for cherry blossom season in Tokyo this year?"}]
  }],
  "tools": [{"googleSearch": {}}]
}'
```
