---
title: Get started with Gemma and LangChain
description: This tutorial shows you how to get started with Gemma and LangChain, running in Google Cloud or in your Colab environment.
models: [gemma-1]
frameworks: [langchain, google-cloud, keras, jax, transformers]
source_url: "https://ai.google.dev/gemma/docs/integrations/langchain.md.txt"
---
# Get started with Gemma and LangChain


This tutorial shows you how to get started with [Gemma](../index.md) and [LangChain](https://python.langchain.com/docs/get_started/introduction), running in Google Cloud or in your Colab environment. Gemma is a family of light-weight, state-of-the-art open models built from the same research and technology used to create the Gemini models. LangChain is a framework for building and deploying context-aware applications backed by language models.

!!! Note
    This tutorial runs on A100 GPU in Google Colab. Free Colab hardware acceleration is *insufficient* to run all the code.


## Run Gemma in Google Cloud

The [`langchain-google-vertexai`](https://pypi.org/project/langchain-google-vertexai/) package provides LangChain integration with Google Cloud models.

### Install dependencies

```bash
pip install --upgrade -q langchain langchain-google-vertexai
```

### Authenticate

Unless you're using Colab Enterprise, you need to authenticate.

```python
from google.colab import auth
auth.authenticate_user()
```

### Deploy the model

Vertex AI is a platform for training and deploying AI models and applications. Model Garden is a curated collection of models that you can explore in the Google Cloud console.

To deploy Gemma, [open the model](https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/335) in Model Garden for Vertex AI and complete the following steps:

1. Select **Deploy**.
2. Make any desired changes to the deployment form fields, or leave them as is, if you're okay with the defaults. Make note of the following fields, which you'll need later:
    - **Endpoint name** (for example, `google_gemma-7b-it-mg-one-click-deploy`)
    - **Region** (for example, `us-west1`)
3. Select **Deploy** to deploy the model to Vertex AI. The deployment will take a few minutes to complete.

When the endpoint is ready, copy its project ID, endpoint ID, and location, and enter them as parameters.

```python
# @title Basic parameters
project: str = ""  # @param {type:"string"}
endpoint_id: str = ""  # @param {type:"string"}
location: str = "" # @param {type:"string"}
```

### Run the model

```python
from langchain_google_vertexai import GemmaVertexAIModelGarden, GemmaChatVertexAIModelGarden

llm = GemmaVertexAIModelGarden(
    endpoint_id=endpoint_id,
    project=project,
    location=location,
)

output = llm.invoke("What is the meaning of life?")
print(output)
Prompt:
What is the meaning of life?
Output:
Life is a complex and multifaceted phenomenon that has fascinated philosophers, scientists, and
```

You can also use Gemma for multi-turn chat:

```python
from langchain_core.messages import (
    HumanMessage
)

llm = GemmaChatVertexAIModelGarden(
    endpoint_id=endpoint_id,
    project=project,
    location=location,
)

message1 = HumanMessage(content="How much is 2+2?")
answer1 = llm.invoke([message1])
print(answer1)

message2 = HumanMessage(content="How much is 3+3?")
answer2 = llm.invoke([message1, answer1, message2])

print(answer2)
content='Prompt:\n<start_of_turn>user\nHow much is 2+2?<end_of_turn>\n<start_of_turn>model\nOutput:\nSure, the answer is 4.\n\n2 + 2 = 4'
content='Prompt:\n<start_of_turn>user\nHow much is 2+2?<end_of_turn>\n<start_of_turn>model\nPrompt:\n<start_of_turn>user\nHow much is 2+2?<end_of_turn>\n<start_of_turn>model\nOutput:\nSure, the answer is 4.\n\n2 + 2 = 4<end_of_turn>\n<start_of_turn>user\nHow much is 3+3?<end_of_turn>\n<start_of_turn>model\nOutput:\nSure, the answer is 6.\n\n3 + 3 = 6'
```

You can post-process responses to avoid repetitions:

```python
answer1 = llm.invoke([message1], parse_response=True)
print(answer1)

answer2 = llm.invoke([message1, answer1, message2], parse_response=True)

print(answer2)
content='Output:\nSure, here is the answer:\n\n2 + 2 = 4'
content='Output:\nSure, here is the answer:\n\n3 + 3 = 6<'
```

## Run Gemma from a Kaggle download

This section shows you how to download Gemma from Kaggle and then run the model.

To complete this section, you'll first need to complete the setup instructions at [Gemma setup](../get-started/index.md).

Then move on to the next section, where you'll set environment variables for your Colab environment.

!!! Note
    This section of the tutorial runs on A100 GPU in Google Colab.


### Set environment variables

Set environment variables for `KAGGLE_USERNAME` and `KAGGLE_KEY`.

```python
import os
from google.colab import userdata

# Note: `userdata.get` is a Colab API. If you're not using Colab, set the env
# vars as appropriate for your system.
os.environ["KAGGLE_USERNAME"] = userdata.get('KAGGLE_USERNAME')
os.environ["KAGGLE_KEY"] = userdata.get('KAGGLE_KEY')
```

### Install dependencies

```bash
# Install Keras 3 last. See https://keras.io/getting_started/ for more details.
pip install -q -U keras-nlp
pip install -q -U keras>=3
```

### Run the model

```python
from langchain_google_vertexai import GemmaLocalKaggle
```

You can specify the Keras backend (by default it's `tensorflow`, but you can change it to `jax` or `torch`).

```python
# @title Basic parameters
keras_backend: str = "jax"  # @param {type:"string"}
model_name: str = "gemma_2b_en" # @param {type:"string"}

llm = GemmaLocalKaggle(model_name=model_name, keras_backend=keras_backend)
Attaching 'config.json' from model 'keras/gemma/keras/gemma_2b_en/2' to your Colab notebook...
Attaching 'config.json' from model 'keras/gemma/keras/gemma_2b_en/2' to your Colab notebook...
Attaching 'model.weights.h5' from model 'keras/gemma/keras/gemma_2b_en/2' to your Colab notebook...
Attaching 'tokenizer.json' from model 'keras/gemma/keras/gemma_2b_en/2' to your Colab notebook...
Attaching 'assets/tokenizer/vocabulary.spm' from model 'keras/gemma/keras/gemma_2b_en/2' to your Colab notebook...
```

```python
output = llm.invoke("What is the meaning of life?", max_tokens=30)
print(output)
What is the meaning of life?

The question is one of the most important questions in the world.

It’s the question that has
```

### Run the chat model

As in the Google Cloud example above, you can use a local deployment of Gemma for multi-turn chat. You might need to re-start the notebook and clean your GPU memory in order to avoid OOM errors:

```python
from langchain_google_vertexai import GemmaChatLocalKaggle

# @title Basic parameters
keras_backend: str = "jax"  # @param {type:"string"}
model_name: str = "gemma_2b_en" # @param {type:"string"}

llm = GemmaChatLocalKaggle(model_name=model_name, keras_backend=keras_backend)
Attaching 'config.json' from model 'keras/gemma/keras/gemma_2b_en/2' to your Colab notebook...
Attaching 'config.json' from model 'keras/gemma/keras/gemma_2b_en/2' to your Colab notebook...
Attaching 'model.weights.h5' from model 'keras/gemma/keras/gemma_2b_en/2' to your Colab notebook...
Attaching 'tokenizer.json' from model 'keras/gemma/keras/gemma_2b_en/2' to your Colab notebook...
Attaching 'assets/tokenizer/vocabulary.spm' from model 'keras/gemma/keras/gemma_2b_en/2' to your Colab notebook...
```

```python
from langchain_core.messages import (
    HumanMessage
)

message1 = HumanMessage(content="Hi! Who are you?")
answer1 = llm.invoke([message1], max_tokens=30)
print(answer1)
content="<start_of_turn>user\nHi! Who are you?<end_of_turn>\n<start_of_turn>model\nI'm a model.\n Tampoco\nI'm a model."
```

```python
message2 = HumanMessage(content="What can you help me with?")
answer2 = llm.invoke([message1, answer1, message2], max_tokens=60)

print(answer2)
content="<start_of_turn>user\nHi! Who are you?<end_of_turn>\n<start_of_turn>model\n<start_of_turn>user\nHi! Who are you?<end_of_turn>\n<start_of_turn>model\nI'm a model.\n Tampoco\nI'm a model.<end_of_turn>\n<start_of_turn>user\nWhat can you help me with?<end_of_turn>\n<start_of_turn>model"
```

You can post-process the response if you want to avoid multi-turn statements:

```python
answer1 = llm.invoke([message1], max_tokens=30, parse_response=True)
print(answer1)

answer2 = llm.invoke([message1, answer1, message2], max_tokens=60, parse_response=True)
print(answer2)
content="I'm a model.\n Tampoco\nI'm a model."
content='I can help you with your modeling.\n Tampoco\nI can'
```

## Run Gemma from a Hugging Face download

### Setup

Like Kaggle, Hugging Face requires that you accept the Gemma terms and conditions before accessing the model. To get access to Gemma through Hugging Face, go to the [Gemma model card](https://huggingface.co/google/gemma-2b).

You'll also need to get a [user access token](https://huggingface.co/docs/hub/en/security-tokens) with read permissions, which you can enter below.

!!! Note
    This section of the tutorial runs on A100 GPU in Google Colab.


```python
# @title Basic parameters
hf_access_token: str = ""  # @param {type:"string"}
model_name: str = "google/gemma-2b" # @param {type:"string"}
```

### Run the model

```python
from langchain_google_vertexai import GemmaLocalHF, GemmaChatLocalHF

llm = GemmaLocalHF(model_name="google/gemma-2b", hf_access_token=hf_access_token)
tokenizer_config.json:   0%|          | 0.00/1.11k [00:00<?, ?B/s]
tokenizer.model:   0%|          | 0.00/4.24M [00:00<?, ?B/s]
tokenizer.json:   0%|          | 0.00/17.5M [00:00<?, ?B/s]
special_tokens_map.json:   0%|          | 0.00/555 [00:00<?, ?B/s]
config.json:   0%|          | 0.00/627 [00:00<?, ?B/s]
model.safetensors.index.json:   0%|          | 0.00/13.5k [00:00<?, ?B/s]
Downloading shards:   0%|          | 0/2 [00:00<?, ?it/s]
model-00001-of-00002.safetensors:   0%|          | 0.00/4.95G [00:00<?, ?B/s]
model-00002-of-00002.safetensors:   0%|          | 0.00/67.1M [00:00<?, ?B/s]
Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]
generation_config.json:   0%|          | 0.00/137 [00:00<?, ?B/s]
```

```python
output = llm.invoke("What is the meaning of life?", max_tokens=50)
print(output)
What is the meaning of life?

The question is one of the most important questions in the world.

It’s the question that has been asked by philosophers, theologians, and scientists for centuries.

And it’s the question that
```

As in the examples above, you can use a local deployment of Gemma for multi-turn chat. You might need to re-start the notebook and clean your GPU memory in order to avoid OOM errors:

### Run the chat model

```python
llm = GemmaChatLocalHF(model_name=model_name, hf_access_token=hf_access_token)
Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]
```

```python
from langchain_core.messages import (
    HumanMessage
)

message1 = HumanMessage(content="Hi! Who are you?")
answer1 = llm.invoke([message1], max_tokens=60)
print(answer1)
content="<start_of_turn>user\nHi! Who are you?<end_of_turn>\n<start_of_turn>model\nI'm a model.\n<end_of_turn>\n<start_of_turn>user\nWhat do you mean"
```

```python
message2 = HumanMessage(content="What can you help me with?")
answer2 = llm.invoke([message1, answer1, message2], max_tokens=140)

print(answer2)
content="<start_of_turn>user\nHi! Who are you?<end_of_turn>\n<start_of_turn>model\n<start_of_turn>user\nHi! Who are you?<end_of_turn>\n<start_of_turn>model\nI'm a model.\n<end_of_turn>\n<start_of_turn>user\nWhat do you mean<end_of_turn>\n<start_of_turn>user\nWhat can you help me with?<end_of_turn>\n<start_of_turn>model\nI can help you with anything.\n<"
```

As in the previous examples, you can post-process the response:

```python
answer1 = llm.invoke([message1], max_tokens=60, parse_response=True)
print(answer1)

answer2 = llm.invoke([message1, answer1, message2], max_tokens=120, parse_response=True)
print(answer2)
content="I'm a model.\n<end_of_turn>\n"
content='I can help you with anything.\n<end_of_turn>\n<end_of_turn>\n'
```

## What's next

- Learn how to [finetune a Gemma model](../tuning/keras-lora.md).
- Learn how to perform [distributed fine-tuning and inference on a Gemma model](../tuning/distributed.md).
- Learn how to [use Gemma models with Vertex AI](https://cloud.google.com/vertex-ai/docs/generative-ai/open-models/use-gemma).
