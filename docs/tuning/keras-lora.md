---
title: Fine-tune Gemma in Keras using LoRA
description: Generative artificial intelligent (AI) models like Gemma are effective at a variety of tasks.
models: [gemma-3]
frameworks: [keras]
source_url: "https://ai.google.dev/gemma/docs/core/lora_tuning.md.txt"
---
# Fine-tune Gemma in Keras using LoRA


Generative artificial intelligent (AI) models like Gemma are effective at a variety of tasks. You can further fine-tune Gemma models with domain-specific data to perform tasks such as sentiment analysis. However, full fine-tuning of generative models by updating billions of parameters is resource intensive, requiring specialized hardware, such as GPUs, processing time, and memory to load the model parameters.

[Low Rank Adaptation](https://arxiv.org/abs/2106.09685) (LoRA) is a fine-tuning technique which greatly reduces the number of trainable parameters for downstream tasks by freezing the weights of the model and inserting a smaller number of new weights into the model. This technique makes training with LoRA much faster and more memory-efficient, and produces smaller model weights (a few hundred MBs), all while maintaining the quality of the model outputs. This tutorial walks you through using Keras to perform LoRA fine-tuning on a Gemma model.

## Setup

To complete this tutorial, you will first need to complete the setup instructions at [Gemma setup](../get-started/index.md). The Gemma setup instructions show you how to do the following:

- Get access to Gemma on [kaggle.com](https://kaggle.com).
- Select a Colab runtime with sufficient resources to tune the Gemma model you want to run. [Learn more](../models/gemma-4/index.md#parameter-sizes-and-quantization).
- Generate and configure a Kaggle username and API key.

After you've completed the Gemma setup, move on to the next section, where you'll set environment variables for your Colab environment.

### Select a Colab runtime

To complete this tutorial, you'll need to have a Colab runtime with sufficient resources to run the Gemma model. In this case, you can use a T4 GPU:

1. In the upper-right of the Colab window, select ▾ (**Additional connection options**).
2. Select **Change runtime type**.
3. Under **Hardware accelerator** , select **T4 GPU**.

### Configure your API key

To use Gemma, you must provide your Kaggle username and a Kaggle API key.

To generate a Kaggle API key, go to the **Account** tab of your Kaggle user profile and select **Create New Token** . This triggers the download of a `kaggle.json` file containing your API credentials.

In Colab, select **Secrets** (🔑) in the left pane and add your Kaggle username and Kaggle API key. Store your username under the name `KAGGLE_USERNAME` and your API key under the name `KAGGLE_KEY`.

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

### Install Keras packages

Install the Keras and KerasHub Python packages.

```bash
pip install -q -U keras-hub
pip install  -q -U keras
```

### Select a backend

Keras is a high-level, multi-framework deep learning API designed for simplicity and ease of use. Using Keras 3, you can run workflows on one of three backends: TensorFlow, JAX, or PyTorch. For this tutorial, configure the backend for JAX as it typically provides the better performance.

```python
os.environ["KERAS_BACKEND"] = "jax"  # Or "torch" or "tensorflow".
# Avoid memory fragmentation on JAX backend.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="1.00"
```

### Import packages

Import the Python packages needed for this tutorial, including Keras and KerasHub.

```python
import keras
import keras_hub
```

## Load model

Keras provides implementations of Gemma and many other popular [model architectures](https://keras.io/keras_hub/api/models/). Use the `Gemma3CausalLM.from_preset()` method to configure an end-to-end Gemma model for causal language modeling. A causal language model predicts the next token based on previous tokens.

```python
gemma_lm = keras_hub.models.Gemma3CausalLM.from_preset("gemma3_instruct_1b")
gemma_lm.summary()
```

The `Gemma3CausalLM.from_preset()` method instantiates the model from a preset architecture and weights. In the code above, the string `"gemma#_xxxxxxx"` specifies a preset version and parameter size for Gemma. You can find the code strings for Gemma models in their **Model Variation** listings on [Kaggle](https://www.kaggle.com/models/keras/gemma3){ data-model="gemma-3" }.

## Inference before fine tuning

Once you have downloaded and configured a Gemma model, you can query it with various prompts to see how it responds.

### Europe trip prompt

Query the model for suggestions on what to do on a trip to Europe.

```python
template = "Instruction:\n{instruction}\n\nResponse:\n{response}"

prompt = template.format(
    instruction="What should I do on a trip to Europe?",
    response="",
)
sampler = keras_hub.samplers.TopKSampler(k=5, seed=2)
gemma_lm.compile(sampler=sampler)
print(gemma_lm.generate(prompt, max_length=256))
Instruction:
What should I do on a trip to Europe?

Response:
The first thing to know is that you will have a great time!

Europe is a great place for a vacation. The countries of Europe are all very different and offer a wide range of activities and attractions. The countries of Europe are also very close to each other, which means you can visit many different places within a short time.

The best way to plan a trip to Europe is to look up the countries you want to visit and see what activities are offered in each country. You can also look for tours and tours that offer a good value for money.

You can also look for hotels and flights that offer good deals. If you are looking for a good value for money, you should look for hotels and flights that offer good deals. This means you will have a great time on your trip!

The next step is to book your tickets to the countries you want to visit. If you are planning to visit many countries, it's a good idea to book your tickets early. This means you’ll be able to get the best deal and avoid the long queues.

The next step is to plan your itinerary. You can use a travel guide to plan your itinerary
```

The model responds with generic tips on how to plan a trip.

### Photosynthesis prompt

Prompt the model to explain photosynthesis in terms simple enough for a 5 year old child to understand.

```python
prompt = template.format(
    instruction="Explain the process of photosynthesis in a way that a child could understand.",
    response="",
)
print(gemma_lm.generate(prompt, max_length=256))
Instruction:
Explain the process of photosynthesis in a way that a child could understand.

Response:
Photosynthesis is a biological process that occurs in plants, algae, and some other organisms. In the process, light energy is captured and converted into the energy stored in the bonds of organic molecules. The process is crucial for life on Earth because it enables plants to use carbon dioxide and water to produce glucose and oxygen, which are essential for all living things.
The process involves several stages:

1. Light Reactions: Light energy is absorbed by pigments in the chloroplasts of the plant, converting it into chemical energy in the form of ATP and reducing power.
2. Carbon Fixation: During this stage, carbon dioxide is combined with hydrogen to form organic molecules such as starch or glucose, which are used as a source of energy.
3. Calvin Cycle: The process of carbon fixation occurs in the stroma of the chloroplasts. It involves the capture and reduction of carbon dioxide, producing glucose and reducing power in the form of ATP and NADPH molecules.
4. Stroma: The stroma is the fluid-filled space where the light reactions occur in the chloroplasts.
5. Chloroplasts: The chloroplasts contain the green pigments that absorb
```

The model response contains words that might not be easy to understand for a child such as chlorophyll.

## LoRA fine-tuning

This section shows you how to do fine-tuning using the Low Rank Adaptation (LoRA) tuning technique. This approach allows you to change the behavior of Gemma models using fewer compute resources.

### Load dataset

Prepare a dataset for tuning by downloading an existing data set and formatting if for use with the the Keras `fit()` fine-tuning method. This tutorial uses the [Databricks Dolly 15k dataset](https://huggingface.co/datasets/databricks/databricks-dolly-15k) for fine-tuning. The dataset contains 15,000 high-quality human-generated prompt and response pairs specifically designed for tuning generative models.

```bash
wget -O databricks-dolly-15k.jsonl https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl
--2025-04-10 20:48:49--  https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl
Resolving huggingface.co (huggingface.co)... 3.163.189.37, 3.163.189.114, 3.163.189.74, ...
Connecting to huggingface.co (huggingface.co)|3.163.189.37|:443... connected.
HTTP request sent, awaiting response... 302 Found
Location: https://cdn-lfs.hf.co/repos/34/ac/34ac588cc580830664f592597bb6d19d61639eca33dc2d6bb0b6d833f7bfd552/2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec?response-content-disposition=inline%3B+filename*%3DUTF-8%27%27databricks-dolly-15k.jsonl%3B+filename%3D%22databricks-dolly-15k.jsonl%22%3B&Expires=1744321729&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc0NDMyMTcyOX19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy5oZi5jby9yZXBvcy8zNC9hYy8zNGFjNTg4Y2M1ODA4MzA2NjRmNTkyNTk3YmI2ZDE5ZDYxNjM5ZWNhMzNkYzJkNmJiMGI2ZDgzM2Y3YmZkNTUyLzJkZjkwODMzMzhiNGFiZDZiY2ViNTYzNTc2NGRhYjVkODMzYjM5M2I1NTc1OWRmZmIwOTU5YjZmY2JmNzk0ZWM%7EcmVzcG9uc2UtY29udGVudC1kaXNwb3NpdGlvbj0qIn1dfQ__&Signature=vh0VIGB-UkK57FSfRikYCREpKuHt%7EnDKPcHHgC1V9rDXLABIRF81nK7olQhAq6zSbAqEtMNnvHgd8IBK1j54mdIYdVLiBwImqez3xu2CPhzYBtKWInnXj9lTXW0p-9GEHcbU%7Eoot22qFSdwyZf1UIdmHZLTHPWjtLhfRkKbg-ptA3CFeegtmvCtY-WG2GffJ%7Em2q2bbs-U1m0yI7cSTW18nD8VSBihxGOMnS1IhkO-LgE4I6GJISXROTk-61%7EJiEIKcagcijL4QGi8j1g9xeQamBXX4hWBdkbJgX5PtX15Ftd0HCM4zCzcJAUrE3ZEJRLe2XRUwfKU3ai7-%7ErPpnSA__&Key-Pair-Id=K3RPWS32NSSJCE [following]
--2025-04-10 20:48:49--  https://cdn-lfs.hf.co/repos/34/ac/34ac588cc580830664f592597bb6d19d61639eca33dc2d6bb0b6d833f7bfd552/2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec?response-content-disposition=inline%3B+filename*%3DUTF-8%27%27databricks-dolly-15k.jsonl%3B+filename%3D%22databricks-dolly-15k.jsonl%22%3B&Expires=1744321729&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc0NDMyMTcyOX19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy5oZi5jby9yZXBvcy8zNC9hYy8zNGFjNTg4Y2M1ODA4MzA2NjRmNTkyNTk3YmI2ZDE5ZDYxNjM5ZWNhMzNkYzJkNmJiMGI2ZDgzM2Y3YmZkNTUyLzJkZjkwODMzMzhiNGFiZDZiY2ViNTYzNTc2NGRhYjVkODMzYjM5M2I1NTc1OWRmZmIwOTU5YjZmY2JmNzk0ZWM%7EcmVzcG9uc2UtY29udGVudC1kaXNwb3NpdGlvbj0qIn1dfQ__&Signature=vh0VIGB-UkK57FSfRikYCREpKuHt%7EnDKPcHHgC1V9rDXLABIRF81nK7olQhAq6zSbAqEtMNnvHgd8IBK1j54mdIYdVLiBwImqez3xu2CPhzYBtKWInnXj9lTXW0p-9GEHcbU%7Eoot22qFSdwyZf1UIdmHZLTHPWjtLhfRkKbg-ptA3CFeegtmvCtY-WG2GffJ%7Em2q2bbs-U1m0yI7cSTW18nD8VSBihxGOMnS1IhkO-LgE4I6GJISXROTk-61%7EJiEIKcagcijL4QGi8j1g9xeQamBXX4hWBdkbJgX5PtX15Ftd0HCM4zCzcJAUrE3ZEJRLe2XRUwfKU3ai7-%7ErPpnSA__&Key-Pair-Id=K3RPWS32NSSJCE
Resolving cdn-lfs.hf.co (cdn-lfs.hf.co)... 18.238.217.63, 18.238.217.81, 18.238.217.120, ...
Connecting to cdn-lfs.hf.co (cdn-lfs.hf.co)|18.238.217.63|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 13085339 (12M) [text/plain]
Saving to: ‘databricks-dolly-15k.jsonl’

databricks-dolly-15 100%[===================>]  12.48M  --.-KB/s    in 0.08s   

2025-04-10 20:48:49 (156 MB/s) - ‘databricks-dolly-15k.jsonl’ saved [13085339/13085339]
```

### Format tuning data

Format the downloaded data for use with the Keras `fit()` method. The following code extracts a subset of the training examples to execute the notebook faster. Consider using more training data for higher quality fine-tuning.

```python
import json

prompts = []
responses = []
line_count = 0

with open("databricks-dolly-15k.jsonl") as file:
    for line in file:
        if line_count >= 1000:
            break  # Limit the training examples, to reduce execution time.

        examples = json.loads(line)
        # Filter out examples with context, to keep it simple.
        if examples["context"]:
            continue
        # Format data into prompts and response lists.
        prompts.append(examples["instruction"])
        responses.append(examples["response"])

        line_count += 1

data = {
    "prompts": prompts,
    "responses": responses
}
```

### Configure LoRA tuning

Activate LoRA tuning using the Keras `model.backbone.enable_lora()` method, including a LoRA rank value. The *LoRA rank* determines the dimensionality of the trainable matrices that are added to the original weights of the LLM. It controls the expressiveness and precision of the fine-tuning adjustments. A higher rank means more detailed changes are possible, but also means more trainable parameters. A lower rank means less computational overhead, but potentially less precise adaptation.

This example uses a LoRA rank of 4. In practice, begin with a relatively small rank (such as 4, 8, 16). This setting is computationally efficient for experimentation. Train your model with this rank and evaluate the performance improvement on your task. Gradually increase the rank in subsequent trials and see if that further boosts performance.

```python
# Enable LoRA for the model and set the LoRA rank to 4.
gemma_lm.backbone.enable_lora(rank=4)
```

Check the model summary after setting the LoRA rank. Notice that enabling LoRA reduces the number of trainable parameters significantly compared to the total number of parameters in the model:

```python
gemma_lm.summary()
```

Configure the rest of the fine-tuning settings, including the preprocessor settings, optimizer, number of tuning epochs, and batch size:

```python
# Limit the input sequence length to 256 (to control memory usage).
gemma_lm.preprocessor.sequence_length = 256
# Use AdamW (a common optimizer for transformer models).
optimizer = keras.optimizers.AdamW(
    learning_rate=5e-5,
    weight_decay=0.01,
)
# Exclude layernorm and bias terms from decay.
optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])

gemma_lm.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizer,
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
```

### Run the fine-tune process

Run the fine-tuning process using the `fit()` method. This process can take several minutes depending on your compute resources, data size, and number of epochs:

```python
gemma_lm.fit(data, epochs=1, batch_size=1)
1000/1000 ━━━━━━━━━━━━━━━━━━━━ 923s 888ms/step - loss: 1.5586 - sparse_categorical_accuracy: 0.5251
<keras.src.callbacks.history.History at 0x799d04393c40>
```

#### Mixed precision fine-tuning on NVIDIA GPUs

Full precision is recommended for fine-tuning. When fine-tuning on NVIDIA GPUs, you can use mixed precision (`keras.mixed_precision.set_global_policy('mixed_bfloat16')`) to speed up training with minimal effect on training quality.

```python
# Uncomment the line below if you want to enable mixed precision training on GPUs
# keras.mixed_precision.set_global_policy('mixed_bfloat16')
```

## Inference after fine-tuning

After fine-tuning, you should see changes in the responses when the tuned model is given the same prompt.

### Europe trip prompt

Try the Europe trip prompt from earlier and note the differences in the response.

```python
prompt = template.format(
    instruction="What should I do on a trip to Europe?",
    response="",
)
sampler = keras_hub.samplers.TopKSampler(k=5, seed=2)
gemma_lm.compile(sampler=sampler)
print(gemma_lm.generate(prompt, max_length=256))
Instruction:
What should I do on a trip to Europe?

Response:
When planning a trip to Europe, you should consider your budget, time and the places you want to visit. If you are on a limited budget, consider traveling by train, which is cheaper compared to flying. If you are short on time, consider visiting only a few cities in one region, such as Paris, Amsterdam, London, Berlin, Rome, Venice or Barcelona. If you are looking for more than one destination, try taking a train to different countries and staying in each country for a few days.
```

The model now provides a shorter response to a question about visiting Europe.

### Photosynthesis prompt

Try the photosynthesis explanation prompt from earlier and note the differences in the response.

```python
prompt = template.format(
    instruction="Explain the process of photosynthesis in a way that a child could understand.",
    response="",
)
print(gemma_lm.generate(prompt, max_length=256))
Instruction:
Explain the process of photosynthesis in a way that a child could understand.

Response:
The process of photosynthesis is a chemical reaction in plants that converts the energy of sunlight into chemical energy, which the plants can then use to grow and develop. During photosynthesis, a plant will absorb carbon dioxide (CO2) from the air and water from the soil and use the energy from the sun to produce oxygen (O2) and sugars (glucose) as a by-product.
```

The model now explains photosynthesis in simpler terms.

## Improving fine-tune results

For demonstration purposes, this tutorial fine-tunes the model on a small subset of the dataset for just one epoch and with a low LoRA rank value. To get better responses from the fine-tuned model, you can experiment with:

1. Increasing the size of the fine-tuning dataset
2. Training for more steps (epochs)
3. Setting a higher LoRA rank
4. Modifying the hyperparameter values such as `learning_rate` and `weight_decay`.

## Summary

This tutorial covered LoRA fine-tuning on a Gemma model using Keras. 

## What's next

<div class="card-grid">
<a class="card" href="../get-started/index.md">
  <span class="card-icon">{{ icon('lucide/rocket') }}</span>
  <div class="card-title">Quickstart</div>
  <p class="card-desc">Run your first Gemma 4 generation.</p>
</a>
<a class="card" href="distributed.md">
  <span class="card-icon">{{ icon('lucide/network') }}</span>
  <div class="card-title">Distributed tuning</div>
  <p class="card-desc">Train across multiple accelerators.</p>
</a>
<a class="card" href="https://cloud.google.com/vertex-ai/docs/generative-ai/open-models/use-gemma" target="_blank" rel="noopener">
  <div class="card-title">Gemma on Vertex AI <span class="card-ext">↗</span></div>
  <p class="card-desc">Use Gemma open models on Google Cloud.</p>
</a>
</div>
