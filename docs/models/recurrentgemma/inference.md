---
title: Inference with RecurrentGemma using JAX and Flax
description: Run sampling and inference with RecurrentGemma 2B Instruct using Google DeepMind's recurrentgemma library and JAX.
models: [recurrentgemma]
frameworks: [jax]
source_url: "https://ai.google.dev/gemma/docs/recurrentgemma/recurrentgemma_jax_inference.md.txt"
---
# Inference with RecurrentGemma using JAX and Flax


This tutorial demonstrates how to perform basic sampling/inference with the [RecurrentGemma](index.md) 2B Instruct model using [Google DeepMind's `recurrentgemma` library](https://github.com/google-deepmind/recurrentgemma) that was written with [JAX](https://jax.readthedocs.io) (a high-performance numerical computing library), [Flax](https://flax.readthedocs.io) (the JAX-based neural network library), [Orbax](https://orbax.readthedocs.io/) (a JAX-based library for training utilities like checkpointing), and [SentencePiece](https://github.com/google/sentencepiece) (a tokenizer/detokenizer library). Although Flax is not used directly in this notebook, Flax was used to create Gemma and RecurrentGemma (the Griffin model).

This notebook can run on Google Colab with the T4 GPU (go to **Edit** \> **Notebook settings** \> Under **Hardware accelerator** select **T4 GPU**).

## Setup

The following sections explain the steps for preparing a notebook to use a RecurrentGemma model, including model access, getting an API key, and configuring the notebook runtime

### Set up Kaggle access for Gemma

To complete this tutorial, you first need to follow the setup instructions *similar* to [Gemma setup](../../get-started/index.md) with a few exceptions:

- Get access to RecurrentGemma (instead of Gemma) on [kaggle.com](https://www.kaggle.com/models/google/recurrentgemma).
- Select a Colab runtime with sufficient resources to run the RecurrentGemma model.
- Generate and configure a Kaggle username and API key.

After you've completed the RecurrentGemma setup, move on to the next section, where you'll set environment variables for your Colab environment.

### Set environment variables

Set environment variables for `KAGGLE_USERNAME` and `KAGGLE_KEY`. When prompted with the "Grant access?" messages, agree to provide secret access.

```python
import os
from google.colab import userdata # `userdata` is a Colab API.

os.environ["KAGGLE_USERNAME"] = userdata.get('KAGGLE_USERNAME')
os.environ["KAGGLE_KEY"] = userdata.get('KAGGLE_KEY')
```

### Install the `recurrentgemma` library

This notebook focuses on using a free Colab GPU. To enable hardware acceleration, click on **Edit** \> **Notebook settings** \> Select **T4 GPU** \> **Save**.

Next, you need to install the Google DeepMind `recurrentgemma` library from [`github.com/google-deepmind/recurrentgemma`](https://github.com/google-deepmind/recurrentgemma). If you get an error about "pip's dependency resolver", you can usually ignore it.

!!! Note
    By installing `recurrentgemma`, you will also install [`flax`](https://flax.readthedocs.io), core [`jax`](https://jax.readthedocs.io), [`optax`](https://optax.readthedocs.io/en/latest/) (the JAX-based gradient processing and optimization library), [`orbax`](https://orbax.readthedocs.io/), and [`sentencepiece`](https://github.com/google/sentencepiece).


```bash
pip install git+https://github.com/google-deepmind/recurrentgemma.git
```

## Load and prepare the RecurrentGemma model

1. Load the RecurrentGemma model with [`kagglehub.model_download`](https://github.com/Kaggle/kagglehub/blob/bddefc718182282882b72f814d407d89e5d178c4/src/kagglehub/models.py#L12), which takes three arguments:

- `handle`: The model handle from Kaggle
- `path`: (Optional string) The local path
- `force_download`: (Optional boolean) Forces to re-download the model

!!! Note
    Be mindful that the `recurrentgemma-2b-it` model is around 3.85Gb in size.


```python
RECURRENTGEMMA_VARIANT = '2b-it' # @param ['2b', '2b-it'] {type:"string"}

import kagglehub

RECURRENTGEMMA_PATH = kagglehub.model_download(f'google/recurrentgemma/flax/{RECURRENTGEMMA_VARIANT}')
```

``` { .no-copy .output }
Downloading from https://www.kaggle.com/api/v1/models/google/recurrentgemma/flax/2b-it/1/download...
100%|██████████| 3.85G/3.85G [00:52<00:00, 78.2MB/s]
Extracting model files...
```

```python
print('RECURRENTGEMMA_PATH:', RECURRENTGEMMA_PATH)
```

``` { .no-copy .output }
RECURRENTGEMMA_PATH: /root/.cache/kagglehub/models/google/recurrentgemma/flax/2b-it/1
```

!!! Note
    The path from the output above is where the model weights and tokenizer are saved locally, you will need them for later.


1. Check the location of the model weights and the tokenizer, then set the path variables. The tokenizer directory will be in the main directory where you downloaded the model, while the model weights will be in a sub-directory. For example:

- The `tokenizer.model` file will be in `/LOCAL/PATH/TO/recurrentgemma/flax/2b-it/1`).
- The model checkpoint will be in `/LOCAL/PATH/TO/recurrentgemma/flax/2b-it/1/2b-it`).

  ```python
  CKPT_PATH = os.path.join(RECURRENTGEMMA_PATH, RECURRENTGEMMA_VARIANT)
  TOKENIZER_PATH = os.path.join(RECURRENTGEMMA_PATH, 'tokenizer.model')
  print('CKPT_PATH:', CKPT_PATH)
  print('TOKENIZER_PATH:', TOKENIZER_PATH)
  ```

  ``` { .no-copy .output }
  CKPT_PATH: /root/.cache/kagglehub/models/google/recurrentgemma/flax/2b-it/1/2b-it
  TOKENIZER_PATH: /root/.cache/kagglehub/models/google/recurrentgemma/flax/2b-it/1/tokenizer.model
  ```

## Perform sampling/inference

1. Load the RecurrentGemma model checkpoint with the [`recurrentgemma.jax.load_parameters`](https://github.com/google-deepmind/recurrentgemma/blob/e4939f9b7edf8baa1d512fb86bfc2e206044d66b/recurrentgemma/jax/utils.py#L31) method. The `sharding` argument set to `"single_device"` loads all model parameters on a single device.

   ```python
   import recurrentgemma
   from recurrentgemma import jax as recurrentgemma

   params = recurrentgemma.load_parameters(checkpoint_path=CKPT_PATH, sharding="single_device")
   ```

1. Load the RecurrentGemma model tokenizer, constructed using [`sentencepiece.SentencePieceProcessor`](https://github.com/google/sentencepiece/blob/4d6a1f41069c4636c51a5590f7578a0dbed83450/python/src/sentencepiece/__init__.py#L423):

   ```python
   import sentencepiece as spm

   vocab = spm.SentencePieceProcessor()
   vocab.Load(TOKENIZER_PATH)
   ```

   ``` { .no-copy .output }
   True
   ```

1. To automatically load the correct configuration from the RecurrentGemma model checkpoint, use [`recurrentgemma.GriffinConfig.from_flax_params_or_variables`](https://github.com/google-deepmind/recurrentgemma/blob/e4939f9b7edf8baa1d512fb86bfc2e206044d66b/recurrentgemma/common.py#L128). Then, instantiate the [Griffin](https://arxiv.org/abs/2402.19427) model with [`recurrentgemma.jax.Griffin`](https://github.com/google-deepmind/recurrentgemma/blob/e4939f9b7edf8baa1d512fb86bfc2e206044d66b/recurrentgemma/jax/griffin.py#L29).

   ```python
   model_config = recurrentgemma.GriffinConfig.from_flax_params_or_variables(
       flax_params_or_variables=params)

   model = recurrentgemma.Griffin(model_config)
   ```

1. Create a `sampler` with [`recurrentgemma.jax.Sampler`](https://github.com/google-deepmind/recurrentgemma/blob/e4939f9b7edf8baa1d512fb86bfc2e206044d66b/recurrentgemma/jax/sampler.py#L74) on top of the RecurrentGemma model checkpoint/weights and the tokenizer:

   ```python
   sampler = recurrentgemma.Sampler(
       model=model,
       vocab=vocab,
       params=params,
   )
   ```

1. Write a prompt in `prompt` and perform inference. You can tweak `total_generation_steps` (the number of steps performed when generating a response --- this example uses `50` to preserve host memory).

!!! Note
    If you run out of memory, click on **Runtime** \> **Disconnect and delete runtime** , and then **Runtime** \> **Run all**.


```python
prompt = [
    "\n# 5+9=?",
]

reply = sampler(input_strings=prompt,
                total_generation_steps=50,
                )

for input_string, out_string in zip(prompt, reply.text):
    print(f"Prompt:\n{input_string}\nOutput:\n{out_string}")
```

``` { .no-copy .output }
/usr/local/lib/python3.10/dist-packages/jax/_src/interpreters/mlir.py:920: UserWarning: Some donated buffers were not usable: ShapedArray(int32[1,8]).
See an explanation at https://jax.readthedocs.io/en/latest/faq.html#buffer_donation.
  warnings.warn("Some donated buffers were not usable:"
Prompt:

# 5+9=?
Output:


# Answer: 14

# Explanation: 5 + 9 = 14.
```

## Learn more

- You can learn more about the Google DeepMind [`recurrentgemma` library on GitHub](https://github.com/google-deepmind/recurrentgemma), which contains docstrings of methods and modules you used in this tutorial, such as [`recurrentgemma.jax.load_parameters`](https://github.com/google-deepmind/recurrentgemma/blob/e4939f9b7edf8baa1d512fb86bfc2e206044d66b/recurrentgemma/jax/utils.py#L31), [`recurrentgemma.jax.Griffin`](https://github.com/google-deepmind/recurrentgemma/blob/e4939f9b7edf8baa1d512fb86bfc2e206044d66b/recurrentgemma/jax/griffin.py#L29), and [`recurrentgemma.jax.Sampler`](https://github.com/google-deepmind/recurrentgemma/blob/e4939f9b7edf8baa1d512fb86bfc2e206044d66b/recurrentgemma/jax/sampler.py#L74).
- The following libraries have their own documentation sites: [core JAX](https://jax.readthedocs.io), [Flax](https://flax.readthedocs.io), and [Orbax](https://orbax.readthedocs.io/).
- For `sentencepiece` tokenizer/detokenizer documentation, check out [Google's `sentencepiece` GitHub repo](https://github.com/google/sentencepiece).
- For `kagglehub` documentation, check out `README.md` on [Kaggle's `kagglehub` GitHub repo](https://github.com/Kaggle/kagglehub).
- Learn how to [use Gemma models with Google Cloud Vertex AI](https://cloud.google.com/vertex-ai/docs/generative-ai/open-models/use-gemma).
- Check out the [RecurrentGemma: Moving Past Transformers
  for Efficient Open Language Models](https://arxiv.org/pdf/2404.07839) paper by Google DeepMind.
- Read the [Griffin: Mixing Gated Linear Recurrences with
  Local Attention for Efficient Language Models](https://arxiv.org/pdf/2402.19427) paper by GoogleDeepMind to learn more about the model architecture used by RecurrentGemma.
