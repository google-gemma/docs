---
title: Fine-tune PaliGemma with JAX and Flax
description: This notebook shows how to fine-tune PaliGemma on a vision-language task with JAX.
models: [paligemma]
frameworks: [jax]
source_url: "https://ai.google.dev/gemma/docs/paligemma/fine-tuning-paligemma.md.txt"
---
# Fine-tune PaliGemma with JAX and Flax


This notebook shows how to fine-tune [PaliGemma](index.md) on a vision-language task with [JAX](https://jax.readthedocs.io/en/latest/index.html). *Fine-tuning* is a process that can improve your model's performance on specific tasks or help the model adhere to specific output requirements when instructions aren't sufficient and you have a set of examples that demonstrate the outputs you want. Gemma-based models like PaliGemma require fine-tuning to produce expected results.

### What's in this notebook

This notebook uses the model reference implementation from [`big_vision`](https://github.com/google-research/big_vision)
and shows how to:

- Install dependencies, and download the PaliGemma model checkpoint and training data
- Load the model onto GPU devices
- Prepare the model's inputs for training and inference
- Fine-tune the model
- Inspect the output

The training data for this notebook consists of 90 pairs of images and long captions describing them. To make it runnable on a T4 colab runtime, you'll only fine-tune the attention layers of the language model and freeze the other parameters.

This example is for learning purposes only. In a real use case, the amount of data, trainable parameters, training steps and hyper-parameters, and obtained results could be significantly different.

### Before you begin

Before going through this notebook, you should be familiar with Python code, as well as how large language models (LLMs) are trained. You don't need to be familiar with JAX, but basic knowledge about JAX (or similar technologies such as Keras) is helpful when reading through the example code.

## Setup

The following sections explain the preliminary steps for getting a notebook to use a PaliGemma model, including model access, getting an API key, and configuring the notebook runtime.

### Get access to PaliGemma

Before using PaliGemma for the first time, you must request access to the model through Kaggle by completing the following steps:

1. Log in to [Kaggle](https://www.kaggle.com), or create a new Kaggle account if you don't already have one.
2. Go to the [PaliGemma model card](https://www.kaggle.com/models/google/paligemma-2){ data-model="paligemma" } and click **Request Access**.
3. Complete the consent form and accept the terms and conditions.

### Configure your API key

To use PaliGemma, you must provide your Kaggle username and a Kaggle API key.

To generate a Kaggle API key, open your [**Settings** page in Kaggle](https://www.kaggle.com/settings) and click **Create New Token** . This triggers the download of a `kaggle.json` file containing your API credentials.

Then, in Colab, select **Secrets** (🔑) in the left pane and add your Kaggle username and Kaggle API key. Store your username under the name `KAGGLE_USERNAME` and your API key under the name `KAGGLE_KEY`.

### Select the runtime

To complete this tutorial, you'll need to have a Colab runtime with sufficient resources to run the PaliGemma model. In this case, you can use a T4 GPU:

1. In the upper-right of the Colab window, click the **▾ (Additional connection options)** dropdown menu.
2. Select **Change runtime type**.
3. Under **Hardware accelerator** , select **T4 GPU**.

### Install Python packages

Run the below cell to install KaggleHub.

```bash
pip install -U -q kagglehub
```

### Set environment variables

Set the environment variables and Kaggle login.

```python
import os
# The T4 runtime is tight on memory to finetune this model. Preallocate
# all memory ahead of time to avoid out-of-memory due to fragmentation.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"

# This will prompt you to enter your Kaggle API token
import kagglehub
kagglehub.login()
```

``` { .no-copy .output }
VBox(children=(HTML(value='<center> <img\nsrc=https://www.kaggle.com/static/images/site-logo.png\nalt=\'Kaggle…
Kaggle credentials set.
Kaggle credentials successfully validated.
```

### Fetch the `big_vision` repository and install related dependencies

Download the `big_vision` repository to your Colab notebook from GitHub and install dependencies related to `big_vision` by running the following code.

```python
import os
import sys

# TPUs with
if "COLAB_TPU_ADDR" in os.environ:
  raise "It seems you are using Colab with remote TPUs which is not supported."

# Fetch big_vision repository if python doesn't know about it and install
# dependencies needed for this notebook.
if not os.path.exists("big_vision_repo"):
  !git clone --quiet --branch=main --depth=1 \
     https://github.com/google-research/big_vision big_vision_repo

# Append big_vision code to python import path
if "big_vision_repo" not in sys.path:
  sys.path.append("big_vision_repo")

# Install missing dependencies. Assume jax~=0.4.25 with GPU available.
!pip3 install -q "overrides" "ml_collections" "einops~=0.7" "sentencepiece"
```

``` { .no-copy .output }
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 76.7/76.7 kB 2.8 MB/s eta 0:00:00
```

### Import JAX and other dependencies

Import JAX and other dependencies required for PaliGemma, like TensorFlow and NumPy.

```python
import base64
import functools
import html
import io
import os
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import ml_collections

import tensorflow as tf
import sentencepiece

from IPython.core.display import display, HTML
from PIL import Image

# Import model definition from big_vision
from big_vision.models.proj.paligemma import paligemma
from big_vision.trainers.proj.paligemma import predict_fns

# Import big vision utilities
import big_vision.datasets.jsonl
import big_vision.utils
import big_vision.sharding

# Don't let TF use the GPU or TPUs
tf.config.set_visible_devices([], "GPU")
tf.config.set_visible_devices([], "TPU")

backend = jax.extend.backend.get_backend()
print(f"JAX version:  {jax.__version__}")
print(f"JAX platform: {backend.platform}")
print(f"JAX devices:  {jax.device_count()}")
```

``` { .no-copy .output }
JAX version:  0.7.2
JAX platform: gpu
JAX devices:  1
```

## Download and configure the model

In this step, you'll download the model checkpoint and configure it so that you can fine-tune it later on. This step shows you how to move model parameters into TPU memory, which is useful for fine-tuning models on devices with limited resources.

### Download the model checkpoint

PaliGemma includes several model variations. For this tutorial, you'll use the base [JAX/FLAX PaliGemma 3B weight model](https://www.kaggle.com/models/google/paligemma-2/jax/paligemma2-3b-pt-224){ data-model="paligemma" }.

Download the model checkpoint from Kaggle by running the following code. This process takes several minutes to complete.

```python
import os
import kagglehub

# Use these for PaliGemma-2 3B 224px²
LLM_VARIANT = "gemma2_2b"
MODEL_PATH = "./paligemma2-3b-pt-224.b16.npz"
KAGGLE_HANDLE = "google/paligemma-2/jax/paligemma2-3b-pt-224"  # Path to fetch from Kaggle.

# Use these for PaliGemma 1:
# LLM_VARIANT = "gemma_2b"
# MODEL_PATH = "./paligemma-3b-pt-224.f16.npz"
# KAGGLE_HANDLE = "google/paligemma/jax/paligemma-3b-pt-224"

if not os.path.exists(MODEL_PATH):
  print("Downloading the checkpoint from Kaggle, this could take a few minutes....")
  MODEL_PATH = kagglehub.model_download(KAGGLE_HANDLE, MODEL_PATH)
  print(f"Model path: {MODEL_PATH}")

TOKENIZER_PATH = "./paligemma_tokenizer.model"
if not os.path.exists(TOKENIZER_PATH):
  print("Downloading the model tokenizer...")
  !gcloud storage cp gs://big_vision/paligemma_tokenizer.model {TOKENIZER_PATH}
  print(f"Tokenizer path: {TOKENIZER_PATH}")

DATA_DIR="./longcap100"
if not os.path.exists(DATA_DIR):
  print("Downloading the dataset...")
  !gcloud storage cp --no-clobber --recursive gs://longcap100/ .
  print(f"Data path: {DATA_DIR}")
```

``` { .no-copy .output }
Downloading the checkpoint from Kaggle, this could take a few minutes....
Downloading to /root/.cache/kagglehub/models/google/paligemma-2/jax/paligemma2-3b-pt-224/1/./paligemma2-3b-pt-224.b16.npz...
100%|██████████| 5.65G/5.65G [00:54<00:00, 112MB/s]
Model path: /root/.cache/kagglehub/models/google/paligemma-2/jax/paligemma2-3b-pt-224/1/./paligemma2-3b-pt-224.b16.npz
Downloading the model tokenizer...
Copying gs://big_vision/paligemma_tokenizer.model...

- [1 files][  4.1 MiB/  4.1 MiB]                                                
Operation completed over 1 objects/4.1 MiB.                                      
Tokenizer path: ./paligemma_tokenizer.model
Downloading the dataset...
Data path: ./longcap100
```

### Configure the model

It's time to actually start configuring the model that you're going to use.

For this notebook, you need to be able to fit your model onto a T4 GPU. Having a limited resource like space constraints means that you have to be mindful of how your model is configured.

If you fine-tune every parameter, your model won't be able to run in the notebook environment. As a result, in this part of the notebook, you'll configure your model so that it has the ability to freeze some of the parameters, and only fine-tune the parameters that really need to be fine-tuned for the model to give you accurate results. In LLMs, parameters are said to be *frozen* when they are no longer actively being used to train the model.

In order to configure your model, you need to:

- Initialize the `model_config` as a [`FrozenConfigDict`](https://github.com/google/ml_collections/tree/master#frozenconfigdict) so that you can freeze some of the parameters and keep memory usage low
- Initialize an instance of the PaliGemma `Model` class using the `model_config` as its configurations
- Load the model parameters into RAM
- Define a `decode` function to sample outputs from the model

This code in this cell takes about a minute to run to completion.

```python
# Define model

# IMPORTANT: Gemma-2 has a "final_logits_softcap" property. Set it to 0.0
# for better transfer results.
model_config = ml_collections.FrozenConfigDict({
    "llm": {"vocab_size": 257_152, "variant": LLM_VARIANT, "final_logits_softcap": 0.0},
    "img": {"variant": "So400m/14", "pool_type": "none", "scan": True, "dtype_mm": "float16"}
})
model = paligemma.Model(**model_config)
tokenizer = sentencepiece.SentencePieceProcessor(TOKENIZER_PATH)

# Load params - this can take up to 1 minute in T4 colabs.
params = paligemma.load(None, MODEL_PATH, model_config)

# Define `decode` function to sample outputs from the model.
decode_fn = predict_fns.get_all(model)['decode']
decode = functools.partial(decode_fn, devices=jax.devices(), eos_token=tokenizer.eos_id())
```

### Move model parameters into GPU/TPU memory

Now you need to move the model parameters into GPU/TPU memory. First, shard the parameters across the available GPUs, then load the parameters. Here, you'll load the parameters sequentially. This process takes longer than loading them simultaneously, but it requires more RAM than you have available in this notebook.

Finally, print out all of the parameters to see what type each individual parameter is cast to. Frozen parameters are kept as `float16`, while the trainable parameters are cast to `float32`. When you inspect the list, you'll see that most of the parameters have been frozen and are `float16`.

```python
# Create a pytree mask of the trainable params.
def is_trainable_param(name, param):  # pylint: disable=unused-argument
  if name.startswith("llm/layers/attn/"):  return True
  if name.startswith("llm/"):              return False
  if name.startswith("img/"):              return False
  raise ValueError(f"Unexpected param name {name}")
trainable_mask = big_vision.utils.tree_map_with_names(is_trainable_param, params)

# If more than one device is available (e.g. multiple GPUs) the parameters can
# be sharded across them to reduce HBM usage per device.
mesh = jax.sharding.Mesh(jax.devices(), ("data"))

data_sharding = jax.sharding.NamedSharding(
    mesh, jax.sharding.PartitionSpec("data"))

params_sharding = big_vision.sharding.infer_sharding(
    params, strategy=[('.*', 'fsdp(axis="data")')], mesh=mesh)

# Yes: Some donated buffers are not usable.
warnings.filterwarnings(
    "ignore", message="Some donated buffers were not usable")

@functools.partial(jax.jit, donate_argnums=(0,), static_argnums=(1,))
def maybe_cast_to_f32(params, trainable):
  # Cast others to float16, since some GPUs don't support bf16.
  return jax.tree.map(lambda p, m: p.astype(jnp.float32)
                      if m else p.astype(jnp.float16),
                      params, trainable)

# Loading all params in simultaneous - albeit much faster and more succinct -
# requires more RAM than the T4 colab runtimes have by default.
# Instead, do it param by param.
params, treedef = jax.tree.flatten(params)
sharding_leaves = jax.tree.leaves(params_sharding)
trainable_leaves = jax.tree.leaves(trainable_mask)
for idx, (sharding, trainable) in enumerate(zip(sharding_leaves, trainable_leaves)):
  params[idx] = big_vision.utils.reshard(params[idx], sharding)
  params[idx] = maybe_cast_to_f32(params[idx], trainable)
  params[idx].block_until_ready()
params = jax.tree.unflatten(treedef, params)

# Print params to show what the model is made of.
def parameter_overview(params):
  for path, arr in big_vision.utils.tree_flatten_with_names(params)[0]:
    print(f"{path:80s} {str(arr.shape):22s} {arr.dtype}")

print(" == Model params == ")
parameter_overview(params)
```

``` { .no-copy .output }
== Model params == 
img/Transformer/encoder_norm/bias                                                (1152,)                float16
img/Transformer/encoder_norm/scale                                               (1152,)                float16
img/Transformer/encoderblock/LayerNorm_0/bias                                    (27, 1152)             float16
img/Transformer/encoderblock/LayerNorm_0/scale                                   (27, 1152)             float16
img/Transformer/encoderblock/LayerNorm_1/bias                                    (27, 1152)             float16
img/Transformer/encoderblock/LayerNorm_1/scale                                   (27, 1152)             float16
img/Transformer/encoderblock/MlpBlock_0/Dense_0/bias                             (27, 4304)             float16
img/Transformer/encoderblock/MlpBlock_0/Dense_0/kernel                           (27, 1152, 4304)       float16
img/Transformer/encoderblock/MlpBlock_0/Dense_1/bias                             (27, 1152)             float16
img/Transformer/encoderblock/MlpBlock_0/Dense_1/kernel                           (27, 4304, 1152)       float16
img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/bias             (27, 16, 72)           float16
img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/kernel           (27, 1152, 16, 72)     float16
img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/bias             (27, 1152)             float16
img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/kernel           (27, 16, 72, 1152)     float16
img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/bias           (27, 16, 72)           float16
img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/kernel         (27, 1152, 16, 72)     float16
img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/bias           (27, 16, 72)           float16
img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/kernel         (27, 1152, 16, 72)     float16
img/embedding/bias                                                               (1152,)                float16
img/embedding/kernel                                                             (14, 14, 3, 1152)      float16
img/head/bias                                                                    (2304,)                float16
img/head/kernel                                                                  (1152, 2304)           float16
img/pos_embedding                                                                (1, 256, 1152)         float16
llm/embedder/input_embedding                                                     (257152, 2304)         float16
llm/final_norm/scale                                                             (2304,)                float16
llm/layers/attn/attn_vec_einsum/w                                                (26, 8, 256, 2304)     float32
llm/layers/attn/kv_einsum/w                                                      (26, 2, 4, 2304, 256)  float32
llm/layers/attn/q_einsum/w                                                       (26, 8, 2304, 256)     float32
llm/layers/mlp/gating_einsum                                                     (26, 2, 2304, 9216)    float16
llm/layers/mlp/linear                                                            (26, 9216, 2304)       float16
llm/layers/post_attention_norm/scale                                             (26, 2304)             float16
llm/layers/post_ffw_norm/scale                                                   (26, 2304)             float16
llm/layers/pre_attention_norm/scale                                              (26, 2304)             float16
llm/layers/pre_ffw_norm/scale                                                    (26, 2304)             float16
```

## Prepare to tune the model

Now that your model is configured, you can tune it. In this step, you'll create your model's inputs as well as the training and validation iterators, view the training examples, and define the training and validation loops.

### Create model inputs

The model checkpoint you're using has already been trained on images of various aspect ratios that have been resized to 224x224 pixels, and to handle tokenized texts.

The code below defines three functions that you'll use in the next step create the model's inputs:

- **`preprocess_image`:** Normalizes the image data. In this case, pre-processing converts the passed-in image to greyscale, removes the alpha layer, and resizes the passed-in image to the size required by the model for image inputs (224x224 pixels).
- **`preprocess_tokens`:** Splits the tokens up and adds flags to mark whether a token is a prefix or suffix token. These flags will be used later on in the code, during the training step and the evaluation loop.
- **`postprocess_tokens`:** Removes any tokens left at and/or after the end-of-sequence (EOS) token and returns the remaining decoded tokens.

  ```python
  def preprocess_image(image, size=224):
    # Model has been trained to handle images of different aspects ratios
    # resized to 224x224 in the range [-1, 1]. Bilinear and antialias resize
    # options are helpful to improve quality in some tasks.
    image = np.asarray(image)
    if image.ndim == 2:  # Convert image without last channel into greyscale.
      image = np.stack((image,)*3, axis=-1)
    image = image[..., :3]  # Remove alpha layer.
    assert image.shape[-1] == 3

    image = tf.constant(image)
    image = tf.image.resize(image, (size, size), method='bilinear', antialias=True)
    return image.numpy() / 127.5 - 1.0  # [0, 255]->[-1,1]

  def preprocess_tokens(prefix, suffix=None, seqlen=None):
    # Model has been trained to handle tokenized text composed of a prefix with
    # full attention and a suffix with causal attention.
    separator = "\n"
    tokens = tokenizer.encode(prefix, add_bos=True) + tokenizer.encode(separator)
    mask_ar = [0] * len(tokens)    # 0 to use full attention for prefix.
    mask_loss = [0] * len(tokens)  # 0 to not use prefix tokens in the loss.

    if suffix:
      suffix = tokenizer.encode(suffix, add_eos=True)
      tokens += suffix
      mask_ar += [1] * len(suffix)    # 1 to use causal attention for suffix.
      mask_loss += [1] * len(suffix)  # 1 to use suffix tokens in the loss.

    mask_input = [1] * len(tokens)    # 1 if it's a token, 0 if padding.
    if seqlen:
      padding = [0] * max(0, seqlen - len(tokens))
      tokens = tokens[:seqlen] + padding
      mask_ar = mask_ar[:seqlen] + padding
      mask_loss = mask_loss[:seqlen] + padding
      mask_input = mask_input[:seqlen] + padding

    return jax.tree.map(np.array, (tokens, mask_ar, mask_loss, mask_input))

  def postprocess_tokens(tokens):
    tokens = tokens.tolist()  # np.array to list[int]
    try:  # Remove tokens at and after EOS if any.
      eos_pos = tokens.index(tokenizer.eos_id())
      tokens = tokens[:eos_pos]
    except ValueError:
      pass
    return tokenizer.decode(tokens)
  ```

### Create the training and validation iterators

Create two iterators:

- A **training iterator** to allow the training process to go through the data in chunks rather than processing it all at once
    - This allows you to do some data pre-processing before use
- A **validation iterator** that allows the training process to iterate over the validation dataset to see how well the tuned model aligned with the provided results

  ```python
  SEQLEN = 128

  train_dataset = big_vision.datasets.jsonl.DataSource(
      os.path.join(DATA_DIR, "data_train90.jsonl"),
      fopen_keys={"image": DATA_DIR})

  val_dataset = big_vision.datasets.jsonl.DataSource(
      os.path.join(DATA_DIR, "data_val10.jsonl"),
      fopen_keys={"image": DATA_DIR})


  def train_data_iterator():
    """Never ending iterator over training examples."""
    # Shuffle examples and repeat so one can train for many epochs.
    dataset = train_dataset.get_tfdata().shuffle(1_000).repeat()
    for example in dataset.as_numpy_iterator():
      image = Image.open(io.BytesIO(example["image"]))
      image = preprocess_image(image)

      prefix = "caption en"  # Could also be a different prefix per example.
      suffix = example["suffix"].decode().lower()
      tokens, mask_ar, mask_loss, _ = preprocess_tokens(prefix, suffix, SEQLEN)

      yield {
          "image": np.asarray(image),
          "text": np.asarray(tokens),
          "mask_ar": np.asarray(mask_ar),
          "mask_loss": np.asarray(mask_loss),
      }


  def validation_data_iterator():
    """Single iterator over validation examples."""
    for example in val_dataset.get_tfdata(ordered=True).as_numpy_iterator():
      image = Image.open(io.BytesIO(example["image"]))
      image = preprocess_image(image)

      prefix = "caption en"  # Could also be a different prefix per example.
      tokens, mask_ar, _, mask_input = preprocess_tokens(prefix, seqlen=SEQLEN)

      yield {
          "image": np.asarray(image),
          "text": np.asarray(tokens),
          "mask_ar": np.asarray(mask_ar),
          "mask_input": np.asarray(mask_input),
      }
  ```

### View training examples

In this notebook, the training data contains 90 images that are paired with long descriptions of what's depicted in the image.

!!! Note
    Normal training data sets that are meant to be used for practical use cases should contain more images, but this notebook limits the number of data points so that you can train the model in a reasonable amount of time for an example.


The code below prints a random selection of images with their descriptions from the training data set so that you can see what the images and descriptions your model is trained on looks like. Each image is displayed in as a 128x128 pixel JPEG, with the description printed next to the image to the right.

```python
def render_inline(image, resize=(128, 128)):
  """Convert image into inline html."""
  image = Image.fromarray(image)
  image.resize(resize)
  with io.BytesIO() as buffer:
    image.save(buffer, format='jpeg')
    image_b64 = str(base64.b64encode(buffer.getvalue()), "utf-8")
    return f"data:image/jpeg;base64,{image_b64}"

def render_example(image, caption):
  image = ((image + 1)/2 * 255).astype(np.uint8)  # [-1,1] -> [0, 255]
  return f"""
    <div style="display: inline-flex; align-items: center; justify-content: center;">
        <img style="width:128px; height:128px;" src="{render_inline(image, resize=(64,64))}" />
        <p style="width:256px; margin:10px; font-size:small;">{html.escape(caption)}</p>
    </div>
    """

html_out = ""
for idx, example in zip(range(8), train_data_iterator()):
  caption = postprocess_tokens(example["text"])  # detokenize model input.
  caption = caption[len("caption en\n"):]        # strip prefix
  html_out += render_example(example["image"], caption)

print("Training examples")
display(HTML(html_out))
```

``` { .no-copy .output }
Training examples
```
![](../../assets/paligemma/fine-tuning-01.jpg)

a car is parked in front of a building with a green umbrella. the building has a green and white sign, a green and white umbrella, and a white sign with black lettering. there is a tall palm tree and a tall tree. the car is parked next to a yellow car and a black car. the road is grey and the sky is white.
![](../../assets/paligemma/fine-tuning-02.jpg)

a blue bicycle is parked next to awooden fence. the bike has a black seat, a black kickstand, and a black tire. the fence is brown and the grass is green. there is a small green bush and a small green tree in the background. the bike has a light on the front and a light on the back. the front tire of the bike is on the ground and the back tire is on the fence. the bike has a black pedal and a black pedal on the bike. the bike has a black seat and a black seat on the bike. the bike has a black kickstand and a
![](../../assets/paligemma/fine-tuning-03.jpg)

a plate of waffles with blueberries and syrup. the waffle is brown. the blueberries are black and are on top of the waffle. the syrup is pouring onto the plate. the plate is white and has a brown rim. the table is white and has a brown wooden fence in the background.
![](../../assets/paligemma/fine-tuning-04.jpg)

a person holds a badminton racket with a shuttlecock attached to it. the racket has a black design, with a white handle and a black net. the shuttlecock is white and red. the person's hand is holding the shuttlecock, and their fingers are gripping the handle. the racket is in the person's hand, and the shuttlecock is in their racket. the person is wearing blue shorts and a red shirt. the background is blurry.
![](../../assets/paligemma/fine-tuning-05.jpg)

a field of sunflowers with a cloudy sky in the background. the sky is blue with white clouds. the flowers are yellow and green, with green leaves on the stems. the sunflower is large and has a yellow center.the flower is in bloom and the petals are yellow.
![](../../assets/paligemma/fine-tuning-06.jpg)

a purple building with a row of chairs outside. the building has a purple door and a purple window. there is a potted plant in the window and a flower pot on the window sill. the chairs are brown and the table is made of wood. there is a flower box on the wall and a flower pot on the window sill. the street is wet and there is a light on the wall. the sign on the building is orange and the sign on the window is purple. the flowers in the pot are pink and yellow.
![](../../assets/paligemma/fine-tuning-07.jpg)

a plate of colorful desserts and a cup of coffee. the plate features a variety of sweet treats, including a macaron with a bite taken out, a green macaron, a pink macaron, and a white macaron. the plate is adorned with a white flower on a tree branch and a green leaf on a plant. the coffee cup has a white handle. the table is covered in a white and gold tablecloth.
![](../../assets/paligemma/fine-tuning-08.jpg)

cinque terre city perched on the side of a towering cliff, overlooking the serene ocean. the sky above is adorned with fluffy white clouds, while the water below mirrors the sky in its tranquility. the buildings below are painted in vibrant hues, ranging from yellow to orange to pink. the hillside is adorned with lush greenery, and the rocky shore provides a picturesque backdrop. the overall atmosphere is serene and captivating, offering a glimpse into the heart of this idyllic town.

### Define the training and evaluation loops

Define the training loop to train the model on the provided dataset, and the evaluation loop to look at all of the examples in the validation dataset and make its predictions.

#### Defining the training loop

The `update_fn` function defines the training step. During the training step, the loss per example is calculated and stochastic gradient descent (SGD) is applied to the trainable parameters.

Recall that earlier in the notebook, you included flags in the `preprocess_tokens` function that included `mask_loss`. You'll use the `mask_loss` flag here to exclude prefix and padded tokens from the loss. Without it, the loss calculation will be skewed. You also need to normalize each example, since each of them has a different number of tokens. After the prefix and padded tokens have been excluded and the examples have been normalized, you can calculate the loss per example.

The training step also includes a function to apply an SGD to optimize the training.

#### Defining the evaluation loop

The `make_predictions` function is your evaluation loop. The evaluation loop is fairly straight forward with one notable change. If you recall from the beginning of the notebook, you only have 90 examples in your training data set. This is a very small amount of training examples, and your model ends up not having enough examples for the batch size when you run the training. This means that in the evaluation loop, you need to pad the batch by repeating examples.

To make sure that your evaluation loop only counts actual examples and not the padded examples, you have to apply a mask to the padded examples that excludes them from the output.

```python
# The main update_fn using a simple stochastic gradient descent (SGD).
@functools.partial(jax.jit, donate_argnums=(0,))
def update_fn(params, batch, learning_rate):
  imgs, txts, mask_ar = batch["image"], batch["text"], batch["mask_ar"]

  def loss_fn(params):
    text_logits, _ = model.apply({"params": params}, imgs, txts[:, :-1], mask_ar[:, :-1], train=True)
    logp = jax.nn.log_softmax(text_logits, axis=-1)

    # The model takes as input txts[:, :-1] but the loss is defined as predicting
    # next tokens txts[:, 1:]. Additionally, mask_loss[:, 1:] indicates which tokens
    # are part of the loss (e.g. prefix and padded tokens are not included).
    mask_loss = batch["mask_loss"][:, 1:]
    targets = jax.nn.one_hot(txts[:, 1:], text_logits.shape[-1])

    # Compute the loss per example. i.e. the mean of per token pplx.
    # Since each example has a different number of tokens, normalize it.
    token_pplx = jnp.sum(logp * targets, axis=-1)  # sum across vocab_size.
    example_loss = -jnp.sum(token_pplx * mask_loss, axis=-1)  # sum across seq_len.
    example_loss /= jnp.clip(jnp.sum(mask_loss, -1), 1)  # weight by num of tokens.

    # batch_loss: mean of per example loss.
    return jnp.mean(example_loss)

  loss, grads = jax.value_and_grad(loss_fn)(params)

  # Apply gradients to trainable params using SGD.
  def apply_grad(param, gradient, trainable):
    if not trainable: return param
    return param - learning_rate * gradient

  params = jax.tree_util.tree_map(apply_grad, params, grads, trainable_mask)

  return params, loss

# Evaluation/inference loop.
def make_predictions(data_iterator, *, num_examples=None,
                     batch_size=4, seqlen=SEQLEN, sampler="greedy"):
  outputs = []
  while True:
    # Construct a list of examples in the batch.
    examples = []
    try:
      for _ in range(batch_size):
        examples.append(next(data_iterator))
        examples[-1]["_mask"] = np.array(True)  # Indicates true example.
    except StopIteration:
      if len(examples) == 0:
        return outputs

    # Not enough examples to complete a batch. Pad by repeating last example.
    while len(examples) % batch_size:
      examples.append(dict(examples[-1]))
      examples[-1]["_mask"] = np.array(False)  # Indicates padding example.

    # Convert list of examples into a dict of np.arrays and load onto devices.
    batch = jax.tree.map(lambda *x: np.stack(x), *examples)
    batch = big_vision.utils.reshard(batch, data_sharding)

    # Make model predictions
    tokens = decode({"params": params}, batch=batch,
                    max_decode_len=seqlen, sampler=sampler)

    # Fetch model predictions to device and detokenize.
    tokens, mask = jax.device_get((tokens, batch["_mask"]))
    tokens = tokens[mask]  # remove padding examples.
    responses = [postprocess_tokens(t) for t in tokens]

    # Append to html output.
    for example, response in zip(examples, responses):
      outputs.append((example["image"], response))
      if num_examples and len(outputs) >= num_examples:
        return outputs
```

## Tune the model

Now that you've set everything up and taken a look at the training data, it's time to finally tune the model. The code below runs the training loop for the model for 64 steps and prints the learning rate (`lr` in the printed output) and loss rate for each step.

Every 16 steps, the model prints what its predictions are at that step in the training. This code prints out predictions for the same set of images so that you can see the model's ability to predict descriptions improve over time.

At earlier steps in the training, there's likely issues with the descriptions, such as repeated sentences as the model gets stuck in its predictive loop or unfinished sentences. The model's predictions become steadily more accurate as training progresses. By step 64, the model's predictions should closely resemble the descriptions provided by the training data.

This process takes around 15 minutes to complete on T4 TPUs.

```python
# Run a short training loop with cosine learning rate schedule.
#
# Note: the first step can be quite slow on some machines (up to several minutes)
# due to XLA compilation of the jax.jit'd function.
#
%%time

BATCH_SIZE = 8
TRAIN_EXAMPLES = 512
LEARNING_RATE = 0.03

TRAIN_STEPS = TRAIN_EXAMPLES // BATCH_SIZE
EVAL_STEPS = TRAIN_STEPS // 4

train_data_it = train_data_iterator()

sched_fn = big_vision.utils.create_learning_rate_schedule(
    total_steps=TRAIN_STEPS+1, base=LEARNING_RATE,
    decay_type="cosine", warmup_percent=0.10)

for step in range(1, TRAIN_STEPS+1):
  # Make list of N training examples.
  examples = [next(train_data_it) for _ in range(BATCH_SIZE)]

  # Convert list of examples into a dict of np.arrays and load onto devices.
  batch = jax.tree.map(lambda *x: np.stack(x), *examples)
  batch = big_vision.utils.reshard(batch, data_sharding)

  # Training step and report training loss
  learning_rate = sched_fn(step)
  params, loss = update_fn(params, batch, learning_rate)

  loss = jax.device_get(loss)
  print(f"step: {step:2d}/{TRAIN_STEPS:2d}   lr: {learning_rate:.5f}   loss: {loss:.4f}")

  if (step % EVAL_STEPS) == 0:
    print(f"Model predictions at step {step}")
    html_out = ""
    for image, caption in make_predictions(
        validation_data_iterator(), num_examples=4, batch_size=4):
      html_out += render_example(image, caption)
    display(HTML(html_out))
```

``` { .no-copy .output }
step:  1/64   lr: 0.00500   loss: 3.6567
step:  2/64   lr: 0.01000   loss: 1.9762
step:  3/64   lr: 0.01500   loss: 1.6299
step:  4/64   lr: 0.02000   loss: 1.5651
step:  5/64   lr: 0.02500   loss: 1.9813
step:  6/64   lr: 0.03000   loss: 1.9996
step:  7/64   lr: 0.02998   loss: 1.8595
step:  8/64   lr: 0.02992   loss: 1.6479
step:  9/64   lr: 0.02981   loss: 1.3693
step: 10/64   lr: 0.02966   loss: 1.3423
step: 11/64   lr: 0.02947   loss: 1.2122
step: 12/64   lr: 0.02924   loss: 1.0602
step: 13/64   lr: 0.02897   loss: 1.1314
step: 14/64   lr: 0.02866   loss: 1.2612
step: 15/64   lr: 0.02831   loss: 1.0132
step: 16/64   lr: 0.02792   loss: 1.2126
Model predictions at step 16
```
![](../../assets/paligemma/fine-tuning-09.jpg)

a woman's hand rests on a white wall, while her hand is on a white wall. the woman's hand is on the wall, and her hand is on the wall. the woman's hand is on the wall, and her hand is on the wall. the woman's hand is on the wall, and her hand is on the wall. the woman's hand is on the wall, and her hand is on the wall. the woman's hand is on the wall, and her hand is on the wall. the woman's hand is on the wall, and her hand is on the wall.
![](../../assets/paligemma/fine-tuning-10.jpg)

a woman in a white dress with a red rose on it stands on a white wall with a white bag on the wall. the woman is wearing a white bag on the wall and a white bag on the wall. the woman is wearing a white bag on the wall and a white bag on the wall. the woman is wearing a white bag on the wall and a white bag on the wall. the woman is wearing a white bag on the wall and a white bag on the wall. the woman is wearing a white bag on the wall and a white bag on the wall. the woman is wearing a white bag on the wall and a white
![](../../assets/paligemma/fine-tuning-11.jpg)

a person wearing a red blazer with a black belt and a black skirt. the person is wearing a black shirt with a white zipper and a black belt with a white zipper. the person is also wearing a black pants with a white zipper and a black belt with a white zipper. the person is wearing a black shirt with a white zipper and a black belt with a white zipper. the person is wearing a black pants with a white zipper and a black belt with a white zipper. the person is wearing a black shirt with a white zipper and a black belt with a white zipper. the person is wearing a black shirt with a white zipper and
![](../../assets/paligemma/fine-tuning-12.jpg)

a woman wearing a pink shirt and a pink bag with a white belt. the woman is standing on a gray concrete wall with a gray concrete wall behind her. the woman is wearing a white belt and a white belt on her jeans. the woman is wearing a white belt on her jeans and a white belt on her shirt. the woman is wearing a white belt on her jeans and a white belt on her shirt. the woman is wearing a white belt on her jeans and a white belt on her shirt. the woman is wearing a white belt on her jeans and a white belt on her shirt. the woman is wearing a white belt on her

``` { .no-copy .output }
step: 17/64   lr: 0.02750   loss: 1.0986
step: 18/64   lr: 0.02704   loss: 0.9461
step: 19/64   lr: 0.02655   loss: 1.2098
step: 20/64   lr: 0.02602   loss: 1.0513
step: 21/64   lr: 0.02546   loss: 1.0979
step: 22/64   lr: 0.02488   loss: 0.9739
step: 23/64   lr: 0.02426   loss: 0.9589
step: 24/64   lr: 0.02362   loss: 0.7053
step: 25/64   lr: 0.02296   loss: 0.7347
step: 26/64   lr: 0.02227   loss: 0.6990
step: 27/64   lr: 0.02156   loss: 0.6736
step: 28/64   lr: 0.02083   loss: 0.6642
step: 29/64   lr: 0.02009   loss: 0.6908
step: 30/64   lr: 0.01933   loss: 0.7257
step: 31/64   lr: 0.01856   loss: 0.6902
step: 32/64   lr: 0.01778   loss: 0.7054
Model predictions at step 32
```
![](../../assets/paligemma/fine-tuning-13.jpg)

a woman in a pink dress leans against a white wall, her hand on the wall. the dress is flowing and the sleeves are gathered. the woman's hand is on the wall. the dress is pink and the sleeves are gathered. the woman is wearing a bracelet and a watch. the dress is long and the sleeves are long. the woman is standing next to the wall.
![](../../assets/paligemma/fine-tuning-14.jpg)

a woman in a white dress with red flowers stands on a pier overlooking the ocean. the dress is long and has a slit in the front. the woman is wearing a straw hat and has a white bag on her shoulder. the dress has a floral pattern and a tie on the front. the woman is standing on the pier and is holding her dress. the dress is flowing in the wind. the woman is wearing a belt on her dress and has a belt on her bag. the dress has a tie on the front and a floral pattern on the front. the woman is wearing a white bag on her shoulder and a white belt on her
![](../../assets/paligemma/fine-tuning-15.jpg)

a person wears a red blazer with a black belt and a black bag. the bag has a zipper and a silver zipper pull. the person wears a black bra and a black belt. the blazer has a button and a single button. the person wears a black pants and a black bag. the bag has a zipper and a silver zipper pull. the person wears a black belt and a silver zipper pull. the blazer has a single button and a single button. the person wears a black bra and a black belt. the bag has a zipper and a silver zipper pull. the person wears a black pants and a black bag. the bag has
![](../../assets/paligemma/fine-tuning-16.jpg)

a woman stands on a stone staircase, her hand on the railing. her jeans are blue, and her shirt is pink. her bag is pink and has a strap. her bracelet is silver and has a silver buckle. her shoes are silver and have a silver buckle.

``` { .no-copy .output }
step: 33/64   lr: 0.01699   loss: 0.7709
step: 34/64   lr: 0.01620   loss: 0.6653
step: 35/64   lr: 0.01540   loss: 0.3811
step: 36/64   lr: 0.01460   loss: 0.3104
step: 37/64   lr: 0.01380   loss: 0.4042
step: 38/64   lr: 0.01301   loss: 0.3904
step: 39/64   lr: 0.01222   loss: 0.3339
step: 40/64   lr: 0.01144   loss: 0.4156
step: 41/64   lr: 0.01067   loss: 0.4085
step: 42/64   lr: 0.00991   loss: 0.3083
step: 43/64   lr: 0.00917   loss: 0.3757
step: 44/64   lr: 0.00844   loss: 0.3813
step: 45/64   lr: 0.00773   loss: 0.3381
step: 46/64   lr: 0.00704   loss: 0.2057
step: 47/64   lr: 0.00638   loss: 0.1287
step: 48/64   lr: 0.00574   loss: 0.1711
Model predictions at step 48
```
![](../../assets/paligemma/fine-tuning-17.jpg)

a woman in a pink dress leans against a white wall, her hand on the wall. the dress is flowing and the fabric is sheer. her hand is gripping the wall. the wall is white and has a shadow on it. the woman is wearing a bracelet and a watch.
![](../../assets/paligemma/fine-tuning-18.jpg)

a woman in a long white floral dress stands on a pier overlooking the ocean. the dress is long and flowing, with a deep v neckline and a slit in the fabric. the woman holds a white woven bag and wears a straw hat. the sky is clear and blue, with a few clouds. the water is calm and blue, with a few waves. the woman's hair is blonde and her nails are painted white.
![](../../assets/paligemma/fine-tuning-19.jpg)

a person wears a red blazer with a black belt and a black skirt. the blazer has a single button on the front. the person wears a white tank top and has their hand in their pocket. the belt is black and has a silver zipper. the bag is a fanny pack and has a zipper on the front. the bag is black and has a zipper on the front. the person is standing next to a green plant.
![](../../assets/paligemma/fine-tuning-20.jpg)

a woman stands on a stone staircase, her hand on her purse. the jeans are blue, and the shirt is pink. the bag is pink and has a strap. the stairs are made of stone, and the wall is made of concrete. the woman is wearing a white cardigan and has a bracelet on her arm. the bag is on her shoulder, and the strap is on her arm. the jeans are ripped, and the knee is on the bag. the woman is standing on a stone staircase.

``` { .no-copy .output }
step: 49/64   lr: 0.00512   loss: 0.1183
step: 50/64   lr: 0.00454   loss: 0.1154
step: 51/64   lr: 0.00398   loss: 0.1967
step: 52/64   lr: 0.00345   loss: 0.1497
step: 53/64   lr: 0.00296   loss: 0.1688
step: 54/64   lr: 0.00250   loss: 0.1878
step: 55/64   lr: 0.00208   loss: 0.1865
step: 56/64   lr: 0.00169   loss: 0.1655
step: 57/64   lr: 0.00134   loss: 0.0911
step: 58/64   lr: 0.00103   loss: 0.1836
step: 59/64   lr: 0.00076   loss: 0.1242
step: 60/64   lr: 0.00053   loss: 0.0814
step: 61/64   lr: 0.00034   loss: 0.0866
step: 62/64   lr: 0.00019   loss: 0.1295
step: 63/64   lr: 0.00008   loss: 0.1053
step: 64/64   lr: 0.00002   loss: 0.0730
Model predictions at step 64
```
![](../../assets/paligemma/fine-tuning-21.jpg)

a woman in a pink dress leans against a white wall, her hand gripping the edge. the dress is flowing and sheer, with long, puffy sleeves. her hand is gripping the wall, and her fingers are curled. the wall is white and has a shadow on it. the woman is standing outside, and the sun is shining on her.
![](../../assets/paligemma/fine-tuning-22.jpg)

a woman in a long white floral dress stands on a pier overlooking the ocean. the dress is long and flowing, with a deep v neckline and a slit in the fabric. the woman holds a white woven bag and wears a bracelet on her wrist. the sky is clear and blue, with a few clouds. the water is calm and blue, with a few waves. the woman's hair is blonde and her nails are painted white.
![](../../assets/paligemma/fine-tuning-23.jpg)

a person wears a red blazer with a black belt and a black skirt. the blazer has a single button on the front. the person wears a white tank top and has their hand in their pocket. the belt is black and has a silver zipper. the bag is a fanny pack and has a zipper on the front. the bag is black and has a zipper on the front. the person is standing next to a green plant.
![](../../assets/paligemma/fine-tuning-24.jpg)

a woman stands on a stone staircase, her hand on her purse. the jeans are blue, and the fabric is ripped. the shirt is pink, and the buttons are silver. the bag is pink and has a strap. the woman is wearing a white cardigan and a silver bracelet on her left arm. the stairs are made of stone, and the wall is made of concrete. the woman is standing on a step, and the bag is on her arm.

``` { .no-copy .output }
CPU times: user 2min 18s, sys: 8.98 s, total: 2min 27s
Wall time: 15min 45s
```

## Output

The validation data for this notebook consists of just 10 images. In normal code, you would likely have many more data points for validation, but for this notebook, run the following code to generate descriptions for all 10 images. After tuning the model, these descriptions should be very similar in form and content coverage to the descriptions included with the training data that you looked at earlier in this notebook.

Run the below code to generate descriptions for the validation data set.

```python
# The validation data consists of 10 images in a different domain than training
# data.
%%time

print("Model predictions")
html_out = ""
for image, caption in make_predictions(validation_data_iterator(), batch_size=4):
  html_out += render_example(image, caption)
display(HTML(html_out))
```

``` { .no-copy .output }
Model predictions
```
![](../../assets/paligemma/fine-tuning-25.jpg)

a woman in a pink dress leans against a white wall, her hand gripping the edge. the dress is flowing and sheer, with long, puffy sleeves. her hand is gripping the wall, and her fingers are curled. the wall is white and has a shadow on it. the woman is standing outside, and the sun is shining on her.
![](../../assets/paligemma/fine-tuning-26.jpg)

a woman in a long white floral dress stands on a pier overlooking the ocean. the dress is long and flowing, with a deep v neckline and a slit in the fabric. the woman holds a white woven bag and wears a bracelet on her wrist. the sky is clear and blue, with a few clouds. the water is calm and blue, with a few waves. the woman's hair is blonde and her nails are painted white.
![](../../assets/paligemma/fine-tuning-27.jpg)

a person wears a red blazer with a black belt and a black skirt. the blazer has a single button on the front. the person wears a white tank top and has their hand in their pocket. the belt is black and has a silver zipper. the bag is a fanny pack and has a zipper on the front. the bag is black and has a zipper on the front. the person is standing next to a green plant.
![](../../assets/paligemma/fine-tuning-28.jpg)

a woman stands on a stone staircase, her hand on her purse. the jeans are blue, and the fabric is ripped. the shirt is pink, and the buttons are silver. the bag is pink and has a strap. the woman is wearing a white cardigan and a silver bracelet on her left arm. the stairs are made of stone, and the wall is made of concrete. the woman is standing on a step, and the bag is on her arm.
![](../../assets/paligemma/fine-tuning-29.jpg)

a person lies on a white bed, wearing a pink sweater with the words "love will save us" written on the front. the sweater is long-sleeved and has a crew neckline. the person's hand is on the bed, and their shoes are white. the jeans are blue and have a belt loop. the blanket is gray.
![](../../assets/paligemma/fine-tuning-30.jpg)

a man stands against a pink wall, his hand on his face. the man has long blonde hair and a black sweater. the wall is pink.
![](../../assets/paligemma/fine-tuning-31.jpg)

a row of white hangers hang on a white metal bar. the hangers have silver hooks and silver metal bars on them. the bar is white and has a silver metal screw on the end of it. the wall is white and has a gray shadow on it. the hanger on the bar has a silver hook and a silver metal bar on it.
![](../../assets/paligemma/fine-tuning-32.jpg)

a white hoodie and white pants hang on a black metal bar. the hoodie has a white drawstring and a white drawstring on the pants. the pants have a white drawstring on the waistband and a white drawstring on the hem. the hanger on the bar is made of wood. the wall has a black circle on it.
![](../../assets/paligemma/fine-tuning-33.jpg)

a woman stands in front of a gray wall, wearing a pair of blue jeans and a black bag. the bag is black and has a silver chain on the handle. the woman's boots are black and are knee-high. the jeans are blue and have a small hole in them. the woman's hand is on the bag. the wall is gray and the concrete is gray.
![](../../assets/paligemma/fine-tuning-34.jpg)

a man stands on a path, wearing a denim jacket, a white t-shirt, and a pair of brown pants. the jacket is open, and the shirt is tucked in. his shoes are white and his pants are brown. the man has his hands in his pockets, and his eyes are closed. the jacket is on the tree.

``` { .no-copy .output }
CPU times: user 1.87 s, sys: 283 ms, total: 2.15 s
Wall time: 39.3 s
```
