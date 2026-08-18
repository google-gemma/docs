---
title: Distributed tuning with Gemma using Keras
description: Gemma is a family of lightweight, state-of-the-art open models built from research and technology used to create Google Gemini models.
models: [gemma-1]
frameworks: [jax, keras]
source_url: "https://ai.google.dev/gemma/docs/core/distributed_tuning.md.txt"
---
# Distributed tuning with Gemma using Keras


Gemma is a family of lightweight, state-of-the-art open models built from research and technology used to create Google Gemini models. Gemma can be further finetuned to suit specific needs. But Large Language Models, such as Gemma, can be very large in size and some of them may not fit on a sing accelerator for finetuning. In this case there are two general approaches for finetuning them:

1. Parameter Efficient Fine-Tuning (PEFT), which seeks to shrink the effective model size by sacrificing some fidelity. LoRA falls in this category and the [Fine-tune Gemma models in Keras using LoRA](keras-lora.md) tutorial demonstrates how to finetune the Gemma 2B model `gemma_2b_en` with LoRA using KerasNLP on a single GPU.
2. Full parameter finetuning with model parallelism. Model parallelism distributes a single model's weights across multiple devices and enables horizontal scaling. You can find out more about distributed training in this [Keras guide](https://keras.io/guides/distribution/).

This tutorial walks you through using Keras with a JAX backend to finetune the Gemma 7B model with LoRA and model-parallism distributed training on Google's Tensor Processing Unit (TPU). Note that LoRA can be turned off in this tutorial for a slower but more accurate full-parameter tuning.

## Using accelerators

Technically you can use either TPU or GPU for this tutorial.

### Notes on TPU environments

Google has 3 products that provide TPUs:

- [Colab](https://colab.sandbox.google.com/) provides TPU v2 for free, which is sufficient for this tutorial.
- [Kaggle](https://www.kaggle.com/) offers TPU v3 for free and they also work for this tutorial.
- [Cloud TPU](https://cloud.google.com/tpu) offers TPU v3 and newer generations. One way to set it up is:
    1. Create a new [TPU VM](https://cloud.google.com/tpu/docs/managing-tpus-tpu-vm#tpu-vms)
    2. Set up [SSH port forwarding](https://cloud.google.com/solutions/connecting-securely#port-forwarding-over-ssh) for your intended Jupyter server port
    3. Install Jupyter and start it on the TPU VM, then connect to Colab through "Connect to a local runtime"

### Notes on multi-GPU setup

Although this tutorial focuses on the TPU use case, you can easily adapt it for your own needs if you have a multi-GPU machine.

If you prefer to work through Colab, it's also possible to provision a multi-GPU VM for Colab directly through "Connect to a custom GCE VM" in the Colab Connect menu.

We will focus on using the **free TPU from Kaggle** here.

## Before you begin

### Kaggle credentials

Gemma models are hosted by Kaggle. To use Gemma, request access on Kaggle:

- Sign in or register at [kaggle.com](https://www.kaggle.com)
- Open the [Gemma model card](https://www.kaggle.com/models/google/gemma) and select *"Request Access"*
- Complete the consent form and accept the terms and conditions

Then, to use the Kaggle API, create an API token:

- Open the [Kaggle settings](https://www.kaggle.com/settings)
- Select *"Create New Token"*
- A `kaggle.json` file is downloaded. It contains your Kaggle credentials

Run the following cell and enter your Kaggle credentials when asked.

```python
# If you are using Kaggle, you don't need to login again.
!pip install ipywidgets
import kagglehub

kagglehub.login()
VBox(children=(HTML(value='<center> <img\nsrc=https://www.kaggle.com/static/images/site-logo.png\nalt=\'Kaggle…
```

An alternative way is to set KAGGLE_USERNAME and KAGGLE_KEY in your environment if kagglehub.login() doesn't work for you.

## Installation

Install Keras and KerasNLP with the Gemma model.

```bash
pip install -q -U keras-nlp
# Work around an import error with tensorflow-hub. The library is not used.
pip install -q -U tensorflow-hub
# Install tensorflow-cpu so tensorflow does not attempt to access the TPU.
pip install -q -U tensorflow-cpu tensorflow-text
# Install keras 3 last. See https://keras.io/getting_started for details.
pip install -q -U keras
```

### Set up Keras JAX backend

Import JAX and run a sanity check on TPU. Kaggle offers TPUv3-8 devices which have 8 TPU cores with 16GB of memory each.

```python
import jax

jax.devices()
[TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0),
 TpuDevice(id=1, process_index=0, coords=(0,0,0), core_on_chip=1),
 TpuDevice(id=2, process_index=0, coords=(1,0,0), core_on_chip=0),
 TpuDevice(id=3, process_index=0, coords=(1,0,0), core_on_chip=1),
 TpuDevice(id=4, process_index=0, coords=(0,1,0), core_on_chip=0),
 TpuDevice(id=5, process_index=0, coords=(0,1,0), core_on_chip=1),
 TpuDevice(id=6, process_index=0, coords=(1,1,0), core_on_chip=0),
 TpuDevice(id=7, process_index=0, coords=(1,1,0), core_on_chip=1)]
```

```python
import os

# The Keras 3 distribution API is only implemented for the JAX backend for now
os.environ["KERAS_BACKEND"] = "jax"
# Pre-allocate 90% of TPU memory to minimize memory fragmentation and allocation
# overhead
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
```

## Load model

```python
import keras
import keras_nlp
```

### Notes on mixed precision training on NVIDIA GPUs

When training on NVIDIA GPUs, mixed precision (`keras.mixed_precision.set_global_policy('mixed_bfloat16')`) can be used to speed up training with minimal effect on training quality. In most case, it is recommended to turn on mixed precision as it saves both memory and time. However, be aware that at small batch sizes, it can inflate memory usage by 1.5x (weights will be loaded twice, at half precision and full precision).

For inference, half-precision (`keras.config.set_floatx("bfloat16")`) will work and save memory while mixed-precision is not applicable.

```python
# Uncomment the line below if you want to enable mixed precision training on GPUs
# keras.mixed_precision.set_global_policy('mixed_bfloat16')
```

To load the model with the weights and tensors distributed across TPUs, first create a new `DeviceMesh`. `DeviceMesh` represents a collection of hardware devices configured for distributed computation and was introduced in Keras 3 as part of the unified distribution API.

The distribution API enables data and model parallelism, allowing for efficient scaling of deep learning models on multiple accelerators and hosts. It leverages the underlying framework (e.g. JAX) to distribute the program and tensors according to the sharding directives through a procedure called single program, multiple data (SPMD) expansion. Check out more details in the new [Keras 3 distribution API guide](https://keras.io/guides/distribution/).

```python
# Create a device mesh with (1, 8) shape so that the weights are sharded across
# all 8 TPUs.
device_mesh = keras.distribution.DeviceMesh(
    (1, 8),
    ["batch", "model"],
    devices=keras.distribution.list_devices())
```

`LayoutMap` from the distribution API specifies how the weights and tensors should be sharded or replicated, using the string keys, for example, `token_embedding/embeddings` below, which are treated like regex to match tensor paths. Matched tensors are sharded with model dimensions (8 TPUs); others will be fully replicated.

```python
model_dim = "model"

layout_map = keras.distribution.LayoutMap(device_mesh)

# Weights that match 'token_embedding/embeddings' will be sharded on 8 TPUs
layout_map["token_embedding/embeddings"] = (model_dim, None)
# Regex to match against the query, key and value matrices in the decoder
# attention layers
layout_map["decoder_block.*attention.*(query|key|value).*kernel"] = (
    model_dim, None, None)

layout_map["decoder_block.*attention_output.*kernel"] = (
    model_dim, None, None)
layout_map["decoder_block.*ffw_gating.*kernel"] = (None, model_dim)
layout_map["decoder_block.*ffw_linear.*kernel"] = (model_dim, None)
```

`ModelParallel` allows you to shard model weights or activation tensors across all devcies on the `DeviceMesh`. In this case, some of the Gemma 7B model weights are sharded across 8 TPU chips according to the `layout_map` defined above. Now load the model in the distributed way.

```python
model_parallel = keras.distribution.ModelParallel(
    layout_map=layout_map, batch_dim_name="batch")

keras.distribution.set_distribution(model_parallel)
gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma_7b_en")
Attaching 'config.json' from model 'keras/gemma/keras/gemma_7b_en/1' to your Kaggle notebook...
Attaching 'config.json' from model 'keras/gemma/keras/gemma_7b_en/1' to your Kaggle notebook...
Attaching 'model.weights.h5' from model 'keras/gemma/keras/gemma_7b_en/1' to your Kaggle notebook...
Attaching 'tokenizer.json' from model 'keras/gemma/keras/gemma_7b_en/1' to your Kaggle notebook...
Attaching 'assets/tokenizer/vocabulary.spm' from model 'keras/gemma/keras/gemma_7b_en/1' to your Kaggle notebook...
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
```

Now verify that the model has been partitioned correctly. Let's take `decoder_block_1` as an example.

```python
decoder_block_1 = gemma_lm.backbone.get_layer('decoder_block_1')
print(type(decoder_block_1))
for variable in decoder_block_1.weights:
  print(f'{variable.path:<58}  {str(variable.shape):<16}  {str(variable.value.sharding.spec)}')
<class 'keras_nlp.src.models.gemma.gemma_decoder_block.GemmaDecoderBlock'>
decoder_block_1/pre_attention_norm/scale                    (3072,)           PartitionSpec(None,)
decoder_block_1/attention/query/kernel                      (16, 3072, 256)   PartitionSpec(None, 'model', None)
decoder_block_1/attention/key/kernel                        (16, 3072, 256)   PartitionSpec(None, 'model', None)
decoder_block_1/attention/value/kernel                      (16, 3072, 256)   PartitionSpec(None, 'model', None)
decoder_block_1/attention/attention_output/kernel           (16, 256, 3072)   PartitionSpec(None, None, 'model')
decoder_block_1/pre_ffw_norm/scale                          (3072,)           PartitionSpec(None,)
decoder_block_1/ffw_gating/kernel                           (3072, 24576)     PartitionSpec('model', None)
decoder_block_1/ffw_gating_2/kernel                         (3072, 24576)     PartitionSpec('model', None)
decoder_block_1/ffw_linear/kernel                           (24576, 3072)     PartitionSpec(None, 'model')
```

## Inference before finetuning

```python
gemma_lm.generate("Best comedy movies in the 90s ", max_length=64)
'Best comedy movies in the 90s 1. The Naked Gun 2½: The Smell of Fear (1991) 2. Wayne’s World (1992) 3. The Naked Gun 33⅓: The Final Insult (1994)'
```

The model generates a list of great comedy movies from the 90s to watch. Now we finetune the Gemma model to change the output style.

## Finetune with IMDB

```python
import tensorflow_datasets as tfds

imdb_train = tfds.load(
    "imdb_reviews",
    split="train",
    as_supervised=True,
    batch_size=2,
)
# Drop labels.
imdb_train = imdb_train.map(lambda x, y: x)

imdb_train.unbatch().take(1).get_single_element().numpy()
Downloading and preparing dataset 80.23 MiB (download: 80.23 MiB, generated: Unknown size, total: 80.23 MiB) to /root/tensorflow_datasets/imdb_reviews/plain_text/1.0.0...
Dl Completed...: 0 url [00:00, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s]
Generating splits...:   0%|          | 0/3 [00:00<?, ? splits/s]
Generating train examples...:   0%|          | 0/25000 [00:00<?, ? examples/s]
Shuffling /root/tensorflow_datasets/imdb_reviews/plain_text/1.0.0.incompleteAJDUZT/imdb_reviews-train.tfrecord…
Generating test examples...:   0%|          | 0/25000 [00:00<?, ? examples/s]
Shuffling /root/tensorflow_datasets/imdb_reviews/plain_text/1.0.0.incompleteAJDUZT/imdb_reviews-test.tfrecord*…
Generating unsupervised examples...:   0%|          | 0/50000 [00:00<?, ? examples/s]
Shuffling /root/tensorflow_datasets/imdb_reviews/plain_text/1.0.0.incompleteAJDUZT/imdb_reviews-unsupervised.t…
Dataset imdb_reviews downloaded and prepared to /root/tensorflow_datasets/imdb_reviews/plain_text/1.0.0. Subsequent calls will reuse this data.
b"This was an absolutely terrible movie. Don't be lured in by Christopher Walken or Michael Ironside. Both are great actors, but this must simply be their worst role in history. Even their great acting could not redeem this movie's ridiculous storyline. This movie is an early nineties US propaganda piece. The most pathetic scenes were those when the Columbian rebels were making their cases for revolutions. Maria Conchita Alonso appeared phony, and her pseudo-love affair with Walken was nothing but a pathetic emotional plug in a movie that was devoid of any real meaning. I am disappointed that there are movies like this, ruining actor's like Christopher Walken's good name. I could barely sit through it."
```

```python
# Use a subset of the dataset for faster training.
imdb_train = imdb_train.take(2000)
```

Perform finetuning using [Low Rank Adaptation](https://arxiv.org/abs/2106.09685) (LoRA). LoRA is a fine-tuning technique which greatly reduces the number of trainable parameters for downstream tasks by freezing the full weights of the model and inserting a smaller number of new trainable weights into the model. Basically LoRA reparameterizes the larger full weight matrices by 2 smaller low-rank matrices AxB to train and this technique makes training much faster and more memory-efficient.

```python
# Enable LoRA for the model and set the LoRA rank to 4.
gemma_lm.backbone.enable_lora(rank=4)

# Fine-tune on the IMDb movie reviews dataset.

# Limit the input sequence length to 128 to control memory usage.
gemma_lm.preprocessor.sequence_length = 128
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
gemma_lm.summary()
gemma_lm.fit(imdb_train, epochs=1)
Preprocessor: "gemma_causal_lm_preprocessor"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Tokenizer (type)                                   ┃                                             Vocab # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ gemma_tokenizer (GemmaTokenizer)                   │                                             256,000 │
└────────────────────────────────────────────────────┴─────────────────────────────────────────────────────┘
Model: "gemma_causal_lm"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                  ┃ Output Shape              ┃         Param # ┃ Connected to               ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ padding_mask (InputLayer)     │ (None, None)              │               0 │ -                          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ token_ids (InputLayer)        │ (None, None)              │               0 │ -                          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ gemma_backbone                │ (None, None, 3072)        │   8,548,748,288 │ padding_mask[0][0],        │
│ (GemmaBackbone)               │                           │                 │ token_ids[0][0]            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ token_embedding               │ (None, None, 256000)      │     786,432,000 │ gemma_backbone[0][0]       │
│ (ReversibleEmbedding)         │                           │                 │                            │
└───────────────────────────────┴───────────────────────────┴─────────────────┴────────────────────────────┘
Total params: 8,548,748,288 (31.85 GB)
Trainable params: 11,067,392 (42.22 MB)
Non-trainable params: 8,537,680,896 (31.81 GB)
/usr/local/lib/python3.10/site-packages/jax/_src/interpreters/mlir.py:756: UserWarning: Some donated buffers were not usable: ShapedArray(float32[256000,384]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,256,384]), ShapedArray(float32[384,24576]), ShapedArray(float32[384,24576]), ShapedArray(float32[24576,384]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,256,384]), ShapedArray(float32[384,24576]), ShapedArray(float32[384,24576]), ShapedArray(float32[24576,384]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,256,384]), ShapedArray(float32[384,24576]), ShapedArray(float32[384,24576]), ShapedArray(float32[24576,384]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,256,384]), ShapedArray(float32[384,24576]), ShapedArray(float32[384,24576]), ShapedArray(float32[24576,384]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,256,384]), ShapedArray(float32[384,24576]), ShapedArray(float32[384,24576]), ShapedArray(float32[24576,384]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,256,384]), ShapedArray(float32[384,24576]), ShapedArray(float32[384,24576]), ShapedArray(float32[24576,384]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,256,384]), ShapedArray(float32[384,24576]), ShapedArray(float32[384,24576]), ShapedArray(float32[24576,384]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,256,384]), ShapedArray(float32[384,24576]), ShapedArray(float32[384,24576]), ShapedArray(float32[24576,384]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,256,384]), ShapedArray(float32[384,24576]), ShapedArray(float32[384,24576]), ShapedArray(float32[24576,384]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,256,384]), ShapedArray(float32[384,24576]), ShapedArray(float32[384,24576]), ShapedArray(float32[24576,384]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,256,384]), ShapedArray(float32[384,24576]), ShapedArray(float32[384,24576]), ShapedArray(float32[24576,384]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,256,384]), ShapedArray(float32[384,24576]), ShapedArray(float32[384,24576]), ShapedArray(float32[24576,384]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,256,384]), ShapedArray(float32[384,24576]), ShapedArray(float32[384,24576]), ShapedArray(float32[24576,384]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,256,384]), ShapedArray(float32[384,24576]), ShapedArray(float32[384,24576]), ShapedArray(float32[24576,384]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,256,384]), ShapedArray(float32[384,24576]), ShapedArray(float32[384,24576]), ShapedArray(float32[24576,384]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,256,384]), ShapedArray(float32[384,24576]), ShapedArray(float32[384,24576]), ShapedArray(float32[24576,384]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,256,384]), ShapedArray(float32[384,24576]), ShapedArray(float32[384,24576]), ShapedArray(float32[24576,384]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,256,384]), ShapedArray(float32[384,24576]), ShapedArray(float32[384,24576]), ShapedArray(float32[24576,384]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,256,384]), ShapedArray(float32[384,24576]), ShapedArray(float32[384,24576]), ShapedArray(float32[24576,384]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,256,384]), ShapedArray(float32[384,24576]), ShapedArray(float32[384,24576]), ShapedArray(float32[24576,384]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,256,384]), ShapedArray(float32[384,24576]), ShapedArray(float32[384,24576]), ShapedArray(float32[24576,384]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,256,384]), ShapedArray(float32[384,24576]), ShapedArray(float32[384,24576]), ShapedArray(float32[24576,384]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,256,384]), ShapedArray(float32[384,24576]), ShapedArray(float32[384,24576]), ShapedArray(float32[24576,384]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,256,384]), ShapedArray(float32[384,24576]), ShapedArray(float32[384,24576]), ShapedArray(float32[24576,384]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,256,384]), ShapedArray(float32[384,24576]), ShapedArray(float32[384,24576]), ShapedArray(float32[24576,384]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,256,384]), ShapedArray(float32[384,24576]), ShapedArray(float32[384,24576]), ShapedArray(float32[24576,384]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,256,384]), ShapedArray(float32[384,24576]), ShapedArray(float32[384,24576]), ShapedArray(float32[24576,384]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,384,256]), ShapedArray(float32[16,256,384]), ShapedArray(float32[384,24576]), ShapedArray(float32[384,24576]), ShapedArray(float32[24576,384]).
See an explanation at https://jax.readthedocs.io/en/latest/faq.html#buffer_donation.
  warnings.warn("Some donated buffers were not usable:"
2000/2000 ━━━━━━━━━━━━━━━━━━━━ 358s 163ms/step - loss: 2.7145 - sparse_categorical_accuracy: 0.4329
<keras.src.callbacks.history.History at 0x7e9cac7f41c0>
```

Note that enabling LoRA reduces the number of trainable parameters significantly, from 7 billion to only 11 million.

## Inference after finetuning

```python
gemma_lm.generate("Best comedy movies in the 90s ", max_length=64)
"Best comedy movies in the 90s \n\nThis is the movie that made me want to be a director. It's a great movie, and it's still funny today. The acting is superb, the writing is excellent, the music is perfect for the movie, and the story is great."
```

After finetuning, the model has learned the style of movie reviews and is now generating output in that style in the context of 90s comedy movies.

## What's next

In this tutorial, you learned how to using KerasNLP JAX backend to finetune a Gemma model on the IMDb dataset in a distributed manner on the powerful TPUs. Here are a few suggestions for what else to learn:

- Learn how to [get started with Keras Gemma](../get-started/index.md).
- Learn how to [finetune the Gemma model on GPU](keras-lora.md).
