---
title: Fine-tuning using JAX and Flax
description: Fine-tune RecurrentGemma 2B Instruct for English-French translation using JAX, Flax, and the MTNT noisy-text dataset.
models: [recurrentgemma]
frameworks: [jax]
source_url: "https://ai.google.dev/gemma/docs/recurrentgemma/recurrentgemma_jax_finetune.md.txt"
---
# Fine-tuning using JAX and Flax

This tutorial demonstrates how to fine-tune the [RecurrentGemma](index.md) 2B Instruct model for an English-French translation task using [Google DeepMind's `recurrentgemma` library](https://github.com/google-deepmind/recurrentgemma){ data-model="recurrentgemma" }, [JAX](https://jax.readthedocs.io) (a high-performance numerical computing library), [Flax](https://flax.readthedocs.io) (the JAX-based neural network library), [Chex](https://chex.readthedocs.io/en/latest/) (a library of utilities for writing reliable JAX code), [Optax](https://optax.readthedocs.io/en/latest/) (the JAX-based gradient processing and optimization library), and the [MTNT (Machine Translation of Noisy Text) dataset](https://arxiv.org/abs/1809.00388). Although Flax is not used directly in this notebook, Flax was used to create Gemma.

The `recurrentgemma` library was written with JAX, Flax, [Orbax](https://orbax.readthedocs.io/) (a JAX-based library for training utilities like checkpointing), and [SentencePiece](https://github.com/google/sentencepiece) (a tokenizer/detokenizer library).

This notebook can run on Google Colab with the T4 GPU (go to **Edit** \> **Notebook settings** \> Under **Hardware accelerator** select **T4 GPU**).

## Setup

The following sections explain the steps for preparing a notebook to use a RecurrentGemma model, including model access, getting an API key, and configuring the notebook runtime.

### Set up Kaggle access for Gemma

To complete this tutorial, you first need to follow the setup instructions *similar* to [Gemma setup](../../get-started/index.md) with a few exceptions:

- Get access to RecurrentGemma (instead of Gemma) on [kaggle.com](https://www.kaggle.com/models/google/recurrentgemma){ data-model="recurrentgemma" }.
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

Free Colab hardware acceleration is currently *insufficient* to run this notebook. If you are using [Colab Pay As You Go or Colab Pro](https://colab.research.google.com/signup), click on **Edit** \> **Notebook settings** \> Select **A100 GPU** \> **Save** to enable hardware acceleration.

Next, you need to install the Google DeepMind `recurrentgemma` library from [`github.com/google-deepmind/recurrentgemma`](https://github.com/google-deepmind/recurrentgemma){ data-model="recurrentgemma" }. If you get an error about "pip's dependency resolver", you can usually ignore it.

!!! Note
    By installing `recurrentgemma`, you will also install [`flax`](https://flax.readthedocs.io), core [`jax`](https://jax.readthedocs.io), [`optax`](https://optax.readthedocs.io/en/latest/) (the JAX-based gradient processing and optimization library), [`orbax`](https://orbax.readthedocs.io/), and [`sentencepiece`](https://github.com/google/sentencepiece).


```bash
pip install -q git+https://github.com/google-deepmind/recurrentgemma.git
```

### Import libraries

This notebook uses [Flax](https://flax.readthedocs.io) (for neural networks), core [JAX](https://jax.readthedocs.io), [SentencePiece](https://github.com/google/sentencepiece) (for tokenization), [Chex](https://chex.readthedocs.io/en/latest/) (a library of utilities for writing reliable JAX code), [Optax](https://optax.readthedocs.io/en/latest/) (the gradient processing and optimization library), and TensorFlow Datasets.

```python
import pathlib
from typing import Any, Mapping, Iterator
import enum
import functools

import chex
import jax
import jax.numpy as jnp
import optax

import tensorflow as tf
import tensorflow_datasets as tfds

import sentencepiece as spm

from recurrentgemma import jax as recurrentgemma
```

## Load the RecurrentGemma model

1. Load the RecurrentGemma model with [`kagglehub.model_download`](https://github.com/Kaggle/kagglehub/blob/bddefc718182282882b72f814d407d89e5d178c4/src/kagglehub/models.py#L12), which takes three arguments:

- `handle`: The model handle from Kaggle
- `path`: (Optional string) The local path
- `force_download`: (Optional boolean) Forces to re-download the model

!!! Note
    Be mindful that the RecurrentGemma 2B (IT) model is around 3.85Gb in size.


```python
RECURRENTGEMMA_VARIANT = '2b-it' # @param ['2b', '2b-it'] {type:"string"}

import kagglehub

RECURRENTGEMMA_PATH = kagglehub.model_download(f'google/recurrentgemma/flax/{RECURRENTGEMMA_VARIANT}')
```

``` { .no-copy .output }
Downloading from https://www.kaggle.com/api/v1/models/google/recurrentgemma/flax/2b-it/1/download...
100%|██████████| 3.85G/3.85G [00:50<00:00, 81.5MB/s]
Extracting model files...
```

```python
print('RECURRENTGEMMA_VARIANT:', RECURRENTGEMMA_VARIANT)
```

``` { .no-copy .output }
RECURRENTGEMMA_VARIANT: 2b-it
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

## Load and prepare the MTNT dataset and the Gemma tokenizer

You will use the [MTNT (Machine Translation of Noisy Text)](https://arxiv.org/abs/1809.00388) dataset, which is available from [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/mtnt).

Download the English-to-French dataset portion of the MTNT dataset, and then sample two examples. Each sample in the dataset contains two entries: `src`: the original English sentence; and `dst`: the corresponding French translation.

```python
ds = tfds.load("mtnt/en-fr", split="train")

ds = ds.take(2)
ds = ds.as_numpy_iterator()

for idx, example in enumerate(ds):
  print(f'Example {idx}:')
  for key, val in example.items():
    print(f'{key}: {val}')
  print()
```

``` { .no-copy .output }
Downloading and preparing dataset 35.08 MiB (download: 35.08 MiB, generated: 11.33 MiB, total: 46.41 MiB) to /root/tensorflow_datasets/mtnt/en-fr/1.0.0...
Dl Completed...: 0 url [00:00, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s]
Extraction completed...: 0 file [00:00, ? file/s]
Generating splits...:   0%|          | 0/3 [00:00<?, ? splits/s]
Generating train examples...:   0%|          | 0/35692 [00:00<?, ? examples/s]
Shuffling /root/tensorflow_datasets/mtnt/en-fr/1.0.0.incompleteJLH33K/mtnt-train.tfrecord*...:   0%|          …
Generating test examples...:   0%|          | 0/1020 [00:00<?, ? examples/s]
Shuffling /root/tensorflow_datasets/mtnt/en-fr/1.0.0.incompleteJLH33K/mtnt-test.tfrecord*...:   0%|          |…
Generating valid examples...:   0%|          | 0/811 [00:00<?, ? examples/s]
Shuffling /root/tensorflow_datasets/mtnt/en-fr/1.0.0.incompleteJLH33K/mtnt-valid.tfrecord*...:   0%|          …
Dataset mtnt downloaded and prepared to /root/tensorflow_datasets/mtnt/en-fr/1.0.0. Subsequent calls will reuse this data.
Example 0:
dst: b'Le groupe de " toutes les \xc3\xa9toiles potentielles de la conf\xc3\xa9rence de l\'Est mais qui ne s\'en sortent pas dans le groupe de l\'Ouest ".'
src: b'The group of \xe2\x80\x9ceastern conference potential all stars but not making it in the West\xe2\x80\x9d group.'

Example 1:
dst: b"Kameron est-elle un peu aigrie de son manque de temps \xc3\xa0 l'\xc3\xa9cran ?"
src: b'Is Kameron a Little Salty About Her Lack of Air Time?'
```

Load the Gemma tokenizer, constructed using [`sentencepiece.SentencePieceProcessor`](https://github.com/google/sentencepiece/blob/4d6a1f41069c4636c51a5590f7578a0dbed83450/python/src/sentencepiece/__init__.py#L423):

```python
vocab = spm.SentencePieceProcessor()
vocab.Load(TOKENIZER_PATH)
```

``` { .no-copy .output }
True
```

Customize the[`SentencePieceProcessor`](https://github.com/google/sentencepiece/blob/4d6a1f41069c4636c51a5590f7578a0dbed83450/python/src/sentencepiece/__init__.py#L423) for the English-to-French translation task. Since you will be fine-tuning the English portion of the RecurrentGemma (Griffin) model, you need to make a few adjustments, such as:

- *The input prefix* : Adding a common prefix to each input signals the translation task. For example, you could use a prompt with a prefix like `Translate this into French: [INPUT_SENTENCE]`.

- *The translation start suffix*: Adding a suffix at the end of each prompt instructs the Gemma model exactly when to begin the translation process. A new line should do the job.

- *Language model tokens*: RecurrentGemma (Griffin) models expect a "beginning of sequence" token at the beginning of each sequence. Similarly, you need to add an "end of sequence" token at the end of each training example.

Build a custom wrapper around the `SentencePieceProcessor` as follows:

```python
class GriffinTokenizer:
  """A custom wrapper around a SentencePieceProcessor."""

  def __init__(self, spm_processor: spm.SentencePieceProcessor):
    self._spm_processor = spm_processor

  @property
  def pad_id(self) -> int:
    """Fast access to the pad ID."""
    return self._spm_processor.pad_id()

  def tokenize(
      self,
      example: str | bytes,
      prefix: str = '',
      suffix: str = '',
      add_eos: bool = True,
  ) -> jax.Array:
    """
    A tokenization function.

    Args:
      example: Input string to tokenize.
      prefix:  Prefix to add to the input string.
      suffix:  Suffix to add to the input string.
      add_eos: If True, add an end of sentence token at the end of the output
               sequence.
    Returns:
      Tokens corresponding to the input string.
    """
    int_list = [self._spm_processor.bos_id()]
    int_list.extend(self._spm_processor.EncodeAsIds(prefix + example + suffix))
    if add_eos:
      int_list.append(self._spm_processor.eos_id())

    return jnp.array(int_list, dtype=jnp.int32)

  def tokenize_tf_op(
      self,
      str_tensor: tf.Tensor,
      prefix: str = '',
      suffix: str = '',
      add_eos: bool = True,
  ) -> tf.Tensor:
    """A TensforFlow operator for the `tokenize` function."""
    encoded = tf.numpy_function(
        self.tokenize,
        [str_tensor, prefix, suffix, add_eos],
        tf.int32)
    encoded.set_shape([None])
    return encoded

  def to_string(self, tokens: jax.Array) -> str:
    """Convert an array of tokens to a string."""
    return self._spm_processor.EncodeIds(tokens.tolist())
```

Try it out by instantiating your new custom `GriffinTokenizer`, and then applying it on a small sample of the MTNT dataset:

```python
def tokenize_source(tokenizer, example: tf.Tensor):
  return tokenizer.tokenize_tf_op(
      example,
      prefix='Translate this into French:\n',
      suffix='\n',
      add_eos=False
  )
def tokenize_destination(tokenizer, example: tf.Tensor):
  return tokenizer.tokenize_tf_op(example, add_eos=True)

tokenizer = GriffinTokenizer(vocab)

ds = tfds.load("mtnt/en-fr",split="train")
ds = ds.take(2)
ds = ds.map(lambda x: {
    'src': tokenize_source(tokenizer, x['src']),
    'dst': tokenize_destination(tokenizer, x['dst'])
  })
ds = ds.as_numpy_iterator()

for idx, example in enumerate(ds):
  print(f'Example {idx}:')
  for key, val in example.items():
    print(f'{key}: {val}')
  print()
```

``` { .no-copy .output }
Example 0:
src: [     2  49688    736   1280   6987 235292    108    651   2778    576
   1080 104745  11982   5736    832   8995    901    780   3547    665
    575    573   4589 235369   2778 235265    108]
dst: [     2   2025  29653    581    664  16298   1437  55563  41435   7840
    581    683 111452    581    533 235303   9776   4108   2459    679
    485 235303    479   6728    579   1806   2499    709  29653    581
    533 235303 101323  16054      1]

Example 1:
src: [     2  49688    736   1280   6987 235292    108   2437  87150    477
    476  11709 230461   8045   3636  40268    576   4252   4897 235336
    108]
dst: [     2 213606    477   1455 235290   3510    748   8268 191017   2809
    581   2032  69972    581  11495   1305    533 235303  65978   1654
      1]
```

Build a data loader for the entire MTNT dataset:

```python
@chex.dataclass(frozen=True)
class TrainingInput:
  # Input tokens provided to the model.
  input_tokens: jax.Array

  # A mask that determines which tokens contribute to the target loss
  # calculation.
  target_mask: jax.Array

class DatasetSplit(enum.Enum):
  TRAIN = 'train'
  VALIDATION = 'valid'


class MTNTDatasetBuilder:
  """A data loader for the MTNT dataset."""

  N_ITEMS = {DatasetSplit.TRAIN: 35_692, DatasetSplit.VALIDATION: 811}

  BUFFER_SIZE_SHUFFLE = 10_000
  TRANSLATION_PREFIX = 'Translate this into French:\n'
  TRANSLATION_SUFFIX = '\n'

  def __init__(self,
               tokenizer : GriffinTokenizer,
               max_seq_len: int):
    """A constructor.

    Args:
      tokenizer: The tokenizer to use.
      max_seq_len: The size of each sequence in a given batch.
    """
    self._tokenizer = tokenizer
    self._base_data = {
        DatasetSplit.TRAIN: tfds.load("mtnt/en-fr",split="train"),
        DatasetSplit.VALIDATION: tfds.load("mtnt/en-fr",split="valid"),
    }
    self._max_seq_len = max_seq_len

  def _tokenize_source(self, example: tf.Tensor):
    """A tokenization function for the source."""
    return self._tokenizer.tokenize_tf_op(
        example, prefix=self.TRANSLATION_PREFIX, suffix=self.TRANSLATION_SUFFIX,
        add_eos=False
    )

  def _tokenize_destination(self, example: tf.Tensor):
    """A tokenization function for the French translation."""
    return self._tokenizer.tokenize_tf_op(example, add_eos=True)

  def _pad_up_to_max_len(self,
                         input_tensor: tf.Tensor,
                         pad_value: int | bool,
                         ) -> tf.Tensor:
    """Pad the given tensor up to sequence length of a batch."""
    seq_len = tf.shape(input_tensor)[0]
    to_pad = tf.maximum(self._max_seq_len - seq_len, 0)
    return tf.pad(
        input_tensor, [[0, to_pad]], mode='CONSTANT', constant_values=pad_value,
    )

  def _to_training_input(
      self,
      src_tokens: jax.Array,
      dst_tokens: jax.Array,
  ) -> TrainingInput:
    """Build a training input from a tuple of source and destination tokens."""

    # The input sequence fed to the model is simply the concatenation of the
    # source and the destination.
    tokens = tf.concat([src_tokens, dst_tokens], axis=0)

    # You want to prevent the model from updating based on the source (input)
    # tokens. To achieve this, add a target mask to each input.
    q_mask = tf.zeros_like(src_tokens, dtype=tf.bool)
    a_mask = tf.ones_like(dst_tokens, dtype=tf.bool)
    mask = tf.concat([q_mask, a_mask], axis=0)

    # If the output tokens sequence is smaller than the target sequence size,
    # then pad it with pad tokens.
    tokens = self._pad_up_to_max_len(tokens, self._tokenizer.pad_id)

    # You don't want to perform the backward on the pad tokens.
    mask = self._pad_up_to_max_len(mask, False)

    return TrainingInput(input_tokens=tokens, target_mask=mask)


  def get_train_dataset(self, batch_size: int, num_epochs: int):
    """Build the training dataset."""

    # Tokenize each sample.
    ds = self._base_data[DatasetSplit.TRAIN].map(
        lambda x : (self._tokenize_source(x['src']),
                    self._tokenize_destination(x['dst']))
    )

    # Convert them to training inputs.
    ds = ds.map(lambda x, y: self._to_training_input(x, y))

    # Remove the samples which are too long.
    ds = ds.filter(lambda x: tf.shape(x.input_tokens)[0] <= self._max_seq_len)

    # Shuffle the dataset.
    ds = ds.shuffle(buffer_size=self.BUFFER_SIZE_SHUFFLE)

    # Repeat if necessary.
    ds = ds.repeat(num_epochs)

    # Build batches.
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds

  def get_validation_dataset(self, batch_size: int):
    """Build the validation dataset."""

    # Same as the training dataset, but no shuffling and no repetition
    ds = self._base_data[DatasetSplit.VALIDATION].map(
        lambda x : (self._tokenize_source(x['src']),
                    self._tokenize_destination(x['dst']))
    )
    ds = ds.map(lambda x, y: self._to_training_input(x, y))
    ds = ds.filter(lambda x: tf.shape(x.input_tokens)[0] <= self._max_seq_len)
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds
```

Try the `MTNTDatasetBuilder` out by instantiating the custom `GriffinTokenizer` again, then applying it on the MTNT dataset, and sampling two examples:

```python
dataset_builder = MTNTDatasetBuilder(tokenizer, max_seq_len=20)
ds = dataset_builder.get_train_dataset(3, 1)
ds = ds.take(2)
ds = ds.as_numpy_iterator()

for idx, example in enumerate(ds):
  print(f'Example {idx}:')
  for key, val in example.items():
    print(f'{key}: {val}')
  print()
```

``` { .no-copy .output }
WARNING:tensorflow:Mapping types may not work well with tf.nest. Prefer using MutableMapping for <class '__main__.TrainingInput'>
WARNING:tensorflow:Mapping types may not work well with tf.nest. Prefer using MutableMapping for <class '__main__.TrainingInput'>
WARNING:tensorflow:Mapping types may not work well with tf.nest. Prefer using MutableMapping for <class '__main__.TrainingInput'>
Example 0:
input_tokens: [[     2  49688    736   1280   6987 235292    108  12583    665 235265
     108      2   6151  94975   1320   6238 235265      1      0      0]
 [     2  49688    736   1280   6987 235292    108   4899  29960  11270
  108282 235265    108      2   4899  79025  11270 108282      1      0]
 [     2  49688    736   1280   6987 235292    108  26620 235265    108
       2  26620 235265      1      0      0      0      0      0      0]]
target_mask: [[False False False False False False False False False False False  True
   True  True  True  True  True  True False False]
 [False False False False False False False False False False False False
  False  True  True  True  True  True  True False]
 [False False False False False False False False False False  True  True
   True  True False False False False False False]]

Example 1:
input_tokens: [[     2  49688    736   1280   6987 235292    108    527   5174   1683
  235336    108      2 206790    581  20726    482   2208   1654      1]
 [     2  49688    736   1280   6987 235292    108  28484 235256 235336
     108      2 120500  13832   1654      1      0      0      0      0]
 [     2  49688    736   1280   6987 235292    108 235324 235304   2705
  235265    108      2 235324 235304  19963 235265      1      0      0]]
target_mask: [[False False False False False False False False False False False False
   True  True  True  True  True  True  True  True]
 [False False False False False False False False False False False  True
   True  True  True  True False False False False]
 [False False False False False False False False False False False False
   True  True  True  True  True  True False False]]
```

## Configure the model

Before you begin fine-tuning the Gemma model, you need to configure it.

Load the RecurrentGemma (Griffin) model checkpoint with the [`recurrentgemma.jax.utils.load_parameters`](https://github.com/google-deepmind/recurrentgemma/blob/e4939f9b7edf8baa1d512fb86bfc2e206044d66b/recurrentgemma/jax/utils.py#L31){ data-model="recurrentgemma" } method:

```python
params =  recurrentgemma.load_parameters(CKPT_PATH, "single_device")
```

To automatically load the correct configuration from the RecurrentGemma model checkpoint, use [`recurrentgemma.GriffinConfig.from_flax_params_or_variables`](https://github.com/google-deepmind/recurrentgemma/blob/e4939f9b7edf8baa1d512fb86bfc2e206044d66b/recurrentgemma/common.py#L128){ data-model="recurrentgemma" }:

```python
config = recurrentgemma.GriffinConfig.from_flax_params_or_variables(params)
```

Instantiate the [Griffin](https://arxiv.org/abs/2402.19427) model with [`recurrentgemma.jax.Griffin`](https://github.com/google-deepmind/recurrentgemma/blob/e4939f9b7edf8baa1d512fb86bfc2e206044d66b/recurrentgemma/jax/griffin.py#L29){ data-model="recurrentgemma" }:

```python
model = recurrentgemma.Griffin(config)
```

Create a `sampler` with [`recurrentgemma.jax.Sampler`](https://github.com/google-deepmind/recurrentgemma/blob/e4939f9b7edf8baa1d512fb86bfc2e206044d66b/recurrentgemma/jax/sampler.py#L74){ data-model="recurrentgemma" } on top of the RecurrentGemma model checkpoint/weights and the tokenizer to check if your model can perform translation:

```python
sampler = recurrentgemma.Sampler(model=model, vocab=vocab, params=params)
```

## Fine-tune the model

In this section, you will:

- Use the `gemma.transformer.Transformer` class to create the forward pass and loss function.
- Build the position and attention mask vectors for tokens
- Build a training step function with Flax.
- Build the validation step without the backwards pass.
- Create the training loop.
- Fine-tune the Gemma model.

Define the forward pass and the loss function using the [`recurrentgemma.jax.griffin.Griffin`](https://github.com/google-deepmind/recurrentgemma/blob/e4939f9b7edf8baa1d512fb86bfc2e206044d66b/recurrentgemma/jax/griffin.py#L29){ data-model="recurrentgemma" }
class. The RecurrentGemma `Griffin` inherits from [`flax.linen.Module`](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html), and offers two essential methods:

- `init`: Initializes the model's parameters.
- `apply`: Executes the model's `__call__` function using a given set of parameters.

Since you are working with pre-trained Gemma weights, you don't need to use the `init` function.

```python
def forward_and_loss_fn(
    params,
    *,
    model: recurrentgemma.Griffin,
    input_tokens: jax.Array,            # Shape [B, L]
    input_mask: jax.Array,              # Shape [B, L]
    positions: jax.Array,               # Shape [B, L]
) -> jax.Array:
  """Forward pass and loss function.

  Args:
    params: model's input parameters.
    model: Griffin model to call.
    input_tokens: input tokens sequence, shape [B, L].
    input_mask: tokens to ignore when computing the loss, shape [B, L].
    positions: relative position of each token, shape [B, L].

  Returns:
    Softmax cross-entropy loss for the next-token prediction task.
  """
  batch_size = input_tokens.shape[0]
  # Forward pass on the input data.
  # No attention cache is needed here.
  # Exclude the last step as it does not appear in the targets.
  logits, _ = model.apply(
        {"params": params},
        tokens=input_tokens[:, :-1],
        segment_pos=positions[:, :-1],
        cache=None,
    )

  # Similarly, the first token cannot be predicteds.
  target_tokens = input_tokens[:, 1:]
  target_mask = input_mask[:, 1:]

  # Convert the target labels into one-hot encoded vectors.
  one_hot = jax.nn.one_hot(target_tokens, logits.shape[-1])

  # Don't update on unwanted tokens.
  one_hot = one_hot * target_mask.astype(one_hot.dtype)[...,None]

  # Normalization factor.
  norm_factor = batch_size * (jnp.sum(target_mask) + 1e-8)

  # Return the negative log-likelihood loss (NLL) function.
  return -jnp.sum(jax.nn.log_softmax(logits) * one_hot) / norm_factor
```

Build the `train_step` function that performs the backward pass and updates the model's parameters accordingly, where:

- [`jax.value_and_grad`](https://jax.readthedocs.io/en/latest/_autosummary/jax.value_and_grad.html) is for evaluating the loss function and gradients during the forward and backward passes.
- [`optax.apply_updates`](https://optax.readthedocs.io/en/latest/api/apply_updates.html#optax.apply_updates) is for updating the parameters.

  ```python
  Params = Mapping[str, Any]

  def get_positions(example: jax.Array, pad_id : int) -> jax.Array:
    """Builds the position vector from the given tokens."""
    pad_mask = example != pad_id
    positions = jnp.cumsum(pad_mask, axis=-1)
    # Subtract one for all positions from the first valid one as they are
    # 0-indexed
    positions = positions - (positions >= 1)
    return positions

  @functools.partial(
      jax.jit,
      static_argnames=['model', 'optimizer'],
      donate_argnames=['params', 'opt_state'],
  )
  def train_step(
      model: recurrentgemma.Griffin,
      params: Params,
      optimizer: optax.GradientTransformation,
      opt_state: optax.OptState,
      pad_id: int,
      example: TrainingInput,
  ) -> tuple[jax.Array, Params, optax.OptState]:
    """The train step.

    Args:
      model: The RecurrentGemma (Griffin) model.
      params: The model's input parameters.
      optimizer: The Optax optimizer to use.
      opt_state: The input optimizer's state.
      pad_id: The ID of the pad token.
      example: The input batch.

    Returns:
      Training loss, updated parameters, updated optimizer state.
    """

    positions = get_positions(example.input_tokens, pad_id)

    # Forward and backward passes.
    train_loss, grads = jax.value_and_grad(forward_and_loss_fn)(
        params,
        model=model,
        input_tokens=example.input_tokens,
        input_mask=example.target_mask,
        positions=positions,
    )
    # Update the parameters.
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return train_loss, params, opt_state
  ```

Build the `validation_step` function without the backward pass:

```python
@functools.partial(jax.jit, static_argnames=['model'])
def validation_step(
    model: recurrentgemma.Griffin,
    params: Params,
    pad_id: int,
    example: TrainingInput,
) -> jax.Array:
  return forward_and_loss_fn(
      params,
      model=model,
      input_tokens=example.input_tokens,
      input_mask=example.target_mask,
      positions=get_positions(example.input_tokens, pad_id),
  )
```

Define the training loop:

```python
def train_loop(
    model: recurrentgemma.Griffin,
    params: Params,
    optimizer: optax.GradientTransformation,
    train_ds: Iterator[TrainingInput],
    validation_ds: Iterator[TrainingInput],
    num_steps: int | None = None,
    eval_every_n: int = 20,
):
  opt_state = jax.jit(optimizer.init)(params)

  step_counter = 0
  avg_loss=0

  # The first round of the validation loss.
  n_steps_eval = 0
  eval_loss = 0
  for val_example in validation_ds.as_numpy_iterator():
    eval_loss += validation_step(
        model, params, dataset_builder._tokenizer.pad_id, val_example
    )
    n_steps_eval += 1
  print(f"Start, validation loss: {eval_loss/n_steps_eval}")

  for train_example in train_ds:
    train_loss, params, opt_state = train_step(
        model=model,
        params=params,
        optimizer=optimizer,
        opt_state=opt_state,
        pad_id=dataset_builder._tokenizer.pad_id,
        example=train_example,
    )

    step_counter += 1
    avg_loss += train_loss
    if step_counter % eval_every_n == 0:
      eval_loss = 0

      n_steps_eval = 0
      val_iterator = validation_ds.as_numpy_iterator()
      for val_example in val_iterator:
        eval_loss += validation_step(
            model,
            params,
            dataset_builder._tokenizer.pad_id,
            val_example,
        )
        n_steps_eval +=1
      avg_loss /= eval_every_n
      eval_loss /= n_steps_eval
      print(f"STEP {step_counter} training loss: {avg_loss} - eval loss: {eval_loss}")
      avg_loss=0
    if num_steps is not None and step_counter > num_steps:
      break
  return params
```

Here you have to choose an (Optax) optimizer. For devices with smaller memory, you should use SGD, as it has a much lower memory footprint. To achieve best fine-tuning performance, try Adam-W. The optimal hyperparameters for each optimizer for the particular task in this notebook are provided in this example for the `2b-it` checkpoint.

```python
def griffin_weight_decay_mask(params_like: optax.Params) -> Any:
  # Don't put weight decay on the RGLRU, the embeddings and any biases
  def enable_weight_decay(path: list[Any], _: Any) -> bool:
    # Parameters in the LRU and embedder
    path = [dict_key.key for dict_key in path]
    if 'rg_lru' in path or 'embedder' in path:
      return False
    # All biases and scales
    if path[-1] in ('b', 'scale'):
      return False
    return True

  return jax.tree_util.tree_map_with_path(enable_weight_decay, params_like)

optimizer_choice = "sgd"

if optimizer_choice == "sgd":
  optimizer = optax.sgd(learning_rate=1e-3)
  num_steps = 300
elif optimizer_choice == "adamw":
  optimizer = optax.adamw(
        learning_rate=1e-4,
        b2=0.96,
        eps=1e-8,
        weight_decay=0.1,
        mask=griffin_weight_decay_mask,
    )
  num_steps = 100
else:
  raise ValueError(f"Unknown optimizer: {optimizer_choice}")
```

Prepare the training and validation datasets:

```python
# Choose a small sequence length size, so that everything fits in memory.
num_epochs = 1
batch_size = 1
sequence_length = 32

# Make the dataset builder.
tokenizer = GriffinTokenizer(vocab)
dataset_builder= MTNTDatasetBuilder(tokenizer, sequence_length + 1)

# Build the training dataset.
train_ds = dataset_builder.get_train_dataset(
    batch_size=batch_size,
    num_epochs=num_epochs,
).as_numpy_iterator()

# Build the validation dataset, with a limited number of samples for this demo.
validation_ds = dataset_builder.get_validation_dataset(
    batch_size=batch_size,
).take(50)
```

Begin fine-tuning the RecurrentGemma (Griffin) model on a limited number of steps (`num_steps`):

```python
trained_params = train_loop(
    model=model,
    params=params,
    optimizer=optimizer,
    train_ds=train_ds,
    validation_ds=validation_ds,
    num_steps=num_steps,
)
```

``` { .no-copy .output }
Start, validation loss: 7.894117832183838
/usr/local/lib/python3.10/dist-packages/jax/_src/interpreters/mlir.py:920: UserWarning: Some donated buffers were not usable: ShapedArray(int32[1,33]), ShapedArray(bool[1,33]), ShapedArray(int32[], weak_type=True).
See an explanation at https://jax.readthedocs.io/en/latest/faq.html#buffer_donation.
  warnings.warn("Some donated buffers were not usable:"
STEP 20 training loss: 4.592616081237793 - eval loss: 2.847407102584839
STEP 40 training loss: 2.7537424564361572 - eval loss: 2.9258534908294678
STEP 60 training loss: 2.835618257522583 - eval loss: 2.4382340908050537
STEP 80 training loss: 2.6322107315063477 - eval loss: 2.3696839809417725
STEP 100 training loss: 1.8703256845474243 - eval loss: 2.355681896209717
STEP 120 training loss: 2.7280433177948 - eval loss: 2.4059958457946777
STEP 140 training loss: 2.3047447204589844 - eval loss: 2.083082914352417
STEP 160 training loss: 2.3432137966156006 - eval loss: 2.095074415206909
STEP 180 training loss: 2.1081202030181885 - eval loss: 2.006460189819336
STEP 200 training loss: 2.5359647274017334 - eval loss: 1.9667452573776245
STEP 220 training loss: 2.202195644378662 - eval loss: 1.9440618753433228
STEP 240 training loss: 2.756615400314331 - eval loss: 2.1073737144470215
STEP 260 training loss: 2.5128934383392334 - eval loss: 2.117241859436035
STEP 280 training loss: 2.73045015335083 - eval loss: 1.9159646034240723
STEP 300 training loss: 2.0918595790863037 - eval loss: 1.9742532968521118
```

Both the training loss and the validation loss should have gone down with each step count.

To ensure your input matches the training format, remember to use the prefix `Translate this into French:\n` and a newline character at the end. This signals the model to begin translation.

```python
sampler.params = trained_params
output = sampler(
    ["Translate this into French:\nHello, my name is Morgane.\n"],
    total_generation_steps=100,
)
print(output.text[0])
```

``` { .no-copy .output }
/usr/local/lib/python3.10/dist-packages/jax/_src/interpreters/mlir.py:920: UserWarning: Some donated buffers were not usable: ShapedArray(int32[1,16]).
See an explanation at https://jax.readthedocs.io/en/latest/faq.html#buffer_donation.
  warnings.warn("Some donated buffers were not usable:"
Mais je m'appelle Morgane.
```

## Learn more

- You can learn more about the Google DeepMind [`recurrentgemma` library on GitHub](https://github.com/google-deepmind/recurrentgemma){ data-model="recurrentgemma" }, which contains docstrings of methods and modules you used in this tutorial, such as [`recurrentgemma.jax.load_parameters`](https://github.com/google-deepmind/recurrentgemma/blob/e4939f9b7edf8baa1d512fb86bfc2e206044d66b/recurrentgemma/jax/utils.py#L31){ data-model="recurrentgemma" }, [`recurrentgemma.jax.Griffin`](https://github.com/google-deepmind/recurrentgemma/blob/e4939f9b7edf8baa1d512fb86bfc2e206044d66b/recurrentgemma/jax/griffin.py#L29){ data-model="recurrentgemma" }, and [`recurrentgemma.jax.Sampler`](https://github.com/google-deepmind/recurrentgemma/blob/e4939f9b7edf8baa1d512fb86bfc2e206044d66b/recurrentgemma/jax/sampler.py#L74){ data-model="recurrentgemma" }.
- The following libraries have their own documentation sites: [core JAX](https://jax.readthedocs.io), [Flax](https://flax.readthedocs.io), [Chex](https://chex.readthedocs.io/en/latest/), [Optax](https://optax.readthedocs.io/en/latest/), and [Orbax](https://orbax.readthedocs.io/).
- For `sentencepiece` tokenizer/detokenizer documentation, check out [Google's `sentencepiece` GitHub repo](https://github.com/google/sentencepiece).
- For `kagglehub` documentation, check out `README.md` on [Kaggle's `kagglehub` GitHub repo](https://github.com/Kaggle/kagglehub).
- Learn how to [use Gemma models with Google Cloud Vertex AI](https://cloud.google.com/vertex-ai/docs/generative-ai/open-models/use-gemma).
- If you are using Google Cloud TPUs (v3-8 and newer), make sure to also update to the latest `jax[tpu]` package (`!pip install -U jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html`), restart the runtime, and check that `jax` and `jaxlib` versions match (`!pip list | grep jax`). This can prevent the `RuntimeError` that can arise because of the `jaxlib` and `jax` version mismatch. For more JAX installation instructions, refer to the [JAX docs](https://jax.readthedocs.io/en/latest/tutorials/installation.html#install-google-tpu).
- Check out the [RecurrentGemma: Moving Past Transformers
  for Efficient Open Language Models](https://arxiv.org/pdf/2404.07839) paper by Google DeepMind.
- Read the [Griffin: Mixing Gated Linear Recurrences with
  Local Attention for Efficient Language Models](https://arxiv.org/pdf/2402.19427) paper by Google DeepMind to learn more about the model architecture used by RecurrentGemma.
