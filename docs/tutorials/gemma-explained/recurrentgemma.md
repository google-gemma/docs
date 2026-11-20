---
title: RecurrentGemma Architecture
description: The Griffin architecture mixing local attention and linear recurrences for fast long-sequence inference.
models: [recurrentgemma]
source_url: https://developers.googleblog.com/gemma-explained-recurrentgemma-architecture/
---

# Gemma Explained: RecurrentGemma Architecture

!!! note
    Originally published on the
    [Google Developers Blog](https://developers.googleblog.com/gemma-explained-recurrentgemma-architecture/).

![RecurrentGemma architecture banner](../../assets/gemma-explained/gemma-var-banner.png)

In the [previous post](/tutorials/gemma-explained/gemma-2/) of the Gemma explained series, we discussed the latest Gemma 2 architecture. In this post, you will explore the RecurrentGemma architecture. Let’s get started!

## RecurrentGemma 2B, 9B

RecurrentGemma is based on [Griffin](https://arxiv.org/abs/2402.19427), a hybrid model that mixes gated linear recurrences with local sliding window attention. This change improves computation and memory and it's better suited for long context prompts.

![Griffin hybrid model architecture](../../assets/gemma-explained/image5_REGZE1l.original.png)

However it comes with the downside of reduced needle in haystack performance due to the fixed-sized state of the Griffin architecture. While it is possible to provide the entire text from a book as input, this approach may not be optimal. Recurrent Neural Networks (RNNs) can encounter difficulties in learning long-range dependencies in exceedingly long sequences, and the model has a limited context window. This means that it can only effectively consider a certain number of preceding tokens when making predictions.

Moreover, recurrent models have not yet received as much attention in terms of inference time optimizations compared to their transformer counterparts. And there’s less research and community support available compared to the well-established transformer architecture.

So, this model will be highly valuable in scenarios when you are concerned about exhausting your LLM’s context window. By prioritizing the most recent information and strategically discarding older data, RecurrentGemma ensures that the LLM's performance remains strong as the context expands.

Below is the architecture diagram for the Recurrent Gemma 2B model.

![Recurrent Gemma 2B model architecture](../../assets/gemma-explained/image7_oX4tL8v.original.png)

Griffin follows the same residual pattern and MLP block as other Transformer baseline. However, unlike both the MQA Transformer baseline and the Hawk model, Griffin uses a blend of recurrent and MQA blocks.

![Layered structure of recurrent and MQA blocks](../../assets/gemma-explained/image2_MB1Gg03.original.png)

Griffin uses a layered structure by alternating two residual blocks with a recurrent block, followed by a residual block that incorporates the local MQA attention block.

The [core parameters](https://storage.googleapis.com/deepmind-media/gemma/recurrentgemma-report.pdf) of the architecture are summarized in the table below.

![Core parameters of the architecture of 2B and 9B models](../../assets/gemma-explained/Screenshot_2024-08-21_at_4.29.54PM.original.png)

### **Non-Embedding params and Embedding params**

Non-embedding parameters are distributed throughout the hidden layers of the model, in components like attention mechanisms and feedforward networks.

**Note:** The naming of the model “2B” comes from this parameter

Embedding Parameters are usually found in the dedicated layer called an embedding layer. This layer is responsible for mapping discrete tokens (like words or characters) into continuous vector representations (embeddings).

**Note:** 0.7B can be calculated as 256k (vocabulary size) x 2560 (model width)

### **Model width and RNN width**

Model width refers to the size of the hidden layers in the model, determining the model’s capacity to represent complex patterns, just like the base Gemma Models.

Recurrent neural network (RNN) width is the size of the hidden state maintained by the Real-Gated Linear Recurrent Unit (RG-LRU). Unlike traditional Transformers, the recurrent block maintains a fixed-size internal state, regardless of the input length. This allows RecurrentGemma to process longer sequences with less memory, making it more efficient for tasks like generating long articles or code.

### **MLP expansion factor**

It’s the same as feedforward hidden dimensions in the base Gemma model. For simplicity, we applied an expansion factor of 3 in the Recurrent Gemma model, resulting in an MLP dimension of 7680 (calculated as 2560 x 3).

### **Local attention window size**

The state maintained by RecurrentGemma has a finite size and does not grow with sequences longer than the local attention window of 2k tokens. This means that while the maximum length of samples generated autoregressively by Gemma is limited by the host system's memory capacity, RecurrentGemma can generate sequences of arbitrary length, overcoming this constraint.

```python
RecurrentGemmaForCausalLM(
  (model): RecurrentGemmaModel(
    (embed_tokens): Embedding(256000, 2560, padding_idx=0)
    (layers): ModuleList(
      (0-1): 2 x RecurrentGemmaDecoderLayer(
        (temporal_pre_norm): RecurrentGemmaRMSNorm()
        (temporal_block): RecurrentGemmaRecurrentBlock(
          (linear_y): Linear(in_features=2560, out_features=2560, bias=True)
          (linear_x): Linear(in_features=2560, out_features=2560, bias=True)
          (linear_out): Linear(in_features=2560, out_features=2560, bias=True)
          (conv_1d): Conv1d(2560, 2560, kernel_size=(4,), stride=(1,), padding=(3,), groups=2560)
          (rg_lru): RecurrentGemmaRglru()
          (act_fn): PytorchGELUTanh()
        )
        (channel_pre_norm): RecurrentGemmaRMSNorm()
        (mlp_block): RecurrentGemmaMlp(
          (gate_proj): Linear(in_features=2560, out_features=7680, bias=True)
          (up_proj): Linear(in_features=2560, out_features=7680, bias=True)
          (down_proj): Linear(in_features=7680, out_features=2560, bias=True)
          (act_fn): PytorchGELUTanh()
        )
      )
      (2): RecurrentGemmaDecoderLayer(
        (temporal_pre_norm): RecurrentGemmaRMSNorm()
        (temporal_block): RecurrentGemmaSdpaAttention(
          (q_proj): Linear(in_features=2560, out_features=2560, bias=False)
          (k_proj): Linear(in_features=2560, out_features=256, bias=False)
          (v_proj): Linear(in_features=2560, out_features=256, bias=False)
          (o_proj): Linear(in_features=2560, out_features=2560, bias=True)
          (rotary_emb): RecurrentGemmaRotaryEmbedding()
        )
        (channel_pre_norm): RecurrentGemmaRMSNorm()
        (mlp_block): RecurrentGemmaMlp(
          (gate_proj): Linear(in_features=2560, out_features=7680, bias=True)
          (up_proj): Linear(in_features=2560, out_features=7680, bias=True)
          (down_proj): Linear(in_features=7680, out_features=2560, bias=True)
          (act_fn): PytorchGELUTanh()
        )
      )

      :

      (23): RecurrentGemmaDecoderLayer(
        (temporal_pre_norm): RecurrentGemmaRMSNorm()
        (temporal_block): RecurrentGemmaSdpaAttention(
          (q_proj): Linear(in_features=2560, out_features=2560, bias=False)
          (k_proj): Linear(in_features=2560, out_features=256, bias=False)
          (v_proj): Linear(in_features=2560, out_features=256, bias=False)
          (o_proj): Linear(in_features=2560, out_features=2560, bias=True)
          (rotary_emb): RecurrentGemmaRotaryEmbedding()
        )
        (channel_pre_norm): RecurrentGemmaRMSNorm()
        (mlp_block): RecurrentGemmaMlp(
          (gate_proj): Linear(in_features=2560, out_features=7680, bias=True)
          (up_proj): Linear(in_features=2560, out_features=7680, bias=True)
          (down_proj): Linear(in_features=7680, out_features=2560, bias=True)
          (act_fn): PytorchGELUTanh()
        )
      )
      (24-25): 2 x RecurrentGemmaDecoderLayer(
        (temporal_pre_norm): RecurrentGemmaRMSNorm()
        (temporal_block): RecurrentGemmaRecurrentBlock(
          (linear_y): Linear(in_features=2560, out_features=2560, bias=True)
          (linear_x): Linear(in_features=2560, out_features=2560, bias=True)
          (linear_out): Linear(in_features=2560, out_features=2560, bias=True)
          (conv_1d): Conv1d(2560, 2560, kernel_size=(4,), stride=(1,), padding=(3,), groups=2560)
          (rg_lru): RecurrentGemmaRglru()
          (act_fn): PytorchGELUTanh()
        )
        (channel_pre_norm): RecurrentGemmaRMSNorm()
        (mlp_block): RecurrentGemmaMlp(
          (gate_proj): Linear(in_features=2560, out_features=7680, bias=True)
          (up_proj): Linear(in_features=2560, out_features=7680, bias=True)
          (down_proj): Linear(in_features=7680, out_features=2560, bias=True)
          (act_fn): PytorchGELUTanh()
        )
      )
    )
    (final_norm): RecurrentGemmaRMSNorm()
  )
  (lm_head): Linear(in_features=2560, out_features=256000, bias=False)
)
```

### **embed_tokens (Embedding Layer)**

Takes the input text as a sequence of tokens and maps each token to a continuous vector representation of size 2560. It has a vocabulary size of 256000 which is the same with base Gemma models.

### **layers**

There are 26 decoder layers in total, grouped into repeating patterns.

The model begins with two residual blocks with a recurrent block (0-1). This sequence is then followed by a residual block (2) and a series of continuous blocks that alternate until the end of the layer (25).

![Recurrent block architecture](../../assets/gemma-explained/image4_7W5j3q2.original.png)

#### **Residual block with a recurrent block**

In the recurrent block (Temporal mixing block), the model takes the input of dimension (Model width) 2560 and applies two linear layers with output dimension (RNN width) 2560 in parallel, creating two branches.

On the first branch (right side), it applies a small separable Conv1D layer with a temporal filter dimension of 4. And the RG-LRU(Real-Gated Linear Recurrent Unit) layer follows.

On the second branch (left side), it applies a GeLU nonlinearity.

And then merge the branches by element-wise multiplication, apply a final linear layer with output dimension (Model width) 2560.

![RecurrentGemma-Residual-block](../../assets/gemma-explained/RecurrentGemma-Residual-block.original.png)

After applying RMSNorm, the MLP block follows.

#### **Residual Block with a local MQA**

After having two residual blocks with a recurrent block (0-1), a residual block with a local MQA (2) follows. One of the key disadvantages of using global attention is that its computational complexity grows quadratically in the sequence length. To address this, RecurrentGemma uses a local sliding window attention. It allows each position to attend only to a fixed number of tokens in the past.

In the local MQA block (Temporal mixing block), the model takes the input of dimension (Model width) 2560. It uses linear projections (*q_proj*, *k_proj*, *v_proj*, *o_proj*) to create query, key, value, and output representations. Note that *out_features* for *k_proj* and *v_proj* is 256 as they share the same head with a size of 256, while *q_proj* and *o_proj* have 10 heads (256 x 10 = 2560) in parallel.

It incorporates *rotary_emb* (RecurrentGemmaRotaryEmbedding) for rotary positional embeddings (RoPE) just like the base Gemma models.

Applying RMSNorm and the MLP block is the same with the previous residual block.

## **What’s Next?**

In this article, you learned about RecurrentGemma.

In the next post, you will explore PaliGemma which is a lightweight open vision-language model (VLM).

Stay tuned and thank you for reading!

## 

## References

### **Papers**

- [Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models](https://arxiv.org/abs/2402.19427)
- [RecurrentGemma: Moving Past Transformers for Efficient Open Language Models](https://storage.googleapis.com/deepmind-media/gemma/recurrentgemma-report.pdf)

### **Code Examples**

- [Getting started with RecurrentGemma](https://colab.research.google.com/github/google-deepmind/recurrentgemma/blob/main/colabs/sampling_tutorial_jax.ipynb)

## 📋 The complete Gemma architecture series

- [Gemma explained: An overview of Gemma model family architectures](/tutorials/gemma-explained/overview/)
- [Gemma explained: What’s new in Gemma 2](/tutorials/gemma-explained/gemma-2/)
- Gemma explained: RecurrentGemma architecture
- [Gemma explained: PaliGemma architecture](/tutorials/gemma-explained/paligemma/)
- [Gemma explained: What’s new in Gemma 3](/tutorials/gemma-explained/gemma-3/)
- [Gemma explained: EmbeddingGemma architecture](/tutorials/gemma-explained/embeddinggemma/)
