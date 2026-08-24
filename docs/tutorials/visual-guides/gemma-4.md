---
title: A Visual Guide to Gemma 4
description: A Visual Guide to Gemma 4 — a visual walkthrough.
models: [gemma-4]
source_url: https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-gemma-4
---

# A Visual Guide to Gemma 4

!!! note
    Originally published on
    [Exploring Language Models](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-gemma-4){ data-model="gemma-4" }.

I’m beyond excited to announce the Gemma 4 family of models! It’s a big part of the reason why I joined Google DeepMind. Developing incredible research and sharing it with the world is something that even to this day, is ingrained in the culture of their research teams.

![](https://substackcdn.com/image/fetch/$s_!Vj_W!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F37fe0e0b-90b8-445b-b2a8-0a402605cc18_6208x4980.png)

There are four models in the Gemma 4 family:

- **[Gemma 4 - E2B](https://huggingface.co/google/gemma-4-e2b-it){ data-model="gemma-4" }** – A dense model with per-layer embeddings, which make them effectively 2 billion parameters
- **[Gemma 4 - E4B](https://huggingface.co/google/gemma-4-e4b-it){ data-model="gemma-4" }** – A dense model with per-layer embeddings, which make them effectively 4 billion parameters
- **[Gemma 4 - 31B](https://huggingface.co/google/gemma-4-31b-it){ data-model="gemma-4" }**– A dense model with 31 billion parameters
- **[Gemma 4 - 26B A4B](https://huggingface.co/google/gemma-4-26B-A4B-it){ data-model="gemma-4" }** – A Mixture of Experts (MoE) model with 26 billion total parameters of which 4 billion are activated during inference.

By having a broad range of model sizes, you can choose whichever model best suits your use case and whether it actually fits on your hardware:

![](https://substackcdn.com/image/fetch/$s_!KYBn!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F107a0d7b-d59b-4815-9f50-65dee5123fc9_1282x304.png)

There are two main architectures that are used across these models, namely dense and Mixture-of-Experts.

![](https://substackcdn.com/image/fetch/$s_!S_JH!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F4c6e4126-0b68-4276-bf59-6f3a3341b597_3932x3352.png)

Aside from a wide range of sizes, all models are multimodal and can reason about input images. They were trained to handle many images of varying sizes (more on that later!).

I’m especially interested in trying out the small models a bit more since they not only support input images and text, but also audio.

![](https://substackcdn.com/image/fetch/$s_!vCrp!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F46450e20-c6c9-4e8b-b9c4-2aa941210628_4272x2088.png)

There is A LOT to cover, but before I go into the specifics of the different sizes and architectures, let’s first explore what all models have in common!

*👈 click on the stack of lines on the left to see a **Table of Contents** (ToC)*

## The Gemma 4 Architecture
The Gemma 4 architecture is in many ways like the Gemma 3 series of models, with some changes here and there. There are a number of things that were changed compared to Gemma 3 that relate to all model sizes:

- **Interleaving Layers** – Global attention is always the last layer
- **K=V** – The Keys are set to be equivalent to the Values only for the global attention
- **p-RoPE** – Low-frequency-pruned RoPE applied to the embeddings

Some of the models are a bit different than others (e.g., only small models use Per-Layer Embeddings). Before we go into the specifics of each model, let’s explore in more detail what they have in common.

### Interleaving Layers
Like Gemma 3, Gemma 4 interleaves layers of local attention (also called “sliding window attention”) with global attention (which is regular or “full” attention).

Remember that in global attention, every token attends to all tokens that came before it. Sliding window attention, however, only attends to tokens within a certain limit. This significantly reduces the compute needed to calculate the full attention.

![](https://substackcdn.com/image/fetch/$s_!KjIB!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ffea15481-672a-4fe2-b803-5f4d493998d5_4770x2460.png)

In practice, that means that when text is processed using a sliding window, it may only see a part of the entire sequence rather than the entire thing. The “sliding” then refers to the idea of continuously moving the sequence in view as the number of tokens are being generated.

![](https://substackcdn.com/image/fetch/$s_!MKWU!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F39531efc-71c3-44b6-97aa-546b108fb4d6_4770x1242.png)

You typically fix the number of tokens it can see previously. In the case of Gemma 4 models, the smaller models (E2B and E4B) have a sliding window of 512 tokens and the larger models (26B A4B and 31B) have a sliding window of 1024 tokens.

Let’s go through an example with a sliding window of 4 tokens to see what is happening at each token generation:

![](https://substackcdn.com/image/fetch/$s_!YTt2!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F3015e48f-a122-4af5-b443-a95681039f9c_4770x2082.png)

In our example, there is only attention given to the last four tokens which at some point in the generation starts to “ignore” the ones that came before that. However, it is actually not forgetting the representations that it calculated in the previous steps. The hidden states allow for the attention to be passed along the attention mechanism from previous layers and steps all the way to the current token.

![](https://substackcdn.com/image/fetch/$s_!AsCY!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff26153b1-ec96-4c04-ad84-717bc134b06b_4020x2046.png)

Although information can be propagated by stacking sliding windows, it is not a perfect recall or attention mechanism. Think of it like a game of telephone, information gets diluted each time it passes through another layer!

Therefore, much like in Gemma 3, local attention and global attention layers are interleaved such that the model does attend to the full sequence at times to better capture the global structure.

In Gemma 3, this interleaving was generally in a 4:1 pattern with 4 layers of local attention followed by a single layer of global attention. However, Gemma 3 - 4B for instance had 34 layers, which means its last layer used local attention rather than global attention. This was changed in the Gemma 4 models to make sure that the last layer is always global attention.

![](https://substackcdn.com/image/fetch/$s_!tQBR!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Feb2f917c-31b6-4461-a881-d1ae4a6d0071_4146x2982.png)

The 4:1 pattern, however, is only for the E2B as all other variants have a 5:1 pattern where they start with 5 layers of local attention followed by a single layer of global attention. We can visualize this pattern side-by-side to also demonstrate the depth of these models.

![](https://substackcdn.com/image/fetch/$s_!CRux!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F3253caa8-0499-42c7-beaf-ff1e828d890e_3012x3510.png)

Note that the context window of the local attention layers were reduced from 1024 to 512 tokens to allow for further efficiency gains.

### Making Global Attention more Efficient
Interleaving local attention with global attention is an interesting way of making a Large Language Model more efficient. However, it is not a free lunch. The global attention layers still attend to the entire context, which is a costly and slow process.

In this section, we explore various tricks that Gemma 4 uses to make those global attention layers more efficient!

#### Grouped Query Attention
Much like Gemma 3, these models use Grouped Query Attention (GQA) which allows the Query heads to share KV-values which reduces the amount of caching that needs to be done. The local attention layers use GQA and have 2 Query heads sharing one KV head.

With Gemma 4, the global attention layers are made more efficient by having 8 Query heads to share one KV head. This drastically reduces the caching needed of the KV values since global attention by itself already has a lot it needs to store (the entire context) compared to the small context of the local attention layers.

![](https://substackcdn.com/image/fetch/$s_!NqV3!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fbe2618d8-6c06-4e87-bc30-e865ccab1f15_5790x2760.png)

Note that reducing the number of keys and values per head may hurt performance and so to compensate for that, the size of the Keys was doubled!

Doubling the dimensions of the Keys does fill up the KV-cache quite a bit, so let’s explore another interesting trick for reducing the KV-cache.

#### K=V
Despite the improvements to the grouping in Grouped Query Attention, the global attention layers still take up quite a bit of memory since they attend to the entire sequence. Grouping into 8 queries helps a bit but there is more that can be done for efficiency!

A neat trick, that does not hurt performance that much, is by using the Keys and Values only in the global attention layers. Effectively, this means that all Keys are equivalent to the Values which further reduces the memory requirements for the KV-Cache (or perhaps more accurately now the K-cache for the global attention layer).

![](https://substackcdn.com/image/fetch/$s_!HBUg!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ffc081644-bfea-4956-a9cd-235eb5c93b0f_4100x2060.png)

#### p-RoPE
An important component of any Large Language Model is how it keeps track of the order of words in a sequence. One of the most common techniques is called Rotary Positional Encodings (RoPE). RoPE takes the Query and Key vectors and slices them up into pairs of two values. Each pair can now be seen as a vector in 2-dimensional space pointing towards a direction. RoPE rotates this direction slightly for each pair of values at decreasing speeds. The first pair has a large rotation compared to the last pair. This rotation allows the model to track the relative distances between words.

![](https://substackcdn.com/image/fetch/$s_!m7Qp!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd981478a-907b-403e-aaf4-fb93aa37955d_4456x2544.png)

Let’s zoom in a bit on what happens if we were to rotate a query embedding. It first gets cut up into pairs of 2 and each subsequent pair is rotated. The rotation itself becomes smaller with each pair. This is referred to as the frequency, where a high frequency means that the rotation is quite large whereas a small frequency results in a smaller rotation.

![](https://substackcdn.com/image/fetch/$s_!HT5D!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd2ef73d6-fd46-407f-94fb-1f294ca6528a_4080x2040.png)

High frequency pairs are very sensitive to small changes in position due to the amount of information they were given at the first place (large rotation). They are great for tracking a word’s position. The low frequency pairs, however, are given a very slight rotation and barely move at all from word to word. Since these low frequency pairs only contain minimal positional information, and are closest to what the original KQ values were, they are more suitable for tracking semantic content. As such, they stay the most true to what they originally were (without positional information).

![](https://substackcdn.com/image/fetch/$s_!uS48!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fce6aa947-cd32-47cc-8d45-acc07984a706_5130x1665.png)

However, as it turns out, the high frequency pairs already contain sufficient positional information. The low frequency pairs contain just a little bit of positional information which is too little to be meaningful. It can even be harmful as models tend to use the low frequency pairs for semantic information, so the added positional information is just noise. Moreover, with long contexts the small rotations stack up and can eventually cause tokens that are far apart to become misaligned. This hinders the model’s ability to connect words across long sequences.

An elegant solution to this problem is by simply applying RoPE to only some of the dimensions of all vectors, which is called p-RoPE. If p is 0.25, then only the first 25% of pairs get positional information and all other pairs are set to 0. This allows the low frequency pairs to preserve meaning.

![](https://substackcdn.com/image/fetch/$s_!IFpH!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc204db00-74ba-46c6-9747-b1203bf2e666_4080x2400.png)

Gemma 4 uses p-RoPE on global attention layers since the context window is much higher there compared to the local attention layers. Moreover, p-RoPE is especially useful in global attention layers since large context windows may result in distances between tokens that the model hadn’t seen before. The space of all possible RoPE rotations is quite large with global attention versus local attention and by limiting the number of rotations, the model can do a much better job of handling a large context.

When we put everything together, you get the following improvements applied to the **global attention layer**:

- The final layer is always a global attention layer
- Has groups of 8 Queries per Key
- The dimensionality of the Keys are doubled
- Keys = Values
- p-RoPE where p=0.25

![](https://substackcdn.com/image/fetch/$s_!udLN!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F79da2845-0098-444d-83f5-61a80e464538_4848x2920.png)

### The Vision Encoder
A big component of the Gemma 4 releases are their multimodal capabilities. All variants support image inputs and allow for handling variable aspect ratios and resolutions!

What makes this possible is a Vision Encoder based on the Vision Transformer (ViT) to process the images. A ViT works similarly to a regular Large Language Model and instead of tokenizing text into sequences of words, it splits up the input image into sequences of patches. Each patch (a subset of the input image) is then passed a Transformer model which spits out an embedding per patch.

![](https://substackcdn.com/image/fetch/$s_!N5tr!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fbe6b733c-9d08-4c26-a4ae-a98cb8ac70bc_5352x2760.png)

Each patch is a snippet of the original image and tends to be 16 by 16 pixels. The original paper was aptly named “An image is worth 16x16 words: Transformers for image recognition at scale”.

#### Variable Aspect Ratio
ViT assumes that the original image is a square that can be divided into patches of 16 by 16 pixels. However, there is a downside to this approach. When you process sequences of patches but the original image can be all different kinds of sizes, the position of a given patch (for instance patch 4) can mean different things depending on the proportion of the original image. For instance, if we assume that the original image will always be a grid of 3 by 3 patches, then patch 4 will always be in the middle. However, if the grid is dynamic and can be 2 by 4 patches for instance, then patch 4 will not be in the middle.

![](https://substackcdn.com/image/fetch/$s_!gXv-!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F987e3dda-3d3d-4f3d-80e6-1b3f2fa8ba20_5560x2820.png)

As such, applying RoPE is not straightforward anymore because the meaning of position (“Patch 4”) is now fully dependent on the aspect ratio of the original image.

An interesting technique applied in Gemma 4 is to replace RoPE by 2D RoPE, which attempts to instill the 2D position of a patch into the positional embeddings rather than seeing them as a 1D sequence of patches. To do so, the embedding of a patch is first split up into two equal-sized parts. RoPE is applied to both parts of the embedding independently, but instead of using m as the position, they now use width (w) and height (h) to add positional information. That way, half of the embedding now contains positional information about its relative width position and the other half contains positional information about its relative height position.

![](https://substackcdn.com/image/fetch/$s_!YBFT!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F1991ddef-1083-401f-b419-33caa3aaed60_6440x2320.png)

Using 2D RoPE is not all that is needed to support variable aspect ratios. The original ViT, for instance, resizes the input to make them into squares so that the processed input nicely fits patches of 16 by 16 pixels. Imagine you have a wide image and you want to force it into a square… that would look either quite weird or crop away much of its content!

Gemma 4 introduces support for different aspect ratios by adaptively resizing the input image so that it still supports patches of 16 by 16 pixels. To maintain the original aspect ratio it will pad the image whenever there couldn’t be a perfect fit of a 16 by 16 pixel box.

![](https://substackcdn.com/image/fetch/$s_!1Us2!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F08bd6389-c503-4c44-8e3a-2e349195d989_4352x2784.png)

The original ViT typically assumed a fixed size for the square and therefore would always get back a fixed number of patches (9 in our example). The adaptive resizing method returns a variable number of patches, which can grow quickly in size. Therefore, the tokens generated by the ViT are pooled based on their spatial location. Patch embeddings close to each other get merged until a fixed number of patch embeddings are left.

#### Variable Resolution
Although variable aspect ratios solve a bunch of problems, how do we decide the number of patches to use? Two images with the same aspect ratio can have different resolutions. Imagine you have two images with both an aspect ratio of 3:2 but one has a lower resolution of 272 by 176 and the other a higher resolution of 544 by 352.

One method would be to always decide on a certain fixed number of patches and resize the image accordingly. However, for certain tasks we might want to support higher resolutions, such as object detection and segmentation whilst for video you might be alright with smaller resolutions to speed up analyzing subsequent frames.

![](https://substackcdn.com/image/fetch/$s_!Ydmv!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F63325403-8aec-4269-86ed-cee314e78d76_5600x2180.png)

Gemma 4 supports different resolutions by introducing a soft token budget. This budget represents the maximum number of patch embeddings (also called soft tokens or visual tokens) are passed to the LLM to process. The user can decide between budget sizes of 70, 140, 280, 560, or 1120 tokens.

Depending on the budget, the input is resized. If you have a higher budget (like 1120 tokens), then your image can maintain a higher resolution and as a result will have many more patches to process. If you have a lower budget (like 70 tokens), then your image needs to be downscaled and you will have fewer patches that need to be processed. With a higher budget (and therefore more tokens), you can capture much more information than with a lower budget.

This budget determines how much the image is resized. Imagine you have a budget of 280 tokens, then the maximum number of patches will be 9 x 280 = 2,520. Why times 9? That’s because in the next step, every 3x3 block of neighboring patches are merged into a single embedding by averaging them.

The budget sizes roughly represent the following resolutions:

![](https://substackcdn.com/image/fetch/$s_!KwmX!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F63b9b1b6-b137-4d52-9eb3-681cea2eba69_1323x496.png)

For simplicity, this assumes a square but with variable aspect ratio it can take different ratios and as a result, the patches will also have different dimensions. Also note how the soft tokens are smaller than the actual number of pooled embeddings, that is because the resolution should be multiples of 48 considering 3 patches of 16 pixels are pooled to a single embedding.

Below is an example of how an initially large number of patches gets averaged (also called pooling) into a lower number of patch embeddings (also called soft tokens).

![](https://substackcdn.com/image/fetch/$s_!Zl1j!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F46207f33-328a-40aa-ac72-61faab484749_3216x3008.png)

The pooled patches are typically smaller than the token budget because not every image will perfectly capture the maximum number of patches.

#### Linear Projection
The patch embeddings generated as a result of the image encoder cannot be given directly to Gemma 4. Like all language models, it expects the input embeddings to have a certain dimensionality to be able to process it. Ideally, you would also want its embeddings to occupy dimensional space in a similar way as the token embeddings such that they can be compared easily.

There is a nice example in the image below where you can see how a given image encoder creates an embedding that is much different from the token embedding created by the text encoder.

![](https://substackcdn.com/image/fetch/$s_!z6fD!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F61ce0dd5-88fb-421b-9718-9cd5f1de1dc5_5640x4000.png)

To make sure that the patch embeddings are aligned correctly with the token embedding the patch embeddings are typically projected using a small neural network so that they have the same dimensions and value distributions as what the language model expects. In Gemma 4, this projection is then followed by RMSNorm to match scale expectations of subsequent Transformer blocks.

![](https://substackcdn.com/image/fetch/$s_!RKWU!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F20e856f9-a1a4-4c79-8f6c-9ee0ece5a361_4656x3928.png)

And that’s how images are processed by Gemma 4! Note that to perform this linear projection, it is trained alongside Gemma 4 to make sure that the projected patch embeddings closely match what Gemma 4 expects.

Moreover, the smaller models (E2B and E4B) all use a vision encoder with 150 million parameters and all other models use a vision encoder with 550 million parameters.

### Putting Everything Together
When we put everything together, the Gemma 4 architecture that all variants share, can be summarized into three major aspects:

- Interleaving Layers of Local and Global Attention
- Either Dense or Mixture of Experts
- Vision encoder

![](https://substackcdn.com/image/fetch/$s_!0-CF!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fecba0193-890d-4f66-ae5a-e6bfdb683841_2424x2672.png)

Note that the feedforward neural networks (FFNNs) are not interleaved with Mixture of Experts (MoE). Any Gemma 4 variants will have either one in all layers or the other.

Now that we have covered the main principles of each variant, let’s go into each set of models to explore them a bit more in-depth:

- Gemma 4 - **[31B](https://huggingface.co/google/gemma-4-31B-it){ data-model="gemma-4" }**
- Gemma 4 - **[26B A4B](https://huggingface.co/google/gemma-4-26B-A4B-it){ data-model="gemma-4" }**
- Gemma 4 - **[E2B](https://huggingface.co/google/gemma-4-E2B-it){ data-model="gemma-4" }** and **[E4B](https://huggingface.co/google/gemma-4-E4B-it){ data-model="gemma-4" }**

## Gemma 4 - 31B - Dense
Despite the popularity of Mixture of Experts (which we cover later), the Gemma 4 family has a dense variant that is quite large with 31B parameters. This model is a nice representation of a more “vanilla” architecture amongst the Gemma 4 variants.

This model is architecturally quite similar to its Gemma 3 counterpart, namely the 27B variant. They both interleave local and global attention and have the same pre- and post- RMSNorm. What it does differently are the aspects we discussed before, like K=V and using P-RoPE. In particular, Gemma 4 - 31B has fewer layers (60 vs 62) but is a wider model.

![](https://substackcdn.com/image/fetch/$s_!cc62!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fbe1a0781-472e-45d8-96f4-8f6cc9911c19_3000x4488.png)

Unlike the Mixture of Experts model and the per-layer embeddings of the tiny models, this variant does not have a “magic” ingredient for us to explore. I think it is a nice representation of the “vanilla” Gemma 4 architecture and allows us to build on this to explore the other models.

## Gemma 4 - 26B A4B - Mixture of Experts
Gemma 4 has a variant that contains “A” in its name, namely the 26 A4B model. The “A” stands for “active parameters” in contrast to the total number of parameters they contain. Specifically, the 26B A4B model contains in total 26 billion parameters (which are all loaded in memory) but only 4 billion parameters are used during inference. These are referred to as the “active parameters”. By only activating a subset of parameters, these models run much faster than their total number of parameters might suggest.

![](https://substackcdn.com/image/fetch/$s_!6B9e!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F8869694f-06de-4318-bd3a-c9359a3be1ae_4000x6128.png)

The architecture that makes this possible is called Mixture of Experts or MoE. Instead of using one big Feedforward Neural Network (FFNN), it uses several smaller ones and dynamically chooses which ones to process a given input.

As such, there are two main components to a MoE:

- **Experts** — A set of FFNNs that can process tokens but only a subset is activated at a time. They tend to be smaller than the FFNN in a dense (“regular”) model.
- **Router** — Determines which tokens are processed by which experts.

A given token embedding is processed in the MoE layer by first passing to the router which selects one or more experts to activate. To do so, expert probabilities are generated which allows for routing the token embedding. The selected experts update the token embedding as a Feedforward Neural Network normally would.

The probabilities allow for some experts to have a larger or smaller influence on the processed embeddings compared to other experts. As such, the processed embedding is multiplied by the expert probability.

![](https://substackcdn.com/image/fetch/$s_!CU2S!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F32e57db5-ea63-4a3e-a193-c430708f2da8_4704x4744.png)

Although the visual shows a single expert, most MoE activate at least several and combine the result. In Gemma 4, the MoE variant has 128 experts that the router can choose from and 8 will be activated at a time.

Moreover, there is a **shared expert**. This expert will always be activated and used to process the input embedding. The underlying idea of having a shared expert is that it contains a lot of general knowledge that should always be activated, whereas the selected experts contain more fine-grained knowledge. In Gemma 4, this shared expert is three times the size to make sure it encompasses the necessary general knowledge.

![](https://substackcdn.com/image/fetch/$s_!Rzhe!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb7b3eb2f-f178-43aa-902c-cd66ace3a131_4704x5272.png)

Now that you have a big picture of Mixture of Experts in Gemma 4, let’s explore a bit more in-depth what it means to have an efficient architecture with sparse and active parameters. The **sparse parameters** are every single parameter of the model which needs to be loaded into memory when you want to load the model. Gemma 4’s 26B billion parameter model for instance is quite large and you might think that it will run slow because of its big size. However, not all of its sparse parameters are actually used by your device (e.g., GPU) when you are running the model. With Mixture of Experts in Gemma 4, only 8 experts and 1 shared expert is actually used for intermediate calculations. All other 119 experts can take a backseat. These are the active parameters and represent the “**A**” in “26B **A**4B”. This means that although this model is large, it runs almost as fast as a 4B parameter model!

![](https://substackcdn.com/image/fetch/$s_!txYk!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Faf9da40e-432c-481c-8d1d-3699896517f1_3660x3870.png)

I’ve had a blast trying this model and I can tell you that it is a quite capable to run. However, you might want to use a smaller model if you want to run it on device like on your phone.

## Gemma 4 - E2B & E4B - Dense
Gemma 4 has a variant that contains “E” in its name, namely the E2B and E4B model. The “E” stands for “effective parameters” in contrast to the total number of parameters they contain.

![](https://substackcdn.com/image/fetch/$s_!X3cx!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ffaff88e2-4cd6-4669-8211-7d5b6039559c_5082x5160.png)

These models are very efficient and great for on-device use cases. What makes these models especially exciting is that they contain an additional audio encoder for processing audio alongside text and images.

Before we go into the specifics of the audio encoder, let’s first explore this “E”!

### Per-Layer Embeddings
An interesting component of the smaller Gemma 4 models is the “E” in “E2B” and “E4B”. It stands for “effective” and is in a way similar to sparse versus active parameters that we explored in Mixture of Experts. “Effective” essentially means that these parameters are used for computation and loaded on to your (V)RAM whereas the other parameters (a lookup table with embeddings) is used only once to quickly find related embeddings.

To explore what this means in practice, let’s first cover what happens at the token embedding layer. Whenever it receives a given token, it looks up its embedding using a lookup table. This table can be quite large since it has to store an embedding for each word in its entire vocabulary. Gemma 4 E2B, for instance, has 262,144 tokens each with an embedding size of 1536 dimensions.

![](https://substackcdn.com/image/fetch/$s_!i3rO!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fea5dd72c-ec58-4da9-94b6-fcc50f46210d_7790x2870.png)

Then, the embeddings for each token gets put through stacks of decoder blocks and iteratively processed. To improve the capabilities of a model, you would normally just add a couple of layers or more parameters. Bigger models tend to outperform smaller ones.

Instead of adding parameters to the model itself, a clever trick is to create an additional set of embeddings at every single layer. It is a bit simplified but imagine that embedding for the token “cat” might emphasize at:

- Layer 2 - “I am a noun, usually followed by a verb.”
- Layer 18 - “I am a small animal and related to terms like ‘pet’ or ‘feline’.”

These embeddings are quite a bit smaller (256 versus 1536 dimensions in E2B and 2056 in E4B) than the original lookup table to save storage. As a result, we now have a lookup table of 262,144 tokens x 256 dimensions x N layers. The benefit of having a larger lookup table is that we can store important information on flash storage rather than needing valuable (V)RAM which is meant for computations in the model.

These are called Per-Layer Embeddings (PLE) and as the name suggests, are embeddings that used for specific layers.

![](https://substackcdn.com/image/fetch/$s_!0AhS!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fe6c18411-5f64-4843-8809-9aa581b4ee7f_4130x3810.png)

At the start of inference, the model grabs the set of embeddings for each input token. Each input token will have an embedding per layer to be used at that specific layer. Note that this lookup is done only once during inference, making this action quite compute efficient since there is no need to lookup the embeddings every time a layer is activated.

![](https://substackcdn.com/image/fetch/$s_!nf1_!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa820d8bb-a7c8-4de1-b350-f74160c5634b_4130x2590.png)

These embeddings are processed between each set of decoder blocks. Here, are gating function is used to decide how to weigh each value in a chosen embedding. This allows the model to additionally focus on certain parts of the embedding retrieved from the lookup table. The resulting embedding, still of dimensionality 256, is then projected up to match the original embedding size (1,536 for E2B and 2,560 for E4B).

After a normalization layer, this weighted representation is then combined with the original output of the previous decoder block. This allows for the processed signal to be “reminded” of what the token embedding represents, rather than it being mixed through a bunch of layers and getting a lot of context added to it.

![](https://substackcdn.com/image/fetch/$s_!otr7!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F2e78b43b-e615-4ba0-a3de-7361138b28e2_5560x8280.png)

The model can now focus its internal dimensions on making sense of the tokens rather than needing to carry around information from the bottom layer. This makes the existing layers much more efficient and expressive than they would be on their own.

But more importantly, these per-layer embeddings, albeit being quite large (262,144 x 35 x 256) are stored in flash memory. Storing it in (V)RAM would be a waste considering we only need a couple of embeddings during inference. Therefore, the “**E**” in “E2B” and “E4B” relate to all parameters except for the per-layer embeddings.

This is an elegant technique that allows these models to run on smaller devices, such as cellphones. There, the RAM is extremely valuable and limited, so anything you can do to minimize its usage is a welcome addition.

### Audio Encoder
The smaller models (E2B and E4B) can also handle audio inputs and are nicely suited for tasks like automatic speech recognition and translation. Much like the vision encoder, the audio encoder transforms the raw audio input into embeddings that can be processed by Gemma 4.

There are several steps to this process but let’s start at the beginning. A common process in converting raw audio into tokens and their embeddings is through a three step process:

1. **Extract features** – Features are extracted from the raw audio through a mel-spectogram which is a 2D visual representation of the raw audio. It represents time (horizontal axis) and frequency bands (vertical axis).
1. **Group into chunks** – The mel features are grouped into chunks as a starting point for the sequence of tokens.
1. **Downsample chunks** - The chunks are overlapped and processed by two 2-dimensional convolutional layers to shorten the sequence length. As such, it converts the 2D chunks into a sequence of embeddings (also called “soft” tokens to represent the continuous nature of the token).

![](https://substackcdn.com/image/fetch/$s_!fjgL!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F93de9d19-38b6-4487-964c-1faa79971366_1856x1660.png)Fun fact: I recorded audio myself and created the mel spectogram for this visual!

This process is much like the linear projection of the image patches to create embeddings. They still need to be processed subsequently by an encoder so that the embeddings are filled with contextual information. The audio encoder used in Gemma 4 is called a conformer and is much like a regular Transformer Encoder but also uses a convolutional module to process the soft tokens. A nice comparison is between the dense Gemma 4 31B and how the Conformer is implemented. Note how it produces embeddings rather than tokens.

![](https://substackcdn.com/image/fetch/$s_!ZU5p!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F7f598397-69e1-4185-bba6-ebe9881c9e7d_2000x2992.png)

Much like the vision encoder, the embeddings produced by the Conformer are projected onto the dimensional space of the embeddings that Gemma 4 would expect. Otherwise, you get a mismatch in embedding size which cannot be processed by Gemma 4.

![](https://substackcdn.com/image/fetch/$s_!KsrJ!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F0f6fd442-0913-4538-b49a-262025b7aa4e_5820x3990.png)

This gives us a pipeline that feels quite similar to what we explored with the vision encoder!

## What's next

<div class="card-grid">
<a class="card" href="../../get-started/index.md">
  <span class="card-icon">{{ icon('lucide/rocket') }}</span>
  <div class="card-title">Quickstart</div>
  <p class="card-desc">Run your first Gemma 4 generation in one command.</p>
</a>
<a class="card" href="../../models/gemma-4/index.md">
  <span class="card-icon">{{ icon('lucide/boxes') }}</span>
  <div class="card-title">Gemma 4</div>
  <p class="card-desc">Sizes, architectures, context windows and memory.</p>
</a>
<a class="card" href="mtp.md">
  <span class="card-icon">{{ icon('lucide/book-open') }}</span>
  <div class="card-title">A Visual Guide to MTP</div>
  <p class="card-desc">Drafter models and speculative decoding, illustrated.</p>
</a>
</div>
