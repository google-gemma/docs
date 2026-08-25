---
title: Gemma 3n model overview
description: Gemma 3n is a generative AI model optimized for use in everyday devices, such as phones, laptops, and tablets.
models: [gemma-3n]
source_url: "https://ai.google.dev/gemma/docs/gemma-3n.md.txt"
---
# Gemma 3n model overview {: #hidden-title style="display: none;" }

<div class="video-embed">
  <iframe src="https://www.youtube-nocookie.com/embed/eJFJRyXEHZ0"
          title="Video" loading="lazy" allowfullscreen
          allow="accelerometer; encrypted-media; gyroscope; picture-in-picture"
          referrerpolicy="strict-origin-when-cross-origin"></iframe>
</div>

Gemma 3n is a generative AI model optimized for use in everyday devices, such as
phones, laptops, and tablets. This model includes innovations in
parameter-efficient processing, including Per-Layer Embedding (PLE) parameter
caching and a MatFormer model architecture that provides the flexibility to
reduce compute and memory requirements. These models feature audio input
handling, as well as text and visual data.

Gemma 3n includes the following key features:

- **Audio input** : Process sound data for speech recognition, translation, and audio data analysis. [Learn more](../../deployment/huggingface.md#generate-text-from-audio-data)
- **Visual and text input** : Multimodal capabilities let you handle vision, sound, and text to help you understand and analyze the world around you. [Learn more](../../deployment/huggingface.md#generate-text-from-image-data)
- **Vision encoder:** High-performance MobileNet-V5 encoder substantially improves speed and accuracy of processing visual data. [Learn more](https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide/#mobilenet-v5:-new-state-of-the-art-vision-encoder){ data-model="gemma-3n" }
- **PLE caching** : Per-Layer Embedding (PLE) parameters contained in these models can be cached to fast, local storage to reduce model memory run costs. [Learn more](index.md#ple-caching)
- **MatFormer architecture:** Matryoshka Transformer architecture allows for selective activation of the models parameters per request to reduce compute cost and response times. [Learn more](index.md#matformer-architecture)
- **Conditional parameter loading:** Bypass loading of vision and audio parameters in the model to reduce the total number of loaded parameters and save memory resources. [Learn more](index.md#conditional-parameter-loading)
- **Wide language support**: Wide linguistic capabilities, trained in over 140 languages.
- **32K token context**: Substantial input context for analyzing data and handling processing tasks.

<div class="cookbook-grid">
<a class="cookbook-card" href="https://www.kaggle.com/models/google/gemma-3n" target="_blank" rel="noopener">
  <div class="cookbook-card-title">Kaggle <span class="cookbook-card-ext">↗</span></div>
  <p class="cookbook-card-desc">Get it on Kaggle</p>
</a>
<a class="cookbook-card" href="https://huggingface.co/collections/google/gemma-3n-685065323f5984ef315c93f4" target="_blank" rel="noopener">
  <div class="cookbook-card-title">Hugging Face <span class="cookbook-card-ext">↗</span></div>
  <p class="cookbook-card-desc">Get it on Hugging Face</p>
</a>
</div>


As with other Gemma models, Gemma 3n is provided with open weights and
licensed for responsible [commercial use][gemma-terms], allowing you to tune
and deploy it in your own projects and applications.

!!! Tip
    If you are interested in building generative AI solutions for Android mobile applications, check out Gemini Nano. For more information, see the Android [Gemini Nano](https://developer.android.com/ai/gemini-nano) developer docs.


## Model parameters and effective parameters

Gemma 3n models are listed with parameter counts, such as **`E2B`** and
**`E4B`** , that are *lower* than the total number of parameters contained in the
models. The **`E`** prefix indicates these models can operate with a reduced set
of Effective parameters. This reduced parameter operation can be achieved using
the flexible parameter technology built into Gemma 3n models to help them run
efficiently on lower resource devices.

The parameters in Gemma 3n models are divided into 4 main groups: text, visual,
audio, and per-layer embedding (PLE) parameters. With standard execution of the
E2B model, over 5 billion parameters are loaded when executing the model.
However, using parameter skipping and PLE caching techniques, this model can be
operated with an effective memory load of just under 2 billion (1.91B)
parameters, as illustrated in Figure 1.

![Gemma 3n diagram of parameter usage](../../assets/gemma-3n/gemma-3n-parameters.png)

**Figure 1.** Gemma 3n E2B model parameters running in standard execution
versus an effectively lower parameter load using PLE caching and parameter
skipping techniques.

Using these parameter offloading and selective activation techniques, you can
run the model with a very lean set of parameters or activate additional
parameters to handle other data types such as visual and audio. These features
enable you to ramp up model functionality or ramp down capabilities based on
device capabilities or task requirements. The following sections explain more
about the parameter efficient techniques available in Gemma 3n models.

## PLE caching

Gemma 3n models include Per-Layer Embedding (PLE) parameters that are used
during model execution to create data that enhances the performance of each
model layer. The PLE data can be generated separately, outside the operating
memory of the model, cached to fast storage, and then added to the model
inference process as each layer runs. This approach allows PLE parameters to be
kept out of the model memory space, reducing resource consumption while still
improving model response quality.

## MatFormer architecture

Gemma 3n models use a Matryoshka Transformer or *MatFormer* model architecture
that contains nested, smaller models within a single, larger model. The nested
sub-models can be used for inferences without activating the parameters of the
enclosing models when responding to requests. This ability to run just the
smaller, core models within a MatFormer model can reduce compute cost, and
response time, and energy footprint for the model. In the case of Gemma 3n, the
E4B model contains the parameters of the E2B model. This architecture also
lets you select parameters and assemble models in intermediate sizes
between 2B and 4B. For more details on this approach, see the
[MatFormer research paper](https://arxiv.org/pdf/2310.07707).
Try using MatFormer techniques to reduce the size of a Gemma 3n model with the
[MatFormer Lab](https://goo.gle/gemma3n-matformer-lab){ data-model="gemma-3n" }
guide.

## Conditional parameter loading

Similar to PLE parameters, you can skip loading of some parameters into memory,
such as audio or visual parameters, in the Gemma 3n model to reduce memory load.
These parameters can be dynamically loaded at runtime if the device has the
required resources. Overall, parameter skipping can further reduce the required
operating memory for a Gemma 3n model, enabling execution on a wider range of
devices and allowing developers to increase resource efficiency for less
demanding tasks.


Ready to start building?
[Get started](../../get-started/index.md)
with Gemma models!
