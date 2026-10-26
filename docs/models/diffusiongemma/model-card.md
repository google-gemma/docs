---
title: DiffusionGemma model card
description: DiffusionGemma is a generative model built by Google DeepMind.
models: [diffusiongemma]
source_url: "https://ai.google.dev/gemma/docs/diffusiongemma/model_card.md.txt"
---
# DiffusionGemma model card

[Hugging Face](https://huggingface.co/google/diffusiongemma-26B-A4B-it){ data-model="diffusiongemma" } \|
[GitHub][gemma-github] \|
[Launch Blog](https://blog.google/innovation-and-ai/technology/developers-tools/diffusion-gemma-faster-text-generation/) \|
[Documentation](index.md)


**License** : [Apache 2.0](https://gemma.ai/resources/apache-2) \| **Authors** : [Google DeepMind](https://deepmind.google/models/gemma/)

DiffusionGemma is a generative model built by Google DeepMind. Based on the 26B
A4B Mixture-of-Experts (MoE) Gemma 4 architecture, DiffusionGemma generates
tokens using discrete diffusion. This open-weights model is multimodal, handling
text, image, and video inputs to generate text output.

Built on a MoE foundation, DiffusionGemma is designed to improve generation
speed (tokens per second) while remaining deployable across various hardware
environments. DiffusionGemma builds upon the architectural and capability
advancements of Gemma 4, introducing several core features:

- **Discrete Text Diffusion** -- Shifts from token-by-token autoregression to block-autoregressive multi-canvas sampling. It generates text by iteratively denoising blocks of tokens (a 'canvas') in parallel, significantly increasing decoding speed.
- **Multimodal Input Processing** -- Processes interleaved text, image (with variable aspect ratio and resolution support), and video inputs to generate text outputs.
- **Encoder-Decoder Architecture** -- Utilizes an autoregressive encoder to process and cache the prompt context, paired with a decoder that applies bidirectional attention over the generation canvas.
- **Mixture-of-Experts (MoE) Efficiency** -- Leverages a sparse MoE design (8 active experts out of 128 total) to provide strong reasoning capabilities while maintaining a low memory footprint suitable for local execution.
- **Thinking Mode (Reasoning)** -- Designed as a highly capable reasoner, with configurable thinking modes.
- **Optimized for Small Batch Size Inference** -- Specifically engineered for low-latency, high-speed generation on a single capable accelerator.
- **Native System Prompt Support** -- As with Gemma 4, it supports updating the `system` role, enabling more structured and controllable conversations.

## **Model Overview**

DiffusionGemma is engineered to reduce the sequential bottlenecks of standard
causal language models. It employs an encoder-decoder architecture specifically
optimized for inference speed.

The encoder operates in a prefill capacity, processing the initial prompt and
generating the KV cache. The decoder then utilizes bidirectional attention to
process an input block (a 'canvas') of tokens, accessing the cached context via
cross-attention.

During inference, DiffusionGemma leverages multi-canvas sampling. Rather than
generating one token at a time, the model iteratively denoises a full block of
tokens using a diffusion sampler. Once a canvas is fully denoised, it is
processed by the encoder and appended to the KV cache, after which the model
generates the next canvas. This block-autoregressive approach facilitates text
generation at higher speeds.

### DiffusionGemma

| Property | DiffusionGemma 26B A4B |
| :---- | :---- |
| **Total Parameters** | 25.2B |
| **Active Parameters** | 3.8B |
| **Layers** | 30 |
| **Sliding Window** | 1024 tokens |
| **Context Length** | Up to 256K tokens |
| **Canvas Length** | 256 |
| **Vocabulary Size** | 262K |
| **Expert Count** | 8 active / 128 total and 1 shared |
| **Supported Modalities** | Text, Image |
| **Vision Encoder Parameters** | ~550M |

## **Benchmark Results**

These models were evaluated against a large collection of different datasets and
metrics to cover different aspects of text generation. Evaluation results marked
in the table are for instruction-tuned models, with the recommended Entropy
Bound (EB) sampler (see Best Practices below).

| Benchmark | DiffusionGemma 26B A4B | Gemma 4 26B A4B |
|---|---|---|
| MMLU Pro | 77.6% | 82.6% |
| AIME 2026 no tools | 69.1% | 88.3% |
| LiveCodeBench v6 | 69.1% | 77.1% |
| Codeforces ELO | 1429 | 1718 |
| GPQA Diamond | 73.2% | 82.3% |
| Tau2 (average over 3) | 56.2% | 68.2% |
| HLE no tools | 11.0% | 8.7% |
| HLE with search | 11.9% | 17.2% |
| BigBench Extra Hard | 47.6% | 64.8% |
| MMMLU | 81.5% | 86.3% |
| **Vision** |   |   |
| MMMU Pro | 54.3% | 73.8% |
| OmniDocBench 1.5 (average edit distance, lower is better) | 0.319 | 0.149 |
| MATH-Vision | 70.5% | 82.4% |
| MedXPertQA MM | 49.0% | 58.1% |
| **Long Context** |   |   |
| MRCR v2 8 needle 128k (average) | 32.0% | 44.1% |

## **Core Capabilities**

DiffusionGemma handles a broad range of tasks across text and vision. Key
capabilities include:

- **High-Speed Generation** - parallel denoising of 256 tokens via diffusion sampling achieves low latency by generating 15-20 tokens per forward pass, unlocking per user generation speeds exceeding 1100 tokens per second in low batch size settings (H100, FP8).
- **Adaptive Inference Time Computation** - Simpler prompts and structured tasks like code require fewer denoising steps, enabling dynamic tokens-per-second speeds based on task complexity.
- **Thinking** -- Built-in reasoning mode that lets the model think step-by-step before answering.
- **Long Context** -- Context windows of up to 256K tokens.
- **Image Understanding** -- Object detection, Document/PDF parsing, screen and UI understanding, chart comprehension, OCR (including multilingual), handwriting recognition, and pointing. Images can be processed at variable aspect ratios and resolutions.
- **Video Understanding** -- Analyzes and describes video content by processing sequences of frames.
- **Interleaved Multimodal Input** -- Mix images, video, and text within a single prompt for context-heavy reasoning.
- **Function Calling** -- Native support for structured tool use, enabling agentic workflows.
- **Coding & Reasoning** -- Capable of code generation, completion, and step-by-step logical reasoning.
- **Multilingual** -- Out-of-the-box support for 35+ languages, pre-trained on 140+ languages.

## **Best Practices**

For the best performance, use these configurations and best practices:

### 1. Diffusion Sampling Settings

Use the following standardized sampling configuration across all use cases:

- **Method:** Diffusion sampling with Entropy-Bounded Denoising and Adaptive Stopping.
- **Sampling Configuration:**
    - Maximum number of Denoising Steps = 48
    - Temperature schedule (for logit shaping): Linear decay from 0.8 → 0.4
    - Token Selection: At each step, the sampler selects the lowest-entropy tokens such that their mutual information bound stays below entropy bound = 0.1
    - Token Renoising: The sampler fully renoises the non-selected tokens
- **Adaptive Stopping:** Sampling terminates early if and only if both of the following conditions are met simultaneously:
    - Confident predictions: The average model entropy over the canvas is below the entropy threshold = 0.005
    - Stable predictions: The highest-probability token predictions remain identical across two consecutive denoising steps

### 2. Thinking Mode Configuration

Similar to Gemma 4 models, we use standard `system`, `assistant`, and `user`
roles. To properly manage the thinking process, use the following control
tokens:

- **Trigger Thinking:** Thinking is enabled by including the `<|think|>` token at the start of the system prompt. To disable thinking, remove the token (note that an empty thinking channel might still be emitted).
- **Standard Generation:** When thinking is enabled, the model will output its internal reasoning followed by the final answer using this structure: `<|channel>thought\n`**\[Internal reasoning\]** `<channel|>`.
- **Disabled Thinking Behavior:** If thinking is disabled, the model will still generate the tags but with an empty thought block: `<|channel>thought\n<channel|>`**\[Final answer\]**.

!!! Note
    Note that many libraries like transformers handle the complexities of the chat
    template for you.


### 3. Multi-Turn Conversations

- **No Thinking Content in History:** In multi-turn conversations, the historical model output should only include the final response. Thoughts from previous model turns must not be added before the next user turn begins.

### 4. Modality order

- For optimal performance with multimodal inputs, place image content **before** the text in your prompt.

### 5. Variable Image Resolution

Aside from variable aspect ratios, DiffusionGemma supports variable image
resolution through a configurable visual token budget, which controls how many
tokens are used to represent an image. A higher token budget preserves more
visual detail at the cost of additional compute, while a lower budget enables
faster inference for tasks that don't require fine-grained understanding.

- The supported token budgets are: **70** , **140** , **280** , **560** , and **1120** .
    - Use *lower budgets* for classification, captioning, or video understanding, where faster inference and processing many frames outweigh fine-grained detail.
    - Use *higher budgets* for tasks like OCR, document parsing, or reading small text.

### 6. Video Length

All models support image inputs and can process videos as frames. Video supports
a maximum of 60 seconds assuming the images are processed at one frame per
second.

## **Model Data**

Data used for model training and how the data was processed.

### **Training Dataset**

Our pre-training dataset is a large-scale, diverse collection of data
encompassing a wide range of domains and modalities, which includes web
documents, code, images, audio, with a cutoff date of January 2025. Here are the
key components:

- **Web Documents:** A diverse collection of web text ensures the model is exposed to a broad range of linguistic styles, topics, and vocabulary. The training dataset includes content in over 140 languages.
- **Code:** Exposing the model to code helps it to learn the syntax and patterns of programming languages, which improves its ability to generate code and understand code-related questions.
- **Mathematics:** Training on mathematical text helps the model learn logical reasoning, symbolic representation, and address mathematical queries.
- **Images:** A wide range of images enables the model to perform image analysis and visual data extraction tasks.

The combination of these diverse data sources is crucial for training a powerful
multimodal model that can handle a wide variety of different tasks and data
formats.

### **Data Preprocessing**

Here are the key data cleaning and filtering methods applied to the training
data:

- **CSAM Filtering:** Rigorous CSAM (Child Sexual Abuse Material) filtering was applied at multiple stages in the data preparation process to ensure the exclusion of harmful and illegal content.
- **Sensitive Data Filtering:** As part of making Gemma pre-trained models safe and reliable, automated techniques were used to filter out certain personal information and other sensitive data from training sets.
- **Additional methods:** Filtering based on content quality and safety in line with [our
  policies](https://ai.google/static/documents/ai-responsibility-update-published-february-2025.pdf).

## **Ethics and Safety**

As open models become central to enterprise infrastructure, provenance and
security are paramount. Developed by Google DeepMind, DiffusionGemma undergoes
the same rigorous safety evaluations as our proprietary Gemini models.

### **Evaluation Approach**

DiffusionGemma was developed in partnership with internal safety and responsible
AI teams. A range of automated as well as human evaluations were conducted to
help improve model safety. These evaluations align with [Google's AI
principles](https://ai.google/principles/), as well as safety policies, which
aim to prevent our generative AI models from generating harmful content,
including:

- Content related to child sexual abuse material and exploitation
- Dangerous content (e.g., promoting suicide, or instructing in activities that could cause real-world harm)
- Sexually explicit content
- Hate speech (e.g., dehumanizing members of protected groups)
- Harassment (e.g., encouraging violence against people)

### **Evaluation Results**

For all areas of safety testing, we saw major improvements in all categories of
content safety relative to previous generations of Gemma models. Overall,
DiffusionGemma, like Gemma 4 models, significantly outperforms Gemma 3 and 3n
models in improving safety, while keeping unjustified refusals low. All testing
was intentionally conducted without safety filters to evaluate the model's raw
capabilities and baseline behaviors. For both text-to-text and image-to-text,
and across all model sizes, the model produced minimal policy violations, and
showed significant improvements over previous Gemma models.

## **Usage and Limitations**

These models have certain limitations that users should be aware of.

### **Intended Usage**

Multimodal models (capable of processing vision, language, and/or audio) have a
wide range of applications across various industries and domains. The following
list of potential uses is not comprehensive. The purpose of this list is to
provide contextual information about the possible use-cases that the model
creators considered as part of model training and development.

- **Content Creation and Communication**
    - **Text Generation:** Generates creative text formats such as poems, scripts, code, marketing copy, and email drafts.
    - **Chatbots and Conversational AI:** Powers conversational interfaces for customer service, virtual assistants, or interactive applications.
    - **Text Summarization:** Generates concise summaries of a text corpus, research papers, or reports.
    - **Image Data Extraction:** Extracts, interprets and summarizes visual data for text communications.
- **Research and Education**
    - **Natural Language Processing (NLP) and VLM Research:** Serves as a foundation for researchers to experiment with VLM and NLP techniques, develop algorithms, and contribute to the advancement of the field.
    - **Language Learning Tools:** Supports interactive language learning experiences, aiding in grammar correction or providing writing practice.
    - **Knowledge Exploration:** Assists researchers in exploring large bodies of text by generating summaries or answering questions about specific topics.

### **Limitations**

- **Training Data**
    - The quality and diversity of the training data significantly influence the model's capabilities. Biases or gaps in the training data can lead to limitations in the model's responses.
    - The scope of the training dataset determines the subject areas the model can handle effectively.
- **Context and Task Complexity**
    - The model performs well on tasks that can be framed with clear prompts and instructions. Open-ended or highly complex tasks might be challenging.
    - The model's performance can be influenced by the amount of context provided (longer context generally leads to better outputs, up to a certain point).
- **Language Ambiguity and Nuance**
    - Natural language is inherently complex. The model might struggle to grasp subtle nuances, sarcasm, or figurative language.
- **Factual Accuracy**
    - The model generates responses based on information it learned from their training datasets, but they are not knowledge bases. It may generate incorrect or outdated factual statements.
- **Common Sense**
    - The model relies on statistical patterns in language. It might lack the ability to apply common sense reasoning in certain situations.

### **Ethical Considerations and Risks**

In creating an open, vision-language model, we have carefully considered the
following:

- **Bias and Fairness**
    - VLMs trained on large-scale, real-world text and image data can reflect socio-cultural biases embedded in the training material. DiffusionGemma underwent careful scrutiny, input data pre-processing, and post-training evaluations as reported in this card to help mitigate the risk of these biases.
- **Misinformation and Misuse**
    - VLMs can be misused to generate text that is false, misleading, or harmful.
    - Guidelines are provided for responsible use with the model, see the [Responsible Generative AI Toolkit][responsible-ai].
- **Transparency and Accountability**
    - This model card summarizes details on the model's architecture, capabilities, limitations, and evaluation processes.
    - A responsibly developed open model offers the opportunity to share innovation by making VLM technology accessible to developers and researchers across the AI ecosystem.

**Risks identified and mitigations**:

- **Generation of harmful content:** Mechanisms and guidelines for content safety are essential. Developers are encouraged to exercise caution and implement appropriate content safety safeguards based on their specific product policies and application use cases.
- **Misuse for malicious purposes:** Technical limitations and developer and end-user education can help mitigate against malicious applications of VLMs. Educational resources and reporting mechanisms for users to flag misuse are provided.
- **Privacy violations:** Models were trained on data filtered for removal of certain personal information and other sensitive data. Developers are encouraged to adhere to privacy regulations with privacy-preserving techniques.
- **Perpetuation of biases:** It's encouraged to perform continuous monitoring (using evaluation metrics, human review) and the exploration of de-biasing techniques during model training, fine-tuning, and other use cases.

### **Benefits**

At the time of release, this is a low-latency, high-performance open
vision-language model that provides a compelling option for developers and those
interested in researching diffusion language models. The model is designed from
the ground up for responsible AI development compared to similarly sized models.
