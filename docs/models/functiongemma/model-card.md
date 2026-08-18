---
title: FunctionGemma model card
description: "Model card for FunctionGemma: function-calling inputs and outputs, training data, evaluation results, and intended use."
models: [functiongemma]
source_url: "https://ai.google.dev/gemma/docs/functiongemma/model_card.md.txt"
---
# Model card

**Model Page** : [FunctionGemma](index.md)

**Resources and Technical Documentation**:

- [Responsible Generative AI Toolkit](https://ai.google.dev/responsible)
- [FunctionGemma on Kaggle](https://www.kaggle.com/models/google/functiongemma/){ data-model="functiongemma" }
- [FunctionGemma on Vertex Model Garden](https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/functiongemma){ data-model="functiongemma" }

**Terms of Use** : [Terms](https://ai.google.dev/gemma/terms)  

**Authors**: Google DeepMind

## Model Information

Summary description and brief definition of inputs and outputs.

### Description

!!! Note
    FunctionGemma is intended to be fine-tuned for your specific function-calling task, including multi-turn use cases.


FunctionGemma is a lightweight, open model from Google, built as a foundation
for creating your own specialized function calling models. FunctionGemma is not
intended for use as a direct dialogue model, and is designed to be highly
performant after further fine-tuning, as is typical of models this size. Built
on the Gemma 3 270M model and with the same research and technology used to
create the Gemini models, FunctionGemma has been trained specifically for
function calling. The model has the same architecture as Gemma 3, but uses a
different chat format. The model is well suited for text-only function calling.
The uniquely small size makes it possible to deploy in environments with limited
resources such as laptops, desktops or your own cloud infrastructure,
democratizing access to state of the art AI models and helping foster innovation
for everyone. Furthermore, akin to the base Gemma 270M, the model has been
optimized to be extremely versatile, performant on a variety of hardware in
single turn scenarios, but should be finetuned on single turn or multiturn task
specific data to achieve best accuracy in specific domains.
To demonstrate how specializing the 270M parameter model can achieve high
performance on specific agentic workflows, we have highlighted two use cases in
the
[Google AI Edge Gallery app](https://play.google.com/store/apps/details?id=com.google.ai.edge.gallery&pcampaignid=web_share).

- **Tiny Garden:** A model fine-tuned to power a voice-controlled
  interactive game. It handles game logic to manage a virtual plot of land,
  decomposing commands like "Plant sunflowers in the top row" and "Water the
  flowers in plots 1 and 2" into app-specific functions (e.g., plant_seed,
  water_plots) and coordinate targets. This demonstrates the model's capacity
  to drive custom app mechanics without server connectivity.

- **Mobile Actions:** To empower developers to build their own expert
  agents, we have published [a
  dataset](https://huggingface.co/datasets/google/mobile-actions) and
  [fine-tuning recipe](https://github.com/google-gemma/cookbook/blob/main/.archive/FunctionGemma/%5BFunctionGemma%5DFinetune_FunctionGemma_270M_for_Mobile_Actions_with_Hugging_Face.ipynb){ data-model="functiongemma" }
  to demonstrate fine-tuning FunctionGemma. It translates user inputs (e.g.,
  "Create a calendar event for lunch," "Turn on the flashlight") into
  function calls that trigger Android OS system tools. This interactive
  notebook demonstrates how to take the base FunctionGemma model and build a
  "Mobile Actions" fine tune from scratch for use in the
  [Google AI Edge gallery app](https://play.google.com/store/apps/details?id=com.google.ai.edge.gallery&pcampaignid=web_share).
  This use case demonstrates the model's ability to act as an offline,
  private agent for personal device tasks.

### Inputs and outputs

- **Input:**
    - Text string, such as a question, a prompt, or a document to be summarized
    - Total input context of 32K tokens
- **Output:**
    - Generated text in response to the input, such as an answer to a question, or a summary of a document
    - Total output context up to 32K tokens per request, subtracting the request input tokens

## Model Data

Data used for model training and how the data was processed.

### Training Dataset

These models were trained on a dataset of text data that includes a wide
variety of sources. The model was trained with 6T tokens. The knowledge cutoff
date for the training data was August 2024. There are the key components:

- Public Tool Definitions - Common APIs found on the web
- Tool Use Interactions - These are a mix of prompts, function calls, function responses, and natural language responses from the model to summarise the function call response, or request clarifications when the prompt is ambiguous or incomplete.

### Data Preprocessing

Here are the key data cleaning and filtering methods applied to the training
data:

- CSAM Filtering: Rigorous CSAM (Child Sexual Abuse Material) filtering was applied at multiple stages in the data preparation process to ensure the exclusion of harmful and illegal content.
- Sensitive Data Filtering: As part of making Gemma pre-trained models safe and reliable, automated techniques were used to filter out certain personal information and other sensitive data from training sets.
- Additional methods: Filtering based on content quality and safety in line with [our policies](https://ai.google/static/documents/ai-responsibility-update-published-february-2025.pdf).

## Implementation Information

Details about the model internals.

### Hardware

Gemma was trained using [Tensor Processing Unit
(TPU)](https://cloud.google.com/tpu/docs/intro-to-tpu) hardware (TPUv4p, TPUv5p
and TPUv5e). Training vision-language models (VLMs) requires significant
computational power. TPUs, designed specifically for matrix operations common in
machine learning, offer several advantages in this domain:

- Performance: TPUs are specifically designed to handle the massive computations involved in training VLMs. They can speed up training considerably compared to CPUs.
- Memory: TPUs often come with large amounts of high-bandwidth memory, allowing for the handling of large models and batch sizes during training. This can lead to better model quality.
- Scalability: TPU Pods (large clusters of TPUs) provide a scalable solution for handling the growing complexity of large foundation models. You can distribute training across multiple TPU devices for faster and more efficient processing.
- Cost-effectiveness: In many scenarios, TPUs can provide a more cost-effective solution for training large models compared to CPU-based infrastructure, especially when considering the time and resources saved due to faster training.
- These advantages are aligned with [Google's commitments to operate sustainably](https://sustainability.google/operating-sustainably/).

### Software

Training was done using [JAX](https://github.com/jax-ml/jax) and
[ML Pathways](https://blog.google/technology/ai/introducing-pathways-next-generation-ai-architecture/).
JAX allows researchers to take advantage of the latest generation of hardware,
including TPUs, for faster and more efficient training of large models. ML
Pathways is Google's latest effort to build artificially intelligent systems
capable of generalizing across multiple tasks. This is specially suitable for
foundation models, including large language models like these ones.  

Together, JAX and ML Pathways are used as described in the [paper about the
Gemini family of models](https://goo.gle/gemma2report); *"the 'single
controller' programming model of Jax and Pathways allows a single Python process
to orchestrate the entire training run, dramatically simplifying the development
workflow."*

## Evaluation

Model evaluation metrics and results.

### Benchmark Results

| **Benchmark** | **n-shot** | **Function Gemma 270m** |
|---|---|---|
| BFCL Simple | 0-shot | 61.6 |
| BFCL Multiple | 0-shot | 63.5 |
| BFCL Parallel | 0-shot | 39 |
| BFCL Parallel Multiple | 0-shot | 29.5 |
| BFCL Live Simple | 0-shot | 36.2 |
| BFCL Live Multiple | 0-shot | 25.7 |
| BFCL Live Parallel | 0-shot | 22.9 |
| BFCL Live Parallel Multiple | 0-shot | 20.8 |
| BFCL Relevance | 0-shot | 61.1 |
| BFCL Irrelevance | 0-shot | 73.7 |

**Impact on Performance after Fine-tuning on Mobile Actions Dataset**   

To demonstrate the value of specialization for small language models, we
compared the base FunctionGemma model against the fine-tuned model using the
"Mobile Actions"
[recipe](https://github.com/google-gemma/cookbook/blob/main/.archive/FunctionGemma/%5BFunctionGemma%5DFinetune_FunctionGemma_270M_for_Mobile_Actions_with_Hugging_Face.ipynb){ data-model="functiongemma" }.
Fine-tuning significantly improved the base FunctionGemma model's ability to
correctly identify and format mobile system calls.

| Model | Eval results for Mobile Actions |
|---|---|
| Base FunctionGemma model | 58% |
| Mobile Actions Fine-Tune | 85% |

**On-Device Performance of the Gemma 270m Fine-tuned Use Cases**   

We evaluated the fine-tuned use cases on a Samsung S25 Ultra to assess on-device
latency and memory footprint.

- **Context:** 512 prefill tokens and 32 decode tokens.
- **Hardware:** S25 Ultra CPU using LiteRT XNNPACK delegate with 4 threads.

Mobile Actions On Device Performance

| Backend | Quantization scheme | Context length | Prefill (tokens per second) | Decode (tokens per second) | Time-to-first-token (seconds) | Model Size (MB) | Peak RSS Memory (MB) |
|---|---|---|---|---|---|---|---|
| CPU | dynamic_int8 | 1024 | 1718 | 125.9 | 0.3 | 288 | 551 |

Tiny Garden On Device Performance

| Backend | Quantization scheme | Context length | Prefill (tokens per second) | Decode (tokens per second) | Time-to-first-token (seconds) | Model Size (MB) | Peak RSS Memory (MB) |
|---|---|---|---|---|---|---|---|
| CPU | dynamic_int8 | 1024 | 1743 | 125.7 | 0.3 | 288 | 549 |

## Ethics and Safety

Ethics and safety evaluation approach and results.

### Evaluation Approach

Our evaluation methods include structured evaluations and internal red-teaming
testing of relevant content policies. Red-teaming was conducted by a number of
different teams, each with different goals and human evaluation metrics. These
models were evaluated against a number of different categories relevant to
ethics and safety, including:

- **Child Safety**: Evaluation of text-to-text and image to text prompts covering child safety policies, including child sexual abuse and exploitation.
- **Content Safety:** Evaluation of text-to-text and image to text prompts covering safety policies including, harassment, violence and gore, and hate speech.
- **Representational Harms**: Evaluation of text-to-text and image to text prompts covering safety policies including bias, stereotyping, and harmful associations or inaccuracies.

### Evaluation Results

For all areas of safety testing, we saw major improvements in the categories of
child safety, content safety, and representational harms relative to previous
Gemma models. All testing was conducted without safety filters to evaluate the
model capabilities and behaviors. The model produced minimal policy violations,
and showed significant improvements over previous Gemma models' performance
with respect to ungrounded inferences. A limitation of our evaluations was they
included only English language prompts.

## Usage and Limitations

These models have certain limitations that users should be aware of.

### Intended Usage

This model is not intended for use as a direct dialogue model.  

Open Large Language Models (LLMs) have a wide range of applications across
various industries and domains. The following list of potential uses is not
comprehensive. The purpose of this list is to provide contextual information
about the possible use-cases that the model creators considered as part of model
training and development.

- Content Creation and Communication
    - Text Generation: These models can be used to generate creative text formats such as poems, scripts, code, marketing copy, and email drafts.
    - Chatbots and Conversational AI: Power conversational interfaces for customer service, virtual assistants, or interactive applications.
    - Text Summarization: Generate concise summaries of a text corpus, research papers, or reports.
- Research and Education
    - Natural Language Processing (NLP) Research: These models can serve as a foundation for researchers to experiment with NLP techniques, develop algorithms, and contribute to the advancement of the field.
    - Language Learning Tools: Support interactive language learning experiences, aiding in grammar correction or providing writing practice.
    - Knowledge Exploration: Assist researchers in exploring large bodies of text by generating summaries or answering questions about specific topics.

### Limitations

- Training Data
    - The quality and diversity of the training data significantly influence the model's capabilities. Biases or gaps in the training data can lead to limitations in the model's responses.
    - The scope of the training dataset determines the subject areas the model can handle effectively.
- Context and Task Complexity
    - Models are better at tasks that can be framed with clear prompts and instructions. Open-ended or highly complex tasks might be challenging.
    - A model's performance can be influenced by the amount of context provided (longer context generally leads to better outputs, up to a certain point).
- Language Ambiguity and Nuance
    - Natural language is inherently complex. Models might struggle to grasp subtle nuances, sarcasm, or figurative language.
- Factual Accuracy
    - Models generate responses based on information they learned from their training datasets, but they are not knowledge bases. They may generate incorrect or outdated factual statements.
- Common Sense
    - Models rely on statistical patterns in language. They might lack the ability to apply common sense reasoning in certain situations.

### Ethical Considerations and Risks

The development of large language models (LLMs) raises several ethical
concerns. In creating an open model, we have carefully considered the
following:

- Bias and Fairness
    - LLMs trained on large-scale, real-world text data can reflect socio-cultural biases embedded in the training material. These models underwent careful scrutiny, input data pre-processing described and posterior evaluations reported in this card.
- Misinformation and Misuse
    - LLMs can be misused to generate text that is false, misleading, or harmful.
    - Guidelines are provided for responsible use with the model, see the [Responsible Generative AI Toolkit](https://ai.google.dev/responsible).
- Transparency and Accountability:
    - This model card summarizes details on the models' architecture, capabilities, limitations, and evaluation processes.
    - A responsibly developed open model offers the opportunity to share innovation by making LLM technology accessible to developers and researchers across the AI ecosystem.

Risks identified and mitigations:

- Perpetuation of biases: It's encouraged to perform continuous monitoring (using evaluation metrics, human review) and the exploration of de-biasing techniques during model training, fine-tuning, and other use cases.
- Generation of harmful content: Mechanisms and guidelines for content safety are essential. Developers are encouraged to exercise caution and implement appropriate content safety safeguards based on their specific product policies and application use cases.
- Misuse for malicious purposes: Technical limitations and developer and end-user education can help mitigate against malicious applications of LLMs. Educational resources and reporting mechanisms for users to flag misuse are provided. Prohibited uses of Gemma models are outlined in the [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy)..
- Privacy violations: Models were trained on data filtered for removal of PII (Personally Identifiable Information). Developers are encouraged to adhere to privacy regulations with privacy-preserving techniques.

### Benefits

At the time of release, this family of models provides high-performance open large language model implementations designed from the ground up for Responsible AI development compared to similarly sized models.
