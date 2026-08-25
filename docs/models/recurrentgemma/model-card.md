---
title: RecurrentGemma model card
description: RecurrentGemma is a family of open language models built on a novel recurrent architecture developed at Google.
models: [recurrentgemma]
source_url: "https://ai.google.dev/gemma/docs/recurrentgemma/model_card.md.txt"
---
# Model card

**Model page:** [RecurrentGemma](index.md)

**Resources and technical documentation:**

- [Responsible Generative AI Toolkit][responsible-ai]
- [RecurrentGemma on Kaggle](https://www.kaggle.com/models/google/recurrentgemma){ data-model="recurrentgemma" }

**Terms of Use:** [Terms](https://www.kaggle.com/models/google/gemma/license/consent)

**Authors:** Google

## Model information

### Model summary

#### Description

RecurrentGemma is a family of open language models built on a [novel recurrent
architecture](https://arxiv.org/abs/2402.19427) developed at Google. Both
pre-trained and instruction-tuned versions are available in English.

Like Gemma, RecurrentGemma models are well-suited for a variety of text
generation tasks, including question answering, summarization, and reasoning.
Because of its novel architecture, RecurrentGemma requires less memory than
Gemma and achieves faster inference when generating long sequences.

#### Inputs and outputs

- **Input:** Text string (e.g., a question, a prompt, or a document to be summarized).
- **Output:** Generated English-language text in response to the input (e.g., an answer to the question, a summary of the document).

#### Citation

```python
@article{recurrentgemma_2024,
    title={RecurrentGemma},
    url={},
    DOI={},
    publisher={Kaggle},
    author={Griffin Team, Alexsandar Botev and Soham De and Samuel L Smith and Anushan Fernando and George-Christian Muraru and Ruba Haroun and Leonard Berrada et al.},
    year={2024}
}
```

### Model data

#### Training dataset and data processing

RecurrentGemma uses the same training data and data processing as used by the
Gemma model family. A full description can be found on the [Gemma model
card](../gemma-1/index.md#model-data).

## Implementation information

### Hardware and frameworks used during training

Like
[Gemma](../gemma-1/index.md#implementation-information),
RecurrentGemma was trained on
[TPUv5e](https://cloud.google.com/tpu/docs/intro-to-tpu?_gl=1*18wi411*_ga*MzE3NDU5OTY1LjE2MzQwNDA4NDY.*_ga_WH2QY8WWF5*MTcxMTA0MjUxMy4xNy4wLjE3MTEwNDI1MTkuMC4wLjA.&_ga=2.239449409.-317459965.1634040846),
using [JAX](https://github.com/jax-ml/jax) and [ML
Pathways](https://blog.google/technology/ai/introducing-pathways-next-generation-ai-architecture/).

## Evaluation information

### Benchmark results

#### Evaluation approach

These models were evaluated against a large collection of different datasets and
metrics to cover different aspects of text generation:

#### Evaluation results

| Benchmark | Metric | RecurrentGemma 2B | RecurrentGemma 9B |
|---|---|---|---|
| [MMLU](https://arxiv.org/abs/2009.03300) | 5-shot, top-1 | 38.4 | 60.5 |
| [HellaSwag](https://arxiv.org/abs/1905.07830) | 0-shot | 71.0 | 80.4 |
| [PIQA](https://arxiv.org/abs/1911.11641) | 0-shot | 78.5 | 81.3 |
| [SocialIQA](https://arxiv.org/abs/1904.09728) | 0-shot | 51.8 | 52.3 |
| [BoolQ](https://arxiv.org/abs/1905.10044) | 0-shot | 71.3 | 80.3 |
| [WinoGrande](https://arxiv.org/abs/1907.10641) | partial score | 67.8 | 73.6 |
| [CommonsenseQA](https://arxiv.org/abs/1811.00937) | 7-shot | 63.7 | 73.2 |
| [OpenBookQA](https://arxiv.org/abs/1809.02789) |   | 47.2 | 51.8 |
| [ARC-e](https://arxiv.org/abs/1911.01547) |   | 72.9 | 78.8 |
| [ARC-c](https://arxiv.org/abs/1911.01547) |   | 42.3 | 52.0 |
| [TriviaQA](https://arxiv.org/abs/1705.03551) | 5-shot | 52.5 | 70.5 |
| [Natural Questions](https://github.com/google-research-datasets/natural-questions) | 5-shot | 11.5 | 21.7 |
| [HumanEval](https://arxiv.org/abs/2107.03374) | pass@1 | 21.3 | 31.1 |
| [MBPP](https://arxiv.org/abs/2108.07732) | 3-shot | 28.8 | 42.0 |
| [GSM8K](https://arxiv.org/abs/2110.14168) | maj@1 | 13.4 | 42.6 |
| [MATH](https://arxiv.org/abs/2103.03874) | 4-shot | 11.0 | 23.8 |
| [AGIEval](https://arxiv.org/abs/2304.06364) |   | 23.8 | 39.3 |
| [BIG-Bench](https://arxiv.org/abs/2206.04615) |   | 35.3 | 55.2 |
| **Average** |   | 44.6 | 56.1 |

## Ethics and safety

### Ethics and safety evaluations

#### Evaluations approach

Our evaluation methods include structured evaluations and internal red-teaming
testing of relevant content policies. Red-teaming was conducted by a number of
different teams, each with different goals and human evaluation metrics. These
models were evaluated against a number of different categories relevant to
ethics and safety, including:

- **Text-to-text content safety:** Human evaluation on prompts covering safety policies including child sexual abuse and exploitation, harassment, violence and gore, and hate speech.
- **Text-to-text representational harms:** Benchmark against relevant academic datasets such as WinoBias and BBQ Dataset.
- **Memorization:** Automated evaluation of memorization of training data, including the risk of personally identifiable information exposure.
- **Large-scale harm:** Tests for "dangerous capabilities," such as chemical, biological, radiological, and nuclear (CBRN) risks; as well as tests for persuasion and deception, cybersecurity, and autonomous replication.

#### Evaluation results

The results of ethics and safety evaluations are within acceptable thresholds
for meeting [internal
policies](https://storage.googleapis.com/gweb-uniblog-publish-prod/documents/2023_Google_AI_Principles_Progress_Update.pdf#page=11)
for categories such as child safety, content safety, representational harms,
memorization, large-scale harms. On top of robust internal evaluations, the
results of well known safety benchmarks like BBQ, Winogender, WinoBias,
RealToxicity, and TruthfulQA are shown here.

| Benchmark | Metric | RecurrentGemma 2B | RecurrentGemma 2B IT | RecurrentGemma 9B | RecurrentGemma 9B IT |
|---|---|---|---|---|---|
| [RealToxicity](https://arxiv.org/abs/2009.11462) | avg | 9.8 | 7.60 | 10.3 | 8.8 |
| [BOLD](https://arxiv.org/abs/2101.11718) |   | 39.3 | 52.3 | 39.8 | 47.9 |
| [CrowS-Pairs](https://aclanthology.org/2020.emnlp-main.154/) | top-1 | 41.1 | 43.4 | 38.7 | 39.5 |
| [BBQ Ambig](https://arxiv.org/abs/2110.08193v2) | top-1 | 62.6 | 71.1 | 95.9 | 67.1 |
| [BBQ Disambig](https://arxiv.org/abs/2110.08193v2) | top-1 | 58.4 | 50.8 | 78.6 | 78.9 |
| [Winogender](https://arxiv.org/abs/1804.09301) | top-1 | 55.1 | 54.7 | 59.0 | 64.0 |
| [TruthfulQA](https://arxiv.org/abs/2109.07958) |   | 35.1 | 42.7 | 38.6 | 47.7 |
| [WinoBias 1_2](https://arxiv.org/abs/1804.06876) |   | 58.4 | 56.4 | 61.5 | 60.6 |
| [WinoBias 2_2](https://arxiv.org/abs/1804.06876) |   | 90.0 | 75.4 | 90.2 | 90.3 |
| [Toxigen](https://arxiv.org/abs/2203.09509) |   | 56.7 | 50.0 | 58.8 | 64.5 |

## Model usage and limitations

### Known limitations

These models have certain limitations that users should be aware of:

- **Training data**
    - The quality and diversity of the training data significantly influence the model's capabilities. Biases or gaps in the training data can lead to limitations in the model's responses.
    - The scope of the training dataset determines the subject areas the model can handle effectively.
- **Context and task complexity**
    - LLMs are better at tasks that can be framed with clear prompts and instructions. Open-ended or highly complex tasks might be challenging.
    - A model's performance can be influenced by the amount of context provided (longer context generally leads to better outputs, up to a certain point).
- **Language ambiguity and nuance**
    - Natural language is inherently complex. LLMs might struggle to grasp subtle nuances, sarcasm, or figurative language.
- **Factual accuracy**
    - LLMs generate responses based on information they learned from their training datasets, but they are not knowledge bases. They may generate incorrect or outdated factual statements.
- **Common sense**
    - LLMs rely on statistical patterns in language. They might lack the ability to apply common sense reasoning in certain situations.

### Ethical considerations and risks

The development of large language models (LLMs) raises several ethical concerns.
In creating an open model, we have carefully considered the following:

- **Bias and fairness**
    - LLMs trained on large-scale, real-world text data can reflect socio-cultural biases embedded in the training material. These models underwent careful scrutiny, input data pre-processing described and posterior evaluations reported in this card.
- **Misinformation and misuse**
    - LLMs can be misused to generate text that is false, misleading, or harmful.
    - Guidelines are provided for responsible use with the model, see the [Responsible Generative AI Toolkit][responsible-ai].
- **Transparency and accountability**
    - This model card summarizes details on the models' architecture, capabilities, limitations, and evaluation processes.
    - A responsibly developed open model offers the opportunity to share innovation by making LLM technology accessible to developers and researchers across the AI ecosystem.

Risks Identified and Mitigations:

- **Perpetuation of biases:** It's encouraged to perform continuous monitoring (using evaluation metrics, human review) and the exploration of de-biasing techniques during model training, fine-tuning, and other use cases.
- **Generation of harmful content:** Mechanisms and guidelines for content safety are essential. Developers are encouraged to exercise caution and implement appropriate content safety safeguards based on their specific product policies and application use cases.
- **Misuse for malicious purposes:** Technical limitations and developer and end-user education can help mitigate against malicious applications of LLMs. Educational resources and reporting mechanisms for users to flag misuse are provided. Prohibited uses of Gemma models are outlined in our [terms of
  use](https://www.kaggle.com/models/google/gemma/license/consent).
- **Privacy violations:** Models were trained on data filtered for removal of PII (Personally Identifiable Information). Developers are encouraged to adhere to privacy regulations with privacy-preserving techniques.

## Intended usage

### Application

Open Large Language Models (LLMs) have a wide range of applications across
various industries and domains. The following list of potential uses is not
comprehensive. The purpose of this list is to provide contextual information
about the possible use-cases that the model creators considered as part of model
training and development.

- **Content creation and communication**
    - **Text generation:** These models can be used to generate creative text formats like poems, scripts, code, marketing copy, email drafts, etc.
    - **Chatbots and conversational AI:** Power conversational interfaces for customer service, virtual assistants, or interactive applications.
    - **Text summarization:** Generate concise summaries of a text corpus, research papers, or reports.
- **Research and education**
    - **Natural Language Processing (NLP) research:** These models can serve as a foundation for researchers to experiment with NLP techniques, develop algorithms, and contribute to the advancement of the field.
    - **Language Learning Tools:** Support interactive language learning experiences, aiding in grammar correction or providing writing practice.
    - **Knowledge Exploration:** Assist researchers in exploring large bodies of text by generating summaries or answering questions about specific topics.

### Benefits

At the time of release, this family of models provides high-performance open
large language model implementations designed from the ground up for Responsible
AI development compared to similarly sized models.

Using the benchmark evaluation metrics described in this document, these models
have shown to provide superior performance to other, comparably-sized open model
alternatives.

In particular, RecurrentGemma models achieve comparable performance to Gemma
models but are faster during inference and require less memory, especially on
long sequences.
