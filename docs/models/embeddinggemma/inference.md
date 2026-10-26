---
title: Generate Embeddings with Sentence Transformers
description: Generate text embeddings, compute semantic similarity, apply task prompts, and build RAG and zero-shot classifiers with EmbeddingGemma 2 and Sentence Transformers.
models: [embeddinggemma-2]
frameworks: [sentence-transformers, transformers]
colab: https://colab.research.google.com/github/google-gemma/docs/blob/main/docs/notebooks/embeddinggemma-inference.ipynb
source_url: "https://ai.google.dev/gemma/docs/embeddinggemma/inference-embeddinggemma-with-sentence-transformers.md.txt"
---
# Generate Embeddings with Sentence Transformers

**EmbeddingGemma 2** is a lightweight, open-source 740M parameter model designed for unified text and multimodal embeddings. Built on the Gemma 4 decoder architecture, it maps inputs into a unified 768-dimensional vector space.

Because of its compact footprint, you can run demanding semantic workflows (e.g. Retrieval-Augmented Generation (RAG), semantic search, and zero-shot classification) locally and offline on consumer GPUs or CPUs.

**What this guide covers:**

- Installing dependencies and loading **EmbeddingGemma 2**
- Generating **text embeddings** , computing semantic similarity, and applying **task-specific prompts**
- Building downstream applications (**RAG** and **zero-shot classification**)
- Optimizing storage and inference speed with **Matryoshka Representation Learning (MRL)**

!!! tip "Looking for image, audio, or video embeddings?"
    Check out [Multimodal Embeddings with EmbeddingGemma 2](multimodal.md) guide.

## Setup and Installation

Install `sentence-transformers` and `transformers`.

Sentence Transformers is a Python framework for using and training embedding and reranker models. For more details on the framework, see the [Sentence Transformers documentation](https://www.sbert.net/).

```bash { .notebook-cell }
pip install -U sentence-transformers transformers pillow soundfile torchcodec
```

## Model Initialization

Instantiate EmbeddingGemma 2 directly using the `SentenceTransformer` class.

```python { .notebook-cell }
from sentence_transformers import SentenceTransformer

model_id = "google/embeddinggemma-2"
model = SentenceTransformer(model_id)

print(f"device: {model.device}")
print(f"dtype: {model.dtype}")
print(model)
print("Total number of parameters in the model:", sum(p.numel() for p in model.parameters()))
```

``` { .no-copy .output }
Loading weights:   0%|          | 0/1376 [00:00<?, ?it/s]
device: cuda:0
dtype: torch.bfloat16
SentenceTransformer(
  (0): Transformer({'transformer_task': 'feature-extraction', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'last_hidden_state'}, 'image': {'method': 'forward', 'method_output_name': 'last_hidden_state'}, 'audio': {'method': 'forward', 'method_output_name': 'last_hidden_state'}, 'video': {'method': 'forward', 'method_output_name': 'last_hidden_state'}, 'message': {'method': 'forward', 'method_output_name': 'last_hidden_state', 'format': 'structured'} }, 'module_output_name': 'token_embeddings', 'architecture': 'EmbeddingGemma2Model'})
  (1): Pooling({'embedding_dimension': 768, 'pooling_mode': 'mean', 'include_prompt': True})
  (2): Normalize({'module_input_name': 'sentence_embedding', 'module_output_name': 'sentence_embedding'})
)
Total number of parameters in the model: 744371488
```

!!! tip "Text-Only Memory Optimization"
    By default, `SentenceTransformer` loads all modality encoders (740M parameters). If your workflow only uses text, you can pass `config_kwargs={"vision_config": None, "audio_config": None}` to skip loading the vision and audio encoders (270M parameters).

!!! note "Note on Precision"
    EmbeddingGemma 2 activations are incompatible with `float16`. Use `bfloat16` (default on supported GPUs) or `float32`.

## Text Embeddings and Task Prompts

Learn how to encode text into dense vector representations, measure semantic similarity between inputs, and apply task-specific prompt prefixes to optimize embedding quality.

### Generating Text Embeddings

An embedding is a dense numerical vector that captures the semantic meaning of an input. Let's encode three words---`["king", "queen", "car"]`---to inspect their 768-dimensional representations.

```python { .notebook-cell }
words = ["king", "queen", "car"]

# Generate embeddings using model.encode()
embeddings = model.encode(words)

print(embeddings)
for idx, embedding in enumerate(embeddings):
    print(f"Embedding {idx+1} (shape): {embedding.shape}")
```

``` { .no-copy .output }
[[ 0.03222656  0.00285339  0.02160645 ...  0.012146    0.03149414
  -0.0067749 ]
 [ 0.00787354  0.01361084  0.02502441 ... -0.01269531  0.02612305
   0.01361084]
 [-0.04052734 -0.00860596  0.04492188 ...  0.00964355 -0.01501465
  -0.00726318]]
Embedding 1 (shape): (768,)
Embedding 2 (shape): (768,)
Embedding 3 (shape): (768,)
```

The key property of embeddings is **the distance between vectors in semantic space** . Because `"king"` and `"queen"` share semantic characteristics as royal titles, their vectors are positioned close together, whereas `"car"` lies further away.

### Determining Semantic Similarity

We can quantify how semantically similar two inputs are by computing the cosine similarity between their embeddings using `model.similarity()`.

Below are three sentence pairs demonstrating high, medium, and low semantic similarity:

- **High Similarity:**
  - *Sentence A:* `"The chef prepared a delicious meal for the guests."`
  - *Sentence B:* `"A tasty dinner was cooked by the chef for the visitors."`
  - *Reasoning:* Both sentences describe the exact same event using active vs. passive voice.
- **Medium Similarity:**
  - *Sentence A:* `"She is an expert in machine learning."`
  - *Sentence B:* `"He has a deep interest in artificial intelligence."`
  - *Reasoning:* Related domains (ML is a subfield of AI), but different subjects and levels of expertise.
- **Low Similarity:**
  - *Sentence A:* `"The weather in Tokyo is sunny today."`
  - *Sentence B:* `"I need to buy groceries for the week."`
  - *Reasoning:* Completely unrelated topics with no semantic overlap.

```python { .notebook-cell }
sentence_high = [
    "The chef prepared a delicious meal for the guests.",
    "A tasty dinner was cooked by the chef for the visitors.",
]
sentence_medium = [
    "She is an expert in machine learning.",
    "He has a deep interest in artificial intelligence.",
]
sentence_low = [
    "The weather in Tokyo is sunny today.",
    "I need to buy groceries for the week.",
]

for pair in [sentence_high, sentence_medium, sentence_low]:
    print("🙋‍♂️", pair)
    pair_embeddings = model.encode(pair)
    score = model.similarity(pair_embeddings[0], pair_embeddings[1]).item()
    print(f" `-> 🤖 similarity score: {score:.4f}")
```

``` { .no-copy .output }
🙋‍♂️ ['The chef prepared a delicious meal for the guests.', 'A tasty dinner was cooked by the chef for the visitors.']
 `-> 🤖 similarity score: 0.9184
🙋‍♂️ ['She is an expert in machine learning.', 'He has a deep interest in artificial intelligence.']
 `-> 🤖 similarity score: 0.7835
🙋‍♂️ ['The weather in Tokyo is sunny today.', 'I need to buy groceries for the week.']
 `-> 🤖 similarity score: 0.6880
```

!!! note "Note on interpreting scores"
    Cosine similarity values for text embeddings tend to skew higher than human intuition might suggest, often rarely dropping below 0.5 in practice. For instance, even completely unrelated sentences (like the "Low Similarity" example above) can yield a similarity score of roughly 0.6880. Keep this higher baseline in mind when evaluating semantic overlap.

### Optimizing Embeddings with Task Prompts

To achieve optimal performance with EmbeddingGemma 2 on text tasks, prepend an **instructional prompt** (task prefix) to your input. Task prompts condition the model for specific objectives (e.g. sentence similarity, asymmetric search retrieval, or classification) and help separate queries from documents.

**Prefix Notation & Usage**

We offer two types of task prefixes, depending on how embeddings are used in the task. There are two types of tasks:

- **Asymmetric Tasks (e.g. retrieval):** Use a query prefix for queries and a document prefix for corpus items.
- **Symmetric Tasks (e.g. classification, similarity):** Apply the same task prefix to all inputs being compared.

| Use Case | Task Type | Query Task Instruction | Document Task Instruction *(use `none` if no title)* |
|---|---|---|---|
| Web / document search | Asymmetric | `task: search result | query: {query}` | `title: {title} | text: {content}` |
| Question answering | Asymmetric | `task: question answering | query: {question}` | `title: {title} | text: {passage}` |
| Fact checking | Asymmetric | `task: fact checking | query: {claim}` | `title: {title} | text: {evidence}` |
| Code search | Asymmetric | `task: code retrieval | query: {query}` | `title: {title or filename} | text: {code}` |
| Text classification | Symmetric | `task: classification | query: {content}` | N/A |
| Clustering | Symmetric | `task: clustering | query: {content}` | N/A |
| Measuring similarity | Symmetric | `task: sentence similarity | query: {content}` | N/A |

**Please note:** Prefixes are used on text inputs only; when encoding images, video, or audio, pass them directly.

You can apply prompts during inference in three ways:

1. **Using `prompt_name` (Recommended):** Select a built-in task prompt registered in `model.prompts`:
   ```python
   embeddings = model.encode(sentences, prompt_name="STS")
   ```
2. **Using `prompt`:** Pass a custom prefix string directly to `model.encode()`:
   ```python
   embeddings = model.encode(sentences, prompt="task: sentence similarity | query: ")
   ```
3. **Using the Default Prompt:** If `default_prompt_name` is configured on the model, calling `model.encode(sentences)` applies it automatically.

Let's inspect all built-in task prompts and re-evaluate our sentence similarity pairs using `prompt_name="STS"`:

```python { .notebook-cell }
print("Available task prompts:")
for name, prefix in model.prompts.items():
    print(f"  {name}: {prefix!r}")
print("-" * 80)

for pair in [sentence_high, sentence_medium, sentence_low]:
    print("🙋‍♂️", pair)
    pair_embeddings = model.encode(pair, prompt_name="STS")
    score = model.similarity(pair_embeddings[0], pair_embeddings[1]).item()
    print(f" `-> 🤖 STS score: {score:.4f}")
```

``` { .no-copy .output }
Available task prompts:
  query: 'task: search result | query: '
  document: 'title: none | text: '
  BitextMining: 'task: search result | query: '
  Classification: 'task: classification | query: '
  Clustering: 'task: clustering | query: '
  CodeRetrieval: 'task: code retrieval | query: '
  Document: 'title: none | text: '
  FactChecking: 'task: fact checking | query: '
  InstructionRetrieval: 'task: code retrieval | query: '
  MultilabelClassification: 'task: classification | query: '
  PairClassification: 'task: sentence similarity | query: '
  QuestionAnswering: 'task: question answering | query: '
  Reranking: 'task: search result | query: '
  Retrieval: 'task: search result | query: '
  Retrieval-document: 'title: none | text: '
  Retrieval-query: 'task: search result | query: '
  STS: 'task: sentence similarity | query: '
  SearchQuery: 'task: search result | query: '
  SentenceSimilarity: 'task: sentence similarity | query: '
  Summarization: 'task: sentence similarity | query: '
---
🙋‍♂️ ['The chef prepared a delicious meal for the guests.', 'A tasty dinner was cooked by the chef for the visitors.']
 `-> 🤖 STS score: 0.9555
🙋‍♂️ ['She is an expert in machine learning.', 'He has a deep interest in artificial intelligence.']
 `-> 🤖 STS score: 0.8628
🙋‍♂️ ['The weather in Tokyo is sunny today.', 'I need to buy groceries for the week.']
 `-> 🤖 STS score: 0.7340
```

## Downstream Applications

Put EmbeddingGemma 2 embeddings into practice to power common real-world workflows, such as document retrieval for Retrieval-Augmented Generation (RAG) and zero-shot text classification.

### Retrieval-Augmented Generation (RAG)

In asymmetric retrieval and RAG pipelines, queries are typically short questions while documents are longer passages. Use dedicated retrieval prompts to optimize both sides of the search index:

- **Search Queries:** Use `prompt_name="Retrieval-query"` (applies `"task: search result | query: "`).
- **Documents:** Use `prompt_name="Retrieval-document"` (applies `"title: none | text: "`), or pass a custom `prompt` string to include document titles (`"title: <Document Title> | text: "`). Including titles significantly improves retrieval accuracy.

```python { .notebook-cell }
query = "How do I reduce memory usage when running EmbeddingGemma 2?"

documents = [
    {
        "title": "Model Initialization and Memory Optimization",
        "text": "You can disable the vision or audio encoders by passing vision_config=None or audio_config=None inside config_kwargs.",
    },
    {
        "title": "Matryoshka Representation Learning",
        "text": "MRL allows truncating 768-dimensional embeddings to 512 or 256 dimensions for faster vector search and smaller indexes.",
    },
    {
        "title": "Audio Processing Specifications",
        "text": "Audio waveforms are automatically resampled to 16 kHz before feature extraction.",
    },
]

# Encode query with Retrieval-query prompt
query_emb = model.encode(query, prompt_name="Retrieval-query")

# Encode documents with their titles embedded in the prompt
doc_embeddings = [
    model.encode(doc["text"], prompt=f"title: {doc['title']} | text: ")
    for doc in documents
]

scores = model.similarity(query_emb, doc_embeddings)[0]
for doc, score in zip(documents, scores):
    print(f"[Score: {score:.4f}] {doc['title']}")
```

``` { .no-copy .output }
[Score: 0.6855] Model Initialization and Memory Optimization
[Score: 0.6386] Matryoshka Representation Learning
[Score: 0.5653] Audio Processing Specifications
```

For a complete end-to-end RAG tutorial, see the [RAG with EmbeddingGemma notebook](https://github.com/google-gemma/cookbook/blob/main/tutorials/RAG_with_EmbeddingGemma.ipynb){ data-model="embeddinggemma-2" } in the Gemma Cookbook.

### Zero-Shot Classification

Zero-shot text classification assigns inputs to predefined categories without task-specific training by selecting the candidate label with the highest embedding similarity to the input text.

When choosing a task prompt for zero-shot classification, consider whether your target labels match standard classification datasets:

- **Custom or Arbitrary Categories (`prompt_name="Clustering"`):** For domain-specific or ad-hoc labels like `"Billing Issue"`, `"Technical Support"`, and `"Sales Inquiry"`, the `"Classification"` prompt does not work well because the training dataset does not include those specific labels. Instead, use `prompt_name="Clustering"` to group inputs and candidate labels by semantic topic similarity.
- **Standard Classification Tasks (`prompt_name="Classification"`):** Common classification prompts---such as sentiment analysis (`"Positive"`, `"Neutral"`, `"Negative"`)---work well with `prompt_name="Classification"` because they align with the classification tasks seen during training.

#### Custom Category Routing with `Clustering`

Use `prompt_name="Clustering"` to route incoming support tickets to custom department categories:

```python { .notebook-cell }
labels = ["Billing Issue", "Technical Support", "Sales Inquiry"]

tickets = [
    "Excuse me, the app freezes on the login screen. It won't work even when I try to reset my password.",
    "I would like to inquire about your enterprise plan pricing and features for a team of 50 people.",
]

# Encode labels and incoming tickets using the Clustering prompt
label_embeddings = model.encode(labels, prompt_name="Clustering")
ticket_embeddings = model.encode(tickets, prompt_name="Clustering")

# Compute similarity matrix and find the best-matching label for each ticket
similarities = model.similarity(ticket_embeddings, label_embeddings)
print("Similarity matrix:\n", similarities)

predicted_indices = similarities.argmax(dim=1)

for ticket, pred_idx in zip(tickets, predicted_indices):
    print(f"\n🙋‍♂️ Ticket: {ticket}")
    print(f" `-> 🤖 Predicted Category: {labels[pred_idx]}")
```

``` { .no-copy .output }
Similarity matrix:
 tensor([[0.8704, 0.8904, 0.8394],
        [0.8426, 0.8566, 0.8758]])

🙋‍♂️ Ticket: Excuse me, the app freezes on the login screen. It won't work even when I try to reset my password.
 `-> 🤖 Predicted Category: Technical Support

🙋‍♂️ Ticket: I would like to inquire about your enterprise plan pricing and features for a team of 50 people.
 `-> 🤖 Predicted Category: Sales Inquiry
```

#### Sentiment Analysis with `Classification`

For standard sentiment analysis, encode the common sentiment labels and input sentences using `prompt_name="Classification"`:

```python { .notebook-cell }
labels = ["Positive", "Neutral", "Negative"]

sentences = [
    # Positive
    "The sunset over the ridge was absolutely breathtaking.",
    "This is hands down the best coffee I have ever tasted.",
    # Neutral
    "The package arrived on Tuesday afternoon.",
    "The meeting is scheduled for 10:00 AM in Conference Room B.",
    # Negative
    "The battery dies after barely two hours of use.",
    "It felt like an utter waste of time and money.",
]

# Encode labels and incoming sentences using the Classification prompt
label_embeddings = model.encode(labels, prompt_name="Classification")
sentence_embeddings = model.encode(sentences, prompt_name="Classification")

# Compute similarity matrix and find the best-matching label for each sentence
similarities = model.similarity(sentence_embeddings, label_embeddings)
print("Similarity matrix:\n", similarities)

predicted_indices = similarities.argmax(dim=1)

for sentence, pred_idx in zip(sentences, predicted_indices):
    print(f"\n🙋‍♂️ Sentence: {sentence}")
    print(f" `-> 🤖 Predicted Sentiment: {labels[pred_idx]}")
```

``` { .no-copy .output }
Similarity matrix:
 tensor([[0.8177, 0.6642, 0.6423],
        [0.8731, 0.6198, 0.6375],
        [0.6581, 0.8198, 0.6425],
        [0.6115, 0.7675, 0.6093],
        [0.6273, 0.6680, 0.7666],
        [0.6103, 0.6236, 0.8660]])

🙋‍♂️ Sentence: The sunset over the ridge was absolutely breathtaking.
 `-> 🤖 Predicted Sentiment: Positive

🙋‍♂️ Sentence: This is hands down the best coffee I have ever tasted.
 `-> 🤖 Predicted Sentiment: Positive

🙋‍♂️ Sentence: The package arrived on Tuesday afternoon.
 `-> 🤖 Predicted Sentiment: Neutral

🙋‍♂️ Sentence: The meeting is scheduled for 10:00 AM in Conference Room B.
 `-> 🤖 Predicted Sentiment: Neutral

🙋‍♂️ Sentence: The battery dies after barely two hours of use.
 `-> 🤖 Predicted Sentiment: Negative

🙋‍♂️ Sentence: It felt like an utter waste of time and money.
 `-> 🤖 Predicted Sentiment: Negative
```

## Matryoshka Representation Learning (MRL)

EmbeddingGemma 2 is trained with **Matryoshka Representation Learning (MRL)**, which concentrates the most important semantic information in the earliest dimensions of the output vector.

Instead of deploying separate models for different latency or storage budgets, you can **truncate** the full 768-dimensional embedding to smaller prefixes---such as **512** , **256** or **128** dimensions---while preserving strong semantic quality. When truncating dimensions, always set `normalize_embeddings=True` so that the truncated vectors are re-normalized to unit length.

Let's compare word similarity scores across **768** , **512** , and **128** dimensions:

```python { .notebook-cell }
words = ["king", "queen", "car"]

def evaluate_mrl(embeddings, sim_fn_name):
    print(f"Embedding shape: {embeddings.shape} | Similarity function: {sim_fn_name}")
    sims = model.similarity(embeddings[0], embeddings[1:])
    for idx, word in enumerate(words[1:]):
        print(f"  🙋‍♂️ 'king' vs. '{word}' -> 🤖 score: {sims[0][idx].item():.4f}")

# 1. Full 768-dimensional embeddings
emb_768 = model.encode(words, prompt_name="STS", normalize_embeddings=True)
evaluate_mrl(emb_768, model.similarity_fn_name)
```

``` { .no-copy .output }
Embedding shape: (3, 768) | Similarity function: cosine
  🙋‍♂️ 'king' vs. 'queen' -> 🤖 score: 0.9128
  🙋‍♂️ 'king' vs. 'car' -> 🤖 score: 0.8354
```

### Truncating to 512 Dimensions

Pass `truncate_dim=512` directly to `model.encode()` to slice and re-normalize vectors on the fly:

```python { .notebook-cell }
# 2. Truncate to 512 dimensions at inference time
emb_512 = model.encode(
    words,
    prompt_name="STS",
    truncate_dim=512,
    normalize_embeddings=True,
)
evaluate_mrl(emb_512, model.similarity_fn_name)
```

``` { .no-copy .output }
Embedding shape: (3, 512) | Similarity function: cosine
  🙋‍♂️ 'king' vs. 'queen' -> 🤖 score: 0.9124
  🙋‍♂️ 'king' vs. 'car' -> 🤖 score: 0.8334
```

### Truncating to 128 Dimensions with Dot-Product Similarity

For maximum speed in high-throughput vector databases, you can configure `truncate_dim=128` and `similarity_fn_name="dot"` at model initialization. Because normalized embeddings have unit length, dot product is mathematically equivalent to cosine similarity while being faster to compute:

```python { .notebook-cell }
# 3. Initialize model with default 128-dim truncation and dot-product similarity
fast_model = SentenceTransformer(
    model_id,
    truncate_dim=128,
    similarity_fn_name="dot",
    device=model.device,
)

emb_128 = fast_model.encode(words, prompt_name="STS", normalize_embeddings=True)
evaluate_mrl(emb_128, fast_model.similarity_fn_name)
```

``` { .no-copy .output }
Loading weights:   0%|          | 0/1376 [00:00<?, ?it/s]
Embedding shape: (3, 128) | Similarity function: dot
  🙋‍♂️ 'king' vs. 'queen' -> 🤖 score: 0.9470
  🙋‍♂️ 'king' vs. 'car' -> 🤖 score: 0.8957
```

!!! note
    EmbeddingGemma 2 supports any output dimension between 128 and 768. However, we recommend using 768, 512, 256, or 128 for optimal performance.

## Summary and Next Steps

You now know how to generate text embeddings using **EmbeddingGemma 2** and the `sentence-transformers` library. You also explored how to apply task-specific prompts, build zero-shot classifiers and RAG retrievers, and compress embeddings using Matryoshka Representation Learning (MRL).

### Further Reading

- [EmbeddingGemma 2 Model Card](model-card-v2.md)
- [Multimodal Embeddings with EmbeddingGemma 2](multimodal.md)
- [Fine-tune EmbeddingGemma with Sentence Transformers](fine-tuning.md)
- [Sentence Transformers Prompt Templates Guide](https://sbert.net/examples/sentence_transformer/applications/computing-embeddings/README.html#prompt-templates)
- [Simple RAG with EmbeddingGemma in the Gemma Cookbook](https://github.com/google-gemma/cookbook/blob/main/tutorials/RAG_with_EmbeddingGemma.ipynb){ data-model="embeddinggemma-2" }

--8<-- "glossary.md"
