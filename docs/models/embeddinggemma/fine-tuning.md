---
title: Fine-tune EmbeddingGemma with Sentence Transformers
description: Adapt EmbeddingGemma 2 to specialized domain terminology using contrastive triplet fine-tuning with Sentence Transformers.
models: [embeddinggemma-2]
frameworks: [sentence-transformers, transformers]
colab: https://colab.research.google.com/github/google-gemma/docs/blob/main/docs/notebooks/embeddinggemma-fine-tuning.ipynb
source_url: "https://ai.google.dev/gemma/docs/embeddinggemma/fine-tuning-embeddinggemma-with-sentence-transformers.md.txt"
---
# Fine-tune EmbeddingGemma with Sentence Transformers

Fine-tuning helps close the gap between a model's general-purpose understanding and the specialized, high-performance accuracy that your application requires. While **EmbeddingGemma 2** provides strong out-of-the-box semantic representations across text and multimodal inputs, domain-specific terminology often benefits from targeted adaptation.

### Example Scenario: Financial Knowledge Base Search

Imagine your company, "Shibuya Financial", offers various complex financial products like investment trusts, **NISA accounts** (Japanese tax-advantaged savings accounts), and home loans. Customer support agents rely on an internal semantic search engine to retrieve relevant policy documents for customer inquiries.

Without fine-tuning, general-purpose embeddings may struggle to separate closely related financial concepts, for example, confusing a query about a *tax-free investment account* (NISA) with a *regular savings account*.

**What this guide covers:**

- Installing dependencies and loading **EmbeddingGemma 2** (`google/embeddinggemma-2`)
- Evaluating baseline retrieval performance before fine-tuning
- Preparing contrastive `(anchor, positive, negative)` training triplets
- Fine-tuning the model using `SentenceTransformerTrainer` and `MultipleNegativesRankingLoss`
- Evaluating post-training improvements and saving your adapted model

## Setup and Installation

Install `sentence-transformers`, `transformers`, and the multimedia libraries (`pillow`, `soundfile`, `torchcodec`) required for processing images, audio, and video.

Sentence Transformers is a Python framework for using and training embedding and reranker models. For more details on the framework, see the [Sentence Transformers documentation](https://www.sbert.net/).

```bash { .notebook-cell }
pip install -U sentence-transformers transformers pillow soundfile torchcodec
```

## (Optional) Hugging Face Authentication

You need a valid Hugging Face Token to publish your model. If you are running inside a Google Colab, you can securely use your Hugging Face Token using the Colab secrets otherwise you can set the token as directly in the login method. Make sure your token has write access too, as you push your model to the Hub during training.

```python { .notebook-cell }
# Login into Hugging Face Hub
from huggingface_hub import login
login()
```

## Load the Base Model

Load `google/embeddinggemma-2` using `SentenceTransformer`.

!!! tip
    If you are fine-tuning exclusively on text data and want to minimize GPU memory usage, you can pass `config_kwargs={"vision_config": None, "audio_config": None}` during initialization to omit the vision and audio encoders.

```python { .notebook-cell }
from sentence_transformers import SentenceTransformer

model_id = "google/embeddinggemma-2"
model = SentenceTransformer(
    model_id,
    config_kwargs={"vision_config": None, "audio_config": None},
)

print(f"Device: {model.device}")
print(f"dtype: {model.dtype}")
print(model)
print("Total number of parameters in the model:", sum([p.numel() for _, p in model.named_parameters()]))
```

``` { .no-copy .output }
Loading weights:   0%|          | 0/413 [00:00<?, ?it/s]
Device: cuda:0
dtype: torch.bfloat16
SentenceTransformer(
  (0): Transformer({'transformer_task': 'feature-extraction', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'last_hidden_state'}, 'image': {'method': 'forward', 'method_output_name': 'last_hidden_state'}, 'audio': {'method': 'forward', 'method_output_name': 'last_hidden_state'}, 'video': {'method': 'forward', 'method_output_name': 'last_hidden_state'}, 'message': {'method': 'forward', 'method_output_name': 'last_hidden_state', 'format': 'structured'} }, 'module_output_name': 'token_embeddings', 'architecture': 'EmbeddingGemma2Model'})
  (1): Pooling({'embedding_dimension': 768, 'pooling_mode': 'mean', 'include_prompt': True})
  (2): Normalize({'module_input_name': 'sentence_embedding', 'module_output_name': 'sentence_embedding'})
)
Total number of parameters in the model: 271002624
```

## Baseline Evaluation (Before Fine-Tuning)

First, let's test how the base model ranks three candidate knowledge base documents when a customer asks:
> *"I want to start a tax-free installment investment, what should I do?"*

The correct match is **Opening a NISA Account**, since NISA is specifically a tax-free investment account.

!!! note "Note on Task Prompts"
    Always include task-specific prompts when encoding text with EmbeddingGemma 2. Here we use `prompt_name="STS"` (sentence similarity). For details on all available prompts, see the [EmbeddingGemma 2 model card](model-card-v2.md#best-practices).

```python { .notebook-cell }
task_name = "STS"

query = "I want to start a tax-free installment investment, what should I do?"
documents = [
    "Opening a NISA Account",
    "Opening a Regular Savings Account",
    "Home Loan Application Guide",
]

def evaluate_retrieval(query_text, docs):
    query_emb = model.encode(query_text, prompt_name=task_name)
    doc_embs = model.encode(docs, prompt_name=task_name)
    similarities = model.similarity(query_emb, doc_embs)[0]

    for doc, score in zip(docs, similarities):
        print(f"Document: {doc:<35} -> 🤖 Score: {score.item():.4f}")

print("Baseline Scores (Before Fine-Tuning):")
evaluate_retrieval(query, documents)
```

``` { .no-copy .output }
Baseline Scores (Before Fine-Tuning):
Document: Opening a NISA Account              -> 🤖 Score: 0.8552
Document: Opening a Regular Savings Account   -> 🤖 Score: 0.8463
Document: Home Loan Application Guide         -> 🤖 Score: 0.8078
```

## Prepare the Fine-Tuning Dataset

To teach the model domain-specific distinctions, we structure our training data as contrastive triplets: **`(anchor, positive, negative)`**:

- **`anchor`**: The user query or reference sentence.
- **`positive`**: A sentence that shares the exact domain intent (e.g., matching "NISA account" with "tax-free investment account").
- **`negative`**: A hard negative sentence from the same domain that is semantically distinct (e.g., "regular savings account" or "investment trust fee").

!!! note
    For demonstration purposes, we create a toy dataset of 3 triplets below. In real-world applications, use hundreds or thousands of domain-specific triplets for robust generalization.

```python { .notebook-cell }
from datasets import Dataset

triplets = [
    [
        "How do I open a NISA account?",
        "What is the procedure for starting a new tax-free investment account?",
        "I want to check the balance of my regular savings account.",
    ],
    [
        "Are there fees for making an early repayment on a home loan?",
        "If I pay back my house loan early, will there be any costs?",
        "What is the management fee for this investment trust?",
    ],
    [
        "What is the coverage for medical insurance?",
        "Tell me about the benefits of the health insurance plan.",
        "What is the cancellation policy for my life insurance?",
    ],
]

# Convert into a Hugging Face Dataset with 'anchor', 'positive', and 'negative' columns
data_as_dicts = [
    {"anchor": row[0], "positive": row[1], "negative": row[2]}
    for row in triplets
]
train_dataset = Dataset.from_list(data_as_dicts)
print(train_dataset)
```

``` { .no-copy .output }
Dataset({
    features: ['anchor', 'positive', 'negative'],
    num_rows: 3
})
```

## Fine-Tuning the Model

We fine-tune the model using `SentenceTransformerTrainer` and **`MultipleNegativesRankingLoss`** , which pulls `(anchor, positive)` pairs closer together in vector space while pushing `(anchor, negative)` pairs (and all other in-batch negatives) apart.

Crucially, we pass `prompts=model.prompts[task_name]` in `SentenceTransformerTrainingArguments` so the model is trained using the exact same task prefix used during inference.

```python { .notebook-cell }
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
from transformers import TrainerCallback

# Initialize contrastive ranking loss
loss = MultipleNegativesRankingLoss(model)

# Configure training arguments
args = SentenceTransformerTrainingArguments(
    output_dir="my-embeddinggemma-2",
    prompts=model.prompts[task_name],  # Train with the target task prompt prefix
    num_train_epochs=5,
    per_device_train_batch_size=1,
    learning_rate=2e-5,
    logging_steps=train_dataset.num_rows,
    report_to="none",
)

class EvaluationCallback(TrainerCallback):
    """Callback that evaluates retrieval scores at each logging step."""

    def __init__(self, eval_fn):
        self.eval_fn = eval_fn

    def on_log(self, args, state, control, **kwargs):
        print(f"\n--- Step {state.global_step} Evaluation ---")
        self.eval_fn()

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    loss=loss,
    callbacks=[EvaluationCallback(lambda: evaluate_retrieval(query, documents))],
)

trainer.train()
```

``` { .no-copy .output }
 [15/15 01:01, Epoch 5/5]
Step	Training Loss
3	0.145459
6	0.013492
9	0.003812
12	0.001927
15	0.001546

--- Step 3 Evaluation ---
Document: Opening a NISA Account              -> 🤖 Score: 0.8542
Document: Opening a Regular Savings Account   -> 🤖 Score: 0.8270
Document: Home Loan Application Guide         -> 🤖 Score: 0.7811

--- Step 6 Evaluation ---
Document: Opening a NISA Account              -> 🤖 Score: 0.8382
Document: Opening a Regular Savings Account   -> 🤖 Score: 0.8064
Document: Home Loan Application Guide         -> 🤖 Score: 0.7507

--- Step 9 Evaluation ---
Document: Opening a NISA Account              -> 🤖 Score: 0.8292
Document: Opening a Regular Savings Account   -> 🤖 Score: 0.7952
Document: Home Loan Application Guide         -> 🤖 Score: 0.7337

--- Step 12 Evaluation ---
Document: Opening a NISA Account              -> 🤖 Score: 0.8253
Document: Opening a Regular Savings Account   -> 🤖 Score: 0.7904
Document: Home Loan Application Guide         -> 🤖 Score: 0.7269

--- Step 15 Evaluation ---
Document: Opening a NISA Account              -> 🤖 Score: 0.8254
Document: Opening a Regular Savings Account   -> 🤖 Score: 0.7902
Document: Home Loan Application Guide         -> 🤖 Score: 0.7273
Writing model shards: 100%
 1/1 [00:13<00:00, 13.14s/it]

--- Step 15 Evaluation ---
Document: Opening a NISA Account              -> 🤖 Score: 0.8254
Document: Opening a Regular Savings Account   -> 🤖 Score: 0.7902
Document: Home Loan Application Guide         -> 🤖 Score: 0.7273
TrainOutput(global_step=15, training_loss=0.033247121175130206, metrics={'train_runtime': 61.8294, 'train_samples_per_second': 0.243, 'train_steps_per_second': 0.243, 'total_flos': 0.0, 'train_loss': 0.033247121175130206, 'epoch': 5.0})
```

## Evaluation After Fine-Tuning

Let's run our test query once more on the fine-tuned model and compare the results against our baseline:

```python { .notebook-cell }
print("Post-Training Scores (After Fine-Tuning):")
evaluate_retrieval(query, documents)
```

``` { .no-copy .output }
Post-Training Scores (After Fine-Tuning):
Document: Opening a NISA Account              -> 🤖 Score: 0.8254
Document: Opening a Regular Savings Account   -> 🤖 Score: 0.7902
Document: Home Loan Application Guide         -> 🤖 Score: 0.7273
```

### Results Comparison

| Candidate Document | Relevance | Score (Before) | Score (After) | Separation Margin vs. Top Hit |
| :--- | :--- | :---: | :---: | :---: |
| **Opening a NISA Account** | **Target Match** | `0.8552` | `0.8254` | — |
| **Opening a Regular Savings Account** | Hard Negative | `0.8463` | `0.7902` | **+0.0352** *(3.9x wider than 0.0091)* |
| **Home Loan Application Guide** | Unrelated Distractor | `0.8078` | `0.7273` | **+0.0981** *(2.2x wider than 0.0443)* |

Even with just a few training steps, fine-tuning significantly increased the margin between the true match (**Opening a NISA Account**) and competing financial documents, making downstream search and RAG pipelines much more reliable.

## Saving and Sharing Your Model

Save your fine-tuned model locally so it can be reloaded directly with `SentenceTransformer`:

```python { .notebook-cell }
# Save locally
model.save_pretrained("my-embeddinggemma-2-finetuned")
```

To share your adapted model with your team or deploy it from the Hugging Face Hub, use `push_to_hub`:

```python { .notebook-cell }
# Optional: Push your fine-tuned model to the Hugging Face Hub
model.push_to_hub("my-embeddinggemma-2-finetuned")
```

## Summary and next steps

You have learned how to adapt **EmbeddingGemma 2** to specialized domain terminology using contrastive triplet fine-tuning with the Sentence Transformers library.

Explore what more you can do with EmbeddingGemma:

- [Sentence Transformers Training Overview](https://sbert.net/docs/sentence_transformer/training_overview.html)
- [Multimodal Embeddings with EmbeddingGemma 2](multimodal.md)
- [Simple RAG with EmbeddingGemma in the Gemma Cookbook](https://github.com/google-gemma/cookbook/blob/main/tutorials/RAG_with_EmbeddingGemma.ipynb){ data-model="embeddinggemma-2" }

--8<-- "glossary.md"
