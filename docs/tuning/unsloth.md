---
title: Unsloth
description: Swap Unsloth's FastVisionModel or FastModel into any single-GPU Gemma 4 tutorial for faster LoRA, QLoRA, and GGUF export.
models: [gemma-4]
frameworks: [unsloth, trl]
---

# Unsloth

Every hands-on guide in this section uses Hugging Face `transformers`, `trl`, and `peft`. We did this to make sure that a single Python environment works across single-GPU and multi-GPU runs. However, when you are only using a single GPU, you can drop [Unsloth](https://unsloth.ai/docs/models/gemma-4){ data-model="gemma-4" } into any tutorial by replacing only the model-loading and LoRA setup step. Everything else, like your dataset formatting, collator, and TRL trainer stays the same.

---

## 1. Install Unsloth

You start by installing `unsloth` alongside the Hugging Face training stack:

```python
# Install Unsloth alongside the standard post-training libraries
%pip install -q -U unsloth "transformers>=5.10.1" "trl>=0.16.0" "peft>=0.19.0" datasets accelerate
```

---

## 2. Swap model loading and LoRA setup

In any tutorial that uses only 1 GPU (which are most), replace:

* `BitsAndBytesConfig`, `AutoModelForMultimodalLM.from_pretrained`, and `LoraConfig` 

with:

* `FastVisionModel.from_pretrained` and `FastVisionModel.get_peft_model`

```python { .model-swap }
import unsloth  # Import unsloth before transformers, trl, or peft
from unsloth import FastVisionModel

# Load the Gemma 4 checkpoint in 4-bit precision with Unsloth gradient checkpointing
model, processor = FastVisionModel.from_pretrained(
    "google/gemma-4-E2B-it",
    load_in_4bit=True,                  # Set load_in_4bit=False, load_in_16bit=True for 16-bit LoRA
    use_gradient_checkpointing="unsloth",
)

# Attach LoRA adapters to the attention and MLP projections
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=False,       # Set True when adapting the vision encoder
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,
    lora_alpha=32,
    lora_dropout=0.0,
    bias="none",
)
```

You can see below what other minor changes are needed for specific tutorials:

| Tutorial | Loader class | Key flags |
| :--- | :--- | :--- |
| [(Q)LoRA](qlora.md), [Continued pre-training](continued-pretraining.md), [Preference optimization](preference-alignment.md), [Verifiable rewards](reinforcement-learning.md) | `FastVisionModel` | `finetune_vision_layers=False, finetune_language_layers=True` |
| [MoE fine-tuning](moe-finetune.md) (26B A4B) | `FastVisionModel` | `finetune_mlp_modules=True` adapts the shared expert and all 128 routed experts while keeping the router frozen |
| [Vision fine-tuning](vision-qlora.md) | `FastVisionModel` | Set `processor.image_processor.max_soft_tokens = 560`; pass `UnslothVisionDataCollator(model, processor)` or the tutorial's `collate_fn` |
| [Audio fine-tuning](audio-qlora.md) | `FastModel` | Set `os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"` before importing `unsloth` so variable-length audio batches do not trigger recompilation |

---

## 3. Train the model

Unsloth is quite handy and already does the configuration of the quantization for you so you only need to run the following to enable training:

```python
from trl import SFTConfig, SFTTrainer

# Enable training mode before running the trainer
FastVisionModel.for_training(model)

# Define and run the trainer with the exact same dataset and SFTConfig
trainer = SFTTrainer(
    model=model,
    args=SFTConfig(output_dir="gemma-4-unsloth-lora", max_length=2048, learning_rate=2e-4, bf16=True),
    train_dataset=dataset["train"],
    processing_class=processor.tokenizer,
)
trainer.train()
```

---

## 4. Export to merged 16-bit or GGUF

Unsloth includes helper methods to fold LoRA adapters into standalone 16-bit SafeTensors or export directly to quantized GGUF files for [llama.cpp](../deployment/llamacpp.md) and [Ollama](../deployment/ollama.md):

```python
# Save a standalone 16-bit merged checkpoint for vLLM or LiteRT-LM conversion
model.save_pretrained_merged("merged_model", processor, save_method="merged_16bit")

# Or export directly to a 4-bit Q4_K_M GGUF file
model.save_pretrained_gguf("gemma-4-gguf", processor.tokenizer, quantization_method="q4_k_m")
```

* For multi-GPU sharding and quantization-aware training, see [Scaling](scaling.md).
* For vLLM and LiteRT-LM mobile export, see [Export](export.md).
* For running Unsloth Studio and dynamic-quant GGUFs locally, see [Run Gemma with Unsloth](../deployment/unsloth.md).

---

## What's next

<div class="card-grid">
<a class="card" href="qlora.md">
  <span class="card-icon">{{ icon('lucide/zap') }}</span>
  <div class="card-title">(Q)LoRA</div>
  <p class="card-desc">Walk through a complete single-GPU QLoRA fine-tuning run from dataset to merged model.</p>
</a>
<a class="card" href="moe-finetune.md">
  <span class="card-icon">{{ icon('lucide/split') }}</span>
  <div class="card-title">MoE fine-tuning</div>
  <p class="card-desc">Adapt all 128 routed experts in Gemma 4 26B A4B while keeping the router frozen.</p>
</a>
<a class="card" href="vision-qlora.md">
  <span class="card-icon">{{ icon('lucide/eye') }}</span>
  <div class="card-title">Vision fine-tuning</div>
  <p class="card-desc">Fine-tune Gemma 4 on structured document and image datasets with selectable token budgets.</p>
</a>
<a class="card" href="export.md">
  <span class="card-icon">{{ icon('lucide/rocket') }}</span>
  <div class="card-title">Export</div>
  <p class="card-desc">Deploy fine-tuned checkpoints across vLLM, GGUF, and LiteRT-LM.</p>
</a>
</div>
