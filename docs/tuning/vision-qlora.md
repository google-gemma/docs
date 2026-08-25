---
title: Fine-Tune Gemma for Vision Tasks using Hugging Face Transformers and QLoRA
description: This guide walks you through how to fine-tune Gemma on a custom image and text dataset for a vision task (generating product descriptions) using Hugging Face Transformers and TRL.
models: [gemma-4]
frameworks: [transformers, peft, trl]
source_url: "https://ai.google.dev/gemma/docs/core/huggingface_vision_finetune_qlora.md.txt"
colab: https://colab.research.google.com/github/google-gemma/docs/blob/main/docs/notebooks/vision-qlora.ipynb
---
# Fine-Tune Gemma for Vision Tasks using Hugging Face Transformers and QLoRA


This guide walks you through how to fine-tune Gemma on a custom image and text dataset for a vision task (generating product descriptions) using Hugging Face [Transformers](https://huggingface.co/docs/transformers/index) and [TRL](https://huggingface.co/docs/trl/index). You will learn:

- What is Quantized Low-Rank Adaptation (QLoRA)
- Setup development environment
- Create and prepare the fine-tuning dataset
- Fine-tune Gemma using TRL and the SFTTrainer
- Test Model Inference and generate product descriptions from images and text.

!!! Note
    This guide requires a GPU which support bfloat16 data type such as NVIDIA L4 or NVIDIA A100 and more than 16GB of memory.


## What is Quantized Low-Rank Adaptation (QLoRA)

This guide demonstrates the use of [Quantized Low-Rank Adaptation (QLoRA)](https://arxiv.org/abs/2305.14314), which emerged as a popular method to efficiently fine-tune LLMs as it reduces computational resource requirements while maintaining high performance. In QloRA, the pretrained model is quantized to 4-bit and the weights are frozen. Then trainable adapter layers (LoRA) are attached and only the adapter layers are trained. Afterwards, the adapter weights can be merged with the base model or kept as a separate adapter.

## Setup development environment

The first step is to install Hugging Face Libraries, including TRL, and datasets to fine-tune open model.

```python { .notebook-cell }
# Install Pytorch & other libraries
%pip install torch tensorboard torchvision
%pip install -U torchao

# Install Transformers
%pip install "transformers>=5.10.1"

# Install Hugging Face libraries
%pip install datasets accelerate evaluate bitsandbytes trl "peft>=0.19.0" protobuf pillow sentencepiece

# COMMENT IN: if you are running on a GPU that supports BF16 data type and flash attn, such as NVIDIA L4 or NVIDIA A100
#%pip install flash-attn
```

*Note: If you are using a GPU with Ampere architecture (such as NVIDIA L4) or newer, you can use Flash attention. Flash Attention is a method that significantly speeds computations up and reduces memory usage from quadratic to linear in sequence length, leading to acelerating training up to 3x. Learn more at [FlashAttention](https://github.com/Dao-AILab/flash-attention/tree/main).*

You need a valid Hugging Face Token to publish your model. If you are running inside a Google Colab, you can securely use your Hugging Face Token using the Colab secrets otherwise you can set the token as directly in the `login` method. Make sure your token has write access too, as you push your model to the Hub during training.

```python { .notebook-cell }
# Login into Hugging Face Hub
from huggingface_hub import login
login()
```

## Create and prepare the fine-tuning dataset

When fine-tuning LLMs, it is important to know your use case and the task you want to solve. This helps you create a dataset to fine-tune your model. If you haven't defined your use case yet, you might want to go back to the drawing board.

As an example, this guide focuses on the following use case:

- Fine-tuning a Gemma model to generate concise, SEO-optimized product descriptions for an ecommerce platform, specifically tailored for mobile search.

This guide uses the [philschmid/amazon-product-descriptions-vlm](https://huggingface.co/datasets/philschmid/amazon-product-descriptions-vlm) dataset, a dataset of Amazon product descriptions, including product images and categories.

Hugging Face TRL supports multimodal conversations. The important piece is the "image" role, which tells the processing class that it should load the image. The structure should follow:

```text
{"messages": [{"role": "system", "content": [{"type": "text", "text":"You are..."}]}, {"role": "user", "content": [{"type": "text", "text": "..."}, {"type": "image"}]}, {"role": "assistant", "content": [{"type": "text", "text": "..."}]}]}
{"messages": [{"role": "system", "content": [{"type": "text", "text":"You are..."}]}, {"role": "user", "content": [{"type": "text", "text": "..."}, {"type": "image"}]}, {"role": "assistant", "content": [{"type": "text", "text": "..."}]}]}
{"messages": [{"role": "system", "content": [{"type": "text", "text":"You are..."}]}, {"role": "user", "content": [{"type": "text", "text": "..."}, {"type": "image"}]}, {"role": "assistant", "content": [{"type": "text", "text": "..."}]}]}
```

You can now use the Hugging Face Datasets library to load the dataset and create a prompt template to combine the image, product name, and category, and add a system message. The dataset includes images as`Pil.Image` objects.

```python { .notebook-cell }
from datasets import load_dataset
from PIL import Image

# System message for the assistant
system_message = "You are an expert product description writer for Amazon."

# User prompt that combines the user query and the schema
user_prompt = """Create a Short Product description based on the provided <PRODUCT> and <CATEGORY> and image.
Only return description. The description should be SEO optimized and for a better mobile search experience.

<PRODUCT>
{product}
</PRODUCT>

<CATEGORY>
{category}
</CATEGORY>
"""

# Convert dataset to OAI messages
def format_data(sample):
    return {
        "messages": [
            {
                "role": "system",
                #"content": [{"type": "text", "text": system_message}],
                "content": system_message,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt.format(
                            product=sample["Product Name"],
                            category=sample["Category"],
                        ),
                    },
                    {
                        "type": "image",
                        "image": sample["image"],
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["description"]}],
            },
        ],
    }

def process_vision_info(messages: list[dict]) -> list[Image.Image]:
    image_inputs = []
    # Iterate through each conversation
    for msg in messages:
        # Get content (ensure it's a list)
        content = msg.get("content", [])
        if not isinstance(content, list):
            content = [content]

        # Check each content element for images
        for element in content:
            if isinstance(element, dict) and (
                "image" in element or element.get("type") == "image"
            ):
                # Get the image and convert to RGB
                if "image" in element:
                    image = element["image"]
                else:
                    image = element
                image_inputs.append(image.convert("RGB"))
    return image_inputs

# Load dataset from the hub
dataset = load_dataset("philschmid/amazon-product-descriptions-vlm", split="train")
dataset = dataset.train_test_split(test_size=0.1)

# Convert dataset to OAI messages
# need to use list comprehension to keep Pil.Image type, .mape convert image to bytes
dataset_train = [format_data(sample) for sample in dataset["train"]]
dataset_test = [format_data(sample) for sample in dataset["test"]]

print(dataset_train[345]["messages"])
[{'role': 'system', 'content': 'You are an expert product description writer for Amazon.'}, {'role': 'user', 'content': [{'type': 'text', 'text': 'Create a Short Product description based on the provided <PRODUCT> and <CATEGORY> and image.\nOnly return description. The description should be SEO optimized and for a better mobile search experience.\n\n<PRODUCT>\nBachmann Trains inches 6 inches Pine Trees with Snow 6 Per Box\n</PRODUCT>\n\n<CATEGORY>\nToys & Games | Hobbies | Trains & Accessories | Accessories | Scenery\n</CATEGORY>\n'}, {'type': 'image', 'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=173x500 at 0x7ED5E4422C00>}]}, {'role': 'assistant', 'content': [{'type': 'text', 'text': 'Enhance your model train landscape with these realistic Bachmann Trains 6-inch pine trees, flocked with snow.  Six highly detailed trees per box, perfect for adding winter charm to your HO scale layout.  Create a captivating snowy scene!'}]}]
```

## Fine-tune Gemma using TRL and the SFTTrainer

You are now ready to fine-tune your model. Hugging Face TRL [SFTTrainer](https://huggingface.co/docs/trl/sft_trainer) makes it straightforward to supervise fine-tune open LLMs. The `SFTTrainer` is a subclass of the `Trainer` from the `transformers` library and supports all the same features, including logging, evaluation, and checkpointing, but adds additional quality of life features, including:

- Dataset formatting, including conversational and instruction formats
- Training on completions only, ignoring prompts
- Packing datasets for more efficient training
- Parameter-efficient fine-tuning (PEFT) support including QloRA
- Preparing the model and tokenizer for conversational fine-tuning (such as adding special tokens)

The following code loads the Gemma model and tokenizer from Hugging Face and initializes the quantization configuration.

```python { .notebook-cell .model-swap }
import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

# Hugging Face model id
model_id = "google/gemma-4-E2B"

# Check if GPU supports bfloat16
if not torch.cuda.is_bf16_supported():
    raise ValueError("GPU does not support bfloat16, please use a GPU that supports bfloat16.")

torch_dtype = torch.bfloat16

# Define model init arguments
model_kwargs = dict(
    dtype=torch_dtype, # What torch dtype to use
    device_map="auto", # Let torch decide how to load the model
)

# BitsAndBytesConfig int-4 config
model_kwargs["quantization_config"] = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_quant_storage=torch_dtype,
)

# Load model and processor
model = AutoModelForMultimodalLM.from_pretrained(model_id, **model_kwargs)
processor = AutoProcessor.from_pretrained("google/gemma-4-E2B-it") # Load the Instruction Tokenizer to use the official Gemma template

# Prepare model for QLoRA training
model = prepare_model_for_kbit_training(model)
Loading weights:   0%|          | 0/1951 [00:00<?, ?it/s]
/usr/local/lib/python3.12/dist-packages/bitsandbytes/backends/cuda/ops.py:213: FutureWarning: _check_is_size will be removed in a future PyTorch release along with guard_size_oblivious.     Use _check(i >= 0) instead.
  torch._check_is_size(blocksize)
```

The `SFTTrainer` supports a built-in integration with `peft`, which makes it straightforward to efficiently tune LLMs using QLoRA. You only need to create a `LoraConfig` and provide it to the trainer.

```python { .notebook-cell }
from peft import LoraConfig

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=16,
    bias="none",
    # no target_modules --- PEFT's Gemma 4 defaults scope to the LM layers
    task_type="CAUSAL_LM",
    modules_to_save=["lm_head", "embed_tokens"], # make sure to save the lm_head and embed_tokens as you train the special tokens
    ensure_weight_tying=True,
)
```

Before you can start your training, you need to define the hyperparameter you want to use in a `SFTConfig` and a custom `collate_fn` to handle the vision processing. The `collate_fn` converts the messages with text and images into a format that the model can understand.

```python { .notebook-cell }
from trl import SFTConfig

args = SFTConfig(
    output_dir="gemma-product-description",     # directory to save and repository id
    num_train_epochs=3,                         # number of training epochs
    per_device_train_batch_size=1,              # batch size per device during training
    per_device_eval_batch_size=1,               # batch size per device during evaluation
    optim="adamw_torch_fused",                  # use fused adamw optimizer
    logging_steps=5,                            # log every 5 steps
    save_strategy="epoch",                      # save checkpoint every epoch
    eval_strategy="epoch",                      # evaluate checkpoint every epoch
    learning_rate=2e-4,                         # learning rate, based on QLoRA paper
    bf16=True,                                  # use bfloat16 precision
    max_grad_norm=0.3,                          # max gradient norm based on QLoRA paper
    lr_scheduler_type="constant",               # use constant learning rate scheduler
    push_to_hub=True,                           # push model to hub
    report_to="tensorboard",                    # report metrics to tensorboard
    dataset_text_field="",                      # need a dummy field for collator
    dataset_kwargs={"skip_prepare_dataset": True}, # important for collator
    remove_unused_columns = False               # important for collator
)

# Create a data collator to encode text and image pairs
def collate_fn(examples):
    texts = []
    images = []
    for example in examples:
        image_inputs = process_vision_info(example["messages"])
        text = processor.apply_chat_template(
            example["messages"], add_generation_prompt=False, tokenize=False
        )
        texts.append(text.strip())
        images.append(image_inputs)

    # Tokenize the texts and process the images
    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    # The labels are the input_ids, and we mask the padding tokens and image tokens in the loss computation
    labels = batch["input_ids"].clone()

    # Mask tokens for not being used in the loss computation
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == processor.tokenizer.boi_token_id] = -100
    labels[labels == processor.tokenizer.image_token_id] = -100
    labels[labels == processor.tokenizer.eoi_token_id] = -100

    batch["labels"] = labels
    return batch
/tmp/ipykernel_26645/1232990497.py:3: FutureWarning: The default `loss_type` will change from `'nll'` to `'chunked_nll'` in TRL 1.7. For standard models this is transparent (same math, lower memory) and no action is needed — you'll get the new default automatically on upgrade. If you use a custom model, check ahead of time that `loss_type='chunked_nll'` runs and yields the same loss as `'nll'`; if it doesn't, pin `loss_type='nll'` to keep the current behavior and please open an issue at https://github.com/huggingface/trl/issues so we can address the edge case.
  args = SFTConfig(
```

You now have every building block you need to create your `SFTTrainer` to start the training of your model.

```python { .notebook-cell }
from trl import SFTTrainer

# Create Trainer object
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset_train,
    eval_dataset=dataset_test,
    peft_config=peft_config,
    processing_class=processor,
    data_collator=collate_fn,
)
```

Start training by calling the `train()` method.

```python { .notebook-cell }
# Start training, the model will be automatically saved to the Hub and the output directory
trainer.train()

# Save the final model again to the Hugging Face Hub
trainer.save_model()
[transformers] The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. The model config and generation config were aligned accordingly, being updated with the tokenizer's values. Updated tokens: {'eos_token_id': 1, 'bos_token_id': 2, 'pad_token_id': 0}.
```
\[3630/3630 57:38, Epoch 3/3\]

| Epoch | Training Loss | Validation Loss | Entropy | Mean Token Accuracy | Num Tokens |
|---|---|---|---|---|---|
| 1 | 0.835934 | 0.844098 | 0.735220 | 0.818571 | 521741.000000 |
| 2 | 0.411868 | 0.886929 | 0.548703 | 0.822062 | 1043482.000000 |
| 3 | 0.488929 | 1.007192 | 0.417825 | 0.822075 | 1565223.000000 |


``` { .no-copy .output }
Processing Files (0 / 0)      : |          |  0.00B /  0.00B
New Data Upload               : |          |  0.00B /  0.00B
...34898.c7e1d0064144.7818.0: 100%|##########| 43.2kB / 43.2kB
...escription/tokenizer.json: 100%|##########| 32.2MB / 32.2MB
...adapter_model.safetensors:   5%|4         | 80.0MB / 1.62GB
...5378.c7e1d0064144.10487.0: 100%|##########|  289kB /  289kB
...ription/training_args.bin: 100%|##########| 5.71kB / 5.71kB
No files have been modified since last commit. Skipping to prevent empty commit.
WARNING:huggingface_hub.hf_api:No files have been modified since last commit. Skipping to prevent empty commit.
```

Before you can test your model, make sure to free the memory.

```python { .notebook-cell }
# free the memory again
del model
del trainer
torch.cuda.empty_cache()
```

When using QLoRA, you only train adapters and not the full model. This means when saving the model during training you only save the adapter weights and not the full model. If you want to save the full model, which makes it easier to use with serving stacks like vLLM or TGI, you can merge the adapter weights into the model weights using the `merge_and_unload` method and then save the model with the `save_pretrained` method. This saves a default model, which can be used for inference.

!!! Note
    It requires more than 30GB of CPU Memory when you want to merge the adapter into the model. You can skip this and continue with Test Model Inference.


```python { .notebook-cell }
from transformers import AutoModelForMultimodalLM, AutoProcessor
from peft import PeftModel

# Load Model base model
model = AutoModelForMultimodalLM.from_pretrained(model_id, low_cpu_mem_usage=True)

# Merge LoRA and base model and save
peft_model = PeftModel.from_pretrained(model, args.output_dir)
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained("merged_model", safe_serialization=True, max_shard_size="2GB")

processor = AutoProcessor.from_pretrained(args.output_dir)
processor.save_pretrained("merged_model")
Loading weights:   0%|          | 0/1951 [00:00<?, ?it/s]
Writing model shards:   0%|          | 0/4 [00:00<?, ?it/s]
['merged_model/processor_config.json']
```

## Test Model Inference and generate product descriptions

After the training is done, you'll want to evaluate and test your model. You can load different samples from the test dataset and evaluate the model on those samples.

!!! Note
    Evaluating Generative AI models is not a trivial task since one input can have multiple correct outputs. This guide only focuses on manual evaluation and vibe checks.


```python { .notebook-cell }
from transformers import AutoModelForMultimodalLM, AutoProcessor

model_id = "merged_model"

# Load Model with PEFT adapter
model = AutoModelForMultimodalLM.from_pretrained(
  model_id,
  device_map="auto",
  dtype="auto",
)
processor = AutoProcessor.from_pretrained(model_id)
Loading weights:   0%|          | 0/1951 [00:00<?, ?it/s]
```

You can test inference by providing a product name, category and image. The `sample` includes a marvel action figure.

```python { .notebook-cell }
import requests
from PIL import Image

# Test sample with Product Name, Category and Image
sample = {
  "product_name": "Hasbro Marvel Avengers-Serie Marvel Assemble Titan-Held, Iron Man, 30,5 cm Actionfigur",
  "category": "Toys & Games | Toy Figures & Playsets | Action Figures",
  "image": Image.open(requests.get("https://m.media-amazon.com/images/I/81+7Up7IWyL._AC_SY300_SX300_.jpg", stream=True).raw).convert("RGB")
}

def generate_description(sample, model, processor):
    # Convert sample into messages and then apply the chat template
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": [
            {"type": "image","image": sample["image"]},
            {"type": "text", "text": user_prompt.format(product=sample["product_name"], category=sample["category"])},
        ]},
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print(text)
    # Process the image and text
    image_inputs = process_vision_info(messages)
    # Tokenize the text and process the images
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    # Move the inputs to the device
    inputs = inputs.to(model.device)

    # Generate the output
    stop_token_ids = [processor.tokenizer.eos_token_id, processor.tokenizer.convert_tokens_to_ids("<turn|>")]
    generated_ids = model.generate(**inputs, max_new_tokens=256, top_p=1.0, do_sample=True, temperature=0.8, eos_token_id=stop_token_ids, disable_compile=True)
    # Trim the generation and decode the output to text
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

# generate the description
description = generate_description(sample, model, processor)
print("MODEL OUTPUT>> \n")
print(description)
<bos><|turn>system
You are an expert product description writer for Amazon.<turn|>
<|turn>user
<|image|>Create a Short Product description based on the provided <PRODUCT> and <CATEGORY> and image.
Only return description. The description should be SEO optimized and for a better mobile search experience.

<PRODUCT>
Hasbro Marvel Avengers-Serie Marvel Assemble Titan-Held, Iron Man, 30,5 cm Actionfigur
</PRODUCT>

<CATEGORY>
Toys & Games | Toy Figures & Playsets | Action Figures
</CATEGORY><turn|>
<|turn>model

MODEL OUTPUT>> 

Bring home the action with the Hasbro Marvel Avengers Assemble Titan-held Iron Man Action Figure! This highly detailed 30.5-centimeter figure is perfect for recreating epic battles.  Collect the entire Marvel Avengers Assemble Titan-held series for an authentic experience.  Shop now!
```

## Summary and next steps

This tutorial covered how to fine-tune a Gemma model for vision tasks using TRL and QLoRA, specifically for generating product descriptions. Check out the following docs next:

- Learn how to [generate text with a Gemma model](../get-started/index.md).
- Learn how to [fine-tune Gemma for text tasks using Hugging Face Transformers](qlora.md).
- Learn how to [full model fine-tune using Hugging Face Transformers](full-finetune.md).
- Learn how to perform [distributed fine-tuning and inference on a Gemma model](distributed.md).
- Learn how to [use Gemma open models with Vertex AI][vertex-use-gemma].
- Learn how to [fine-tune Gemma using KerasNLP and deploy to Vertex AI](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/model_garden/model_garden_gemma_kerasnlp_to_vertexai.ipynb).
