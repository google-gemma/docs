---
title: Fine-Tune Gemma using Hugging Face Transformers and QloRA
description: This guide walks you through how to fine-tune Gemma on a custom text-to-sql dataset using Hugging Face Transformers and TRL.
models: [gemma-4]
frameworks: [transformers, peft, trl]
source_url: "https://ai.google.dev/gemma/docs/core/huggingface_text_finetune_qlora.md.txt"
colab: https://colab.research.google.com/github/google-gemma/docs/blob/main/docs/notebooks/qlora-gemma-4-e2b.ipynb
---
# Fine-Tune Gemma using Hugging Face Transformers and QloRA


This guide walks you through how to fine-tune Gemma on a custom text-to-sql dataset using Hugging Face [Transformers](https://huggingface.co/docs/transformers/index) and [TRL](https://huggingface.co/docs/trl/index). You will learn:

- What is Quantized Low-Rank Adaptation (QLoRA)
- Setup development environment
- Create and prepare the fine-tuning dataset
- Fine-tune Gemma using TRL and the SFTTrainer
- Test Model Inference and generate SQL queries

!!! Note
    This guide was created to run on a Google colaboratory account using a NVIDIA T4 GPU with 16GB and Gemma 1B, but can be adapted to run on bigger GPUs and bigger models.


## What is Quantized Low-Rank Adaptation (QLoRA)

This guide demonstrates the use of [Quantized Low-Rank Adaptation (QLoRA)](https://arxiv.org/abs/2305.14314), which emerged as a popular method to efficiently fine-tune LLMs as it reduces computational resource requirements while maintaining high performance. In QloRA, the pretrained model is quantized to 4-bit and the weights are frozen. Then trainable adapter layers (LoRA) are attached and only the adapter layers are trained. Afterwards, the adapter weights can be merged with the base model or kept as a separate adapter.

## Setup development environment

The first step is to install Hugging Face Libraries, including TRL, and datasets to fine-tune open model, including different RLHF and alignment techniques.

```python { .notebook-cell }
# Install Pytorch & other libraries
%pip install torch tensorboard
%pip install -U torchao

# Install Transformers
%pip install "transformers>=5.10.1"

# Install Hugging Face libraries
%pip install datasets accelerate evaluate bitsandbytes trl "peft>=0.19.0" protobuf sentencepiece

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

- Fine-tune a natural language to SQL model for seamless integration into a data analysis tool. The objective is to significantly reduce the time and expertise required for SQL query generation, enabling even non-technical users to extract meaningful insights from data.

Text-to-SQL can be a good use case for fine-tuning LLMs, as it is a complex task that requires a lot of (internal) knowledge about the data and the SQL language.

Once you have determined that fine-tuning is the right solution, you need a dataset to fine-tune. The dataset should be a diverse set of demonstrations of the task(s) you want to solve. There are several ways to create such a dataset, including:

- Using existing open-source datasets, such as [Spider](https://huggingface.co/datasets/spider)
- Using synthetic datasets created by LLMs, such as [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)
- Using datasets created by humans, such as [Dolly](https://huggingface.co/datasets/databricks/databricks-dolly-15k).
- Using a combination of the methods, such as [Orca](https://huggingface.co/datasets/Open-Orca/OpenOrca)

Each of the methods has its own advantages and disadvantages and depends on the budget, time, and quality requirements. For example, using an existing dataset is the easiest but might not be tailored to your specific use case, while using domain experts might be the most accurate but can be time-consuming and expensive. It is also possible to combine several methods to create an instruction dataset, as shown in [Orca: Progressive Learning from Complex Explanation Traces of GPT-4.](https://arxiv.org/abs/2306.02707)

This guide uses an already existing dataset ([philschmid/gretel-synthetic-text-to-sql](https://huggingface.co/datasets/philschmid/gretel-synthetic-text-to-sql)), a high quality synthetic Text-to-SQL dataset including natural language instructions, schema definitions, reasoning and the corresponding SQL query.

[Hugging Face TRL](https://huggingface.co/docs/trl/en/index) supports automatic templating of conversation dataset formats. This means you only need to convert your dataset into the right json objects, and `trl` takes care of templating and putting it into the right format. With the prompt-completion format below, `trl` also computes the loss on the completion only, so the model learns to write the SQL query rather than to repeat the prompt.

```text
{"prompt": [{"role": "system", "content": "You are..."}, {"role": "user", "content": "..."}], "completion": [{"role": "assistant", "content": "..."}]}
{"prompt": [{"role": "system", "content": "You are..."}, {"role": "user", "content": "..."}], "completion": [{"role": "assistant", "content": "..."}]}
{"prompt": [{"role": "system", "content": "You are..."}, {"role": "user", "content": "..."}], "completion": [{"role": "assistant", "content": "..."}]}
```

The [philschmid/gretel-synthetic-text-to-sql](https://huggingface.co/datasets/philschmid/gretel-synthetic-text-to-sql) contains over 100k samples. To keep the guide small, it is downsampled to only use 10,000 samples.

You can now use the Hugging Face Datasets library to load the dataset and create a prompt template to combine the natural language instruction, schema definition and add a system message for your assistant.

```python { .notebook-cell }
from datasets import load_dataset

# System message for the assistant
system_message = """You are a text to SQL query translator. Users will ask you questions in English and you will generate a SQL query based on the provided SCHEMA."""

# User prompt that combines the user query and the schema
user_prompt = """Given the <USER_QUERY> and the <SCHEMA>, generate the corresponding SQL command to retrieve the desired data, considering the query's syntax, semantics, and schema constraints.

<SCHEMA>
{context}
</SCHEMA>

<USER_QUERY>
{question}
</USER_QUERY>
"""
def create_conversation(sample, idx):
  return {
    "prompt": [
      {"role": "system", "content": system_message},
      {"role": "user", "content": user_prompt.format(question=sample["sql_prompt"], context=sample["sql_context"])},
    ],
    "completion": [
      {"role": "assistant", "content": sample["sql"]}
    ],
  }

# Load dataset from the hub
dataset = load_dataset("philschmid/gretel-synthetic-text-to-sql", split="train")
dataset = dataset.select(range(1250))

# Convert dataset to OAI messages
dataset = dataset.map(create_conversation, with_indices=True, remove_columns=dataset.features)
# split dataset into 80% training samples and 20% test samples
dataset = dataset.train_test_split(test_size=0.2, shuffle=False)

# Print formatted user prompt
for item in dataset["train"][0]["prompt"] + dataset["train"][0]["completion"]:
  print(item)
```

```{ .no-copy .output }
{'role': 'system', 'content': 'You are a text to SQL query translator. Users will ask you questions in English and you will generate a SQL query based on the provided SCHEMA.'}
{'role': 'user', 'content': "Given the <USER_QUERY> and the <SCHEMA>, generate the corresponding SQL command to retrieve the desired data, considering the query's syntax, semantics, and schema constraints.\n\n<SCHEMA>\nCREATE TABLE salesperson (salesperson_id INT, name TEXT, region TEXT); INSERT INTO salesperson (salesperson_id, name, region) VALUES (1, 'John Doe', 'North'), (2, 'Jane Smith', 'South'); CREATE TABLE timber_sales (sales_id INT, salesperson_id INT, volume REAL, sale_date DATE); INSERT INTO timber_sales (sales_id, salesperson_id, volume, sale_date) VALUES (1, 1, 120, '2021-01-01'), (2, 1, 150, '2021-02-01'), (3, 2, 180, '2021-01-01');\n</SCHEMA>\n\n<USER_QUERY>\nWhat is the total volume of timber sold by each salesperson, sorted by salesperson?\n</USER_QUERY>\n"}
{'role': 'assistant', 'content': 'SELECT salesperson_id, name, SUM(volume) as total_volume FROM timber_sales JOIN salesperson ON timber_sales.salesperson_id = salesperson.salesperson_id GROUP BY salesperson_id, name ORDER BY total_volume DESC;'}
```

## Fine-tune Gemma using TRL and the SFTTrainer

You are now ready to fine-tune your model. Hugging Face TRL [SFTTrainer](https://huggingface.co/docs/trl/sft_trainer) makes it straightforward to supervise fine-tune open LLMs. The `SFTTrainer` is a subclass of the `Trainer` from the `transformers` library and supports all the same features, including logging, evaluation, and checkpointing, but adds additional quality of life features, including:

- Dataset formatting, including conversational and instruction formats
- Training on completions only, ignoring prompts
- Packing datasets for more efficient training
- Parameter-efficient fine-tuning (PEFT) support including QloRA
- Preparing the model and tokenizer for conversational fine-tuning (such as adding special tokens)

The following code defines the quantization configuration and loads the tokenizer. The `SFTTrainer` loads and quantizes the model for you.

```python { .notebook-cell .model-swap }
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig

# Hugging Face model id
model_id = "google/gemma-4-E2B"

# Check if GPU supports bfloat16
if torch.cuda.is_bf16_supported():
    dtype = "bfloat16"
else:
    dtype = "float16"

# BitsAndBytesConfig: Enables 4-bit quantization to reduce model size/memory usage
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=dtype,
    bnb_4bit_quant_storage=dtype,
)

# The base model ships no chat template, so load the tokenizer of the instruction-tuned model
tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E2B-it")
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

Before you can start your training, you need to define the hyperparameter you want to use in a `SFTConfig` instance.

```python { .notebook-cell }
from trl import SFTConfig

args = SFTConfig(
    output_dir="gemma-text-to-sql",         # directory to save and repository id
    num_train_epochs=3,                     # number of training epochs
    per_device_train_batch_size=1,          # batch size per device during training
    per_device_eval_batch_size=1,           # batch size per device during evaluation
    optim="adamw_torch_fused",              # use fused adamw optimizer
    logging_steps=10,                       # log every 10 steps
    save_strategy="epoch",                  # save checkpoint every epoch
    eval_strategy="epoch",                  # evaluate checkpoint every epoch
    learning_rate=2e-4,                     # learning rate
    fp16=dtype == "float16",                # use float16 precision
    bf16=dtype == "bfloat16",               # use bfloat16 precision
    model_init_kwargs={"dtype": dtype},     # dtype of the weights that are not quantized
    lr_scheduler_type="constant",           # use constant learning rate scheduler
    push_to_hub=True,                       # push model to hub
    report_to="tensorboard",                # report metrics to tensorboard
)
```

You now have every building block you need to create your `SFTTrainer` to start the training of your model.

```python { .notebook-cell }
from trl import SFTTrainer

# Create Trainer object
trainer = SFTTrainer(
    model=model_id,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
    quantization_config=quantization_config,
    peft_config=peft_config,
)
```

Start training by calling the `train()` method.

```python { .notebook-cell }
# Start training, the model will be automatically saved to the Hub and the output directory
trainer.train()

# Save the final model again to the Hugging Face Hub
trainer.save_model()
```

```{ .no-copy .output }
[3000/3000 1:51:30, Epoch 3/3]

| Epoch | Training Loss | Validation Loss | Entropy | Mean Token Accuracy | Num Tokens |
|---|---|---|---|---|---|
| 1 | 0.315478 | 0.267880 | 0.204330 | 0.927141 | 246435.000000 |
| 2 | 0.139383 | 0.297303 | 0.131811 | 0.929568 | 492870.000000 |
| 3 | 0.150697 | 0.315715 | 0.110690 | 0.927032 | 739305.000000 |
```

Before you can test your model, make sure to free the memory.

```python { .notebook-cell }
# free the memory again
del trainer
torch.cuda.empty_cache()
```

When using QLoRA, you only train adapters and not the full model. This means when saving the model during training you only save the adapter weights and not the full model. If you want to save the full model, which makes it easier to use with serving stacks like vLLM or TGI, you can merge the adapter weights into the model weights using the `merge_and_unload` method and then save the model with the `save_pretrained` method. This saves a default model, which can be used for inference.

!!! Note
    It requires more than 30GB of CPU Memory when you want to merge the adapter into the model. You can skip this and continue with Test Model Inference.


```python { .notebook-cell .model-swap }
from transformers import AutoModelForMultimodalLM, AutoProcessor
from peft import PeftModel

# Load Model base model
model = AutoModelForMultimodalLM.from_pretrained(model_id, low_cpu_mem_usage=True)

# Merge LoRA and base model and save
peft_model = PeftModel.from_pretrained(model, args.output_dir)
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained("merged_model", safe_serialization=True, max_shard_size="2GB")

processor = AutoProcessor.from_pretrained("google/gemma-4-E2B-it")
processor.save_pretrained("merged_model")
```

```{ .no-copy .output }
Loading weights:   0%|          | 0/1951 [00:00<?, ?it/s]
Writing model shards:   0%|          | 0/4 [00:00<?, ?it/s]
['merged_model/processor_config.json']
```

## Test Model Inference and generate SQL queries

After the training is done, you'll want to evaluate and test your model. You can load different samples from the test dataset and evaluate the model on those samples.

!!! Note
    Evaluating generative AI models is not a trivial task since one input can have multiple correct outputs. This guide only focuses on manual evaluation and vibe checks.


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
```

```{ .no-copy .output }
Loading weights:   0%|          | 0/1951 [00:00<?, ?it/s]
```

Let's load a random sample from the test dataset and generate a SQL command.

```python { .notebook-cell }
from random import randint
import re
from transformers import pipeline, GenerationConfig, pipeline

config = GenerationConfig.from_pretrained(model_id)
config.max_new_tokens = 256
config.eos_token_id = [processor.tokenizer.convert_tokens_to_ids("<turn|>")]

# Load the model and tokenizer into the pipeline
pipe = pipeline("text-generation", model=model, tokenizer=processor.tokenizer)

# Load a random sample from the test dataset
rand_idx = randint(0, len(dataset["test"]))
test_sample = dataset["test"][rand_idx]

# Convert as test example into a prompt with the Gemma template
prompt = processor.tokenizer.apply_chat_template(test_sample["prompt"], tokenize=False, add_generation_prompt=True)
print(prompt)

# Generate our SQL query.
outputs = pipe(text_inputs=prompt, generation_config=config)

# Extract the user query and original answer
print(f"Context:\n", re.search(r'<SCHEMA>\n(.*?)\n</SCHEMA>', test_sample['prompt'][1]['content'], re.DOTALL).group(1).strip())
print(f"Query:\n", re.search(r'<USER_QUERY>\n(.*?)\n</USER_QUERY>', test_sample['prompt'][1]['content'], re.DOTALL).group(1).strip())
print(f"Original Answer:\n{test_sample['completion'][0]['content']}")
print(f"Generated Answer:\n{outputs[0]['generated_text'][len(prompt):].strip().removesuffix("<turn|>")}")
```

```{ .no-copy .output }
<bos><|turn>system
You are a text to SQL query translator. Users will ask you questions in English and you will generate a SQL query based on the provided SCHEMA.<turn|>
<|turn>user
Given the <USER_QUERY> and the <SCHEMA>, generate the corresponding SQL command to retrieve the desired data, considering the query's syntax, semantics, and schema constraints.

<SCHEMA>
CREATE TABLE Auto_Shows (id INT, show_name VARCHAR(255), show_year INT, location VARCHAR(255)); INSERT INTO Auto_Shows (id, show_name, show_year, location) VALUES (1, 'New York International Auto Show', 2019, 'United States'); INSERT INTO Auto_Shows (id, show_name, show_year, location) VALUES (2, 'Chicago Auto Show', 2019, 'United States'); INSERT INTO Auto_Shows (id, show_name, show_year, location) VALUES (3, 'North American International Auto Show', 2018, 'United States');
</SCHEMA>

<USER_QUERY>
How many auto shows were held in the United States in the year 2019?
</USER_QUERY><turn|>
<|turn>model

Context:
 CREATE TABLE Auto_Shows (id INT, show_name VARCHAR(255), show_year INT, location VARCHAR(255)); INSERT INTO Auto_Shows (id, show_name, show_year, location) VALUES (1, 'New York International Auto Show', 2019, 'United States'); INSERT INTO Auto_Shows (id, show_name, show_year, location) VALUES (2, 'Chicago Auto Show', 2019, 'United States'); INSERT INTO Auto_Shows (id, show_name, show_year, location) VALUES (3, 'North American International Auto Show', 2018, 'United States');
Query:
 How many auto shows were held in the United States in the year 2019?
Original Answer:
SELECT COUNT(*) FROM Auto_Shows WHERE show_year = 2019 AND location = 'United States';
Generated Answer:
SELECT COUNT(*) FROM Auto_Shows WHERE location = 'United States' AND show_year = 2019;
```

## Summary and next steps

This tutorial covered how to fine-tune a Gemma model using TRL and QLoRA. Check out the following docs next:

- Learn how to [generate text with a Gemma model](../get-started/index.md).
- Learn how to [fine-tune Gemma for vision tasks using Hugging Face Transformers](vision-qlora.md).
- Learn how to [use Gemma open models with Vertex AI][vertex-use-gemma].
- Learn how to [fine-tune Gemma using KerasNLP and deploy to Vertex AI](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/model_garden/model_garden_gemma_kerasnlp_to_vertexai.ipynb).
