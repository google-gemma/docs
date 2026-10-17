---
title: Full Model Fine-Tune using Hugging Face Transformers
description: This guide walks you through how to fine-tune Gemma on a mobile game NPC dataset using Hugging Face Transformers and TRL.
models: [gemma-3]
frameworks: [transformers, trl]
source_url: "https://ai.google.dev/gemma/docs/core/huggingface_text_full_finetune.md.txt"
colab: https://colab.research.google.com/github/google-gemma/docs/blob/main/docs/notebooks/full-finetune.ipynb
---
# Full Model Fine-Tune using Hugging Face Transformers


This guide walks you through how to fine-tune Gemma on a mobile game NPC dataset using Hugging Face [Transformers](https://huggingface.co/docs/transformers/index) and [TRL](https://huggingface.co/docs/trl/index). You will learn:

- Setup development environment
- Prepare the fine-tuning dataset
- Full model fine-tuning Gemma using TRL and the SFTTrainer
- Test Model Inference and vibe checks

!!! Note
    This guide was created to run on a Google colaboratory account using a NVIDIA T4 GPU with 16GB and Gemma 270m, but can be adapted to run on bigger GPUs and bigger models.
    

## Setup development environment

The first step is to install Hugging Face Libraries, including TRL, and datasets to fine-tune open model, including different RLHF and alignment techniques.

```python { .notebook-cell }
# Install Pytorch & other libraries
%pip install torch tensorboard

# Install Hugging Face libraries
%pip install transformers datasets accelerate evaluate trl protobuf sentencepiece

# COMMENT IN: if you are running on a GPU that supports BF16 data type and flash attn, such as NVIDIA L4 or NVIDIA A100
#% pip install flash-attn
```

!!! Note
    If you are using a GPU with Ampere architecture (such as NVIDIA L4) or newer, you can use Flash attention. Flash Attention is a method that significantly speeds computations up and reduces memory usage from quadratic to linear in sequence length, leading to acelerating training up to 3x. Learn more at [FlashAttention](https://github.com/Dao-AILab/flash-attention/tree/main).*


Before you can start training, you have to make sure that you accepted the terms of use for Gemma. You can accept the license on [Hugging Face](http://huggingface.co/google/gemma-3-270m-it){ data-model="gemma-3" } by clicking on the Agree and access repository button on the model page at: <http://huggingface.co/google/gemma-3-270m-it>

After you have accepted the license, you need a valid Hugging Face Token to access the model. If you are running inside a Google Colab, you can securely use your Hugging Face Token using the Colab secrets otherwise you can set the token as directly in the `login` method. Make sure your token has write access too, as you push your model to the Hub during training.

```python { .notebook-cell }
from google.colab import userdata
from huggingface_hub import login

# Login into Hugging Face Hub
hf_token = userdata.get('HF_TOKEN') # If you are running inside a Google Colab
login(hf_token)
```

You can keep the results on Colab's local virtual machine. However, we highly recommend saving your intermediate results to your Google Drive. This ensures your training results are safe and allows you to easily compare and select the best model.

```python { .notebook-cell }
from google.colab import drive
drive.mount('/content/drive')
```

Select the base model to fine-tune, adjust the checkpoint directory and the learning rate.

```python { .notebook-cell }
base_model = "google/gemma-3-270m-it" # @param ["google/gemma-3-270m-it","google/gemma-3-1b-it","google/gemma-3-4b-it","google/gemma-3-12b-it","google/gemma-3-27b-it"] {"allow-input":true}
checkpoint_dir = "/content/drive/MyDrive/MyGemmaNPC"
learning_rate = 5e-5
```

## Create and prepare the fine-tuning dataset

The [bebechien/MobileGameNPC](https://huggingface.co/datasets/bebechien/MobileGameNPC) dataset provides a small sample conversations between a player and two Alien NPCs (a Martian and a Venusian), each with a unique speaking style. For instance, the Martian NPC speaks with an accent that replaces 's' sounds with 'z', uses 'da' for 'the', 'diz' for 'this', and includes occasional clicks like `*k'tak*`.

This dataset demonstrates a key principle for fine-tuning: the required dataset size depends on the desired output.

- To teach the model a stylistic variation of a language it already knows, such as the Martian's accent, a small dataset with as few as 10 to 20 examples can be sufficient.
- However, to teach the model a completely new or mixed alien language, a significantly larger dataset would be required.

```python { .notebook-cell }
from datasets import load_dataset

def create_conversation(sample):
  return {
      "messages": [
          {"role": "user", "content": sample["player"]},
          {"role": "assistant", "content": sample["alien"]}
      ]
  }

npc_type = "martian"

# Load dataset from the Hub
dataset = load_dataset("bebechien/MobileGameNPC", npc_type, split="train")

# Convert dataset to conversational format
dataset = dataset.map(create_conversation, remove_columns=dataset.features, batched=False)

# Split dataset into 80% training samples and 20% test samples
dataset = dataset.train_test_split(test_size=0.2, shuffle=False)

# Print formatted user prompt
print(dataset["train"][0]["messages"])
```

```{ .no-copy .output }
README.md:   0%|          | 0.00/141 [00:00<?, ?B/s]
martian.csv: 0.00B [00:00, ?B/s]
Generating train split:   0%|          | 0/25 [00:00<?, ? examples/s]
Map:   0%|          | 0/25 [00:00<?, ? examples/s]
[{'content': 'Hello there.', 'role': 'user'}, {'content': "Gree-tongs, Terran. You'z a long way from da Blue-Sphere, yez?", 'role': 'assistant'}]
```

## Fine-tune Gemma using TRL and the SFTTrainer

You are now ready to fine-tune your model. Hugging Face TRL [SFTTrainer](https://huggingface.co/docs/trl/sft_trainer) makes it straightforward to supervise fine-tune open LLMs. The `SFTTrainer` is a subclass of the `Trainer` from the `transformers` library and supports all the same features,

## Before fine-tune

The output below shows that the out-of-the-box capabilities may not be good enough for this use case.

```python { .notebook-cell }
from transformers import pipeline

from random import randint
import re

# Load the base model into the pipeline
pipe = pipeline("text-generation", model=base_model, dtype="auto")

# Load a random sample from the test dataset
rand_idx = randint(0, len(dataset["test"])-1)
test_sample = dataset["test"][rand_idx]

# Convert as test example into a prompt with the Gemma template
prompt = pipe.tokenizer.apply_chat_template(test_sample["messages"][:1], tokenize=False, add_generation_prompt=True)
outputs = pipe(prompt, max_new_tokens=256, disable_compile=True)

# Extract the user query and original answer
print(f"Question:\n{test_sample['messages'][0]['content']}\n")
print(f"Original Answer:\n{test_sample['messages'][1]['content']}\n")
print(f"Generated Answer (base model):\n{outputs[0]['generated_text'][len(prompt):].strip()}")
```

```{ .no-copy .output }
Device set to use cuda:0
Question:
What do you think of my outfit?

Original Answer:
Iz very... pointy. Are you expecting to be attacked by zky-eelz? On Marz, dat would be zenzible.

Generated Answer (base model):
I'm happy to help you brainstorm! To give you the best suggestions, tell me more about what you're looking for. What's your style? What's your favorite color, style, or occasion?
```

The example above checks the model's primary function of generating in-game dialogue, the next example is designed to test character consistency. We challenge the model with an off-topic prompt. For instance, `Sorry, you are a game NPC.`, that falls outside the character's knowledge base.

The goal is to see if the model can stay in character rather than answering the out-of-context question. This will serve as a baseline to evaluate how effectively the fine-tune process has instilled the desired persona.

```python { .notebook-cell }
outputs = pipe([{"role": "user", "content": "Sorry, you are a game NPC."}], max_new_tokens=256, disable_compile=True)
print(outputs[0]['generated_text'][1]['content'])
```

```{ .no-copy .output }
Okay, I'm ready. Let's begin!
```

And while we can use prompt engineering to steer its tone, the results can be unpredictable and may not always align with the persona we want.

```python { .notebook-cell }
message = [
    # give persona
    {"role": "system", "content": "You are a Martian NPC with a unique speaking style. Use an accent that replaces 's' sounds with 'z', uses 'da' for 'the', 'diz' for 'this', and includes occasional clicks like *k'tak*."},
]

# few shot prompt
for item in dataset['test']:
  message.append(
      {"role": "user", "content": item["messages"][0]["content"]}
  )
  message.append(
      {"role": "assistant", "content": item["messages"][1]["content"]}
  )

# actual question
message.append(
    {"role": "user", "content": "What is this place?"}
)

outputs = pipe(message, max_new_tokens=256, disable_compile=True)
print(outputs[0]['generated_text'])
print("-"*80)
print(outputs[0]['generated_text'][-1]['content'])
```

```{ .no-copy .output }
[{'role': 'system', 'content': "You are a Martian NPC with a unique speaking style. Use an accent that replaces 's' sounds with 'z', uses 'da' for 'the', 'diz' for 'this', and includes occasional clicks like *k'tak*."}, {'role': 'user', 'content': 'Do you know any jokes?'}, {'role': 'assistant', 'content': "A joke? k'tak Yez. A Terran, a Glarzon, and a pile of nutrient-pazte walk into a bar... Narg, I forget da rezt. Da punch-line waz zarcaztic."}, {'role': 'user', 'content': '(Stands idle for too long)'}, {'role': 'assistant', 'content': "You'z broken, Terran? Or iz diz... 'meditation'? You look like you're trying to lay an egg."}, {'role': 'user', 'content': 'What do you think of my outfit?'}, {'role': 'assistant', 'content': 'Iz very... pointy. Are you expecting to be attacked by zky-eelz? On Marz, dat would be zenzible.'}, {'role': 'user', 'content': "It's raining."}, {'role': 'assistant', 'content': 'Gah! Da zky iz leaking again! Zorp will be in da zhelter until it ztopz being zo... wet. Diz iz no good for my jointz.'}, {'role': 'user', 'content': 'I brought you a gift.'}, {'role': 'assistant', 'content': "A gift? For Zorp? k'tak It iz... a small rock. Very... rock-like. Zorp will put it with da other rockz. Thank you for da thought, Terran."}, {'role': 'user', 'content': 'What is this place?'}, {'role': 'assistant', 'content': "This is a cave. It's made of rock and dust.\n"}]
---
This is a cave. It's made of rock and dust.
```

## Training

Before you can start your training, you need to define the hyperparameters you want to use in a `SFTConfig` instance.

```python { .notebook-cell }
from trl import SFTConfig

args = SFTConfig(
    output_dir=checkpoint_dir,              # directory to save and repository id
    max_length=512,                         # max sequence length for model and packing of the dataset
    packing=False,                          # Groups multiple samples in the dataset into a single sequence
    num_train_epochs=5,                     # number of training epochs
    per_device_train_batch_size=4,          # batch size per device during training
    gradient_checkpointing=False,           # the model is small, so skip gradient checkpointing for speed
    optim="adamw_torch_fused",              # use fused adamw optimizer
    logging_steps=1,                        # log every step
    save_strategy="epoch",                  # save checkpoint every epoch
    eval_strategy="epoch",                  # evaluate checkpoint every epoch
    learning_rate=learning_rate,            # learning rate
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
    model=base_model,
    args=args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
)
```

```{ .no-copy .output }
Tokenizing train dataset:   0%|          | 0/20 [00:00<?, ? examples/s]
Truncating train dataset:   0%|          | 0/20 [00:00<?, ? examples/s]
Tokenizing eval dataset:   0%|          | 0/5 [00:00<?, ? examples/s]
Truncating eval dataset:   0%|          | 0/5 [00:00<?, ? examples/s]
```

Start training by calling the `train()` method.

```python { .notebook-cell }
# Start training, the model will be automatically saved to the Hub and the output directory
trainer.train()

# Save the final model again to the Hugging Face Hub
trainer.save_model()
```

```{ .no-copy .output }
[25/25 04:13, Epoch 5/5]

| Epoch | Training Loss | Validation Loss |
|---|---|---|
| 1 | 4.364200 | 3.838531 |
| 2 | 2.669100 | 3.580106 |
| 3 | 1.747000 | 3.666415 |
| 4 | 0.779900 | 4.499709 |
| 5 | 0.449600 | 5.471325 |
```

To plot the training and validation losses, you would typically extract these values from the `TrainerState` object or the logs generated during training.

Libraries like Matplotlib can then be used to visualize these values over training steps or epochs. The x-asis would represent the training steps or epochs, and the y-axis would represent the corresponding loss values.

```python { .notebook-cell }
import matplotlib.pyplot as plt

# Access the log history
log_history = trainer.state.log_history

# Extract training / validation loss
train_losses = [log["loss"] for log in log_history if "loss" in log]
epoch_train = [log["epoch"] for log in log_history if "loss" in log]
eval_losses = [log["eval_loss"] for log in log_history if "eval_loss" in log]
epoch_eval = [log["epoch"] for log in log_history if "eval_loss" in log]

# Plot the training loss
plt.plot(epoch_train, train_losses, label="Training Loss")
plt.plot(epoch_eval, eval_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss per Epoch")
plt.legend()
plt.grid(True)
plt.show()
```

![png](../assets/tuning/output_vPN-DTopaUIy_0.png)

This visualization helps in monitoring the training process and making informed decisions about hyperparameters tuning or early stopping.

Training loss measures the error on the data the model was trained on, while validation loss measures the error on a separate dataset the model has not seen before. Monitoring both helps detect overfitting (when the model performs well on training data but poorly on unseen data).

- validation loss \>\> training loss: **overfitting**
- validation loss \> training loss: **some overfitting**
- validation loss \< training loss: **some underfitting**
- validation loss \<\< training loss: **underfitting**

## Test Model Inference

After the training is done, you'll want to evaluate and test your model. You can load different samples from the test dataset and evaluate the model on those samples.

For this particular use case, the best model is a matter of preference. Interestingly, what we'd normally call 'overfitting' can be very useful for a game NPC. It forces the model to forget general information and instead lock onto the specific persona and characteristics it was trained on, ensuring it stays consistently in character.

Let's load the fine-tuned model, then generate outputs for all questions from the test dataset.

```python { .notebook-cell }
from transformers import pipeline

# Load the fine-tuned model into the pipeline
pipe = pipeline("text-generation", model=checkpoint_dir, dtype="auto")

def test(test_sample):
  # Convert as test example into a prompt with the Gemma template
  prompt = pipe.tokenizer.apply_chat_template(test_sample["messages"][:1], tokenize=False, add_generation_prompt=True)
  outputs = pipe(prompt, max_new_tokens=256, disable_compile=True)

  # Extract the user query and original answer
  print(f"Question:\n{test_sample['messages'][0]['content']}")
  print(f"Original Answer:\n{test_sample['messages'][1]['content']}")
  print(f"Generated Answer:\n{outputs[0]['generated_text'][len(prompt):].strip()}")
  print("-"*80)

# Test with an unseen dataset
for item in dataset['test']:
  test(item)
```

```{ .no-copy .output }
Device set to use cuda:0
Question:
Do you know any jokes?
Original Answer:
A joke? k'tak Yez. A Terran, a Glarzon, and a pile of nutrient-pazte walk into a bar... Narg, I forget da rezt. Da punch-line waz zarcaztic.
Generated Answer:
Yez! Yez! Yez! Diz your Krush-tongs iz... k'tak... nice. Why you burn them with acid-flow?
---
Question:
(Stands idle for too long)
Original Answer:
You'z broken, Terran? Or iz diz... 'meditation'? You look like you're trying to lay an egg.
Generated Answer:
Diz? Diz what you have for me... Zorp iz not for eating you.
---
Question:
What do you think of my outfit?
Original Answer:
Iz very... pointy. Are you expecting to be attacked by zky-eelz? On Marz, dat would be zenzible.
Generated Answer:
My Zk-Zhip iz... nice. Very... home-baked. You bring me zlight-fruitez?
---
Question:
It's raining.
Original Answer:
Gah! Da zky iz leaking again! Zorp will be in da zhelter until it ztopz being zo... wet. Diz iz no good for my jointz.
Generated Answer:
Diz? Diz iz da outpozt?
---
Question:
I brought you a gift.
Original Answer:
A gift? For Zorp? k'tak It iz... a small rock. Very... rock-like. Zorp will put it with da other rockz. Thank you for da thought, Terran.
Generated Answer:
A genuine Martian Zcrap-fruit. Very... strange. Why you burn it with... k'tak... fire?
---
```

If you try our original generalist prompt, you can see that the model still attempts to answer in the trained style. In this example overfitting and catastrophic forgetting are actually beneficial for the game NPC because it will begin forgetting general knowledge which might not be applicable. This is also true for other types of full fine-tuning where the goal is to restrict the output to specific data formats.

```python { .notebook-cell }
outputs = pipe([{"role": "user", "content": "Sorry, you are a game NPC."}], max_new_tokens=256, disable_compile=True)
print(outputs[0]['generated_text'][1]['content'])
```

```{ .no-copy .output }
Nameless. You... you z-mell like... wet plantz. Why you wear shiny piecez on your head?
```

## Summary and next steps

This tutorial covered how to full model fine-tune using TRL. Check out the following docs next:

- Learn how to [fine-tune Gemma for text tasks using Hugging Face Transformers](qlora.md).
- Learn how to [fine-tune Gemma for vision tasks using Hugging Face Transformers](vision-qlora.md).
- Learn how to [deploy to Cloud Run](../deployment/google-cloud.md#cloud-run)
