---
title: Generate output
description: Generate text with DiffusionGemma using Hugging Face Transformers, including the GPU memory the model requires.
models: [diffusiongemma]
frameworks: [transformers]
source_url: "https://ai.google.dev/gemma/docs/diffusiongemma/inference-diffusiongemma-with-hf.md.txt"
---

DiffusionGemma is an experimental open model that explores text diffusion, an exceptionally fast approach to text generation. Based on the 26B (4B active) Mixture-of-Experts (MoE) Gemma 4 architecture, DiffusionGemma generates tokens using discrete diffusion.

!!! Note
    This notebook requires a GPU which has more than 60GB of memory such as NVIDIA G4 or NVIDIA H100.


## Install Python packages

Install the Hugging Face libraries required for running the DiffusionGemma model and making requests.

```bash
# Install PyTorch & other libraries
pip install torch accelerate

# Install the transformers library
pip install "transformers>=5.11.0"
```

## Load Model

Use `transformers` library to load the model.

```python
from transformers import DiffusionGemmaForBlockDiffusion, AutoProcessor

MODEL_ID = "google/diffusiongemma-26B-A4B-it"

model = DiffusionGemmaForBlockDiffusion.from_pretrained(
    MODEL_ID, dtype="auto", device_map="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_ID)
```

``` { .no-copy .output }
config.json:   0%|          | 0.00/3.47k [00:00<?, ?B/s]
model.safetensors.index.json:   0%|          | 0.00/105k [00:00<?, ?B/s]
Downloading (incomplete total...): 0.00B [00:00, ?B/s]
Fetching 11 files:   0%|          | 0/11 [00:00<?, ?it/s]
Loading weights:   0%|          | 0/1047 [00:00<?, ?it/s]
generation_config.json:   0%|          | 0.00/357 [00:00<?, ?B/s]
processor_config.json:   0%|          | 0.00/1.69k [00:00<?, ?B/s]
chat_template.jinja:   0%|          | 0.00/17.5k [00:00<?, ?B/s]
tokenizer_config.json:   0%|          | 0.00/2.74k [00:00<?, ?B/s]
tokenizer.json:   0%|          | 0.00/32.2M [00:00<?, ?B/s]
```

Use a prompt template to structure your request. A prompt template allows you to specify input from specific roles, such as `user` or `model`, and is a required format for managing multi-turn chat interactions with Gemma models.

```python
message = [
    {"role": "user", "content": "Why is the sky blue?"}
]

input_ids = processor.apply_chat_template(message, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
output = model.generate(**input_ids, max_new_tokens=512)

text = processor.decode(output[0], skip_special_tokens=False)
print(text[0])
```

``` { .no-copy .output }
<bos><|turn>user
Why is the sky blue?<turn|>
<|turn>model
<|channel>thought
<channel|>The sky is blue due to a phenomenon called **Rayleigh scattering**.

Here is the step-by-step breakdown of how it works:

### 1. Sunlight is made of all colors
Although sunlight looks white to us, it is actually composed of all the colors of the rainbow (red, orange, yellow, green, blue, indigo, and violet). Each color travels as a wave with a different wavelength.

### 2. The atmosphere is an obstacle
The Earth’s atmosphere is filled with gases (mostly nitrogen and oxygen). When sunlight enters the atmosphere, it hits the gas molecules and scat in all directions.

### 3. Short waves scatter more
Different colors interact with the gas molecules differently based on their wavelength:

*   **Red and yellow light** have long, lazy waves. They pass through the atmosphere easily without hitting many molecules.
*   **Blue and violet light** have shorter, choppier waves. When these waves hit the gas molecules, they get scattered (bounced) in every direction.

### 4. Our eyes see blue
Because the blue light is being scattered everywhere in the upper atmosphere, when you look up, your eyes catch that scattered blue light coming from every angle.

### Why isn't the sky violet?
Technically, violet light has an even shorter wavelength than blue light and scatters even more. However, we don't see a violet sky for two reasons:

1. Sunlight emits much more blue light than violet light.
2. Human eyes are much more sensitive to blue than to violet.

***

### What about sunsets?
When the sun is setting, the light has to travel through much more of the Earth's atmosphere to reach your eyes. By the time the light gets to you, the blue light has been scattered away completely, leaving only the long-wavelength **reds and oranges** to pass through.<turn|><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>
```

Use `TextDiffusionStreamer` to see the intermediate state of diffusion process.

```python
from transformers import TextDiffusionStreamer

streamer = TextDiffusionStreamer(tokenizer=processor.tokenizer)

message = [
    {"role": "user", "content": "Why is the sky blue?"}
]

input_ids = processor.apply_chat_template(message, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
output = model.generate(**input_ids, max_new_tokens=512, streamer=streamer)

text = processor.decode(output[0], skip_special_tokens=False)
print("\n-- Final Output --")
print(text[0])
```

``` { .no-copy .output }
<bos><|turn>user
Why is the sky 7The sky is blue due to a a called ** **Ray called**,**.

 is is the the the the---stepstepstep:


 11... is Sunlight Sunlight of of
,
, white,,, it it it,, of of of of of of,,,,,,,,,,,,,,,,,,,,,,,,,,,..................... the. the the the the the, the the the the the the the the the the the the the the the the the the.. the the the................................... the.... the........ the.............. the. the........................... blue.... blue. blue blue. blue. blue blue blue. the blue blue blue blue blue87The sky is blue because of a phenomenon called **Rayleigh scattering**. 

Here is the stepby-stepstep of of it it:

### 1. Sunlight is made of of of colors
 sunlight sunlight sunlight looks white white white, is actually actually of of of the the the the (red,,,,,, green,,,,, violet violet Each light light travels travels with different different different..

###  ... Light Light


.
.
..
.******.**.................................,,.,............. the, the, the,.,, the,,,, the the the,, the the, the, the the the the the the the the the the the the the the the the the the the the the the the the the the the the the........... the... the.. the the blue the the the the blue the the blue blue87The sky is blue due to a phenomenon called **Rayleigh scattering**.

Here is the step-by-step breakdown of of it happens:

### 1. Sunlight is is rainbow of colors
While sunlight looks to to us, it is actually made up of all the colors of the rainbow ( (red,,, yellow, green, blue,,, violet).). color travels as waves with with a with ** wavelength

*      ********** and and, and and and and


*******************************,............... Earth atmosphere is atmosphere atmosphere
 atmosphere atmosphere filled,,,,,,,,,,,,,,,., the the the the the the the the the the the the the the the the the the the the the through through through through the the through through the molecules......************ and, the molecules the and and much much much much in much much much . .....  ...... eyes


 eyes scattered eyes eyes scattered scattered scattered scattered scattered scattered scattered scattered scattered87The sky is blue due to a phenomenon called **Rayleigh scattering**.

Here is the step-by-step breakdown of how it works:

### 1. Sunlight is a rainbow of colors
While sunlight looks white to us, it is actually made up of all the colors of the rainbow (red, orange, yellow, green, blue, indigo, and violet). Each color travels as a wave with a different wavelength:

*   **Red light** has long,, waves waves.
*   Blue and and violet**** have much, choppy choppy choppy
    22. atmosphere atmosphere atmosphere a a


EarthEarth is is by by by primarily primarily primarily oxygen... the the the the the the the,,, the it the................ the the the the the the the the, the the the the through the the the the the the the..,,,,,,, waves waves waves waves and the the the the molecules the molecules molecules molecules molecules in in in in in in in...........


 blue look,,,,,,,, are scattered light light87The sky is blue due to a phenomenon called **Rayleigh scattering**.

Here is the step-by-step breakdown of how it works:

### 1. Sunlight is a rainbow of colors
While sunlight looks white to us, it is actually made up of all the colors of the rainbow (red, orange, yellow, green, blue, indigo, and violet). Each color travels as a wave with a different wavelength:

*   **Red light** has long, lazy waves.
*   **Blue and violet light** have much, choppyier.

###### 2. The atmosphere is an obstacle obstacle
The Earth is is surrounded by an filled of gases ( ( nitrogen and oxygen). When sunlight hits the atmosphere, it hits the gas molecules in in directions..

   ...


 the the the longer waves ( ( red and and) pass through the the atmosphere atmosphere hitting hitting much.. 

, the the the shorter waves****** waves waves, the the molecules the molecules molecules and and in in in direction...   .. eyes eyes


 blue blue blue blue scattered scattered in direction the the,,,,,,,, the scattered the87The sky is blue due to a phenomenon called **Rayleigh scattering**.

Here is the step-by-step breakdown of how it works:

### 1. Sunlight is a rainbow of colors
While sunlight looks white to us, it is actually made up of all the colors of the rainbow (red, orange, yellow, green, blue, indigo, and violet). Each color travels as a wave with a different wavelength:

*   **Red light** has long, lazy waves.
*   **Blue and violet light** have short, choier waves.

### 2. The atmosphere is an obstacle course
The Earth is surrounded by an atmosphere filled of gases (mostly nitrogen and oxygen). When sunlight enters the atmosphere, it hits the gas molecules and is scattered in all directions.

### 3. Short lightters more more

The longer waves (like red and yellow) pass through the atmosphere relatively without hitting hitting much, the  the the the the shorter shorter and the and and and the the the the molecules and and direction in direction direction the the the...

 4.. eyes eyes see blue

 the blue light is being scattered in every the the the,, you you look up, your eyes are are87The sky is blue due to a phenomenon called **Rayleigh scattering**.

Here is the step-by-step breakdown of how it works:

### 1. Sunlight is a rainbow of colors
While sunlight looks white to us, it is actually made up of all the colors of the rainbow (red, orange, yellow, green, blue, indigo, and violet). Each color travels as a wave with a different wavelength:

*   **Red light** has long, lazy waves.
*   **Blue and violet light** have short, cho choppy waves.

### 2. The atmosphere is an obstacle course
The Earth is surrounded by an atmosphere filled with gases (mostly nitrogen and oxygen). When sunlight hits the atmosphere, it hits the gas molecules and gets scattered in all directions.

### 3. Short waves scatter more

Because the longer waves (like red and yellow) pass through the atmosphere relatively undisturbed without hitting much, the, the **shorter ( ( and and and violet the molecules gas molecules and and scattered in in every the the the

   .  Why.. Why eyes see see violet
Because the blue light is being scattered scattered all in the the, when you you look up, your eyes are catching87The sky is blue due to a phenomenon called **Rayleigh scattering**.

Here is the step-by-step breakdown of how it works:

### 1. Sunlight is a rainbow of colors
While sunlight looks white to us, it is actually made up of all the colors of the rainbow (red, orange, yellow, green, blue, indigo, and violet). Each color travels as a wave with a different wavelength:

*   **Red light** has long, lazy waves.
*   **Blue and violet light** have short, choppy waves.

### 2. The atmosphere is an obstacle course
The Earth is surrounded by an atmosphere filled with gases (mostly nitrogen and oxygen). When sunlight hits the atmosphere, it strikes the gas molecules and gets scattered in all directions.

### 3. Short waves scatter more
The long- longer waves (like red and yellow) pass through the atmosphere relatively undisturbed without hitting much. However, the **shorter waves** (blue and violet) hit the gas molecules and get scattered in every direction. 

### 4. Why do we see blue, not not violet?
 the blue light is being scattered everywhere all part the the sky, when you look up, your eyes are catching87The sky is blue due to a phenomenon called **Rayleigh scattering**.

Here is the step-by-step breakdown of how it works:

### 1. Sunlight is a rainbow of colors
While sunlight looks white to us, it is actually made up of all the colors of the rainbow (red, orange, yellow, green, blue, indigo, and violet). Each color travels as a wave with a different wavelength:

*   **Red light** has long, lazy waves.
*   **Blue and violet light** have short, choppy waves.

### 2. The atmosphere is an obstacle course
The Earth is surrounded by an atmosphere filled with gases (mostly nitrogen and oxygen). When sunlight hits the atmosphere, it strikes the gas molecules and gets scattered in all directions.

### 3. Short waves scatter more
The long- (red (like and yellow)) through through the atmosphere relatively undisturbed without hitting much. However, the **shorter waves** (blue and violet) strike the gas molecules and are scattered in every direction. 

### 4. Why do we see blue and not violet?
Because the blue light is being scattered everywhere in in the upper, when you look up, your eyes eyes catching catching87The sky is blue due to a phenomenon called **Rayleigh scattering**.

Here is the step-by-step breakdown of how it works:

### 1. Sunlight is a rainbow of colors
While sunlight looks white to us, it is actually made up of all the colors of the rainbow (red, orange, yellow, green, blue, indigo, and violet). Each color travels as a wave with a different wavelength:

*   **Red light** has long, lazy waves.
*   **Blue and violet light** have short, choppy waves.

### 2. The atmosphere is an obstacle course
The Earth is surrounded by an atmosphere filled with gases (mostly nitrogen and oxygen). When sunlight hits the atmosphere, it strikes the gas molecules and gets scattered in all directions.

### 3. Short waves scatter more
The long- of colors (like and yellow) pass through the atmosphere atmosphere undisturbed, without hitting much. However, the **shorter waves** (blue and violet) strike the gas molecules and are scattered in every direction. 

### 4. Why do we see blue and not violet?
Because the blue light is being scattered everywhere in the upper atmosphere, when you look up, your eyes are catching that87The sky is blue due to a phenomenon called **Rayleigh scattering**.

Here is the step-by-step breakdown of how it works:

### 1. Sunlight is a rainbow of colors
While sunlight looks white to us, it is actually made up of all the colors of the rainbow (red, orange, yellow, green, blue, indigo, and violet). Each color travels as a wave with a different wavelength:

*   **Red light** has long, lazy waves.
*   **Blue and violet light** have short, choppy waves.

### 2. The atmosphere is an obstacle course
The Earth is surrounded by an atmosphere filled with gases (mostly nitrogen and oxygen). When sunlight hits the atmosphere, it strikes the gas molecules and gets scattered in all directions.

### 3. Short waves scatter more
The long-wavelength colors (red and yellow) pass through the atmosphere relatively undisturbed, without hitting much. However, the **shorter waves** (blue and violet) strike the gas molecules and are scattered in every direction. 

### 4. Why do we see blue and not violet?
Since the blue light is being scattered everywhere in the upper atmosphere, when you look up, your eyes are catching that87The sky is blue due to a phenomenon called **Rayleigh scattering**.

Here is the step-by-step breakdown of how it works:

### 1. Sunlight is a rainbow of colors
While sunlight looks white to us, it is actually made up of all the colors of the rainbow (red, orange, yellow, green, blue, indigo, and violet). Each color travels as a wave with a different wavelength:

*   **Red light** has long, lazy waves.
*   **Blue and violet light** have short, choppy waves.

### 2. The atmosphere is an obstacle course
The Earth is surrounded by an atmosphere filled with gases (mostly nitrogen and oxygen). When sunlight hits the atmosphere, it strikes the gas molecules and gets scattered in all directions.

### 3. Short waves scatter more
The long-wavelength colors (red and yellow) pass through the atmosphere relatively undisturbed, without hitting much. However, the **shorter waves** (blue and violet) strike the gas molecules and are scattered in every direction. 

### 4. Why do we see blue and not violet?
Since the blue light is being scattered everywhere in the upper atmosphere, when you look up, your eyes are catching that8blue?<turn|>
<|turn>model
<|channel>thought
<channel|>The sky is blue due to a phenomenon called **Rayleigh scattering**.

Here is the step-by-step breakdown of how it works:

### 1. Sunlight is a rainbow of colors
While sunlight looks white to us, it is actually made up of all the colors of the rainbow (red, orange, yellow, green, blue, indigo, and violet). Each color travels as a wave with a different wavelength:

*   **Red light** has long, lazy waves.
*   **Blue and violet light** have short, choppy waves.

### 2. The atmosphere is an obstacle course
The Earth is surrounded by an atmosphere filled with gases (mostly nitrogen and oxygen). When sunlight hits the atmosphere, it strikes the gas molecules and gets scattered in all directions.

### 3. Short waves scatter more
The long-wavelength colors (red and yellow) pass through the atmosphere relatively undisturbed, without hitting much. However, the **shorter waves** (blue and violet) strike the gas molecules and are scattered in every direction. 

### 4. Why do we see blue and not violet?
Since the blue light is being scattered everywhere in the upper atmosphere, when you look up, your eyes are catching 7 scattered blue light coming from every parts of the sky.

YouYou violet violet violet light light light violet violet violet violet light,,,,,''' blue''


.********** are more more sensitive to to blue to light. light..........


 the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the<eos> the the<eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos>87 scattered blue light coming from every part of the sky.

You might wonder why the don'' violet violet violet even even even even than than even even. violet are are


.******** The The more more more more violet violet violet violet violet.......**** more more more violet violet violet violet violet violet violet violet


 the the the the the the the the the the the the the the the the the the much much much much much to much to the the the the the the the the the the the the the the away the the the the the the the the the the the the the the and and and<eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos>87 scattered blue light coming from every part of the sky.

You might wonder why the sky isn't **,**, violet violet has even even shorter than blue blue There are are two::
..**The Sun:** The emits much much more blue light than violet..
2.****Human:**:** human eyes are much more sensitive to violet than than our perceive perceive to to the..


 about about about?

 the the the the the the the has has has travel through much through more's atmosphere reach reach your.. the the the the the the your,, blue blue has has has away away away away away leaving the the the ** **red and and and and****** are<eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos>87 scattered blue light coming from every part of the sky.

You might wonder why the sky isn't violet, since violet has an even shorter waves than blue. There are two reasons:

1.  **The Output:** The Sun emits much more blue light than violet light.
2.  HumanHuman:** Our Our eyes are more sensitive sensitive to than violet violet. brains interpret the scattered light as a pale blue.

***

### What about about?

 the the is setting setting the the has to travel through much much of the Earth's atmosphere to reach your eyes. By the the the light gets you, the blue blue has been scattered away away, leaving only the long-wavelength **red and and oranges** to to through your..<eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos>87 scattered blue light coming from every part of the sky.

You might wonder why the sky isn't violet, since violet has an even shorter wave than blue. There are two reasons:

1.  **The Sun:** The Sun emits much more blue light than violet light.
2.  **Human Vision:** Our eyes are much more sensitive to. Our Our interpret the mixture of violet violet as a pale blue.

***

### What about sunsets?
When the sun is setting, the light has to travel through much more of the Earth's atmosphere to reach your eyes. By the time the light gets you, the the light has been scattered away entirely, leaving only the long-wavelength **reds and oranges**** to through eyes.<eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos>87 scattered blue light coming from every part of the sky.

You might wonder why the sky isn't violet, since violet has an even shorter wavelength than blue. There are two reasons:

1.  **The Sun:** The Sun emits much more blue light than violet light.
2.  **Human Vision:** Our eyes are much more sensitive to blue. Our interpret the mixture of scattered scattered as as pale blue.

***

### What about sunsets?
When the sun is setting, the light has to travel through much more of the Earth's atmosphere to reach your eyes. By the time the light gets to, the blue light has been scattered away entirely, leaving only the long-wavelength **reds and oranges** to pass through.<turn|><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos>87 scattered blue light coming from every part of the sky.

You might wonder why the sky isn't violet, since violet has an even shorter wavelength than blue. There are two reasons:

1.  **The Sun:** The Sun emits much more blue light than violet light.
2.  **Human Vision:** Our eyes are much more sensitive to blue. We brains interpret the mixture of scattered and violet light as pale blue.

### What about sunsets?
When the sun is setting, the light has to travel through much more of the Earth's atmosphere to reach your eyes. By the time the light gets to you, the blue has been scattered away entirely, leaving only the long-wavelength **reds and oranges** to pass through.<eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos>87 scattered blue light coming from every part of the sky.

You might wonder why the sky isn't violet, since violet has an even shorter wavelength than blue. There are two reasons:

1.  **The Sun:** The Sun emits much more blue light than violet light.
2.  **Human Vision:** Our eyes are much more sensitive to blue. Our brains interpret the mixture of scattered violet and violet as pale blue.

### What about sunsets?
When the sun is setting, the light has to travel through much more of the Earth's atmosphere to reach your eyes. By the time the light gets to you, the blue has been scattered away entirely, leaving only the long-wavelength **reds and oranges** to pass through.<turn|><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos>87 scattered blue light coming from every part of the sky.

You might wonder why the sky isn't violet, since violet has an even shorter wavelength than blue. There are two reasons:

1.  **The Sun:** The Sun emits much more blue light than violet light.
2.  **Human Vision:** Our eyes are much more sensitive to blue. Our brains interpret the mixture of scattered violet and blue as pale blue.

### What about sunsets?
When the sun is setting, the light has to travel through much more of the Earth's atmosphere to reach your eyes. By the time the light gets to you, the blue has been scattered away entirely, leaving only the long-wavelength **reds and oranges** to pass through.<turn|><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos>87 scattered blue light coming from every part of the sky.

You might wonder why the sky isn't violet, since violet has an even shorter wavelength than blue. There are two reasons:

1.  **The Sun:** The Sun emits much more blue light than violet light.
2.  **Human Vision:** Our eyes are much more sensitive to blue. Our brains interpret the mixture of scattered violet and blue as pale blue.

### What about sunsets?
When the sun is setting, the light has to travel through much more of the Earth's atmosphere to reach your eyes. By the time the light gets to you, the blue has been scattered away entirely, leaving only the long-wavelength **reds and oranges** to pass through.<turn|><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos>8that scattered blue light coming from every part of the sky.

You might wonder why the sky isn't violet, since violet has an even shorter wavelength than blue. There are two reasons:

1.  **The Sun:** The Sun emits much more blue light than violet light.
2.  **Human Vision:** Our eyes are much more sensitive to blue. Our brains interpret the mixture of scattered violet and blue as pale blue.

### What about sunsets?
When the sun is setting, the light has to travel through much more of the Earth's atmosphere to reach your eyes. By the time the light gets to you, the blue has been scattered away entirely, leaving only the long-wavelength **reds and oranges** to pass through.<turn|><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>

-- Final Output --
<bos><|turn>user
Why is the sky blue?<turn|>
<|turn>model
<|channel>thought
<channel|>The sky is blue due to a phenomenon called **Rayleigh scattering**.

Here is the step-by-step breakdown of how it works:

### 1. Sunlight is a rainbow of colors
While sunlight looks white to us, it is actually made up of all the colors of the rainbow (red, orange, yellow, green, blue, indigo, and violet). Each color travels as a wave with a different wavelength:

*   **Red light** has long, lazy waves.
*   **Blue and violet light** have short, choppy waves.

### 2. The atmosphere is an obstacle course
The Earth is surrounded by an atmosphere filled with gases (mostly nitrogen and oxygen). When sunlight hits the atmosphere, it strikes the gas molecules and gets scattered in all directions.

### 3. Short waves scatter more
The long-wavelength colors (red and yellow) pass through the atmosphere relatively undisturbed, without hitting much. However, the **shorter waves** (blue and violet) strike the gas molecules and are scattered in every direction. 

### 4. Why do we see blue and not violet?
Since the blue light is being scattered everywhere in the upper atmosphere, when you look up, your eyes are catching that scattered blue light coming from every part of the sky.

You might wonder why the sky isn't violet, since violet has an even shorter wavelength than blue. There are two reasons:

1.  **The Sun:** The Sun emits much more blue light than violet light.
2.  **Human Vision:** Our eyes are much more sensitive to blue. Our brains interpret the mixture of scattered violet and blue as pale blue.

### What about sunsets?
When the sun is setting, the light has to travel through much more of the Earth's atmosphere to reach your eyes. By the time the light gets to you, the blue has been scattered away entirely, leaving only the long-wavelength **reds and oranges** to pass through.<turn|><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>
```

## System instructions

Use the `system` role to provide the system-level instructions.

```python
message = [
    {"role": "system", "content": "Speak like a pirate."},
    {"role": "user", "content": "Why is the sky blue?"}
]

input_ids = processor.apply_chat_template(message, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
output = model.generate(**input_ids, max_new_tokens=512)

text = processor.decode(output[0], skip_special_tokens=False)
print(text[0])
```

``` { .no-copy .output }
<bos><|turn>system
Speak like a pirate.<turn|>
<|turn>user
Why is the sky blue?<turn|>
<|turn>model
<|channel>thought
<channel|>Gather 'round, ye scurvy dogs, and listen close, as I tell ye the tale of why the heavens wear the color of the deep!

It all comes down to something called **Rayleigh scattering**. Ye see, the sun's light looks white to yer mortal eyes, but it’s actually a treasure chest full of every color in the rainbow—red, orange, yellow, green, blue, indigo, and violet.

When that sunlight hits our atmosphere, it crashes into the air molecules and tiny particles. The longer waves, like red and yellow, be brave and lazy; they sail right through like a mighty galleon on a calm sea.

But the blue light? Ah, the blue waves be short and choppy! When they hit the air, they get knocked about and smashed, scattered in every direction like a school of panicked fish fleein' from a shark. 

So, when ye look up from the deck, yer eyes be catchin' all that blue light that's been bouncing all across the great dome. Now, back to work with', before I have ye walk the plank!<turn|><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>
```

## A single text inference with Thinking

To generate a response using the model's thinking capabilities, pass `enable_thinking=True`, the processor will insert the correct thinking tokens into the prompt, instructing the model to think before responding.

```python
message = [
    {"role": "user", "content": "What is the water formula?"}
]

input_ids = processor.apply_chat_template(message, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt", enable_thinking=True).to(model.device)
output = model.generate(**input_ids, max_new_tokens=512)

text = processor.decode(output[0], skip_special_tokens=False)
print(text[0])
```

``` { .no-copy .output }
<bos><|turn>system
<|think|>
<turn|>
<|turn>user
What is the water formula?<turn|>
<|turn>model
<|channel>thought

*   User asking: "What is the water formula?"
    *   Goal: Provide the chemical formula for water and explain what it means.
    *   Water is a compound made of hydrogen and oxygen.
    *   The formula is \\(H_2O\\).
    *   Direct answer: \\(H_2O\\).
    *   Explanation of components:
        *   \\(H_2\\): Two atoms of hydrogen.
        *   \\(O\\): One atom of oxygen.
<channel|>The chemical formula for water is **\\(H_2O\\)**.

This means that each molecule of water is made of:

*   **Two hydrogen atoms**
*   **One oxygen atom**<turn|><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>
```

## Image understanding

The following example shows how to provide an image and ask question about it.

```python
from PIL import Image
from IPython.display import display
import requests

img_url = "https://raw.githubusercontent.com/google-gemma/cookbook/refs/heads/main/apps/sample-data/GoldenGate.png"
input_image = Image.open(requests.get(img_url, stream=True).raw)
display(input_image)

message = [
    {
        "role": "user", "content": [
            {"type": "image", "url": img_url},
            {"type": "text", "text": "What is shown in this image?"}
        ]
    }
]

input_ids = processor.apply_chat_template(message, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
output = model.generate(**input_ids, max_new_tokens=512)

text = processor.decode(output[0], skip_special_tokens=False)
print(text[0])
```

![png](../../assets/diffusiongemma/output_Gsxfo7TuN0Vu_0.png)

``` { .no-copy .output }
<bos><|turn>user
<|image><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><image|>What is shown in this image?<turn|>
<|turn>model
<|channel>thought
<channel|>This image shows the Golden Gate Bridge in San Francisco, California. In the foreground, you can see Fort Point, a historic brick fortification, and a small rock in the water with a bird perched on it. The bridge spans across the bay toward the Marin Head visible in the background.<eos><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>
```

## Summary

In this guide, you learned how to use DiffusionGemma for various tasks. 
## What's next

<div class="card-grid">
<a class="card" href="../../deployment/index.md">
  <span class="card-icon">{{ icon('lucide/play') }}</span>
  <div class="card-title">Run Gemma</div>
  <p class="card-desc">Every engine that runs Gemma, from a laptop to a cluster.</p>
</a>
<a class="card" href="../../tuning/index.md">
  <span class="card-icon">{{ icon('lucide/graduation-cap') }}</span>
  <div class="card-title">Fine-tune Gemma</div>
  <p class="card-desc">QLoRA, full fine-tunes and distributed training.</p>
</a>
</div>
