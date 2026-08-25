---
title: Gemma model fine-tuning
description: Fine-tuning a generative artificial intelligence (AI) model such as Gemma modifies the behavior of the model.
models: [gemma-4]
source_url: "https://ai.google.dev/gemma/docs/tune.md.txt"
---
# Gemma model fine-tuning

Fine-tuning a generative artificial intelligence (AI) model such as
Gemma modifies the behavior of the model. You typically fine-tune Gemma with the
intent of improving its performance on a specific task or domain, or to better
fulfill a role, such as customer service. Gemma models are released with open
weights, which means you can modify those weights, which then changes the
behavior of the model. The general steps for fine-tuning a Gemma model are as
follows:

<ol class="steps" markdown>
<li markdown>
**[Choose a framework](index.md#choose-a-framework)**

Each one is tied to a model format and to how much hardware you have. Pick before you
prepare data, because the format follows from it.
</li>
<li markdown>
**[Collect data](index.md#collect-data)**

Pairs of input and expected response. Start small — as few as 20 pairs can shift behaviour.
</li>
<li markdown>
**[Tune and test the model](index.md#tune-and-test-the-model)**

Choose between full fine-tuning and a parameter-efficient method such as LoRA, then test
against success, failure and boundary cases.
</li>
<li markdown>
**[Deploy the model](index.md#deploy-the-model)**

Ship the tuned weights — and for LoRA, the base model alongside its adapter.
</li>
</ol>

## Choose a framework

Gemma models are compatible with a variety of AI tuning frameworks. Each is tied to a
particular model format and to how much hardware you need, so this is the decision that shapes
the rest. Make sure your intended deployment format — Keras, Safetensors or GGUF — is
supported as an output by whichever one you pick.

<!-- Generated from data/frameworks.yml -- edit that file, not these grids. -->
{% set fwmap = {} %}
{%- for fw in frameworkdata.frameworks %}{% set _ = fwmap.update({fw.id: fw}) %}{% endfor %}
{% macro paths(ids) %}
<ul class="engine-grid engine-grid--detailed">
{% for id in ids %}{% for path in frameworkdata.tuning_paths if path.id == id %}
{% set fw = fwmap.get(path.framework) if path.get('framework') else none %}
{% set page = path.get('page') %}
  <li><a class="engine" href="{% if page %}../{{ page }}{% else %}{{ path.url }}{% endif %}"
         {% if not page %}target="_blank" rel="noopener"{% endif %}
         {%- if path.get('pinned') %} data-model="{{ path.pinned }}"{% endif %}>
    <span class="engine-mark" aria-hidden="true">
      {%- if fw and fw.get('icon') %}{{ icon(fw.icon, 'engine-logo') }}{% else %}{{ path.get('mark') or fw.mark }}{% endif -%}
    </span>
    <span class="engine-body">
      <span class="engine-name">{{ path.label }}{% if not page %} ↗{% endif %}</span>
      <span class="engine-blurb">{{ path.blurb }}</span>
    </span>
  </a></li>
{% endfor %}{% endfor %}
</ul>
{% endmacro %}

### Tune in Python

The libraries you reach for when tuning is part of a training pipeline you control.

{{ paths(["keras-lora", "transformers-peft", "gemma-jax"]) }}

### Start from a notebook

Ready-made cookbook notebooks that run end to end. These were written for earlier Gemma
generations — the approach carries over, but expect to update model names and dependencies.

{{ paths(["unsloth-notebook", "axolotl", "llama-factory", "xtuner"]) }}

### Tune at scale

For runs that outgrow a single machine, or that you would rather not operate yourself.

{{ paths(["keras-distributed", "gke", "vertex-ai"]) }}

!!! note
    The JAX fine-tuning guide has been replaced by the [Gemma library fine-tuning](https://gemma-llm.readthedocs.io/en/latest/colab_sampling.html) guide.

## Collect data

Model tuning requires data. Tuning data typically consists of pairs of input
data with the expected response. There are many public datasets available online
for training on various tasks or output. For example, if you wanted to train a
Gemma model to translate car part descriptions to part numbers, your dataset
might include the following:

```python
training_data = [
  {"input_text": "Part number for A4 brake caliper", "output_text": "4M0615107BS"},
  {"input_text": "Part number for Beetle fuel pump", "output_text": "6A127026H"},
  {"input_text": "Part number for Camaro cylinder head", "output_text": "12711770"},
]
```

If you want a Gemma model to perform a specific set of tasks or role, you
typically need to compile a dataset demonstrating multiple variations of that
task. How much data you need to tune a model depends on your goals, particularly
how much of a behavioral change you want from the model and how well you want
the model to perform based on the task to be accomplished and the level of
variation in the input data.

In general, you should start with a small set of data for your task tuning,
adjust training parameters, and add data until you achieve the task performance
that meets your needs. Some of our example applications show that you can
influence the behavior of a Gemma model with as few as 20 prompt and response
pairs. For more details, see
[Tasks in spoken languages with Gemma](spoken-language.md).

## Tune and test the model

Once you have a tuning framework and tuning data in place, you can begin the
Gemma model tuning process. When performing tuning, you have some options in how
you tune which affects the resources you need to complete it. You also should
have a testing plan for your tuned model to evaluate if it is performing the way
you want it to after tuning.

### Parameter-efficient tuning

When fine-tuning an open weights model such as Gemma, you have the option to
[tune all the parameters the model](full-finetune.md)
or use a less resource intensive parameter efficient tuning technique which
updates a subset of them. A *full* tuning approach means that as you apply your
tuning data, you calculate new weights for all parameters of the model. This
approach is compute intensive and memory intensive, since you are performing
these calculations for billions of parameters. Using less resource intensive
tuning approaches, called *parameter efficient fine-tuning (PEFT)* , including
techniques like Low Rank Adapter (LoRA) tuning can produce similar results with
less compute resources. For details on how to perform tuning with less resources
using LoRA, see [Fine-tune Gemma models in Keras using
LoRA](keras-lora.md) and [Fine-Tuning Gemma Models in Hugging
Face](https://huggingface.co/blog/gemma-peft).

### Testing tuned models

Once you have tuned a model for a specific task you should test its performance
against the set of tasks you want it to perform. You should test your model with
tasks or requests that it was not specifically trained on. How you test your
tuned model depends on the task you want it to perform and how closely you
manage the inputs and outputs for the model. A common way to manage generative
model testing is to use success, failure, and borderline cases:

- **Success tests**: Requests that the tuned model should always be able to perform successfully.
- **Failure tests**: Requests that the tuned model should always not be able to perform, or explicitly refuse to perform, if requested.
- **Boundary tests**: Requests that the tuned model should be able to perform, if they fall within a defined boundary, or set of boundaries, of acceptable output behavior.

When testing failure or boundary conditions for your generative AI application,
you should also apply generative AI safety approaches, techniques, and tools as
described in the [Responsible Generative AI Toolkit][responsible-ai].

## Deploy the model

After completing your tuning and successful completion of your testing, it's
time to deploy your model. You can typically refer to the documentation for your
[chosen framework](index.md#choose-a-framework) for how to deploy a tuned
model.

If you are deploying a model with LoRA tuned weights, note that with this
technique you typically deploy *both* the original model and its weights with
the LoRA weights as an additional calculation layer for the model.
