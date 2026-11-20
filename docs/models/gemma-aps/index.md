---
title: Gemma for abstractive proposition segmentation (APS)
seo_title: Gemma APS proposition segmentation
description: Abstractive proposition segmentation (APS) is an analysis technique that breaks down text information into component parts.
models: [gemma-aps]
source_url: "https://ai.google.dev/gemma/docs/gemma-aps.md.txt"
---
# Gemma for abstractive proposition segmentation (APS)

## Gemma-APS


Abstractive proposition segmentation (APS) is an analysis technique that breaks down text
information into component parts. Gemma-APS is a variant of Gemma that takes a given
passage of text and attempts to segment the content into the individual
facts, statements, and ideas expressed in the text, and restates them in full sentences
with small changes to the original text.


This model can be used for research where there is a need to break down text content into
meaningful components. Applications include generative content grounding, content
retrieval, fact-checking, and evaluation of generative output, such as summarization,
where it is useful to divide up individual factual data so that they can be verified
independently. For more information, check out the
[research paper](https://arxiv.org/pdf/2406.19803).

- **Analyze content for component parts** — Apply generative artificial intelligence (AI) to break down content into discrete statements for analysis or further processing.
- **Combine with larger content analysis** — Use generated content segments in larger content analysis projects such as grounding, retrieval, fact-checking, evaluation of generation tasks, and duplicate content detection.

## Learn more

### [View 2B model](https://huggingface.co/google/gemma-2b-aps-it)

View more code, notebooks, information, and discussions about the Gemma-APS 2B model on Hugging Face.

### [View 7B model](https://huggingface.co/google/gemma-7b-aps-it)

View more code, notebooks, information, and discussions about the Gemma-APS 7B model on Hugging Face.

### [Read the research](https://arxiv.org/pdf/2406.19803)

Get more insights into the research, development, and operation of the Gemma-APS model.
