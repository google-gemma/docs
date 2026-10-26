---
title: DataGemma
description: DataGemma is a research tool that lets users ask questions in plain language and receive answers based on publicly available statistical data in the Data Commons repository.
models: [datagemma]
source_url: "https://ai.google.dev/gemma/docs/datagemma.md.txt"
---
# DataGemma

DataGemma is a research tool that lets users ask questions in plain language and receive
answers based on publicly available statistical data in the
[Data Commons](https://datacommons.org/) repository. The tool uses
specially built versions of [Gemma 2](../gemma-2/index.md),
the [Gemini API](https://ai.google.dev/gemini-api) with Gemini 1.5 Pro, and a set of libraries
specifically designed to work with Data Commons.


This research tool provides two separate techniques for answering questions based on Data
Commons statistical data:

- **Retrieval-Interleaved Generation (RIG)** - This approach uses a variant of Gemma 2 that is fine-tuned to recognize when it needs to replace a generated number with more accurate information from Data Commons. For more details, see the Colab [notebook](https://colab.research.google.com/github/datacommonsorg/llm-tools/blob/main/notebooks/datagemma_rig.ipynb){ data-model="datagemma" } and models on [Kaggle](https://www.kaggle.com/models/google/datagemma-rig){ data-model="datagemma" } or [Hugging Face](https://huggingface.co/google/datagemma-rig-27b-it){ data-model="datagemma" }.
- **Retrieval-Augmented Generation (RAG)** - This approach uses a variant of Gemma 2 that retrieves relevant information from Data Commons and then uses that information to create an extended prompt for the Gemini 1.5 Pro model. For more details, see the Colab [notebook](https://colab.research.google.com/github/datacommonsorg/llm-tools/blob/main/notebooks/datagemma_rag.ipynb){ data-model="datagemma" } and models on [Kaggle](https://www.kaggle.com/models/google/datagemma-rag){ data-model="datagemma" } or [Hugging Face](https://huggingface.co/google/datagemma-rag-27b-it){ data-model="datagemma" }.


For more research and technical details on DataGemma, see the
[DataGemma technical paper](http://datacommons.org/link/DataGemmaPaper){ data-model="datagemma" }.

- **Generate answers with real data** — Apply generative artificial intelligence (AI) to a vast repository of public statistical data to explore and uncover new insights.
- **Evaluate AI data grounding techniques** — Investigate ways to guide generative AI model output with retrieval-augmented and data-interleaved techniques.

## Learn more

### [View RIG on Kaggle](https://www.kaggle.com/models/google/datagemma-rig){ data-model="datagemma" }

View more code, notebooks, information, and discussions about the DataGemma RIG model on Kaggle.

### [Run RIG in Colab](https://colab.research.google.com/github/datacommonsorg/llm-tools/blob/main/notebooks/datagemma_rig.ipynb){ data-model="datagemma" }

Try DataGemma using the retrieval-interleaved technique to answer questions.

### [Run RAG in Colab](https://colab.research.google.com/github/datacommonsorg/llm-tools/blob/main/notebooks/datagemma_rag.ipynb){ data-model="datagemma" }

Try DataGemma using the retrieval-augmented technique to answer questions.
