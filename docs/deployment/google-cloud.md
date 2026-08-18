---
title: Google Cloud
description: "Deploy, serve, and fine-tune Gemma 4 on Google Cloud: Cloud Run, GKE, the Agent Development Kit, vLLM on TPUs, MaxText, and Sovereign Cloud."
models: [gemma-4]
frameworks: [google-cloud, vllm]
source_url: "https://ai.google.dev/gemma/docs/integrations/google-cloud.md.txt"
---
# Gemma on Google Cloud

Google Cloud offers several routes for deploying, serving and fine-tuning Gemma 4. Pick by what
you are doing, not by product name — each card jumps to the detail below.

### Serve it

<div class="card-grid">
<a class="card" href="google-cloud.md#cloud-run">
  <div class="card-title">Cloud Run</div>
  <p class="card-desc">Fully managed, scale-to-zero, pay-per-use GPU inference.</p>
</a>
<a class="card" href="google-cloud.md#google-kubernetes-engine-gke">
  <div class="card-title">GKE</div>
  <p class="card-desc">Your own serving cluster on TPUs or GPUs, for high throughput and low latency.</p>
</a>
<a class="card" href="google-cloud.md#vllm-with-tpus">
  <div class="card-title">vLLM on TPUs</div>
  <p class="card-desc">State-of-the-art serving performance on Cloud TPUs.</p>
</a>
<a class="card" href="google-cloud.md#sovereign-cloud">
  <div class="card-title">Sovereign Cloud</div>
  <p class="card-desc">Extra control and compliance for sensitive workloads.</p>
</a>
</div>

### Build with it

<div class="card-grid">
<a class="card" href="google-cloud.md#gemini-enterprise-agent-platform">
  <div class="card-title">Agent Platform</div>
  <p class="card-desc">Test and deploy straight from Model Garden in the console.</p>
</a>
<a class="card" href="google-cloud.md#agent-development-kit-adk">
  <div class="card-title">Agent Development Kit</div>
  <p class="card-desc">Orchestrate agents around Gemma's reasoning and function calling.</p>
</a>
</div>

### Train it

<div class="card-grid">
<a class="card" href="google-cloud.md#gemini-enterprise-agent-platform-training-clusters">
  <div class="card-title">Training Clusters</div>
  <p class="card-desc">Managed infrastructure for large-scale fine-tuning of open models.</p>
</a>
<a class="card" href="google-cloud.md#maxtext">
  <div class="card-title">MaxText</div>
  <p class="card-desc">High-performance JAX implementation for Cloud TPUs.</p>
</a>
</div>

## Gemini Enterprise Agent Platform

[Gemini Enterprise Agent Platform](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/open-models/use-gemma) is a Google Cloud platform for rapidly
building and scaling machine learning projects. Gemma 4 is available in [Model
Garden](https://console.cloud.google.com/agent-platform/publishers/google/model-garden/gemma4), a curated collection of models on Gemini Enterprise Agent Platform.
You can test and deploy models directly from the console.

To learn more, refer to the following pages:

- [Agent Platform overview](https://docs.cloud.google.com/gemini-enterprise-agent-platform/overview): Get started with Gemini Enterprise Agent Platform.
- [Gemma with Gemini Enterprise Agent Platform](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/open-models/use-gemma): Use Gemma open models with Gemini Enterprise Agent Platform.

## Cloud Run

[Cloud Run](https://cloud.google.com/run) is a fully managed platform to run your code or containers on
top of Google's highly scalable infrastructure.
[Deploy Gemma 4 on Cloud Run](https://docs.cloud.google.com/run/docs/run-gemma-on-cloud-run) using GPUs for scale-to-zero, pay-per-use
inference.

For larger mode sizes, [leverage advanced configurations](https://codelabs.developers.google.com/codelabs/cloud-run/cloud-run-gpu-rtx-pro-6000-gemma4-vllm){ data-model="gemma-4" } with RTX 6000 Pro
GPUs and Model Streaming.

## Google Kubernetes Engine (GKE)

[Google Kubernetes Engine](https://cloud.google.com/kubernetes-engine) (GKE) is a managed Kubernetes service from Google
Cloud. [Run Gemma 4 on GKE](https://cloud.google.com/kubernetes-engine/docs/tutorials/serve-gemma-gpu-vllm) for enterprise-grade container orchestration.
Use TPUs and GPUs to serve models with high throughput and low latency.

## Agent Development Kit (ADK)

Build and orchestrate AI agents with Gemma 4 and the [Agent Development Kit
(ADK)](https://adk.dev/agents/models/google-gemma/). Gemma 4's strong reasoning and function-calling capabilities make it
ideal for agentic workflows.

## Gemini Enterprise Agent Platform Training Clusters

[Fine-tune Gemma 4 using Gemini Enterprise Agent Platform Training
Clusters](https://discuss.google.dev/t/end-to-end-guide-fine-tuning-and-serving-gemma-4-on-vertex-ai/345865). Training Clusters provides optimized infrastructure for
large-scale training and fine-tuning of open models.

## vLLM with TPUs

[Serve Gemma 4 on Google Cloud TPUs](https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html#cloud-tpu-deployment-via-docker){ data-model="gemma-4" } for state-of-the-art serving
performance.

## MaxText

Gemma 4 is supported in [MaxText](https://github.com/google/maxtext), a high-performance, arbitrary-sized JAX
LLM implementation for Google Cloud TPUs.

## Sovereign Cloud

Gemma 4 is available on [Sovereign Cloud](https://cloud.google.com/sovereign-cloud) solutions, providing enhanced
control and compliance for sensitive workloads.
