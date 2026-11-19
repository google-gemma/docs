<!-- GENERATED from data/glossary.yml by scripts/generate_snippets.py -- do not edit. -->

*[MTP]: Multi-Token Prediction — a small drafter model proposes several tokens at once and the main model verifies them in one pass
*[QAT]: Quantization-Aware Training — quantization is simulated during training so the model learns to compensate for it
*[PTQ]: Post-Training Quantization — compressing a finished model, which loses more quality than QAT at the same bit width
*[GGUF]: The file format llama.cpp and its ecosystem use for quantized weights
*[LoRA]: Low-Rank Adaptation — fine-tuning a small set of added weights instead of the whole model
*[QLoRA]: LoRA applied on top of a quantized base model, so fine-tuning fits on a single consumer GPU
*[SFT]: Supervised Fine-Tuning — training on prompt and response pairs
*[DPO]: Direct Preference Optimization — aligning a model from preferred and rejected response pairs
*[MoE]: Mixture of Experts — only a fraction of the model's parameters are active for any given token
*[PLE]: Per-Layer Embeddings — layer-specific token embedding tables kept in flash storage rather than VRAM
*[per-layer embeddings]: Layer-specific token embedding lookup tables that can stay in flash storage rather than VRAM
*[Per-layer embeddings]: Layer-specific token embedding lookup tables that can stay in flash storage rather than VRAM
*[KV cache]: The stored attention keys and values for tokens already processed; it grows with context length
*[IT]: Instruction-Tuned — trained to follow instructions, so it responds usefully without further training
*[PT]: Pre-Trained — the base model, before any instruction tuning
*[VLM]: Vision-Language Model — one that takes images alongside text
*[RAG]: Retrieval-Augmented Generation — retrieving relevant documents and putting them in the prompt
*[TPU]: Tensor Processing Unit — Google's accelerator for machine learning workloads
*[PEFT]: Parameter-Efficient Fine-Tuning — tuning a small subset of parameters rather than updating all model weights
*[FT]: Fine-Tuned — specialized for a specific benchmark, dataset, or domain task
*[ASR]: Automatic Speech Recognition — transcribing spoken audio into text
*[VQA]: Visual Question Answering — answering questions about visual content in an image
*[OCR]: Optical Character Recognition — extracting written or printed text directly from images
*[RLHF]: Reinforcement Learning from Human Feedback — aligning model behavior with human preferences
*[CPT]: Continued Pre-Training — training a base model on unlabeled domain or language corpora before instruction tuning
*[GRPO]: Group Relative Policy Optimization — a reinforcement learning algorithm that normalizes rewards across sampled group outputs without a critic model
*[ORPO]: Odds Ratio Preference Optimization — a single-stage method that combines supervised fine-tuning and preference alignment without a reference model
*[KTO]: Kahneman-Tversky Optimization — a preference alignment method that learns from binary desirable or undesirable signals rather than paired responses
*[rsLoRA]: Rank-Stabilized LoRA — scales adapter outputs by alpha divided by the square root of rank to keep high-rank training stable
*[FSDP]: Fully Sharded Data Parallel — a multi-GPU strategy that shards model weights, gradients, and optimizer states across devices
*[DDP]: Distributed Data Parallel — a multi-GPU strategy that replicates the model on each device and averages gradients across batches
*[RLVR]: Reinforcement Learning with Verifiable Rewards — training reasoning models against deterministic verifiers such as math or code execution checks
*[FFN]: Feed-forward Network — sub-layer that processes the token and produces the output vector for the layer
*[RoPE]: Rotary Position Encoding — method of encoding the absolute and relative position of tokens in the sequence
