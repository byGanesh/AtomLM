# AtomLM

> A decoder-only Transformer language model built completely from scratch in PyTorch for learning, research, and experimentation.

AtomLM is a research project exploring how modern Large Language Models are built from tokenization and attention mechanisms to distributed training, instruction tuning, reasoning, and tool use.

Unlike projects that simply fine-tune existing LLMs, AtomLM is trained from scratch with a custom architecture and training pipeline.

## Features

* Decoder-only Transformer
* RoPE (Rotary Positional Embeddings)
* FlashAttention
* Grouped Query Attention (GQA)
* RMSNorm
* SwiGLU Feed Forward Network
* Mixed Precision (FP16)
* Distributed Data Parallel (DDP)
* Byte-Level BPE tokenizer
* Cosine Learning Rate Scheduler
* Gradient Accumulation
* Checkpointing & Resume Training


## Current Base Model

| Property        | Value             |
| --------------- | ----------------- |
| Parameters      | 52.3M             |
| Layers          | 16                |
| Hidden Size     | 512               |
| Attention Heads | 8                 |
| KV Heads        | 2 (GQA)           |
| Context Length  | 1024              |
| Vocabulary      | 8K Byte-Level BPE |
| Framework       | PyTorch           |


## Current Status

The current checkpoint is a **base pretrained model**.

It has learned:

* English grammar
* Story generation
* Basic mathematical reasoning patterns
* Long-form text generation
* Structured reasoning format

It has **not yet** been instruction-tuned or aligned, so factual accuracy, reasoning quality, and instruction following remain limited.


## Roadmap

**Foundation**  

- [x] Prototype Language Model (2.2M parameters)
- [x] Transformer Baseline (8.8M parameters, TinyStories pretraining)
- [x] AtomLM Base (52M parameters)
  - Decoder-only Transformer
  - RoPE
  - FlashAttention
  - Grouped Query Attention (GQA)
  - RMSNorm
  - SwiGLU
  - Pretrained on 343M tokens

**Model Training**  

- [ ] Continued Pretraining
- [ ] Supervised Instruction Fine-Tuning (SFT)
- [ ] Reasoning Fine-Tuning
- [ ] Tool-Use Training (Search, Python, Calculator)
- [ ] Preference Alignment (RLHF / DPO / GRPO)

**Optimization**  

- [ ] Quantization & Efficient Inference
- [ ] Mobile Deployment
- [ ] Long-Context Support
- [ ] Retrieval-Augmented Generation (RAG)

**Scaling**  

- [ ] AtomLM-100M
- [ ] AtomLM-300M
- [ ] AtomLM-1B

Training logs, generations, loss curves, and experiment history are available in:

```
results/LOGS.md
```

## Project Structure

```

AtomLM/
├── src/
│   ├── model.py
│   ├── train.py
│   ├── dataset.py
│   ├── generate.py
│   ├── config.py
│   └── tokenizer.py
│
├── checkpoints/
├── tokenizer/
├── results/
└── README.md

```

## Author

[Ganesh Kumar](https://byganesh.com)  
MIT
