# AtomLM

A small decoder-only Transformer language model built from scratch using PyTorch.

The goal of AtomLM is to understand how language models work by implementing, training, and experimenting with every part of the pipeline.

## About

It is an experiment to study:
- Transformer architecture
- Training behavior
- Data scaling
- Model limitations
- Reasoning experiments

The project follows a staged approach:

Language → Knowledge → Reasoning → Tools


## Architecture

#### Stage 1

| Component | Details |
|---|---|
| Architecture | Decoder-only Transformer |
| Parameters | 8.8M |
| Layers | 6 |
| Attention heads | 8 |
| d_model | 256 |
| FFN dimension | 1024 |
| Vocabulary size | 8000 |
| Tokenizer | ByteLevel BPE |
| Context length | 128 tokens |
| Activation | GELU |
| Normalization | Pre-LayerNorm |

## Training

### Stage 1 Experiment 002

Dataset:
- TinyStories
- 2.1M stories
- ~85M tokens

Hardware:
- Kaggle T4 x2 GPU

Configuration:
- Batch size: 128
- Learning rate: 3e-4
- Context length: 128

Results:

| Epoch | Average Loss | Time |
|---|---|---|
| 1 | 2.7389 | 1609s |
| 2 | ~2.10 | 1622s |
| 3 | 1.9828 | 1622s |

Training was interrupted during epoch 4.

**Capabilities**  

After TinyStories training, AtomLM can:
- Generate short stories
- Create simple dialogue
- Maintain characters over short contexts
- Follow common story patterns

Example:

Prompt:
```
Once upon a time
```

Output:
```
Once upon a time, there was a little boy named Timmy.
Timmy loved to play with his toys in his room.
```

**Limitations**

Current AtomLM:

- Does not have factual knowledge
- Cannot perform reliable reasoning
- Has a 128 token context window
- Mostly performs pattern completion

Example:

Prompt:
```
Ganesh loves artificial intelligence and he
```

Output:
```
Ganesh loves artificial intelligence and he is like
a princess and a boat.
```

The model learned language patterns but does not understand the meaning.


## Planned Experiments

### Stage 2: Knowledge

Train on:
- Wikipedia
- Educational text

Goal:
- Improve factual knowledge

### Stage 3: Reasoning

Train on:
- GSM8K
- MATH
- Mathematical proofs

Goal:
- Improve mathematical reasoning

### Stage 4: Tools

Add:
- SymPy
- Lean
- External memory

Goal:
- Build a tool-assisted reasoning system

---

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

---

## Tech Stack

- PyTorch
- HuggingFace Tokenizers
- HuggingFace Datasets
- Kaggle GPUs

---

## Author

[Ganesh Kumar](https://byganesh.com)  
MIT
