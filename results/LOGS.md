# AtomLM Experiment Log

## Exp 001 — 2026-07-12

**Config:** 2.8M params, d_model=128, 4 layers, 4 heads, seq_len=128, bs=32, lr=3e-4  
**Data:** TinyStories 50k stories, 2M tokens, 3 epochs  
**Hardware:** CPU only

| Epoch | Loss | Time |
|---|---|---|
| 1 | 5.29 | 261s |
| 2 | 4.12 | 251s |
| 3 | 3.77 | 252s |

**Observations:** Learned story grammar and openings fast. Repetition problem at temp=0.8. Short prompts exit early. Loss still high — needs more data and epochs.

**Generations:**  
Prompt: Once upon a time  
Response: Once upon a time, there was a little girl named Lily. She loved to play with her to play with her friends. One day, Lily were her mom's mom too. One day, Lily went to the jungle. One day, she found a little girl named Sarah. She loved to play with her dad. One day, a big hug and Tim went to the ground. "That made a big boy. I think you, Lily, let's okay, Lily. You are playing, but you would try


## Exp 002 — 2026-07-13

**Config:** 8.8M params, d_model=256, 6 layers, 8 heads, seq_len=128, bs=128, lr=3e-4
**Data:** TinyStories full 2.1M stories, 85M tokens
**Hardware:** T4 x2 GPU

| Epoch | Avg Loss | Time |
|---|---|---|
| 1 | 2.7389 | 1609s |
| 2 | ~2.1 | 1622s |
| 3 | 1.9828 | 1622s |

*Interrupted during epoch 4 step 2000*

**Observations:** Coherent complete stories. Consistent characters. Natural dialogue.
No more repetition loops. Clear quality jump from Exp 001.

**Generations (temp=0.8):**

| Prompt | Response |
|---|---|
| Once upon a time | Once upon a time, there was a little boy named Timmy. Timmy loved to play with his toys in his room. One day, Timmy's mom made him a big, fluffy sandwich for lunch. Timmy was so happy to eat it all up! |
| The little boy | The little boy smiled. "Thank you," he asked the little girl, giving her a bigger grin. |
| One day a girl named | One day a girl named Emily was walking in the park with her dad when he found his wallet on the ground. He was very confused. He didn't know what it was, but he thought it was a good looking. |
| Ganesh is a boy and loves | Ganesh is a boy and loves nature and playing with them. |
| Ganesh loves artificial intelligence and he | Ganesh loves artificial intelligence and he is like a princess and a boat. |
| AtomLM is a helpful model that | AlomLM is a helpful model that helps me with anything. I want to see how I am in the store and what I want to buy. |

**Notable:** Model has no factual knowledge — "Ganesh loves AI and he is like a princess and a boat" shows pure pattern completion, no understanding. Expected at Stage 1.

**Next:** Stage 2 with RoPE + math data.

## Exp 003 - 2026-07-14 to 2026-07-17

**Config:** 52.3M params, d_model=512, 16 layers, 8 query heads, 2 KV heads (GQA), FFN=1536, RoPE, RMSNorm, SwiGLU, FlashAttention, seq_len=1024, bs=16, grad_accum=8, lr=3e-4 (cosine), fp16

**Data:** 343.6M tokens
DATASET_WEIGHTS = {
    "tinystories": 0.20,   # grammar
    "metamathqa":  0.40,   # math reasoning
    "gsm8k":       0.05,   # math reasoning
    "arc":         0.05,   # science reasoning
    "openwebmath": 0.30,   # math + science + logic from the web
}

**Hardware:** T4 x2 GPU
**Training Time:** ~7 Hours

| Epoch | Avg Loss |
|---|---:|
| 1 | ~3.39 |
| 2 | ~2.07 |
| 3 | ~1.65 |

**Final checkpoint:** Step 3940

**Major architecture upgrades**  
- ✓ RoPE positional embeddings
- ✓ FlashAttention
- ✓ Grouped Query Attention (GQA)
- ✓ RMSNorm
- ✓ SwiGLU feed-forward
- ✓ Cosine LR scheduler with warmup
- ✓ Gradient accumulation
- ✓ Mixed precision (FP16)
- ✓ Distributed Data Parallel (DDP)

**Observations**  
- First successful modern Transformer implementation.
- Training remained stable throughout all three epochs.
- Loss decreased smoothly from ~9.0 at initialization to ~1.65.
- Story generation quality improved noticeably.
- Learned the `<|problem|>`, `<|step|>`, `<|answer|>`, and `<|end|>` format.
- Began producing multi-step mathematical reasoning traces.
- Frequently produced plausible reasoning despite incorrect final answers.
- Strong tendency to overfit toward mathematical reasoning because most training data contained reasoning traces.
- General knowledge and instruction following remained weak.
- Hallucinations were common outside the training distribution.

**Example generations**  
| Prompt | Response |
|---|---|
| Once upon a time there was a... | Generated coherent multi-paragraph children's stories with consistent grammar and characters. |
| John has 10 apples... | Correctly solved many simple arithmetic word problems and showed reasoning steps. |
| What is science? | Produced fluent but hallucinated mathematical explanations unrelated to the prompt. |
| Can an elephant fly? | Switched into mathematical reasoning instead of answering logically. |
| Solve x² + 5x − 6 = 0 | Generated reasonable-looking derivations but often incorrect mathematics. |
| Essay on cow | Drifted into numerical reasoning instead of essay writing. |

**Notable findings**  
- Model learned English grammar far better than factual knowledge.
- Mathematical reasoning format emerged surprisingly early.
- Chain-of-thought style generation appeared without explicit RLHF.
- The model often "looked intelligent" while making incorrect logical inferences.
- Storytelling quality was significantly stronger than Exp 002.
- Instruction following was weak because no instruction tuning had been performed.
- The base model behaved like a pretrained language model rather than a chatbot.

**Lessons learned**  
- Architecture improvements (RoPE + FlashAttention + GQA) worked correctly.
- Data quality is now the primary bottleneck rather than model architecture.
- Continued pretraining is likely to provide much larger gains than immediately increasing parameter count.
- Instruction tuning is required before evaluating conversational ability.
- Specialized reasoning datasets alone bias the model toward answering everything as a math problem.

**Next**  
- Continued pretraining on several billion high-quality general-language tokens.
- Instruction tuning using high-quality instruction-response datasets.
- Reasoning tuning using curated mathematics and coding datasets.
- Tool-use training (search, calculator, code execution).
