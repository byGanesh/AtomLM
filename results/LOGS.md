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
**Hardware:** Kaggle T4 x2 GPU

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
