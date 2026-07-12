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

**Next:** Train on 300k stories
