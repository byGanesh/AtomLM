# src/eval.py
import torch
from tokenizers import ByteLevelBPETokenizer
from collections import Counter
from model import AtomLM
from config import (
    TOKENIZER_DIR, CHECKPOINT_DIR, MAX_SEQ_LEN,
    PROBLEM_TOKEN_ID, STEP_TOKEN_ID, ANSWER_TOKEN_ID, END_TOKEN_ID
)

# ── Load ──────────────────────────────────────────────────────────────────────

tokenizer = ByteLevelBPETokenizer(
    str(TOKENIZER_DIR / "vocab.json"),
    str(TOKENIZER_DIR / "merges.txt"),
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = AtomLM().to(device)

ckpt = torch.load(CHECKPOINT_DIR / "latest.pt", map_location=device, weights_only=True)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"Loaded checkpoint from step {ckpt['step']}")

# ── Generate ──────────────────────────────────────────────────────────────────

def generate(prompt, max_new_tokens=200, temperature=0.8, top_k=50):
    ids = tokenizer.encode(prompt).ids
    x   = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(
        x,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        stop_token_id=END_TOKEN_ID,
    )
    return tokenizer.decode(out[0].tolist())

# ── Test 1: Structure fidelity ────────────────────────────────────────────────
# Does the model generate <|step|> and <|answer|> tokens at all?
# Pass = model learned the reasoning format

def test_structure(prompts):
    print("\n" + "="*50)
    print("TEST 1 — Structure Fidelity")
    print("="*50)

    passed = 0
    for prompt in prompts:
        out     = generate(prompt)
        has_step   = "<|step|>"   in out
        has_answer = "<|answer|>" in out
        has_end    = "<|end|>"    in out
        ok      = has_step and has_answer

        print(f"\nPrompt:  {prompt}")
        print(f"Output:  {out[:200]}")
        print(f"<|step|>: {has_step}  <|answer|>: {has_answer}  <|end|>: {has_end}")
        print(f"{'PASS' if ok else 'FAIL'}")

        if ok:
            passed += 1

    print(f"\nStructure: {passed}/{len(prompts)} passed")
    return passed / len(prompts)

# ── Test 2: Self-consistency ──────────────────────────────────────────────────
# Sample 8 times, majority vote on answer
# Gap between greedy and voted = model's latent knowledge

def extract_answer(text):
    if "<|answer|>" not in text:
        return None
    after = text.split("<|answer|>")[-1]
    ans   = after.split("<|end|>")[0].strip()
    return ans if ans else None

def test_self_consistency(problems, n_samples=8):
    print("\n" + "="*50)
    print("TEST 2 — Self Consistency")
    print(f"          {n_samples} samples per problem, majority vote")
    print("="*50)

    greedy_correct = 0
    voted_correct  = 0

    for problem, expected in problems:
        prompt = f"<|problem|>{problem}"

        # greedy
        greedy_out = generate(prompt, temperature=0.0001)
        greedy_ans = extract_answer(greedy_out)
        greedy_ok  = greedy_ans is not None and expected.lower() in greedy_ans.lower()

        # sample n times
        answers = []
        for _ in range(n_samples):
            out = generate(prompt, temperature=0.8, top_k=50)
            ans = extract_answer(out)
            if ans:
                answers.append(ans)

        # majority vote
        voted_ans = None
        voted_ok  = False
        if answers:
            voted_ans = Counter(answers).most_common(1)[0][0]
            voted_ok  = expected.lower() in voted_ans.lower()

        print(f"\nProblem:  {problem}")
        print(f"Expected: {expected}")
        print(f"Greedy:   {greedy_ans}  {'✓' if greedy_ok else '✗'}")
        print(f"Voted:    {voted_ans}  {'✓' if voted_ok else '✗'}")
        print(f"Samples:  {answers}")

        if greedy_ok: greedy_correct += 1
        if voted_ok:  voted_correct  += 1

    n = len(problems)
    print(f"\nGreedy accuracy:  {greedy_correct}/{n} = {greedy_correct/n*100:.1f}%")
    print(f"Voted accuracy:   {voted_correct}/{n}  = {voted_correct/n*100:.1f}%")
    print(f"Self-consistency gain: +{(voted_correct - greedy_correct)/n*100:.1f}%")

# ── Test 3: Cross-domain ──────────────────────────────────────────────────────
# One prompt per domain — does quality hold across all?

def test_cross_domain():
    print("\n" + "="*50)
    print("TEST 3 — Cross Domain")
    print("="*50)

    prompts = {
        "math":    "<|problem|>A store has 48 apples. They sell 15. How many remain?",
        "code":    "<|problem|>Write a Python function to check if a number is even.",
        "grammar": "<|problem|>Continue the story: The little dog sat by the river...",
        "science": "<|problem|>Why does ice float on water?",
        "logic":   "<|problem|>All cats are animals. Whiskers is a cat. What is Whiskers?",
    }

    for domain, prompt in prompts.items():
        out = generate(prompt)
        has_structure = "<|step|>" in out and "<|answer|>" in out
        print(f"\n[{domain.upper()}]")
        print(f"Prompt: {prompt}")
        print(f"Output: {out[:300]}")
        print(f"Structure: {'✓' if has_structure else '✗'}")

# ── Run all tests ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Test 1 prompts — just problem starters
    structure_prompts = [
        "<|problem|>John has 5 apples. He gives 2 to Mary. How many does he have?",
        "<|problem|>Write a function to reverse a string in Python.",
        "<|problem|>What is the boiling point of water?",
        "<|problem|>Continue the story: Once there was a brave knight...",
        "<|problem|>If all birds can fly and a penguin is a bird, can a penguin fly?",
    ]

    # Test 2 problems — (question, expected answer)
    consistency_problems = [
        ("What is 12 + 15?",                          "27"),
        ("A bag has 10 red and 5 blue balls. How many total?", "15"),
        ("If x + 3 = 7, what is x?",                  "4"),
        ("What is 8 × 7?",                             "56"),
        ("A train travels 60km in 1 hour. Speed?",     "60"),
    ]

    test_structure(structure_prompts)
    test_self_consistency(consistency_problems, n_samples=8)
    test_cross_domain()
