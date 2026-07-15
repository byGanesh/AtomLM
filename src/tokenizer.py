from tokenizers import ByteLevelBPETokenizer
from datasets import load_dataset
from pathlib import Path
from config import VOCAB_SIZE, SPECIAL_TOKENS, TOKENIZER_DIR

def get_texts():
    print("MetaMathQA...")
    ds = load_dataset("meta-math/MetaMathQA", split="train")
    for i, row in enumerate(ds):
        yield row['query'] + " " + row['response']
        if i>= 50000:
            break

    print("TinyStories...")
    ds = load_dataset("roneneldan/TinyStories", split="train")
    for i,row in enumerate(ds):
        yield row['text']
        if i>= 20000:
            break

    print("GSM8K...")
    ds = load_dataset("roneneldan/TinyStories", split="train")
    for row in ds:
        yield row['question'] + " " + row['answer']

    print("ARC...")
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
    for row in ds:
        yield row['question'] + " " + " ".join(row['choices']['text'])

# corpus
Path("data").mkdir(exist_ok=True)
corpus_path = "data/corpus.txt"

print("writing corpus...")
with open(corpus_path, "w") as f:
    for text in get_texts():
        f.write(text.replace('\n', " ") + '\n')

# Train
TOKENIZER_DIR.mkdir(exist_ok=True)
tokenizer = ByteLevelBPETokenizer()

print("Training tokenizer...")
tokenizer.train(
    files =[corpus_path],
    vocab_size = VOCAB_SIZE,
    min_frequency = 2,
    special_tokens = SPECIAL_TOKENS,
    show_progress = True,
)

tokenizer.save_model(str(TOKENIZER_DIR))
print(f"Saved to {TOKENIZER_DIR}/")

# verification
for text in [
    "<|problem|>John has 3 apples.<|step|>He gets 4 more.<|answer|>7<|end|>",
    "def is_prime(n):",
    "3 + 4 = 7",
]:
    enc = tokenizer.encode(text)
    print(f"\n{text!r}")
    print(f"tokens: {enc.tokens}")
    print(f"count: {len(enc.ids)}")
