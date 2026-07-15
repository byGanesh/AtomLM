import torch
from torch.utils.data import Dataset
from tokenizers import ByteLevelBPETokenizer
from datasets import load_dataset
from pathlib import Path
from config import (
    TOKENIZER_DIR, DATA_DIR, MAX_SEQ_LEN, END_TOKEN_ID, PAD_TOKEN_ID, DATASET_WEIGHTS
)

# Format functions

def format_math(row):
    return f"<|problem|>{row['query']}<|step|>{row['response']}<|end|>"

def format_gsm8k(row):
    return f"<|problem|>{row['question']}<|step|>{row['answer']}<|end|>"

def format_tinystories(row):
    return f"<|problem|>Continue the story:<|step|>{row['text']}<|end|>"

def format_arc(row):
    choices = " ".join(row['choices']['text'])
    return f"<|problem|>{row['question']}<|step|>{choices}<|end|>"

# Stream functions
def stream_metamathqa(n_samples):
    ds = load_dataset("meta-math/MetaMathQA", split="train")
    for i, row in enumerate(ds):
        yield format_math(row)
        if i >= n_samples: break

def stream_gsm8k(n_samples):
    ds = load_dataset("openai/gsm8k", "main", split="train")
    for i, row in enumerate(ds):
        yield format_gsm8k(row)
        if i >= n_samples: break

def stream_tinystories(n_samples):
    ds = load_dataset("roneneldan/TinyStories", split="train")
    for i, row in enumerate(ds):
        yield format_tinystories(row)
        if i >= n_samples: break

def stream_arc(n_samples):
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
    for i, row in enumerate(ds):
        yield format_arc(row)
        if i >= n_samples: break

DATASET_STREAMS = {
    "metamathqa": stream_metamathqa,
    "gsm8k": stream_gsm8k,
    "tinystories": stream_tinystories,
    "arc":stream_arc,
}

# build func: stream raw data, tokenize it, pack into binrary file
def build(total_samples=500_000):
    tokenizer = ByteLevelBPETokenizer(
        str(TOKENIZER_DIR / "vocab.json"),
        str(TOKENIZER_DIR / "merges.txt"),
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR/"tokens.bin"

    all_tokens = []
    total = 0

    for name, weight in DATASET_WEIGHTS.items():
        if name not in DATASET_STREAMS:
            print(f"Skipping {name}... no stream function")
            continue
        n = int(total_samples * weight)
        print(f"\n {name}: {n} samples (weight={weight})")

        stream_fn = DATASET_STREAMS[name]
        count = 0

        for text in stream_fn(n):
            ids = tokenizer.encode(text).ids
            all_tokens.extend(ids + [END_TOKEN_ID])
            count += 1
            if count % 10_000 == 0:
                print(f"{count}/{n} ... {len(all_tokens):,} tokens so far.")

        total += count
        print(f"done: {count} samples")

    print(f"\nTotal tokens: {len(all_tokens):,}")
    print(f"Total samples: {total:,}")

    tokens = torch.tensor(all_tokens, dtype=torch.int32)
    torch.save(tokens, out_path)
    print(f"Saved to {out_path}")
    return out_path

class AtomDataset(Dataset):
    def __init__(self, path = None):
        path = path or DATA_DIR / "tokens.bin"
        print(f"Loading tokens from {path}...")
        tokens = torch.load(path)
        print(f"Total tokens: {len(tokens):,}")

        self.samples = []
        for i in range(0, len(tokens) - MAX_SEQ_LEN, MAX_SEQ_LEN):
            chunk = tokens[i : i + MAX_SEQ_LEN + 1]
            if len(chunk) == MAX_SEQ_LEN + 1:
                self.samples.append(chunk)

        print(f"Total samples: {len(self.samples):,}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        chunk = self.samples[idx].long()
        return chunk[:-1], chunk[1:]

if __name__ == "__main__":
    build(total_samples=500_000)
    ds = AtomDataset()
    x, y = ds[0]
    print(f"\nSample 0:")
    print(f" x shape: {x.shape}")
    print(f" y shape: {y.shape}")
    print(f" x[:10]: {x[:10].tolist()}")
    print(f" y[:10]: {y[:10].tolist()}")
