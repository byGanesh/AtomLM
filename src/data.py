import torch
import numpy as np
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


def stream_metamathqa(n_samples=10_000_000):
    ds = load_dataset("meta-math/MetaMathQA", split="train")
    for i, row in enumerate(ds):
        yield format_math(row)
        if i >= n_samples: break

def stream_gsm8k(n_samples=10_000_000):
    ds = load_dataset("openai/gsm8k", "main", split="train")
    for i, row in enumerate(ds):
        yield format_gsm8k(row)
        if i >= n_samples: break

def stream_tinystories(n_samples=10_000_000):
    ds = load_dataset("roneneldan/TinyStories", split="train")
    for i, row in enumerate(ds):
        yield format_tinystories(row)
        if i >= n_samples: break

def stream_arc(n_samples=10_000_000):
    ds1 = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
    ds2 = load_dataset("allenai/ai2_arc", "ARC-Easy", split="train")
    count = 0
    for row in list(ds1) + list(ds2):
        yield format_arc(row)
        count += 1
        if count >= n_samples: break

def stream_openwebmath(n_samples=10_000_000):
    ds = load_dataset("open-web-math/open-web-math", split="train", streaming=True)
    count = 0
    for row in ds:
        text = row['text'].strip()
        if len(text) > 100:
            yield f"<|problem|>Reason through this:<|step|>{text}<|end|>"
            count += 1
            if count >= n_samples: break

DATASET_STREAMS = {
    "tinystories": stream_tinystories,
    "metamathqa":  stream_metamathqa,
    "gsm8k":       stream_gsm8k,
    "arc":         stream_arc,
    "openwebmath": stream_openwebmath,
}


def build(total_tokens=500_000_000):
    tokenizer = ByteLevelBPETokenizer(
        str(TOKENIZER_DIR / "vocab.json"),
        str(TOKENIZER_DIR / "merges.txt"),
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "tokens.bin"

    f = open(out_path, 'wb')
    total_written = 0

    for name, weight in DATASET_WEIGHTS.items():
        if name not in DATASET_STREAMS:
            print(f"Skipping {name}... no stream function")
            continue

        token_budget = int(total_tokens * weight)
        print(f"\n{name}: token budget = {token_budget:,} (weight={weight})", flush=True)

        stream_fn = DATASET_STREAMS[name]
        count = 0
        tokens_so_far = 0
        chunk = []

        for text in stream_fn():
            ids = tokenizer.encode(text).ids
            chunk.extend(ids + [END_TOKEN_ID])
            tokens_so_far += len(ids) + 1
            count += 1

            if len(chunk) >= 100_000:
                np.array(chunk, dtype=np.int32).tofile(f)
                total_written += len(chunk)
                chunk = []

            if tokens_so_far % 50_000_000 < 2500:
                print(f"  {count} samples — {tokens_so_far:,} / {token_budget:,} tokens", flush=True)

            if tokens_so_far >= token_budget:
                print(f"  token budget reached at {count} samples", flush=True)
                break

        # flush remaining chunk
        if chunk:
            np.array(chunk, dtype=np.int32).tofile(f)
            total_written += len(chunk)

        print(f"  done: {count} samples, {tokens_so_far:,} tokens", flush=True)

    f.close()
    print(f"\nTotal tokens written: {total_written:,}")
    print(f"Saved to {out_path}")
    return out_path


class AtomDataset(Dataset):
    def __init__(self, path=None):
        path = path or DATA_DIR / "tokens.bin"
        print(f"Loading tokens from {path}...")

        tokens = np.memmap(path, dtype=np.int32, mode='r')
        print(f"Total tokens: {len(tokens):,}")

        self.samples = []
        for i in range(0, len(tokens) - MAX_SEQ_LEN, MAX_SEQ_LEN):
            chunk = tokens[i : i + MAX_SEQ_LEN + 1]
            if len(chunk) == MAX_SEQ_LEN + 1:
                self.samples.append(torch.tensor(chunk.astype(np.int64)))

        print(f"Total samples: {len(self.samples):,}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        chunk = self.samples[idx]
        return chunk[:-1], chunk[1:]


if __name__ == "__main__":
    build(total_tokens=500_000_000)
    ds = AtomDataset()
    x, y = ds[0]
    print(f"\nSample 0:")
    print(f"  x shape: {x.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  x[:10]: {x[:10].tolist()}")
    print(f"  y[:10]: {y[:10].tolist()}")
