import torch
import numpy as np
from torch.utils.data import Dataset
from tokenizers import ByteLevelBPETokenizer
from datasets import load_dataset
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))
from config_cpt import (
    TOKENIZER_DIR, MAX_SEQ_LEN, END_TOKEN_ID, DATASET_WEIGHTS
)

DATA_DIR = Path("data/processed")
CPT_DATA_PATH = DATA_DIR / "tokens_cpt.bin"


def format_wikipedia(row):
    text = row['text'].strip()
    if len(text) < 200:
        return None
    return f"{text}<|end|>"

def format_books(row):
    text = row['text'].strip()
    if len(text) < 100:
        return None
    return f"{text}<|end|>"

def format_tinystories(row):
    text = row['text'].strip()
    if len(text) < 50:
        return None
    return f"{text}<|end|>"

def format_math(row):
    return f"{row['query'].strip()}\n{row['response'].strip()}<|end|>"

def format_gsm8k(row):
    return f"{row['question'].strip()}\n{row['answer'].strip()}<|end|>"


def stream_wikipedia(n_samples=10_000_000):
    ds = load_dataset(
        "wikimedia/wikipedia", "20231101.en",
        split="train", streaming=True
    )
    count = 0
    for row in ds:
        text = format_wikipedia(row)
        if text:
            yield text
            count += 1
            if count >= n_samples:
                break

def stream_books(n_samples=10_000_000):
    ds = load_dataset(
        "bookcorpus/bookcorpus",
        split="train", streaming=True
    )
    count = 0
    for row in ds:
        text = format_books(row)
        if text:
            yield text
            count += 1
            if count >= n_samples:
                break

def stream_tinystories(n_samples=10_000_000):
    ds = load_dataset("roneneldan/TinyStories", split="train")
    count = 0
    for row in ds:
        text = format_tinystories(row)
        if text:
            yield text
            count += 1
            if count >= n_samples:
                break

def stream_metamathqa(n_samples=10_000_000):
    ds = load_dataset("meta-math/MetaMathQA", split="train")
    count = 0
    for row in ds:
        yield format_math(row)
        count += 1
        if count >= n_samples:
            break

def stream_gsm8k(n_samples=10_000_000):
    ds = load_dataset("openai/gsm8k", "main", split="train")
    count = 0
    for row in ds:
        yield format_gsm8k(row)
        count += 1
        if count >= n_samples:
            break

DATASET_STREAMS = {
    "tinystories": stream_tinystories,
    "wikipedia":   stream_wikipedia,
    "bookcorpus":  stream_books,
    "metamathqa":  stream_metamathqa,
    "gsm8k":       stream_gsm8k,
}


def build(total_tokens=300_000_000):
    tokenizer = ByteLevelBPETokenizer(
        str(TOKENIZER_DIR / "vocab.json"),
        str(TOKENIZER_DIR / "merges.txt"),
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    f = open(CPT_DATA_PATH, 'wb')
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
            chunk.extend(ids)
            tokens_so_far += len(ids)
            count += 1

            if len(chunk) >= 100_000:
                np.array(chunk, dtype=np.int32).tofile(f)
                total_written += len(chunk)
                chunk = []

            if tokens_so_far % 10_000_000 < 500:
                print(f"  {count} samples — {tokens_so_far:,} / {token_budget:,} tokens", flush=True)

            if tokens_so_far >= token_budget:
                print(f"  budget reached at {count} samples", flush=True)
                break

        if chunk:
            np.array(chunk, dtype=np.int32).tofile(f)
            total_written += len(chunk)

        print(f"  done: {count} samples, {tokens_so_far:,} tokens", flush=True)

    f.close()
    print(f"\nTotal tokens written: {total_written:,}")
    print(f"Saved to {CPT_DATA_PATH}")
    return CPT_DATA_PATH


class AtomDataset(Dataset):
    def __init__(self, path=None):
        path = path or CPT_DATA_PATH
        print(f"Loading tokens from {path}...")
        self.tokens = np.memmap(path, dtype=np.int32, mode='r')
        self.n_samples = (len(self.tokens) - 1) // MAX_SEQ_LEN
        print(f"Total tokens: {len(self.tokens):,}")
        print(f"Total samples: {self.n_samples:,}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * MAX_SEQ_LEN
        chunk = self.tokens[start : start + MAX_SEQ_LEN + 1].astype(np.int64)
        x = torch.tensor(chunk[:-1])
        y = torch.tensor(chunk[1:])
        return x, y


if __name__ == "__main__":
    build(total_tokens=300_000_000)
    ds = AtomDataset()
    x, y = ds[0]
    print(f"\nSample 0:")
    print(f"  x shape: {x.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  x[:10]: {x[:10].tolist()}")
    print(f"  y[:10]: {y[:10].tolist()}")
