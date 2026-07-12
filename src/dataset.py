import torch
import torch.nn as nn
from torch.utils.data import Dataset

class StoryDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_seq_len, max_samples):
        print("Loading data...")
        with open(data_file, "r", encoding="utf-8") as f:
            text = f.read()

        stories = [s.strip() for s in text.split("\n\n") if s.strip()]
        stories = stories[:max_samples]

        print(f"Tokenizing {len(stories)} stories...")


        all_tokens = []
        for story in stories:
            ids = tokenizer.encode(story).ids
            all_tokens.extend(ids + [3])

        print(f"Total tokens: {len(all_tokens):,}")


        self.samples = []
        for i in range(0, len(all_tokens) - max_seq_len, max_seq_len):
            chunk = all_tokens[i:i + max_seq_len + 1]
            self.samples.append(chunk)

        print(f"Total samples: {len(self.samples):,}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        chunk = torch.tensor(self.samples[idx], dtype=torch.long)
        return chunk[:-1], chunk[1:]
