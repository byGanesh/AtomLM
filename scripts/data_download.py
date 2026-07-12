from pathlib import Path
from datasets import load_dataset

# Create output directory
output_dir = Path("datasets/stories")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "tinystories.txt"

# Download dataset
dataset = load_dataset("roneneldan/TinyStories")

# Save all splits into one text file
with output_file.open("w", encoding="utf-8") as f:
    for split_name, split in dataset.items():
        print(f"Writing {split_name}...")

        for example in split:
            text = example["text"].strip()
            if text:
                f.write(text)
                f.write("\n\n")

print(f"\nSaved to: {output_file.resolve()}")
