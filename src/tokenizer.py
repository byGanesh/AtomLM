from tokenizers import ByteLevelBPETokenizer
from pathlib import Path

data_file = "datasets/stories/tinystories.txt"
save_dir = Path("tokenizer")
save_dir.mkdir(exist_ok=True)

tokenizer = ByteLevelBPETokenizer()
print("training....")
tokenizer.train(
    files = [data_file],
    vocab_size = 8000,
    min_frequency = 2,
    special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]
)

tokenizer.save_model(str(save_dir))
print("Tokenizer trained and saved")

# testing
encoded = tokenizer.encode("Once upon a time there was a little girl")
print(f"Tokens: {encoded.tokens}")
print(f"IDs: {encoded.ids}")
print(f"Decoded: {tokenizer.decode(encoded.ids)}")
