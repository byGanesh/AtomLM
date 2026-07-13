from pathlib import Path

# Model
VOCAB_SIZE = 8000
D_MODEL = 256
N_HEADS = 8
N_LAYERS = 6
FFN_DIM = 1024
MAX_SEQ_LEN = 128

# Training
BATCH_SIZE = 128
EPOCHS = 5
LR = 3e-4

# Dataset
DATA_FILE = Path("datasets/stories/tinystories.txt")
MAX_SAMPLES = 2_100_000

# Paths
TOKENIZER_DIR = Path("tokenizer")
CHECKPOINT_DIR = Path("checkpoints")
