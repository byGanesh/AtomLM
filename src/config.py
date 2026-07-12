from pathlib import Path

# Model
VOCAB_SIZE = 8000
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 4
FFN_DIM = 512
MAX_SEQ_LEN = 128

# Training
BATCH_SIZE = 32
EPOCHS = 3
LR = 3e-4

# Dataset
DATA_FILE = Path("datasets/stories/tinystories.txt")
MAX_SAMPLES = 50000

# Paths
TOKENIZER_DIR = Path("tokenizer")
CHECKPOINT_DIR = Path("checkpoints")
