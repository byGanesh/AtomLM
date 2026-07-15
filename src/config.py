from pathlib import Path

# Model
VOCAB_SIZE = 8004
D_MODEL = 768
N_HEADS = 12
N_KV_HEADS = 3
N_LAYERS = 16
FFN_DIM = 2048
MAX_SEQ_LEN = 1024
ROPE_THETA = 10000.0
TIE_EMBEDDINGS = True

# Special token IDs
PAD_TOKEN_ID = 0
PROBLEM_TOKEN_ID = 1
STEP_TOKEN_ID = 2
ANSWER_TOKEN_ID = 3
END_TOKEN_ID = 4
SPECIAL_TOKENS = ["<|pad|>", "<|problem|>", "<|step|>", "<|answer|>", "<|end|>"]

# Training
BATCH_SIZE = 16
GRAD_ACCUM = 8
EPOCHS = 3
LR = 2e-4
MIN_LR = 2e-5
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
CHECKPOINT_EVERY = 500

# Precision
PRECISION = "fp16"

# Datasets
DATASET_WEIGHTS = {
    "tinystories": 0.10,
    "metamathqa": 0.45,
    "gsm8k": 0.05,
    "python_code": 0.30,
    "arc": 0.10,
}


# Paths
TOKENIZER_DIR = Path("tokenizer")
CHECKPOINT_DIR = Path("checkpoints")
DATA_DIR = Path("data/processed")
LOG_DIR = Path("results")

# Derived
HEAD_DIM = D_MODEL // N_HEADS # 64
KV_DIM = N_KV_HEADS * HEAD_DIM  #192
N_QUERIES_PER_KV = N_HEADS // N_KV_HEADS # 4

# Sanity Checks
assert D_MODEL % N_HEADS == 0, "D_MODEL must be divisible by N_HEADS"
assert N_HEADS % N_KV_HEADS == 0, "N_HEADS must be divisible by N_KV_HEADS"
assert HEAD_DIM % 2 == 0, "HEAD_DIM must be even for RoPE"
