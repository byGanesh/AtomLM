from pathlib import Path

# Model
VOCAB_SIZE = 8004
D_MODEL = 512
N_HEADS = 8
N_KV_HEADS = 2
N_LAYERS = 16
FFN_DIM = 1536
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
BATCH_SIZE = 8
GRAD_ACCUM = 16
EPOCHS = 3
LR = 3e-4
MIN_LR = 3e-5
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
CHECKPOINT_EVERY = 500
GRAD_CHECKPOINT = True
COMPILE = True

# Precision
PRECISION = "fp16"

# Datasets
DATASET_WEIGHTS = {
    "tinystories": 0.20,   # grammar
    "metamathqa":  0.40,   # math reasoning
    "gsm8k":       0.05,   # math reasoning
    "arc":         0.05,   # science reasoning
    "openwebmath": 0.30,   # math + science + logic from the web
}


# Paths
TOKENIZER_DIR = Path("tokenizer")
CHECKPOINT_DIR = Path("checkpoints")
DATA_DIR = Path("data/processed")
LOG_DIR = Path("results")

# Derived
HEAD_DIM = D_MODEL // N_HEADS # 64
KV_DIM = N_KV_HEADS * HEAD_DIM  #128
N_QUERIES_PER_KV = N_HEADS // N_KV_HEADS # 4

# Sanity Checks
assert D_MODEL % N_HEADS == 0, "D_MODEL must be divisible by N_HEADS"
assert N_HEADS % N_KV_HEADS == 0, "N_HEADS must be divisible by N_KV_HEADS"
assert HEAD_DIM % 2 == 0, "HEAD_DIM must be even for RoPE"
