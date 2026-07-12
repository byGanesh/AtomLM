import torch
from tokenizers import ByteLevelBPETokenizer
from model import AtomLM


VOCAB_SIZE = 8000
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 4
FFN_DIM = 512
MAX_SEQ_LEN = 128

tokenizer = ByteLevelBPETokenizer(
    "tokenizer/vocab.json",
    "tokenizer/merges.txt"
)


model = AtomLM(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, FFN_DIM, MAX_SEQ_LEN)
model.load_state_dict(torch.load("checkpoints/atomlm_stage1.pt", weights_only=True))
model.eval()

def generate(prompt, max_new_tokens=100, temperature=0.8):
    ids = tokenizer.encode(prompt).ids
    x = torch.tensor([ids], dtype=torch.long)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(x[:, -MAX_SEQ_LEN:])
            next_logits = logits[0, -1, :] / temperature
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, next_token.unsqueeze(0)], dim=1)
            if next_token.item() == 3:  # <eos>
                break

    return tokenizer.decode(x[0].tolist())

print(generate("Once upon a time", temperature=0.5))
print("temperature 1.2-------")
print(generate("Once upon a time", temperature=1.2))
