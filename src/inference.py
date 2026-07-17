import torch
from tokenizers import ByteLevelBPETokenizer

from model import AtomLM
from config import (
    TOKENIZER_DIR,
    CHECKPOINT_DIR,
    END_TOKEN_ID
)


# ----------------------------
# Device
# ----------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print(f"Device: {device}")


# ----------------------------
# Tokenizer
# ----------------------------

tokenizer = ByteLevelBPETokenizer(
    str(TOKENIZER_DIR / "vocab.json"),
    str(TOKENIZER_DIR / "merges.txt"),
)


# ----------------------------
# Model
# ----------------------------

model = AtomLM().to(device)


weights = torch.load(
    CHECKPOINT_DIR / "latest.pt",
    map_location=device,
    weights_only=True
)


model.load_state_dict(weights)

model.eval()

print("AtomLM loaded successfully")
print("-" * 50)



# ----------------------------
# Generation
# ----------------------------

def generate(
    prompt,
    max_new_tokens=1000,
    temperature=0.8,
    top_k=50
):

    ids = tokenizer.encode(prompt).ids

    x = torch.tensor(
        [ids],
        dtype=torch.long,
        device=device
    )


    with torch.no_grad():

        output = model.generate(
            x,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            stop_token_id=END_TOKEN_ID
        )


    text = tokenizer.decode(
        output[0].tolist()
    )

    return text



# ----------------------------
# Interactive loop
# ----------------------------

print("\nType your prompt.")
print("Type 'exit' to quit.\n")


while True:

    prompt = input(">>> ")

    if prompt.lower() == "exit":
        break


    # Add your training format
    formatted = (
        "<|problem|>"
        + prompt
        + "<|step|>"
    )


    result = generate(
        formatted,
        max_new_tokens=200,
        temperature=0.7,
        top_k=40
    )


    print("\nAtomLM:")
    print(result)

    print("\n" + "-"*50)
