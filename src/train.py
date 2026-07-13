import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tokenizers import ByteLevelBPETokenizer
from pathlib import Path
from model import AtomLM
from dataset import StoryDataset
import config
import time

tokenizer = ByteLevelBPETokenizer(
    str(config.TOKENIZER_DIR / "vocab.json"),
    str(config.TOKENIZER_DIR / "merges.txt")
)


dataset = StoryDataset(
    config.DATA_FILE,
    tokenizer,
    config.MAX_SEQ_LEN,
    config.MAX_SAMPLES
)


loader = DataLoader(
    dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = AtomLM(
    config.VOCAB_SIZE,
    config.D_MODEL,
    config.N_HEADS,
    config.N_LAYERS,
    config.FFN_DIM,
    config.MAX_SEQ_LEN
).to(device)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.LR
)


loss_fn = nn.CrossEntropyLoss()

print(f"\nParameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Batches per epoch: {len(loader)}")
print("\nTraining...\n")

for epoch in range(config.EPOCHS):
    total_loss = 0
    start = time.time()

    for step, (x, y) in enumerate(loader):
        x,y = x.to(device), y.to(device)
        logits = model(x)

        loss = loss_fn(logits.view(-1, config.VOCAB_SIZE), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

        if step % 100 == 0:
            print(f"Epoch {epoch+1} | Step {step} | Loss {loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    elapsed = time.time() - start
    print(f"\nEpoch {epoch+1} done | Avg Loss {avg_loss:.4f} | Time {elapsed:.1f}s\n")

    torch.save(model.state_dict(), f"checkpoints/atomlm_epoch{epoch+1}.pt")
    print(f"Saved checkpoints/atomlm_epoch{epoch+1}.pt")

# save
Path("checkpoints").mkdir(exist_ok=True)
torch.save(model.state_dict(), "checkpoints/atomlm_stage1.pt")
print("Model saved to checkpoints/atomlm_stage1.pt")
