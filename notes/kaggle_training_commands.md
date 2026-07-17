Here are the full instructions from start to finish.

---

**Step 1 — Set up Kaggle notebook**

New notebook → Settings (right panel):
- Accelerator: `GPU T4 x2` (gives you 2 GPUs for DDP, free tier)
- Internet: `On`
- Persistence: `Files` (keeps your outputs between sessions)

---

**Step 2 — First cell: clone your repo and install deps**

```python
!git clone https://github.com/YOUR_USERNAME/AtomLM.git
%cd AtomLM
!pip install tokenizers datasets huggingface_hub -q
```

---

**Step 3 — Download base checkpoint from HuggingFace**

```python
from huggingface_hub import hf_hub_download
import shutil
from pathlib import Path

# Download the final model (210MB, not the training checkpoint)
path = hf_hub_download(
    repo_id="byGanesh/AtomLM-v0.1",
    filename="AtomLM_final.pt",
)

# Put it where train_cpt.py expects it
Path("checkpoints/base").mkdir(parents=True, exist_ok=True)
shutil.copy(path, "checkpoints/base/latest.pt")
print("Base checkpoint ready.")

# Download tokenizer files
for f in ["tokenizer/vocab.json", "tokenizer/merges.txt"]:
    p = hf_hub_download(repo_id="byGanesh/AtomLM-v0.1", filename=f)
    dest = Path(f)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(p, dest)

print("Tokenizer ready.")
```

Use `AtomLM_final.pt` (210MB), not `training_checkpoint_latest.pt` (629MB). The training checkpoint has optimizer state baked in which you don't want — you're resetting that anyway.

---

**Step 4 — Verify the checkpoint loads**

```python
import torch
ckpt = torch.load("checkpoints/base/latest.pt", map_location='cpu')
print(ckpt.keys())
print("loss:", ckpt.get('loss'))
print("step:", ckpt.get('step'))
```

If `model_state_dict` is in the keys, you're good. If the file just contains raw weights (no dict), you'll need to wrap it:

```python
# Only run this if keys() doesn't show 'model_state_dict'
torch.save({'model_state_dict': ckpt, 'loss': 0, 'step': 0},
           "checkpoints/base/latest.pt")
print("Wrapped and re-saved.")
```

---

**Step 5 — Build CPT dataset**

```python
%cd /kaggle/working/AtomLM/cpt
!python data_cpt.py
```

This will take 30–60 minutes depending on download speed. Wikipedia alone is a few GB. Watch the output — it prints token counts per dataset as it goes.

If you hit a memory issue during build, reduce in `dataset_cpt.py`:

```python
def build(total_tokens=150_000_000):  # drop to 150M if 300M is too slow
```

---

**Step 6 — Smoke test before full training**

```python
import sys
sys.path.insert(0, '/kaggle/working/AtomLM/src')
sys.path.insert(0, '/kaggle/working/AtomLM/cpt')

import torch
from model import AtomLM
import config_cpt as config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AtomLM().to(device)

# Load base checkpoint
ckpt = torch.load("checkpoints/base/latest.pt", map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'])
print("Model loaded OK")
print(f"Params: {model.num_params():,}")

# Quick forward pass
x = torch.randint(0, config.VOCAB_SIZE, (2, 64)).to(device)
y = torch.randint(0, config.VOCAB_SIZE, (2, 64)).to(device)
with torch.no_grad():
    _, loss = model(x, y)
print(f"Forward pass OK — loss: {loss.item():.4f}")
```

---

**Step 7 — Run CPT training**

For single GPU:
```python
%cd /kaggle/working/AtomLM/cpt
!python train_cpt.py
```

For 2 GPUs (T4 x2):
```python
%cd /kaggle/working/AtomLM/cpt
!torchrun --nproc_per_node=2 train_cpt.py
```

---

**Step 8 — Save outputs before session ends**

Kaggle kills sessions after 12 hours. CPT on 300M tokens will likely need more than one session. Your checkpoints save to `checkpoints/cpt/` — copy them to `/kaggle/working/` so they persist:

```python
import shutil
shutil.copytree(
    "/kaggle/working/AtomLM/checkpoints/cpt",
    "/kaggle/working/cpt_checkpoints",
    dirs_exist_ok=True
)
print("Checkpoints backed up.")
```

To resume next session, just re-run from Step 2 (clone + install), Step 3 (download base), then copy your saved checkpoints back:

```python
shutil.copytree(
    "/kaggle/working/cpt_checkpoints",
    "/kaggle/working/AtomLM/checkpoints/cpt",
    dirs_exist_ok=True
)
```

Then run Step 7 — `train_cpt.py` will detect `checkpoints/cpt/latest.pt` and resume automatically.

---

**Watch for these during training:**

Loss should start around where your base model ended and drop within the first 200–300 steps. If it spikes above 5–6 immediately, the LR is too high — drop `LR` to `1e-5` in `config_cpt.py`. If loss goes NaN, it's usually a bad batch — the OOM handler in `train_cpt.py` will skip it but check the logs.
