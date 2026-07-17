import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
import time
import os
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from model import AtomLM
from data_cpt import AtomDataset
import config_cpt as config

# ── DDP ─────────────────────────────────────────────────────────
def setup_ddp():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def is_master():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0

def get_device():
    if dist.is_available() and dist.is_initialized():
        return torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── LR schedule ──────────────────────────────────────────────────
def get_lr(step, total_steps):
    if step < config.WARMUP_STEPS:
        return config.LR * max(step, 1) / config.WARMUP_STEPS
    progress = (step - config.WARMUP_STEPS) / max(total_steps - config.WARMUP_STEPS, 1)
    cosine   = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item())
    return config.MIN_LR + (config.LR - config.MIN_LR) * cosine

# ── Checkpointing ────────────────────────────────────────────────
def save_checkpoint(model, optimizer, step, loss):
    raw_model = model.module if hasattr(model, 'module') else model
    config.CPT_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = {
        'step': step,
        'model_state_dict': raw_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(ckpt, config.CPT_CHECKPOINT_DIR / f"step_{step}.pt")
    torch.save(ckpt, config.CPT_CHECKPOINT_DIR / "latest.pt")
    print(f"[ckpt] saved step_{step}.pt", flush=True)

def load_base_checkpoint(model):
    """
    Load base pretrained weights only — no optimizer state.
    This is the key difference from normal resume.
    """
    path = config.BASE_CHECKPOINT_PATH
    if not path.exists():
        raise FileNotFoundError(f"Base checkpoint not found at {path}")
    print(f"Loading base checkpoint from {path}", flush=True)
    ckpt = torch.load(path, map_location='cpu')
    raw_model = model.module if hasattr(model, 'module') else model
    raw_model.load_state_dict(ckpt['model_state_dict'])
    base_loss = ckpt.get('loss', float('inf'))
    print(f"Base checkpoint loaded — loss was {base_loss:.4f}", flush=True)

def load_cpt_checkpoint(model, optimizer):
    """
    Resume from a CPT checkpoint (if CPT was interrupted).
    """
    path = config.CPT_CHECKPOINT_DIR / "latest.pt"
    if not path.exists():
        return 0
    print(f"Resuming CPT from {path}", flush=True)
    ckpt = torch.load(path, map_location='cpu')
    raw_model = model.module if hasattr(model, 'module') else model
    raw_model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    print(f"Resumed from step {ckpt['step']}, loss {ckpt['loss']:.4f}", flush=True)
    return ckpt['step']

# ── Main ─────────────────────────────────────────────────────────
def main():
    print("train_cpt.py started", flush=True)

    ddp = "LOCAL_RANK" in os.environ
    if ddp:
        setup_ddp()

    device = get_device()

    if ddp and dist.get_rank() != 0:
        dist.barrier()

    dataset = AtomDataset()

    if ddp and dist.get_rank() == 0:
        dist.barrier()

    sampler = DistributedSampler(dataset) if ddp else None
    loader  = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=(sampler is None),
        sampler=sampler,
        pin_memory=True,
        num_workers=2,
    )

    model     = AtomLM().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )

    # ── Key CPT logic: what to load ──────────────────────────────
    cpt_exists = (config.CPT_CHECKPOINT_DIR / "latest.pt").exists()

    if cpt_exists:
        # CPT was interrupted — resume with optimizer state
        global_step = load_cpt_checkpoint(model, optimizer)
    else:
        # Fresh CPT start — load base weights, reset optimizer
        load_base_checkpoint(model)
        global_step = 0
        print("Optimizer reset — fresh CPT start", flush=True)

    if ddp:
        model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])])

    scaler      = torch.amp.GradScaler('cuda', enabled=(config.PRECISION == "fp16"))
    total_steps = len(loader) * config.EPOCHS // config.GRAD_ACCUM

    if is_master():
        raw = model.module if hasattr(model, 'module') else model
        print(f"\n{'='*50}", flush=True)
        print(f"AtomLM CPT", flush=True)
        print(f"{'='*50}", flush=True)
        print(f"Parameters     : {raw.num_params():,}", flush=True)
        print(f"Device         : {device}", flush=True)
        print(f"Dataset size   : {len(dataset):,} samples", flush=True)
        print(f"Total steps    : {total_steps:,}", flush=True)
        print(f"Effective batch: {config.BATCH_SIZE * config.GRAD_ACCUM}", flush=True)
        print(f"Peak LR        : {config.LR:.2e}", flush=True)
        print(f"Min LR         : {config.MIN_LR:.2e}", flush=True)
        print(f"Resuming from  : step {global_step}", flush=True)
        if torch.cuda.is_available():
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU memory     : {mem:.1f}GB", flush=True)
        print(f"{'='*50}\n", flush=True)

    config.CPT_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    steps_per_epoch = max(len(loader) // config.GRAD_ACCUM, 1)
    start_epoch     = global_step // steps_per_epoch

    for epoch in range(start_epoch, config.EPOCHS):
        if ddp:
            sampler.set_epoch(epoch)

        model.train()
        total_loss  = 0.0
        accum_count = 0
        t0          = time.time()

        for step, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)

            try:
                with torch.amp.autocast('cuda', enabled=(config.PRECISION == "fp16")):
                    logits, loss = model(x, y)
                    loss         = loss / config.GRAD_ACCUM

                scaler.scale(loss).backward()

            except torch.cuda.OutOfMemoryError:
                print(f"[OOM] step {global_step} batch {step} — skipping", flush=True)
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                continue

            total_loss  += loss.item() * config.GRAD_ACCUM
            accum_count += 1

            if (step + 1) % config.GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)

                lr = get_lr(global_step, total_steps)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                global_step += 1

                # Fixed logging — uses accum_count not (step+1)
                if is_master() and global_step % 100 == 0:
                    avg_loss = total_loss / accum_count
                    elapsed  = time.time() - t0
                    print(
                        f"epoch {epoch+1} | step {global_step}/{total_steps} | "
                        f"loss {avg_loss:.4f} | "
                        f"lr {lr:.2e} | "
                        f"{elapsed:.0f}s elapsed",
                        flush=True
                    )

                if is_master() and global_step % config.CHECKPOINT_EVERY == 0:
                    save_checkpoint(model, optimizer, global_step, total_loss / accum_count)

        if is_master():
            avg  = total_loss / max(accum_count, 1)
            secs = time.time() - t0
            print(f"\nepoch {epoch+1} done | avg loss {avg:.4f} | {secs:.0f}s\n", flush=True)
            save_checkpoint(model, optimizer, global_step, avg)

    if is_master():
        print("CPT complete.", flush=True)

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()
