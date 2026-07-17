import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
import time
import os
from model import AtomLM
from data import AtomDataset
import config

# DDP
def setup_ddp():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def is_master():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0

def get_device():
    if dist.is_available() and dist.is_initialized():
        return torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LR schedule
def get_lr(step, total_steps):
    # linear warmup
    if step < config.WARMUP_STEPS:
        return config.LR * step / config.WARMUP_STEPS
    # cosine decay to MIN_LR
    progress = (step - config.WARMUP_STEPS) / (total_steps - config.WARMUP_STEPS)
    cosine   = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item())
    return config.MIN_LR + (config.LR - config.MIN_LR) * cosine

# checkpoint
def save_checkpoint(model, optimizer, step, loss):
    raw_model = model.module if hasattr(model, 'module') else model
    config.CHECKPOINT_DIR.mkdir(exist_ok=True)
    path = config.CHECKPOINT_DIR / f"step_{step}.pt"
    torch.save({
        'step': step,
        'model_state_dict': raw_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

    torch.save({
        'step': step,
        'model_state_dict': raw_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, config.CHECKPOINT_DIR / 'latest.pt')
    print(f"checkpoint saved. step_{step}.pt")

def load_checkpoint(model, optimizer):
    path = config.CHECKPOINT_DIR / "latest.pt"
    if not path.exists():
        return 0, float('inf')
    print(f"Resuming from {path}")
    ckpt = torch.load(path, map_location='cpu')
    raw_model = model.module if hasattr(model, 'module') else model
    raw_model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    print(f"resumed from step {ckpt['step']}, loss {ckpt['loss']:.4f}")
    return ckpt['step'], ckpt['loss']

# Main
def main():
    print("train.py started", flush=True)

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

    start_step, _ = load_checkpoint(model, optimizer)

    if ddp:
        model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])])

    scaler      = torch.amp.GradScaler('cuda', enabled=(config.PRECISION == "fp16"))
    total_steps = len(loader) * config.EPOCHS // config.GRAD_ACCUM

    if is_master():
        raw = model.module if hasattr(model, 'module') else model
        print(f"AtomLM params  : {raw.num_params():,}", flush=True)
        print(f"Device         : {device}", flush=True)
        print(f"Total steps    : {total_steps}", flush=True)
        print(f"Resuming from  : step {start_step}", flush=True)
        if torch.cuda.is_available():
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU memory     : {mem:.1f}GB", flush=True)
        print("\nTraining...\n", flush=True)

    config.CHECKPOINT_DIR.mkdir(exist_ok=True)
    steps_per_epoch = len(loader) // config.GRAD_ACCUM
    start_epoch     = start_step // steps_per_epoch
    global_step     = start_step

    for epoch in range(start_epoch, config.EPOCHS):
        if ddp:
            sampler.set_epoch(epoch)

        model.train()
        total_loss = 0
        t0         = time.time()

        for step, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)

            try:
                with torch.amp.autocast('cuda', enabled=(config.PRECISION == "fp16")):
                    logits, loss = model(x, y)
                    loss         = loss / config.GRAD_ACCUM

                scaler.scale(loss).backward()

            except torch.cuda.OutOfMemoryError:
                print(f"OOM at step {global_step} batch {step} — skipping", flush=True)
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                continue

            total_loss += loss.item() * config.GRAD_ACCUM

            # if is_master() and step % 50 == 0:
            #     print(f"  batch {step} | loss {loss.item() * config.GRAD_ACCUM:.4f}", flush=True)

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

                if is_master() and global_step % 100 == 0:
                    print(f"epoch {epoch+1} | step {global_step} | "
                          f"loss {total_loss / (step+1):.4f} | "
                          f"lr {lr:.2e}", flush=True)

                if is_master() and global_step % config.CHECKPOINT_EVERY == 0:
                    save_checkpoint(model, optimizer, global_step, loss.item())

        if is_master():
            avg  = total_loss / len(loader)
            secs = time.time() - t0
            print(f"\nepoch {epoch+1} done | avg loss {avg:.4f} | {secs:.0f}s\n", flush=True)
            save_checkpoint(model, optimizer, global_step, avg)

    if ddp:
        destroy_process_group()


# SMOKE TEST
def smoke_test():
    """
    Run 10 steps using the real training pipeline,
    but with a tiny batch and sequence length.
    """
    print("\n" + "=" * 50)
    print("SMOKE TEST — 10 steps")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AtomLM().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR)
    scaler = torch.cuda.amp.GradScaler(
        enabled=(torch.cuda.is_available() and config.PRECISION == "fp16")
    )

    # Tiny settings (instead of config values)
    BATCH_SIZE = 2
    SEQ_LEN = 32
    STEPS = 10

    losses = []

    model.train()

    for step in range(STEPS):

        x = torch.randint(
            0,
            config.VOCAB_SIZE,
            (BATCH_SIZE, SEQ_LEN),
            device=device,
        )

        y = torch.randint(
            0,
            config.VOCAB_SIZE,
            (BATCH_SIZE, SEQ_LEN),
            device=device,
        )

        with torch.cuda.amp.autocast(
            enabled=(torch.cuda.is_available() and config.PRECISION == "fp16")
        ):
            _, loss = model(x, y)
            loss = loss / config.GRAD_ACCUM

        scaler.scale(loss).backward()

        if (step + 1) % config.GRAD_ACCUM == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.GRAD_CLIP,
            )

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        losses.append(loss.item() * config.GRAD_ACCUM)

        print(f"step {step+1:2d} | loss {losses[-1]:.4f}")

    # Verify loss
    assert not any(torch.isnan(torch.tensor(losses))), "NaN loss detected"
    assert not any(torch.isinf(torch.tensor(losses))), "Inf loss detected"
    print("✓ Loss OK")

    # Save checkpoint
    config.CHECKPOINT_DIR.mkdir(exist_ok=True)
    save_checkpoint(model, optimizer, step=STEPS, loss=losses[-1])
    assert (config.CHECKPOINT_DIR / "latest.pt").exists()
    print("✓ Checkpoint Save OK")

    # Load checkpoint
    model2 = AtomLM().to(device)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=config.LR)

    loaded_step, loaded_loss = load_checkpoint(model2, optimizer2)

    assert loaded_step == STEPS
    print("✓ Checkpoint Load OK")

    # Generation
    prompt = torch.randint(
        0,
        config.VOCAB_SIZE,
        (1, 8),
        device=device,
    )

    with torch.no_grad():
        generated = model.generate(prompt, max_new_tokens=8)

    assert generated.shape == (1, 16)
    print("✓ Generation OK")

    print("\n✓ Smoke test passed — safe to train\n")

if __name__ == "__main__":
    import sys
    if "--smoke" in sys.argv:
        smoke_test()
    else:
        main()
