"""
Quick benchmark to find max batch size and measure throughput.
Run: python bench.py
"""
import time
import torch
from config import Config
from model import AGHindiSLM, build_optimizer


def measure(cfg: Config, device, steps: int = 20):
    model = AGHindiSLM(cfg, device=device).to(device)
    optimizer = build_optimizer(model, cfg)
    model.train()

    # warmup
    for _ in range(3):
        x = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.max_seq_len - 1), device=device)
        y = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.max_seq_len - 1), device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    for _ in range(steps):
        x = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.max_seq_len - 1), device=device)
        y = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.max_seq_len - 1), device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    elapsed = time.time() - t0
    vram = torch.cuda.max_memory_allocated() / 1e9
    tokens = steps * cfg.batch_size * (cfg.max_seq_len - 1)
    tok_per_sec = tokens / elapsed

    print(f"  batch={cfg.batch_size:3d} | seq={cfg.max_seq_len} | "
          f"VRAM {vram:.2f} GB | {tok_per_sec/1000:.1f}K tok/s")

    del model, optimizer
    torch.cuda.empty_cache()
    return vram, tok_per_sec


def main():
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU  : {torch.cuda.get_device_name(0)}")
    print(f"VRAM : {vram_total:.1f} GB\n")

    # --- sweep batch sizes at seq_len=1024 ---
    print("=== Sweep batch size (seq_len=1024) ===")
    for bs in [2, 4, 8, 16]:
        cfg = Config()
        cfg.batch_size = bs
        cfg.max_seq_len = 1024
        try:
            vram, _ = measure(cfg, device)
            if vram > vram_total * 0.95:
                print(f"  batch={bs} too close to VRAM limit, stopping sweep")
                break
        except torch.cuda.OutOfMemoryError:
            print(f"  batch={bs:3d} | seq=1024 | OOM")
            torch.cuda.empty_cache()
            break

    # --- sweep seq lengths at batch=4 ---
    print("\n=== Sweep seq_len (batch=4) ===")
    for seq in [512, 1024]:
        cfg = Config()
        cfg.batch_size = 4
        cfg.max_seq_len = seq
        try:
            measure(cfg, device)
        except torch.cuda.OutOfMemoryError:
            print(f"  batch=4   | seq={seq} | OOM")
            torch.cuda.empty_cache()
            break

    # --- recommended config ---
    print("\n=== Recommended config (batch=8, seq=1024) ===")
    cfg = Config()
    cfg.batch_size = 8
    cfg.max_seq_len = 1024
    try:
        vram, tok_s = measure(cfg, device, steps=50)
        total_tokens = cfg.batch_size * cfg.grad_accum * (cfg.max_seq_len - 1) * 5000
        eta_h = total_tokens / tok_s / 3600
        print(f"  → ETA for 5000 steps: {eta_h:.1f} hours")
    except torch.cuda.OutOfMemoryError:
        print("  OOM — stick with batch=4")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
