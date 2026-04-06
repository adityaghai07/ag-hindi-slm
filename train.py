import os
import math
import time
import torch


from config import Config, get_lr
from model import AGHindiSLM, build_optimizer

from data import load_tokenizer, load_data, BatchSampler



def sanity_check(model, cfg, tokenizer, device):

    B, T = 1, 64

    x = torch.randint(0, cfg.vocab_size, (B, T), device=device)

    y = torch.randint(0, cfg.vocab_size, (B, T), device=device)


    with torch.autocast("cuda", dtype=torch.bfloat16):

        _, loss0 = model(x, y)

    expected = math.log(cfg.vocab_size)

    print(f"[1] Initial loss : {loss0.item():.3f}  (expected ≈ {expected:.3f})")

    assert abs(loss0.item() - expected) < 2.0


    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):

        logits, _ = model(x)

    assert logits.shape == (B, T, cfg.vocab_size)

    print(f"[2] Logits shape : {logits.shape}  ✓")


    opt_test = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for _ in range(5):

        opt_test.zero_grad(set_to_none=True)

        with torch.autocast("cuda", dtype=torch.bfloat16):

            _, loss_t = model(x, y)

        loss_t.backward()
        opt_test.step()

    print(f"[3] Loss after 5 steps: {loss_t.item():.3f}")
    del opt_test


    prompt_ids = torch.tensor([tokenizer.encode(cfg.sample_prompt)], device=device)

    with torch.no_grad():

        out = model.generate(prompt_ids, max_new=20, tokenizer=tokenizer)

    print(f"[4] Generation   : {tokenizer.decode(out[0].tolist(), skip_special_tokens=True)}")

    torch.cuda.empty_cache()

    print("Sanity checks passed ✓\n")



def train():

    device = torch.device("cuda")

    assert torch.cuda.is_available()

    print(f"GPU: {torch.cuda.get_device_name(0)}  |  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")


    cfg = Config()


    tokenizer = load_tokenizer()

    cfg.vocab_size = len(tokenizer)

    print(f"Vocab size: {cfg.vocab_size}")


    token_chunks = load_data(cfg, tokenizer)

    sampler = BatchSampler(token_chunks, cfg, device)


    print("Building model...")
    model = AGHindiSLM(cfg, device=device).to(device)

    print(f"Params: {model.num_params()/1e6:.2f}M")


    sanity_check(model, cfg, tokenizer, device)


    optimizer = build_optimizer(model, cfg)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    start_step = 1
    if cfg.resume_from and os.path.isfile(cfg.resume_from):
        ckpt = torch.load(cfg.resume_from, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"] + 1
        print(f"Resumed from {cfg.resume_from} at step {ckpt['step']}")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    t_start = time.time()

    for step in range(start_step, cfg.max_steps + 1):
        t_step = time.time()

        lr = get_lr(step, cfg)

        for pg in optimizer.param_groups:

            pg["lr"] = lr


        step_loss = 0.0

        for micro in range(cfg.grad_accum):

            x, y = sampler.get_batch(step * cfg.grad_accum + micro)

            with torch.autocast("cuda", dtype=torch.bfloat16):

                _, loss = model(x, y)

                loss = loss / cfg.grad_accum

            loss.backward()

            step_loss += loss.item()


        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        optimizer.step()

        optimizer.zero_grad(set_to_none=True)

        losses.append(step_loss)

        tok_per_sec = cfg.batch_size * cfg.grad_accum * (cfg.max_seq_len - 1) / (time.time() - t_step)


        if step % cfg.log_every == 0:

            avg_loss = sum(losses[-cfg.log_every:]) / cfg.log_every

            vram = torch.cuda.max_memory_allocated() / 1e9
            print(

                f"step {step:5d}/{cfg.max_steps} | loss {avg_loss:.4f} | "

                f"gnorm {grad_norm:.3f} | lr {lr:.2e} | "

                f"{tok_per_sec/1000:.1f}K tok/s | VRAM {vram:.1f}GB | "

                f"elapsed {(time.time()-t_start)/60:.1f}m"
            )


        if step % cfg.sample_every == 0:

            for temp in [0.7, 1.0]:

                prompt_ids = torch.tensor([tokenizer.encode(cfg.sample_prompt)], device=device)

                with torch.no_grad():

                    out = model.generate(prompt_ids, max_new=80, temperature=temp, top_k=50, tokenizer=tokenizer)

                print(f"  T={temp}: {tokenizer.decode(out[0].tolist(), skip_special_tokens=True)}")
            model.train()


        if step % cfg.save_every == 0:

            path = os.path.join(cfg.ckpt_dir, f"step_{step:05d}.pt")

            torch.save({"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": cfg, "loss": step_loss}, path)

            print(f"  Checkpoint → {path}")


    total = time.time() - t_start

    print(f"\nDone in {total/60:.1f}m | final loss {losses[-1]:.4f} | best {min(losses):.4f}")


    for prompt in ["भारत एक महान देश", "हिन्दी भाषा", "दिल्ली में"]:

        ids = torch.tensor([tokenizer.encode(prompt)], device=device)

        with torch.no_grad():

            out = model.generate(ids, max_new=100, temperature=0.8, tokenizer=tokenizer)

        print(f"  '{prompt}' → {tokenizer.decode(out[0].tolist(), skip_special_tokens=True)}")



if __name__ == "__main__":
    train()

