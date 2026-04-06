"""

DDP training — launch with:

  torchrun --nproc_per_node=4 train_ddp.py
"""
import os
import time

import torch

import torch.distributed as dist

from contextlib import contextmanager

from torch.nn.parallel import DistributedDataParallel as DDP


from config import Config, get_lr
from model import AGHindiSLM, build_optimizer

from data import load_tokenizer, load_data, BatchSampler



def setup_ddp():

    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()

    world_size = dist.get_world_size()

    torch.cuda.set_device(rank)

    return rank, world_size



def cleanup_ddp():

    dist.destroy_process_group()



def train():

    rank, world_size = setup_ddp()

    device = torch.device(f"cuda:{rank}")

    is_master = rank == 0


    cfg = Config()

    # Scale effective batch: each rank handles cfg.batch_size, so effective = batch * grad_accum * world_size

    # Optionally reduce per-rank batch if needed for VRAM

    # cfg.batch_size = cfg.batch_size // world_size  # uncomment to keep same effective batch


    tokenizer = load_tokenizer()

    cfg.vocab_size = len(tokenizer)


    if is_master:

        print(f"World size: {world_size}")

        print(f"Vocab size: {cfg.vocab_size}")


    # All ranks load data (streaming, so no bottleneck)

    token_chunks = load_data(cfg, tokenizer)

    sampler = BatchSampler(token_chunks, cfg, device, rank=rank, world_size=world_size)


    if is_master:

        print("Building model...")

    model = AGHindiSLM(cfg, device=device).to(device)

    model = DDP(model, device_ids=[rank])

    raw_model = model.module  # unwrapped for checkpointing / generation


    if is_master:

        print(f"Params: {raw_model.num_params()/1e6:.2f}M")


    optimizer = build_optimizer(raw_model, cfg)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    start_step = 1
    if cfg.resume_from and os.path.isfile(cfg.resume_from):
        ckpt = torch.load(cfg.resume_from, map_location=device)
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"] + 1
        if is_master:
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

            # Only sync gradients on the last micro-step

            sync = micro == cfg.grad_accum - 1

            ctx = model.no_sync() if not sync else contextlib_nullcontext()

            with ctx:

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


        if is_master:

            tok_per_sec = cfg.batch_size * world_size * cfg.grad_accum * (cfg.max_seq_len - 1) / (time.time() - t_step)


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

                        out = raw_model.generate(prompt_ids, max_new=80, temperature=temp, top_k=50, tokenizer=tokenizer)

                    print(f"  T={temp}: {tokenizer.decode(out[0].tolist(), skip_special_tokens=True)}")
                model.train()


            if step % cfg.save_every == 0:

                path = os.path.join(cfg.ckpt_dir, f"step_{step:05d}.pt")

                torch.save({

                    "step": step,

                    "model": raw_model.state_dict(),

                    "optimizer": optimizer.state_dict(),

                    "config": cfg,

                    "loss": step_loss,

                }, path)

                print(f"  Checkpoint → {path}")


    if is_master:

        total = time.time() - t_start

        print(f"\nDone in {total/60:.1f}m | final loss {losses[-1]:.4f} | best {min(losses):.4f}")


    cleanup_ddp()



@contextmanager

def contextlib_nullcontext():

    yield



if __name__ == "__main__":
    train()

