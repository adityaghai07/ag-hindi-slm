import math
from dataclasses import dataclass


@dataclass
class Config:
    vocab_size: int = 32000

    d_model: int = 1024
    n_layers: int = 16
    n_heads: int = 16

    kv_lora_rank: int = 512
    d_ff: int = 2730
    max_seq_len: int = 1024
    rope_theta: float = 500_000.0
    attn_res_block_size: int = 6
    norm_eps: float = 1e-5

    batch_size: int = 16
    grad_accum: int = 8
    max_steps: int = 20000
    lr_max: float = 3e-4
    lr_min: float = 3e-5
    warmup_steps: int = 200
    grad_clip: float = 1.0
    weight_decay: float = 0.1

    log_every: int = 10
    sample_every: int = 200
    save_every: int = 1000
    sample_prompt: str = "भारत एक महान देश"
    ckpt_dir: str = "checkpoints"
    resume_from: str = ""   # path to checkpoint to resume from, empty = fresh start

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


def get_lr(step: int, cfg: Config) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr_max * step / max(1, cfg.warmup_steps)
    t = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.lr_min + 0.5 * (cfg.lr_max - cfg.lr_min) * (1 + math.cos(math.pi * t))
