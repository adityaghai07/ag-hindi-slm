"""
Run once to build and cache the token buffer to disk.
All DDP ranks then load from cache instead of re-streaming.

  python prepare_data.py
"""
import torch
from config import Config
from data import load_tokenizer, load_data

cfg = Config()
tokenizer = load_tokenizer()
cfg.vocab_size = len(tokenizer)

token_chunks = load_data(cfg, tokenizer, target_tokens=500_000_000)

torch.save(token_chunks, "token_cache.pt")
print(f"Saved token_cache.pt — shape: {token_chunks.shape}")
print(f"Size on disk: {token_chunks.numel() * 8 / 1e9:.2f} GB")
