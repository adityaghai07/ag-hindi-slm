import torch
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset

from config import Config


def load_tokenizer(name: str = "adityaghai07/ag_hindi_bpe_tokenizer_32k") -> PreTrainedTokenizerFast:
    tokenizer = PreTrainedTokenizerFast.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    return tokenizer


def build_token_buffer(dataset, tokenizer, target_tokens: int, seq_len: int) -> torch.Tensor:
    eos = tokenizer.eos_token_id or 1
    buffer = []
    print(f"  Building token buffer (~{target_tokens // 1_000_000}M tokens)...")

    for article in dataset:
        text = article.get("text", "").strip()
        if len(text) < 100:
            continue
        buffer.extend(tokenizer.encode(text, add_special_tokens=False))
        buffer.append(eos)
        if len(buffer) >= target_tokens:
            break

    tokens = torch.tensor(buffer[:target_tokens], dtype=torch.long)
    n_chunks = len(tokens) // seq_len
    tokens = tokens[: n_chunks * seq_len]
    print(f"  Packed {len(tokens):,} tokens → {n_chunks:,} chunks of {seq_len}")
    return tokens


def _stream(name, config=None, data_dir=None):
    kwargs = dict(split="train", streaming=True)
    if config:
        return load_dataset(name, config, **kwargs)
    if data_dir:
        return load_dataset(name, data_dir=data_dir, **kwargs)
    return load_dataset(name, **kwargs)


def load_data(cfg: Config, tokenizer, target_tokens: int = 500_000_000, cache_path: str = "token_cache.pt"):
    import os
    if cache_path and os.path.isfile(cache_path):
        print(f"Loading token cache from {cache_path}...")
        token_chunks = torch.load(cache_path)
        print(f"  token_chunks shape: {token_chunks.shape}")
        return token_chunks
    from itertools import chain

    sources = [
        _stream("wikimedia/wikipedia", config="20231101.hi"),
        _stream("ai4bharat/sangraha", data_dir="verified/hin"),
        _stream("ai4bharat/sangraha", data_dir="unverified/hin"),
    ]
    combined = chain(*sources)
    token_buffer = build_token_buffer(combined, tokenizer, target_tokens, cfg.max_seq_len)
    token_chunks = token_buffer.view(-1, cfg.max_seq_len)
    print(f"  token_chunks shape: {token_chunks.shape}")
    return token_chunks


class BatchSampler:
    def __init__(self, token_chunks: torch.Tensor, cfg: Config, device, rank: int = 0, world_size: int = 1):
        self.chunks = token_chunks
        self.cfg = cfg
        self.device = device
        self.rank = rank
        self.world_size = world_size

    def get_batch(self, step: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = torch.Generator()
        rng.manual_seed(step * 1000 + 42 + self.rank)
        n = self.chunks.shape[0]
        idx = torch.randint(0, n - 1, (self.cfg.batch_size,), generator=rng)
        seqs = self.chunks[idx].to(self.device)
        return seqs[:, :-1], seqs[:, 1:]
