# AG Hindi SLM

A ~178M parameter Hindi language model trained from scratch.

**Architecture:** Multi-head Latent Attention (MLA) + Block Attention Residuals + SwiGLU FFN  
**Target language:** Hindi  
**Training data:** Hindi Wikipedia + ai4bharat/sangraha (verified + unverified)

---

## Architecture

- MLA attention — KV compressed to latent dim (512) before expansion, smaller KV cache at inference
- Block AttnRes — replaces standard residuals, each layer can route from any earlier block snapshot
- SwiGLU FFN — gated activation, d_ff = 2730
- RoPE positional encoding, theta = 500k for extended context
- RMSNorm pre-norm, no biases anywhere, weight-tied embeddings

```
d_model   : 1024
n_layers  : 16
n_heads   : 16
kv_lora   : 512
d_ff      : 2730
seq_len   : 1024
params    : ~178M
```

## Training

Single GPU:
```bash
python train.py
```

Multi-GPU (4x):
```bash
torchrun --nproc_per_node=4 train_ddp.py
```

Background (keep alive after SSH disconnect):
```bash
nohup torchrun --nproc_per_node=4 train_ddp.py > train.log 2>&1 &
tail -f train.log
```

## Benchmarking

```bash
python bench.py
```

Sweeps batch sizes and seq lengths, reports VRAM usage and tok/s.

## Export

```bash
# local
python export.py --ckpt checkpoints/step_20000.pt --out ag_hindi_slm_final.pt

# push to HuggingFace
python export.py --ckpt checkpoints/step_20000.pt --push --repo username/ag-hindi-slm
```

## Inference

```python
import torch
from model import AGHindiSLM
from data import load_tokenizer

ckpt = torch.load("ag_hindi_slm_final.pt", map_location="cuda")
model = AGHindiSLM(ckpt["config"]).cuda()
model.load_state_dict(ckpt["model"])
model.eval()

tokenizer = load_tokenizer()
prompt = torch.tensor([tokenizer.encode("भारत एक")], device="cuda")
out = model.generate(prompt, max_new=100, tokenizer=tokenizer)
print(tokenizer.decode(out[0].tolist(), skip_special_tokens=True))
```

## Requirements

```
torch>=2.2.0
transformers>=4.40.0
datasets>=2.18.0
sentencepiece
huggingface_hub
```
