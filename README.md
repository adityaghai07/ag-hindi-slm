# AG Hindi SLM

A 225M parameter Hindi language model trained from scratch on ~4B tokens.

**Architecture:** Multi-head Latent Attention (MLA) + Block Attention Residuals + SwiGLU FFN  
**Language:** Hindi  
**Training data:** Hindi Wikipedia + ai4bharat/sangraha (verified + unverified)  
**Training:** 10,000 steps × 3 GPUs (RTX 4090) — Chinchilla optimal for 225M params  
**Tokenizer:** Custom Hindi BPE — [ag-hindi-llm-tokenizer](https://github.com/adityaghai07/ag-hindi-llm-tokenizer) | 32K vocab | fertility ~1.5–1.8 tokens/word

---

## Loss Curve

![Loss Curve](logs/loss_curve.png)

| Step  | Loss  | Tokens seen |
|-------|-------|-------------|
| 10    | 10.15 | ~4M         |
| 1000  | 3.54  | ~393M       |
| 4000  | 3.02  | ~1.57B      |
| 7000  | 2.85  | ~2.75B      |
| 10000 | 2.73  | ~3.93B      |

Chinchilla optimal for 225M params = 4.5B tokens (~11.5K steps). Training was stopped at 10K steps (~3.93B tokens), just under optimal.

---

## Architecture

```
d_model   : 1024
n_layers  : 16
n_heads   : 16
kv_lora   : 512
d_ff      : 2730
seq_len   : 1024
params    : 225.77M
```

**MLA (Multi-head Latent Attention)**  
KV is down-projected to a rank-512 latent before expansion — smaller KV cache at inference vs standard MHA. Inspired by DeepSeek-V2.

**Block Attention Residuals**  
Replaces standard residual connections. Each sub-layer attends over snapshots from all previous blocks via a learned pseudo-query vector, letting the model selectively route information from any earlier representation. Implementation reference: [open-attention-residuals](https://github.com/wdlctc/open-attention-residuals)

**SwiGLU FFN**  
Gated activation: `out = down(silu(gate(x)) * up(x))`, d_ff = 2730 = (2/3)×4×1024.

**RoPE** with theta=500k for extended context. RMSNorm pre-norm. Weight-tied embeddings. No biases anywhere.

---

## Quickstart (Colab / local)

```python
!pip install transformers sentencepiece huggingface_hub torch -q

from huggingface_hub import hf_hub_download
import shutil, os

# download required files
for f in ["pipeline.py", "model.py", "config.py"]:
    shutil.copy(hf_hub_download("adityaghai07/ag-hindi-slm", f), f)

from pipeline import AGHindiPipeline

pipe = AGHindiPipeline.from_pretrained("adityaghai07/ag-hindi-slm")

# generate
print(pipe("भारत एक"))
print(pipe("हिन्दी भाषा", max_new=150, temperature=0.7))
print(pipe("दिल्ली में", max_new=100, temperature=0.9, top_k=40))
```

---

## Training

Single GPU:
```bash
python train.py
```

Multi-GPU (3×):
```bash
torchrun --nproc_per_node=3 train_ddp.py
```

Background (survives SSH disconnect):
```bash
nohup torchrun --nproc_per_node=3 train_ddp.py > train.log 2>&1 &
tail -f train.log
```

## Export

```bash
# push checkpoint to HuggingFace
python export.py --ckpt checkpoints/step_10000.pt --push --repo adityaghai07/ag-hindi-slm
```

## Requirements

```
torch>=2.2.0
transformers>=4.40.0
datasets>=2.18.0
sentencepiece
huggingface_hub
matplotlib
```
