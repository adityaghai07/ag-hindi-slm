"""
Export trained model — two modes:

  # Save weights locally (for inference / fine-tuning later)
  python export.py --ckpt checkpoints/step_20000.pt --out ag_hindi_slm_final.pt

  # Push to HuggingFace Hub
  python export.py --ckpt checkpoints/step_20000.pt --push --repo adityaghai07/ag-hindi-slm
"""
import argparse
import torch
from config import Config
from model import AGHindiSLM


def load_model(ckpt_path: str, device="cpu") -> tuple[AGHindiSLM, Config]:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg: Config = ckpt["config"]
    model = AGHindiSLM(cfg, device=torch.device(device))
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded step={ckpt['step']} | loss={ckpt['loss']:.4f} | params={model.num_params()/1e6:.2f}M")
    return model, cfg


def save_local(model: AGHindiSLM, cfg: Config, out_path: str):
    torch.save({"model": model.state_dict(), "config": cfg}, out_path)
    print(f"Saved → {out_path}")


def push_to_hub(model: AGHindiSLM, cfg: Config, repo_id: str):
    from huggingface_hub import HfApi, upload_file
    import tempfile, os

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        # Save weights
        weights_path = os.path.join(tmp, "model.pt")
        torch.save({"model": model.state_dict(), "config": cfg}, weights_path)

        # Simple model card
        card_path = os.path.join(tmp, "README.md")
        with open(card_path, "w") as f:
            f.write(f"""---
language: hi
tags:
  - hindi
  - language-model
  - pretraining
---

# AG Hindi SLM

~{model.num_params()/1e6:.0f}M parameter Hindi language model.

Architecture: MLA Attention + Block AttnRes + SwiGLU  
Training data: Hindi Wikipedia + ai4bharat/sangraha (verified + unverified)  
Params: d_model={cfg.d_model}, n_layers={cfg.n_layers}, n_heads={cfg.n_heads}

## Load model

```python
import torch
from model import AGHindiSLM

ckpt = torch.load("model.pt", map_location="cpu")
model = AGHindiSLM(ckpt["config"])
model.load_state_dict(ckpt["model"])
model.eval()
```
""")

        upload_file(path_or_fileobj=weights_path, path_in_repo="model.pt", repo_id=repo_id)
        upload_file(path_or_fileobj=card_path,    path_in_repo="README.md", repo_id=repo_id)
        # Also upload model.py and config.py so the repo is self-contained
        for fname in ["model.py", "config.py"]:
            if os.path.exists(fname):
                upload_file(path_or_fileobj=fname, path_in_repo=fname, repo_id=repo_id)

    print(f"Pushed → https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",  required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--out",   default="ag_hindi_slm_final.pt", help="Local output path")
    parser.add_argument("--push",  action="store_true", help="Push to HuggingFace Hub")
    parser.add_argument("--repo",  default="", help="HuggingFace repo id, e.g. username/ag-hindi-slm")
    args = parser.parse_args()

    model, cfg = load_model(args.ckpt)

    if args.push:
        assert args.repo, "Provide --repo username/model-name"
        push_to_hub(model, cfg, args.repo)
    else:
        save_local(model, cfg, args.out)


if __name__ == "__main__":
    main()
