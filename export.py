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


def load_model(ckpt_path: str, device="cpu") -> tuple[AGHindiSLM, Config, int]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg: Config = ckpt["config"]
    model = AGHindiSLM(cfg, device=torch.device(device))
    model.load_state_dict(ckpt["model"])
    model.eval()
    step = ckpt.get("step", 0)
    print(f"Loaded step={step} | loss={ckpt.get('loss', '?')} | params={model.num_params()/1e6:.2f}M")
    return model, cfg, step


def save_local(model: AGHindiSLM, cfg: Config, out_path: str):
    torch.save({"model": model.state_dict(), "config": cfg}, out_path)
    print(f"Saved → {out_path}")


def push_to_hub(model: AGHindiSLM, cfg: Config, repo_id: str, step: int = 0):
    from huggingface_hub import HfApi, upload_file
    import tempfile, os

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    # store each checkpoint in its own subfolder e.g. step-5000/model.pt
    subfolder = f"step-{step:05d}" if step else "latest"

    with tempfile.TemporaryDirectory() as tmp:
        weights_path = os.path.join(tmp, "model.pt")
        torch.save({"model": model.state_dict(), "config": cfg, "step": step}, weights_path)

        upload_file(path_or_fileobj=weights_path, path_in_repo=f"{subfolder}/model.pt", repo_id=repo_id)

    # always keep latest/ in sync too
    if step:
        with tempfile.TemporaryDirectory() as tmp:
            weights_path = os.path.join(tmp, "model.pt")
            torch.save({"model": model.state_dict(), "config": cfg, "step": step}, weights_path)
            upload_file(path_or_fileobj=weights_path, path_in_repo="latest/model.pt", repo_id=repo_id)

    # upload code files once (idempotent)
    for fname in ["model.py", "config.py", "pipeline.py"]:
        if os.path.exists(fname):
            upload_file(path_or_fileobj=fname, path_in_repo=fname, repo_id=repo_id)

    print(f"Pushed → https://huggingface.co/{repo_id}/{subfolder}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",  required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--out",   default="ag_hindi_slm_final.pt", help="Local output path")
    parser.add_argument("--push",  action="store_true", help="Push to HuggingFace Hub")
    parser.add_argument("--repo",  default="", help="HuggingFace repo id, e.g. username/ag-hindi-slm")
    args = parser.parse_args()

    model, cfg, step = load_model(args.ckpt)

    if args.push:
        assert args.repo, "Provide --repo username/model-name"
        push_to_hub(model, cfg, args.repo, step=step)
    else:
        save_local(model, cfg, args.out)


if __name__ == "__main__":
    main()
