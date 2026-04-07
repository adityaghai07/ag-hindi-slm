"""
AG Hindi SLM — quick inference helper.

Usage:
    from pipeline import AGHindiPipeline
    pipe = AGHindiPipeline.from_pretrained("adityaghai07/ag-hindi-slm")
    print(pipe("भारत एक"))
"""
import torch
from huggingface_hub import hf_hub_download


class AGHindiPipeline:
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @classmethod
    def from_pretrained(cls, repo_id: str = "adityaghai07/ag-hindi-slm", device: str = None):
        import sys, shutil, os, importlib

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # download model files
        for fname in ["model.pt", "model.py", "config.py"]:
            path = hf_hub_download(repo_id, fname)
            shutil.copy(path, fname)

        # import after copying
        if "model" in sys.modules: del sys.modules["model"]
        if "config" in sys.modules: del sys.modules["config"]
        from model import AGHindiSLM
        from config import Config

        torch.serialization.add_safe_globals([Config])
        ckpt = torch.load("model.pt", map_location=device, weights_only=False)
        model = AGHindiSLM(ckpt["config"], device=torch.device(device)).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        from transformers import PreTrainedTokenizerFast
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            "adityaghai07/ag_hindi_bpe_tokenizer_32k"
        )
        print(f"Loaded AG Hindi SLM — {model.num_params()/1e6:.1f}M params on {device}")
        return cls(model, tokenizer, device)

    def __call__(self, prompt: str, max_new: int = 100, temperature: float = 0.8, top_k: int = 50) -> str:
        ids = torch.tensor([self.tokenizer.encode(prompt)], device=self.device)
        with torch.no_grad():
            out = self.model.generate(ids, max_new=max_new, temperature=temperature,
                                      top_k=top_k, tokenizer=self.tokenizer)
        return self.tokenizer.decode(out[0].tolist(), skip_special_tokens=True)
