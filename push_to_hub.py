"""
Push README and loss curve to HuggingFace.
Run: python push_to_hub.py
"""
import os
from huggingface_hub import HfApi

HF_TOKEN = os.environ.get("HF_TOKEN", "")
REPO_ID = "adityaghai07/ag-hindi-slm"

api = HfApi(token=HF_TOKEN)

api.upload_file(path_or_fileobj="README.md", path_in_repo="README.md", repo_id=REPO_ID)
print("Uploaded README.md")

api.upload_file(path_or_fileobj="loss_curve.png", path_in_repo="logs/loss_curve.png", repo_id=REPO_ID)
print("Uploaded loss_curve.png")

api.upload_file(path_or_fileobj="pipeline.py", path_in_repo="pipeline.py", repo_id=REPO_ID)
print("Uploaded pipeline.py")

print(f"Done → https://huggingface.co/{REPO_ID}")
