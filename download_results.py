"""On your laptop: download every results file from the private Hugging Face dataset into the repo and unpack it.
  python download_results.py
Needs HF_TOKEN set to a token for the same account."""
import os, pathlib, tarfile
from huggingface_hub import HfApi, hf_hub_download
REPO = os.environ.get("RESULTS_REPO", "prithsk/quantprint-results")
for name in HfApi().list_repo_files(REPO, repo_type="dataset"):
    if name.endswith(".tar.gz"):
        with tarfile.open(hf_hub_download(REPO, name, repo_type="dataset")) as t:
            t.extractall(pathlib.Path(__file__).parent)
        print(f"unpacked {name}")
