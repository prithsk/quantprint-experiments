"""Copy finished results off the pod the moment they exist, so a stopped pod never takes them with it. Uploads every
/workspace/results-*.tar.gz that is not uploaded yet to a private Hugging Face dataset, prithsk/quantprint-results by
default. Needs HF_TOKEN to be a token with WRITE access. Without one, it warns and the run continues.
  python upload_results.py
Download later at https://huggingface.co/datasets/prithsk/quantprint-results/tree/main or with download_results.py.
"""
import glob, os, pathlib
REPO = os.environ.get("RESULTS_REPO", "prithsk/quantprint-results")

def main():
    token = os.environ.get("HF_TOKEN")
    files = sorted(glob.glob("/workspace/results-*.tar.gz")) + sorted(glob.glob("/workspace/logs*.tar.gz"))
    if not files:
        print("upload: no results files yet"); return
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
    except Exception as e:
        print(f"UPLOAD WARNING: could not reach {REPO} ({str(e).splitlines()[0][:120]}). Use a WRITE token as HF_TOKEN. "
              "Results stay only on this pod, so download them before it stops."); return
    marker = pathlib.Path("/workspace/.uploaded"); sent = set(marker.read_text().split()) if marker.exists() else set()
    for f in files:
        name = pathlib.Path(f).name; stamp = f"{name}:{os.path.getsize(f)}"
        if stamp in sent:
            continue
        try:
            api.upload_file(path_or_fileobj=f, path_in_repo=name, repo_id=REPO, repo_type="dataset", commit_message=f"add {name}")
            sent.add(stamp); marker.write_text("\n".join(sorted(sent)))
            print(f"uploaded {name} to huggingface.co/datasets/{REPO}", flush=True)
        except Exception as e:
            print(f"UPLOAD WARNING: {name} failed ({str(e).splitlines()[0][:120]}); it stays on the pod only", flush=True)

if __name__ == "__main__":
    main()
