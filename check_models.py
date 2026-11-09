"""Run this on the GPU machine before any collection. It confirms that every repository in a family exists and that
your Hugging Face token can download it, so a missing or gated model never stops a run halfway.
  python check_models.py --family llama3.1-8b
Exit code 1 means the BF16 reference is unreachable; other missing setups are only skipped.
"""
import argparse, os, sys
import families

def main(a):
    from huggingface_hub import HfApi
    api, token = HfApi(), os.environ.get("HF_TOKEN")
    ok = []
    for name, cfg in families.configs(a.family).items():
        try:
            info = api.model_info(cfg["model"], token=token, files_metadata=False)
            gated = getattr(info, "gated", False)
            api.hf_hub_download(cfg["model"], "config.json", token=token)      # proves the token can actually download
            print(f"OK       {name:11s} {cfg['model']}" + ("  (gated, access works)" if gated else ""))
            ok.append(name)
        except Exception as e:
            msg = str(e).splitlines()[0][:110]
            hint = "accept the license on huggingface.co and set HF_TOKEN" if "gated" in str(e).lower() or "401" in str(e) or "403" in str(e) else "repository missing or misnamed"
            print(f"SKIP     {name:11s} {cfg['model']}\n         {msg}\n         fix: {hint}")
    print("\nREADY: " + ",".join(ok))
    if "bf16" not in ok:
        sys.exit(1)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--family", default="qwen2.5-7b")
    main(ap.parse_args())
