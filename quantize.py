"""Build an INT4 GPTQ checkpoint with llm-compressor, the quantization tool the vLLM project maintains. Calibration uses
256 filler prompts from Dolly, never the probe prompts. The output loads in vLLM as a compressed-tensors model.
  python quantize.py --family mistral-7b --out ckpt/mistral-7b-w4a16
Run it from its own environment, since llm-compressor pins its own library versions; jobs/quantize.sh does that.
"""
import argparse, os, pathlib, shutil, sys
ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT))
import collect, families

def main(a):
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier
    base = families.configs(a.family)["bf16"]["model"]
    tok = AutoTokenizer.from_pretrained(base)
    _, fillers = collect.prompts(120, a.samples)                     # the first 120 rows are probes, so skip them
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in fillers]
    ds = Dataset.from_dict(dict(tok(texts, truncation=True, max_length=a.max_len, add_special_tokens=False)))
    model = None
    for loader in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):            # Muse Glimmer has a vision encoder
        try:
            import transformers
            model = getattr(transformers, loader).from_pretrained(base, torch_dtype=torch.bfloat16, device_map="auto")
            print(f"loaded {base} with {loader}", flush=True); break
        except Exception as e:
            print(f"{loader} could not load {base}: {str(e).splitlines()[0][:150]}", flush=True)
    if model is None:
        raise SystemExit("STOP: no transformers class could load the model")
    recipe = GPTQModifier(targets="Linear", scheme="W4A16",
                          ignore=["lm_head", "re:.*vision.*", "re:.*visual.*", "re:.*multi_modal.*", "re:.*mm_projector.*"])
    oneshot(model=model, dataset=ds, recipe=recipe, max_seq_length=a.max_len, num_calibration_samples=len(texts))
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, save_compressed=True); tok.save_pretrained(out)
    from huggingface_hub import snapshot_download                     # processor and template files vLLM may need
    src = snapshot_download(base, allow_patterns=["*.json", "*.jinja", "*.model", "*.txt"], token=os.environ.get("HF_TOKEN"))
    for f in pathlib.Path(src).iterdir():
        if f.is_file() and not (out / f.name).exists():
            shutil.copy(f, out / f.name)
    print(f"wrote {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=256); ap.add_argument("--max-len", type=int, default=1024)
    main(ap.parse_args())
