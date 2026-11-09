"""Collect greedy top-k log-probabilities for one serving configuration with vLLM, at several batch sizes.

Each configuration runs in its own process so GPU memory is fully released between models:
  python collect.py --config bf16 --batches 1,8,64 --repeats 3
Batch composition changes between repeats, since the filler prompts that share each batch are reshuffled. That is the
batch nondeterminism a production provider produces, and the noise floor in the paper comes from it.
Output: data/<config>.jsonl, one line per probe per run.
"""
import argparse, json, os, pathlib, random, time

MODEL = "Qwen/Qwen2.5-7B-Instruct"
import families
CONFIGS = families.FAMILIES["qwen2.5-7b"]     # kept for scripts that import it
ROOT = pathlib.Path(__file__).parent

def prompts(n_probe, n_filler, seed=7):
    """Probe and filler prompts from Dolly, CC BY-SA 3.0: open questions with no context, disjoint sets."""
    from datasets import load_dataset
    rows = [r["instruction"].strip() for r in load_dataset("databricks/databricks-dolly-15k", split="train")
            if r["category"] in ("open_qa", "general_qa", "brainstorming") and not r["context"].strip()
            and 20 <= len(r["instruction"]) <= 300]
    random.Random(seed).shuffle(rows)
    return rows[:n_probe], rows[n_probe:n_probe + n_filler]

def load_llm(cfg, gpu_mem=0.85, prefix_cache=False, max_len=2048):
    """Load one setup in vLLM. bitsandbytes setups retry with the load format that older vLLM versions need."""
    from vllm import LLM
    kw = dict(model=cfg["model"], quantization=cfg["quantization"], dtype=cfg["dtype"], max_model_len=max_len,
              gpu_memory_utilization=gpu_mem, seed=0, enable_prefix_caching=prefix_cache)
    if cfg.get("kv_cache_dtype"):
        kw["kv_cache_dtype"] = cfg["kv_cache_dtype"]
    try:
        return LLM(**kw)
    except Exception:
        if cfg["quantization"] != "bitsandbytes":
            raise
        print("retrying bitsandbytes with load_format=bitsandbytes, which older vLLM versions need", flush=True)
        return LLM(**kw, load_format="bitsandbytes")

def main(a):
    from vllm import LLM, SamplingParams
    table = families.configs(a.family)
    if a.config not in table:
        raise SystemExit(f"{a.config} is not a setup of {a.family}: {', '.join(table)}")
    cfg = table[a.config]
    a.out = a.out or families.data_dir(a.family)
    probes, fillers = prompts(a.probes, 2000)
    # Prefix caching is off by default: with it on, every run after the first reuses cached prompt computations,
    # which gives all later runs a shared shift against the fresh first run and inflates the noise correlation.
    llm = load_llm(cfg, a.gpu_mem, a.prefix_cache)
    # Record exactly what produced this data, so a paper never has to guess the version later.
    import torch, vllm, datetime, subprocess as _sp
    gpu = _sp.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    (ROOT / a.out).mkdir(exist_ok=True)
    (ROOT / a.out / f"{a.config}.meta.json").write_text(json.dumps({
        "config": a.config, "family": a.family, "model": cfg["model"], "vllm": vllm.__version__, "torch": torch.__version__, "gpu": gpu,
        "prefix_cache": a.prefix_cache, "probes": a.probes, "tokens": a.tokens, "k": a.k,
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}, indent=2))
    tok = llm.get_tokenizer()
    chat = lambda p: tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
    probes_c, fillers_c = [chat(p) for p in probes], [chat(p) for p in fillers]
    sp = SamplingParams(temperature=0.0, max_tokens=a.tokens, logprobs=a.k)
    out = (ROOT / a.out); out.mkdir(exist_ok=True)
    path = out / f"{a.config}.jsonl"
    with open(path, "a") as fh:
        for b in [int(x) for x in a.batches.split(",")]:
            for rep in range(a.repeats):
                rng = random.Random(1000 * b + rep)
                order = list(range(len(probes_c))); rng.shuffle(order)
                per_call = min(b, 8)                              # up to 8 probes per batch, the rest fillers
                t0 = time.time()
                for i in range(0, len(order), per_call):
                    ids = order[i:i + per_call]
                    batch = [probes_c[j] for j in ids] + rng.sample(fillers_c, b - len(ids))
                    rng.shuffle(batch_positions := list(range(len(batch))))
                    batch = [batch[p] for p in batch_positions]
                    res = llm.generate(batch, sp, use_tqdm=False)
                    for pos, r in zip(batch_positions, res):
                        if pos >= len(ids):
                            continue                              # a filler
                        o = r.outputs[0]; toks, lp, topk, text = list(o.token_ids), [], [], []
                        for t, d in zip(toks, o.logprobs):
                            lp.append(d[t].logprob); text.append(d[t].decoded_token)
                            topk.append(sorted(([tid, x.logprob] for tid, x in d.items() if x.rank is not None and x.rank <= a.k),
                                               key=lambda z: -z[1]))
                        fh.write(json.dumps({"config": a.config, "batch": b, "rep": rep, "pid": ids[pos],
                                             "tokens": toks, "text": text, "lp": lp, "topk": topk}) + "\n")
                print(f"{a.config} batch {b} repeat {rep}: {len(order)} probes in {time.time() - t0:.0f} s", flush=True)
    print(f"wrote {path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="qwen2.5-7b", choices=list(families.FAMILIES))
    ap.add_argument("--config", required=True)
    ap.add_argument("--batches", default="1,8,64"); ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--probes", type=int, default=200); ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--k", type=int, default=5); ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--prefix-cache", action="store_true", help="turn vLLM prefix caching on, as the first data set had it")
    ap.add_argument("--out", default=None, help="output folder; defaults to data-<family>, or data-nocache for qwen2.5-7b")
    main(ap.parse_args())
