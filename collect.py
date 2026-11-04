"""Collect greedy top-k log-probabilities for one serving configuration with vLLM, at several batch sizes.

Each configuration runs in its own process so GPU memory is fully released between models:
  python collect.py --config bf16 --batches 1,8,64 --repeats 3
Batch composition changes between repeats, since the filler prompts that share each batch are reshuffled. That is the
batch nondeterminism a production provider produces, and the noise floor in the paper comes from it.
Output: data/<config>.jsonl, one line per probe per run.
"""
import argparse, json, os, pathlib, random, time

MODEL = "Qwen/Qwen2.5-7B-Instruct"
CONFIGS = {
    # The reference, plus an FP16 control: quantized checkpoints compute in FP16, so this separates the effect of FP16
    # arithmetic from the effect of quantization itself.
    "bf16":      dict(model=MODEL, quantization=None, dtype="bfloat16"),
    "fp16":      dict(model=MODEL, quantization=None, dtype="float16"),
    "fp8":       dict(model=MODEL, quantization="fp8", dtype="bfloat16"),
    "int8-gptq": dict(model=MODEL + "-GPTQ-Int8", quantization=None, dtype="float16"),
    "int4-gptq": dict(model=MODEL + "-GPTQ-Int4", quantization=None, dtype="float16"),
    "int4-awq":  dict(model=MODEL + "-AWQ", quantization=None, dtype="float16"),
}
ROOT = pathlib.Path(__file__).parent

def prompts(n_probe, n_filler, seed=7):
    """Probe and filler prompts from Dolly, CC BY-SA 3.0: open questions with no context, disjoint sets."""
    from datasets import load_dataset
    rows = [r["instruction"].strip() for r in load_dataset("databricks/databricks-dolly-15k", split="train")
            if r["category"] in ("open_qa", "general_qa", "brainstorming") and not r["context"].strip()
            and 20 <= len(r["instruction"]) <= 300]
    random.Random(seed).shuffle(rows)
    return rows[:n_probe], rows[n_probe:n_probe + n_filler]

def main(a):
    from vllm import LLM, SamplingParams
    cfg = CONFIGS[a.config]
    probes, fillers = prompts(a.probes, 2000)
    llm = LLM(model=cfg["model"], quantization=cfg["quantization"], dtype=cfg["dtype"], max_model_len=2048,
              gpu_memory_utilization=a.gpu_mem, seed=0)
    tok = llm.get_tokenizer()
    chat = lambda p: tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
    probes_c, fillers_c = [chat(p) for p in probes], [chat(p) for p in fillers]
    sp = SamplingParams(temperature=0.0, max_tokens=a.tokens, logprobs=a.k)
    out = (ROOT / "data"); out.mkdir(exist_ok=True)
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
    ap.add_argument("--config", choices=list(CONFIGS), required=True)
    ap.add_argument("--batches", default="1,8,64"); ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--probes", type=int, default=200); ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--k", type=int, default=5); ap.add_argument("--gpu-mem", type=float, default=0.85)
    main(ap.parse_args())
