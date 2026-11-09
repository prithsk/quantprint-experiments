"""The rank-based uniformity test of Zhu et al. (ICLR 2026), reimplemented from their paper, since their repository
holds no code yet. Settings follow their Section 5.1: temperature 0.5, at most 30 new tokens, one target response and
m = 100 reference responses per prompt, the log-rank score, randomized ranks and a Cramer-von Mises test.
  python rut.py sample --family qwen2.5-7b --config int8-gptq --role target      (once per target setup, bf16 too)
  python rut.py sample --family qwen2.5-7b --config bf16 --role reference
  python rut.py score --family qwen2.5-7b                                       (BF16 scores every sample)
  python rut.py analyze --family qwen2.5-7b --quantprint data-nocache           (CPU; compares both tests)
"""
import argparse, glob, json, math, pathlib, random, statistics, sys
import numpy as np
ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT))
import families

def folder(a):
    d = ROOT / (a.out or f"data-rut-{a.family}"); d.mkdir(exist_ok=True); return d

def chats(tok, n):
    import collect
    probes, _ = collect.prompts(n, 0)
    return [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in probes]

def sample(a):
    import collect
    from vllm import SamplingParams
    cfg = families.configs(a.family)[a.config]
    llm = collect.load_llm(cfg, a.gpu_mem); tok = llm.get_tokenizer()
    texts = chats(tok, a.prompts)
    n = a.m if a.role == "reference" else 1
    seed = 12345 if a.role == "reference" else 777 + families.ORDER.index(a.config)
    sp = SamplingParams(temperature=a.temperature, max_tokens=a.max_tokens, n=n, seed=seed)
    outs = llm.generate(texts, sp)
    path = folder(a) / f"{a.role}-{a.config}.jsonl"
    with open(path, "w") as fh:
        for pid, o in enumerate(outs):
            for idx, c in enumerate(o.outputs):
                fh.write(json.dumps({"config": a.config, "role": a.role, "pid": pid, "idx": idx, "response": list(c.token_ids)}) + "\n")
    print(f"wrote {path}: {len(outs)} prompts x {n} responses")

def score(a):
    """Log-rank of every sampled response under the BF16 reference: the mean of log(rank) over response tokens."""
    import collect
    from vllm import SamplingParams
    d = folder(a); cfg = families.configs(a.family)["bf16"]
    llm = collect.load_llm(cfg, a.gpu_mem, max_len=4096); tok = llm.get_tokenizer()
    texts = chats(tok, a.prompts)
    prompt_ids = [tok(t, add_special_tokens=False).input_ids for t in texts]
    done = set()
    sp_path = d / "scores.jsonl"
    if sp_path.exists():
        done = {(r["config"], r["role"], r["pid"], r["idx"]) for r in map(json.loads, open(sp_path))}
    todo = []
    for f in sorted(glob.glob(str(d / "*-*.jsonl"))):
        if f.endswith("scores.jsonl"): continue
        for r in map(json.loads, open(f)):
            if (r["config"], r["role"], r["pid"], r["idx"]) not in done and r["response"]:
                todo.append(r)
    print(f"scoring {len(todo)} responses", flush=True)
    sp = SamplingParams(max_tokens=1, prompt_logprobs=1)
    with open(sp_path, "a") as fh:
        for i in range(0, len(todo), a.chunk):
            part = todo[i:i + a.chunk]
            reqs = [{"prompt_token_ids": prompt_ids[r["pid"]] + r["response"]} for r in part]
            outs = llm.generate(reqs, sp, use_tqdm=False)
            for r, o in zip(part, outs):
                start, ids = len(prompt_ids[r["pid"]]), prompt_ids[r["pid"]] + r["response"]
                ranks = []
                for j in range(start, len(ids)):
                    entry = o.prompt_logprobs[j][ids[j]]
                    if getattr(entry, "rank", None) is None:
                        raise SystemExit(f"vLLM returned no rank for prompt logprobs: {entry!r}. Paste this to Claude.")
                    ranks.append(entry.rank)
                fh.write(json.dumps({"config": r["config"], "role": r["role"], "pid": r["pid"], "idx": r["idx"],
                                     "logrank": float(np.mean(np.log(ranks))), "tokens": len(ranks)}) + "\n")
            print(f"  {min(i + a.chunk, len(todo))} of {len(todo)}", flush=True)
    print(f"wrote {sp_path}")

def rut_power(ref_scores, tgt_scores, n, alpha, rng, trials=500):
    """Share of trials in which RUT rejects equality. Ranks follow equation 4 of Zhu et al. with uniform tie breaking."""
    from scipy.stats import cramervonmises
    pids = [p for p in tgt_scores if p in ref_scores and len(ref_scores[p]) > 0]
    if len(pids) < n: return float("nan")
    hits = 0
    for _ in range(trials):
        r = []
        for p in rng.sample(pids, n):
            s, refs = tgt_scores[p], ref_scores[p]
            r.append((sum(x < s for x in refs) + rng.random() * sum(x == s for x in refs)) / len(refs))
        hits += cramervonmises(r, "uniform").pvalue < alpha
    return 100 * hits / trials

def quantprint_power(data, cfg, n, alpha, rng):
    import analyze as A
    from other_setup import corr_scores, tau
    runs = A.load(sorted(glob.glob(str(ROOT / data / "*.jsonl")))); ref = runs[A.REF]
    if (cfg, 1, 0) not in runs: return float("nan")
    fp = runs[(cfg, 1, 0)]
    neg = [s for k in runs if k[0] == "bf16" and k != A.REF and k[1] != 1 for s in corr_scores(runs[k], fp, ref).values()]
    pos = [s for k in runs if k[0] == cfg and k[1] == 64 for s in corr_scores(runs[k], fp, ref).values()]
    if len(pos) < n or len(neg) < n: return float("nan")
    t = tau(neg, n, alpha, rng)
    return 100 * float(np.mean([statistics.mean(rng.sample(pos, n)) > t for _ in range(500)]))

def analyze(a):
    d = folder(a); rows = [json.loads(l) for l in open(d / "scores.jsonl")]
    ref = {}
    for r in rows:
        if r["role"] == "reference": ref.setdefault(r["pid"], []).append(r["logrank"])
    targets = {}
    for r in rows:
        if r["role"] == "target": targets.setdefault(r["config"], {})[r["pid"]] = r["logrank"]
    ns = [int(x) for x in a.ns.split(",")]; rng = random.Random(0); res = {}
    lines = [f"# RUT and QuantPrint on {a.family}", "", f"RUT: {len(ref)} prompts with {statistics.mean(len(v) for v in ref.values()):.0f} reference "
             "responses each. A BF16 target row is a false positive rate, and every other row is detection power.", ""]
    for alpha in (0.05, 0.01):
        lines += [f"## alpha = {alpha:.0%}", "", "| Setup | Test | " + " | ".join(f"{n} prompts" for n in ns) + " |", "|---|---|" + "---|" * len(ns)]
        for c in sorted(targets, key=lambda c: families.ORDER.index(c) if c in families.ORDER else 99):
            rp = [rut_power(ref, targets[c], n, alpha, rng) for n in ns]
            res[f"rut/{c}/{alpha}"] = rp
            lines.append(f"| {families.NAMES.get(c, c)} | RUT | " + " | ".join("n/a" if math.isnan(x) else f"{x:.0f}%" for x in rp) + " |")
            if a.quantprint and c != "bf16":
                qp = [quantprint_power(a.quantprint, c, n, alpha, rng) for n in ns]
                res[f"quantprint/{c}/{alpha}"] = qp
                lines.append(f"| {families.NAMES.get(c, c)} | QuantPrint direction | " + " | ".join("n/a" if math.isnan(x) else f"{x:.0f}%" for x in qp) + " |")
        lines.append("")
    report = "\n".join(lines); print(report)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"rut-{a.family}.md").write_text(report); (ROOT / "results" / f"rut-{a.family}.json").write_text(json.dumps(res, indent=1))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("sample", "score", "analyze"):
        s = sub.add_parser(name)
        s.add_argument("--family", default="qwen2.5-7b"); s.add_argument("--out", default=None)
        s.add_argument("--prompts", type=int, default=300); s.add_argument("--gpu-mem", type=float, default=0.85)
        if name == "sample":
            s.add_argument("--config", required=True); s.add_argument("--role", choices=["target", "reference"], required=True)
            s.add_argument("--m", type=int, default=100); s.add_argument("--temperature", type=float, default=0.5)
            s.add_argument("--max-tokens", type=int, default=30)
        if name == "score":
            s.add_argument("--chunk", type=int, default=2000)
        if name == "analyze":
            s.add_argument("--quantprint", default=None, help="data folder for the same family, to compare both tests")
            s.add_argument("--ns", default="5,10,20,50,100")
    a = ap.parse_args(); {"sample": sample, "score": score, "analyze": analyze}[a.cmd](a)
