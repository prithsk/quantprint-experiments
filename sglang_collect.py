"""Collect greedy log-probabilities from a running SGLang server, in the same record format as collect.py, so
other_setup.py can score SGLang runs against the vLLM fingerprints. jobs/sglang.sh starts and stops the servers.
  python sglang_collect.py --config int4-awq --url http://127.0.0.1:30000
Batches go through SGLang's native /generate endpoint as one request, so the engine sees them together.
"""
import argparse, json, pathlib, random, sys, time, urllib.request
ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT))
import families

def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())

def parse(item, k):
    """One SGLang result -> tokens, text, chosen log-probabilities and top-k, or a clear stop if the format differs."""
    try:
        mi = item["meta_info"]; chosen = mi["output_token_logprobs"]; top = mi["output_top_logprobs"]
        lp = [float(x[0]) for x in chosen]; toks = [int(x[1]) for x in chosen]; text = [x[2] for x in chosen]
        topk = [sorted([[int(t[1]), float(t[0])] for t in pos if t[0] is not None], key=lambda z: -z[1])[:k] for pos in top]
        return toks, text, lp, topk
    except (KeyError, IndexError, TypeError) as e:
        raise SystemExit(f"Unexpected SGLang response format ({e}). Paste this to Claude:\n{json.dumps(item)[:1500]}")

def main(a):
    import collect
    from transformers import AutoTokenizer
    cfg = families.configs(a.family)[a.config]
    tok = AutoTokenizer.from_pretrained(families.configs(a.family)["bf16"]["model"])
    probes, fillers = collect.prompts(a.probes, 2000)
    chat = lambda p: tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
    probes_c, fillers_c = [chat(p) for p in probes], [chat(p) for p in fillers]
    out = ROOT / a.out; out.mkdir(exist_ok=True)
    try:
        info = json.loads(urllib.request.urlopen(a.url + "/get_server_info", timeout=30).read())
    except Exception:
        info = {}
    (out / f"{a.config}.meta.json").write_text(json.dumps({"engine": "sglang", "config": a.config, "model": cfg["model"],
        "sglang": info.get("version"), "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2))
    path = out / f"{a.config}.jsonl"
    with open(path, "a") as fh:
        for b in [int(x) for x in a.batches.split(",")]:
            for rep in range(a.repeats):
                rng = random.Random(1000 * b + rep); order = list(range(len(probes_c))); rng.shuffle(order)
                per_call = min(b, 8); t0 = time.time()
                for i in range(0, len(order), per_call):
                    ids = order[i:i + per_call]
                    batch = [probes_c[j] for j in ids] + rng.sample(fillers_c, b - len(ids))
                    positions = list(range(len(batch))); rng.shuffle(positions); batch = [batch[p] for p in positions]
                    res = post(a.url + "/generate", {"text": batch, "sampling_params": {"temperature": 0, "max_new_tokens": a.tokens},
                                                     "return_logprob": True, "top_logprobs_num": a.k, "return_text_in_logprobs": True})
                    res = res if isinstance(res, list) else [res]
                    for pos, item in zip(positions, res):
                        if pos >= len(ids): continue
                        toks, text, lp, topk = parse(item, a.k)
                        fh.write(json.dumps({"config": a.config, "batch": b, "rep": rep, "pid": ids[pos], "tokens": toks,
                                             "text": text, "lp": lp, "topk": topk}) + "\n")
                print(f"{a.config} batch {b} repeat {rep}: {len(order)} probes in {time.time() - t0:.0f} s", flush=True)
    print(f"wrote {path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="qwen2.5-7b"); ap.add_argument("--config", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:30000"); ap.add_argument("--batches", default="1,64")
    ap.add_argument("--repeats", type=int, default=2); ap.add_argument("--probes", type=int, default=120)
    ap.add_argument("--tokens", type=int, default=64); ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default="data-sglang")
    main(ap.parse_args())
