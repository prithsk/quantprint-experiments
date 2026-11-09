"""RQ3: query each OpenRouter provider that serves the model, with fallbacks off, and save its top-k log-probabilities
in the same format as collect.py. Each provider's declared precision comes from OpenRouter's endpoint listing.
  python audit_openrouter.py --model qwen/qwen-2.5-7b-instruct --probes 120
  python audit_score.py --ref data-nocache --audit data-audit
Needs OPENROUTER_API_KEY in .env. A run costs well under a dollar: 120 prompts x 64 tokens per provider.
"""
import argparse, json, os, pathlib, time, urllib.request
import collect

ROOT = pathlib.Path(__file__).parent
API = "https://openrouter.ai/api/v1"

def load_env():
    f = ROOT / ".env"
    if f.exists():
        for line in f.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip())

def http(method, url, body=None):
    req = urllib.request.Request(url, method=method, data=json.dumps(body).encode() if body else None,
                                 headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())

def endpoints(model):
    data = http("GET", f"{API}/models/{model}/endpoints")
    eps = data.get("data", {}).get("endpoints", [])
    out = []
    for e in eps:
        params = e.get("supported_parameters") or []
        out.append({"provider": e.get("provider_name") or e.get("name"), "declared": e.get("quantization") or "unknown",
                    "logprobs": "logprobs" in params and "top_logprobs" in params})
    return out

def to_record(resp, provider, declared, pid):
    """Convert an OpenAI-style logprobs response into collect.py's text-only record."""
    content = resp["choices"][0].get("logprobs", {}) or {}
    toks = content.get("content") or []
    return {"config": f"api:{provider}", "declared": declared, "batch": 0, "rep": 0, "pid": pid, "tokens": [],
            "text": [t["token"] for t in toks], "lp": [t["logprob"] for t in toks],
            "topk": [[[x["token"], x["logprob"]] for x in t.get("top_logprobs", [])] for t in toks]}

def main(a):
    load_env()
    if "OPENROUTER_API_KEY" not in os.environ:
        raise SystemExit("add OPENROUTER_API_KEY to .env")
    probes, _ = collect.prompts(a.probes, 0)
    eps = endpoints(a.model)
    print("providers:", ", ".join(f"{e['provider']} ({e['declared']}{'' if e['logprobs'] else ', no logprobs'})" for e in eps))
    if a.list:
        return
    out = ROOT / (a.out or f"data-audit-{a.model.split('/')[-1]}"); out.mkdir(exist_ok=True)
    print(f"saving to {out.name}")
    for e in [e for e in eps if e["logprobs"]]:
        path = out / f"api_{e['provider'].replace(' ', '_').replace('/', '_')}.jsonl"
        n = 0
        with open(path, "w") as fh:
            for pid, p in enumerate(probes):
                body = {"model": a.model, "messages": [{"role": "user", "content": p}], "temperature": 0,
                        "max_tokens": a.tokens, "logprobs": True, "top_logprobs": a.k,
                        "provider": {"order": [e["provider"]], "allow_fallbacks": False}}
                try:
                    rec = to_record(http("POST", f"{API}/chat/completions", body), e["provider"], e["declared"], pid)
                except Exception as ex:
                    print(f"  {e['provider']} prompt {pid}: {ex}"); continue
                if rec["lp"]:
                    fh.write(json.dumps(rec) + "\n"); n += 1
                time.sleep(a.pause)
        print(f"{e['provider']}: {n} of {len(probes)} prompts returned log-probabilities -> {path.name}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen/qwen-2.5-7b-instruct"); ap.add_argument("--probes", type=int, default=120)
    ap.add_argument("--tokens", type=int, default=64); ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--pause", type=float, default=0.2); ap.add_argument("--out", default=None, help="defaults to data-audit-<model>, so each model keeps its own folder")
    ap.add_argument("--list", action="store_true", help="only list providers, their declared precision and logprob support")
    main(ap.parse_args())
