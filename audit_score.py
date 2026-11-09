"""Score each audited provider with the direction rule. Run audit_openrouter.py first.
  python audit_score.py --ref data-nocache --audit data-audit
A provider with too few scorable prompts gets no verdict. A prompt is scorable when the provider's first five or more
tokens match the reference. A different chat template, system prompt or decoding setting changes the first tokens of
nearly every response, while quantization still leaves many scorable prompts, so token match alone would wrongly
excuse heavily quantized providers.
"""
import argparse, glob, json, pathlib, random, statistics, sys
import numpy as np
ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT)); import analyze as A, families
from other_setup import corr_scores, tau

def label(scores, cfgs, taus, pids):
    m = {c: statistics.mean(scores[c][p] for p in pids) for c in cfgs}
    over = [c for c in cfgs if m[c] > taus[c]]
    return (max(over, key=lambda c: m[c]) if over else "bf16"), m

def main(a):
    rng = random.Random(0)
    runs = A.load(sorted(glob.glob(str(ROOT / a.ref / "*.jsonl")))); ref = runs[A.REF]
    cfgs = [c for c in families.ORDER if c != "bf16" and (c, 1, 0) in runs]
    fps = {c: runs[(c, 1, 0)] for c in cfgs}
    noise = [k for k in runs if k[0] == "bf16" and k != A.REF and k[1] != 1]
    neg = {c: [s for k in noise for s in corr_scores(runs[k], fps[c], ref).values()] for c in cfgs}
    api = A.load(sorted(glob.glob(str(ROOT / a.audit / "api_*.jsonl"))))
    if not api:
        raise SystemExit(f"no api_*.jsonl files in {a.audit}: run audit_openrouter.py first")
    rows, out = [], {}
    for k, recs in sorted(api.items()):
        provider = k[0].removeprefix("api:"); declared = next(iter(recs.values())).get("declared", "unknown")
        match = statistics.mean(A.compare(r, ref[p])[3] for p, r in recs.items() if p in ref) if recs else 0.0
        scores = {c: corr_scores(recs, fps[c], ref) for c in cfgs}
        pids = [p for p in scores[cfgs[0]] if all(p in scores[c] for c in cfgs)]
        if len(pids) < a.min_prompts:
            rows.append(f"| {provider} | {declared} | no verdict | --- | {len(pids)} | {match:.0%} |")
            out[provider] = dict(declared=declared, detected=None, match=match, prompts=len(pids)); continue
        taus = {c: tau(neg[c], min(len(pids), len(neg[c])), a.alpha / len(cfgs), rng) for c in cfgs}
        got, means = label(scores, cfgs, taus, pids)
        boot = [label(scores, cfgs, taus, [rng.choice(pids) for _ in pids])[0] for _ in range(1000)]
        conf = boot.count(got) / len(boot)
        rows.append(f"| {provider} | {declared} | {families.NAMES.get(got, got)} | {conf:.0%} | {len(pids)} | {match:.0%} |")
        out[provider] = dict(declared=declared, detected=got, confidence=conf, match=match, prompts=len(pids), mean_scores=means)
    report = "\n".join(["# Provider audit", "", f"Direction rule with a Bonferroni threshold, {a.alpha:.0%} overall. Confidence is the share of "
                        "1000 bootstrap resamples of prompts that give the same verdict.", "",
                        "| Provider | Declared | Detected | Confidence | Prompts scored | Token match |", "|---|---|---|---|---|---|"] + rows)
    print(report)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "audit.md").write_text(report); (ROOT / "results" / "audit.json").write_text(json.dumps(out, indent=1, default=str))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="data-nocache"); ap.add_argument("--audit", default="data-audit")
    ap.add_argument("--alpha", type=float, default=0.01); ap.add_argument("--min-prompts", type=int, default=10)
    main(ap.parse_args())
