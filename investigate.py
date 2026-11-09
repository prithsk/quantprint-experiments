"""1. Why does the INT4 AWQ fingerprint flag honest BF16 from other setups? Compare mean scores per setup.
2. Can an auditor fix it by calibrating with BF16 runs from more than one setup? Leave-one-setup-out: thresholds from
   the L4 noise plus every other setup's BF16, tested on the held-out setup's BF16 and on its quantized runs.
3. RUT calibrated on its own BF16 trials, so its false positive rate is exactly alpha."""
import glob, json, random, statistics, sys
import numpy as np
sys.path.insert(0, "/tmp/qp")
import analyze as A, families
from other_setup import corr_scores, tau
runs = A.load(sorted(glob.glob("data-nocache/*.jsonl"))); ref = runs[A.REF]
cfgs = [c for c in families.ORDER if c != "bf16" and (c, 1, 0) in runs]; fps = {c: runs[(c, 1, 0)] for c in cfgs}
l4noise = [k for k in runs if k[0] == "bf16" and k != A.REF and k[1] != 1]
setups = {"H100": A.load(["data-h100/bf16.jsonl"]), "A100": A.load(sorted(glob.glob("data-a100q/*.jsonl"))), "SGLang": A.load(sorted(glob.glob("data-sglang/*.jsonl")))}
sc = lambda run, c: list(corr_scores(run, fps[c], ref).values())
print("1. Mean INT4 AWQ fingerprint score on honest BF16, with the share of prompts scoring above 0.5")
pools = {"L4 noise": [s for k in l4noise for s in sc(runs[k], "int4-awq")]}
for name, d in setups.items():
    pools[name] = [s for k in d if k[0] == "bf16" for s in sc(d[k], "int4-awq")]
for name, p in pools.items():
    print(f"   {name:9s} mean {statistics.mean(p):+.3f}  sd {statistics.stdev(p):.3f}  above 0.5: {100*np.mean(np.array(p) > 0.5):.0f}%  n={len(p)}")
print("\n2. Leave-one-setup-out calibration at 1% per fingerprint, 50 prompts")
rng = random.Random(0)
for held in setups:
    neg = {c: [s for k in l4noise for s in sc(runs[k], c)] + [s for n, d in setups.items() if n != held for k in d if k[0] == "bf16" for s in sc(d[k], c)] for c in cfgs}
    d = setups[held]; fa = []; det = []
    for c in cfgs:
        t = tau(neg[c], 50, 0.01, rng)
        test = [s for k in d if k[0] == "bf16" for s in sc(d[k], c)]
        fa.append(f"{families.NAMES[c]} {100*np.mean([statistics.mean(rng.sample(test, 50)) > t for _ in range(500)]):.0f}%")
        pos = [s for k in d if k[0] == c for s in sc(d[k], c)]
        if len(pos) >= 50:
            t5 = tau(neg[c], 5, 0.01, rng)
            det.append(f"{families.NAMES[c]} {100*np.mean([statistics.mean(rng.sample(pos, 5)) > t5 for _ in range(500)]):.0f}%")
    print(f"   held out {held:7s} false alarms: " + ", ".join(fa))
    if det: print(f"   {'':16s} detection at 5 prompts: " + ", ".join(det))
print("\n3. RUT with its threshold calibrated on its own BF16 trials")
from scipy.stats import cramervonmises
for fam in ("qwen2.5-7b", "mistral-7b"):
    rows = [json.loads(l) for l in open(f"data-rut-{fam}/scores.jsonl")]
    refsc = {}; tg = {}
    for r in rows:
        if r["role"] == "reference": refsc.setdefault(r["pid"], []).append(r["logrank"])
        else: tg.setdefault(r["config"], {})[r["pid"]] = r["logrank"]
    def pvals(c, n, trials=1000, seed=1):
        rr = random.Random(seed); pids = [p for p in tg[c] if p in refsc]; out = []
        for _ in range(trials):
            r = [(sum(x < tg[c][p] for x in refsc[p]) + rr.random() * sum(x == tg[c][p] for x in refsc[p])) / len(refsc[p]) for p in rr.sample(pids, n)]
            out.append(cramervonmises(r, "uniform").pvalue)
        return out
    print(f"   {fam}")
    for n in (5, 20, 50, 100):
        null = sorted(pvals("bf16", n, seed=2)); cut = null[int(0.05 * len(null))]
        row = []
        for c in [c for c in families.ORDER if c in tg and c != "bf16"]:
            p = pvals(c, n, trials=500, seed=3); row.append(f"{families.NAMES[c]} {100*np.mean(np.array(p) < cut):.0f}%")
        print(f"      n={n:3d}, 5% empirical threshold: " + ", ".join(row))
