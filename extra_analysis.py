"""Two analyses for the paper, run on the published data:
1. AUC of the magnitude statistic against batched BF16 noise only, batch 8 and 64. A BF16 repeat at batch 1 matches the
   reference exactly, so its zero shifts would make any precision look more separable than it is.
2. Direction-based attribution with no training: score n prompts against every fingerprint, name the precision with the
   highest mean score among those above their 1% threshold, or BF16 if none clears it. Quantized runs are tested only at
   batch 64, so they never match their batch-1 fingerprint run exactly. Each BF16 noise run is tested with thresholds
   built from the other noise runs, and the H100 BF16 runs are tested as an extra honest class.
"""
import sys, glob, random, statistics, collections
import numpy as np
import pathlib
ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT)); import analyze as A
import families
DATA = sys.argv[sys.argv.index("--data") + 1] if "--data" in sys.argv else "data-nocache"
HW = sys.argv[sys.argv.index("--hw") + 1] if "--hw" in sys.argv else ("data-h100" if DATA == "data-nocache" else "")  # H100 data is Qwen 7B only
from sklearn.metrics import roc_auc_score

runs = A.load(sorted(glob.glob(str(ROOT / DATA / "*.jsonl")))); ref = runs[A.REF]
hw = {k: v for k, v in A.load([str(ROOT / HW / "bf16.jsonl")]).items() if k[0] == "bf16"}
cfgs = ["fp16", "fp8", "int8-gptq", "int4-gptq", "int4-awq"]
noise = [k for k in runs if k[0] == "bf16" and k != A.REF and k[1] != 1]       # batched noise only

print("1. Magnitude AUC against batched BF16 noise")
negS = [s for k in noise for s in A.per_prompt_S(runs[k], ref).values()]
for c in cfgs:
    posS = [s for k in runs if k[0] == c for s in A.per_prompt_S(runs[k], ref).values()]
    print(f"   {c:10s} AUC {roc_auc_score([0]*len(negS) + [1]*len(posS), negS + posS):.3f}")

fps = {c: runs[(c, 1, 0)] for c in cfgs}
score = {}                                                     # (run key, cfg) -> {pid: rho}
def scores_for(key, src, c):
    out = {}
    for pid, rec in src[key].items():
        if pid not in ref or pid not in fps[c]: continue
        d = A.compare(rec, ref[pid])[0]; f = A.compare(fps[c][pid], ref[pid])[0]; n = min(len(d), len(f))
        if n >= 5 and np.std(d[:n]) > 0 and np.std(f[:n]) > 0:
            out[pid] = float(np.corrcoef(d[:n], f[:n])[0, 1])
    return out
tests = [(k, runs, k[0]) for k in runs if k[0] in cfgs and k[1] == 64] + [(k, runs, "bf16") for k in noise] \
        + [(("h100",) + k[1:], hw, "bf16-h100") for k in hw]
allsrc = {**{k: (runs, k) for k in runs}, **{("h100",) + k[1:]: (hw, k) for k in hw}}
for key, (src, real) in allsrc.items():
    if key[0] in ("bf16", "h100") or key[0] in cfgs:
        for c in cfgs:
            score[(key, c)] = scores_for(real, src, c)

alpha, trials = 0.01, 2000
def tau(c, n, noise_keys, rng):
    pool = [s for k in noise_keys for s in score[(k, c)].values()]
    return sorted(statistics.mean(rng.sample(pool, n)) for _ in range(trials))[int((1 - alpha) * trials) - 1]

print("\n2. Direction-based attribution, no training")
for n in (5, 20, 50):
    rng = random.Random(n); conf = collections.defaultdict(collections.Counter)
    base_tau = {c: tau(c, n, noise, rng) for c in cfgs}
    for key, src, truth in tests:
        tk = {c: tau(c, n, [k for k in noise if k != key], rng) for c in cfgs} if truth == "bf16" else base_tau
        pids = [p for p in score[(key, cfgs[0])] if all(p in score[(key, c)] for c in cfgs)]
        for _ in range(200):
            pick = rng.sample(pids, n)
            means = {c: statistics.mean(score[(key, c)][p] for p in pick) for c in cfgs}
            over = [c for c in cfgs if means[c] > tk[c]]
            conf[truth][max(over, key=lambda c: means[c]) if over else "bf16"] += 1
    rows = ["bf16", "bf16-h100"] + cfgs
    correct = sum(conf[t][("bf16" if t.startswith("bf16") else t)] for t in rows); total = sum(sum(conf[t].values()) for t in rows)
    print(f"   n={n}: accuracy {100*correct/total:.1f}%")
    for t in rows:
        tot = sum(conf[t].values()); want = "bf16" if t.startswith("bf16") else t
        print(f"      {t:10s} correct {100*conf[t][want]/tot:5.1f}%   " + ", ".join(f"{k} {100*v/tot:.0f}%" for k, v in conf[t].most_common() if k != want))
