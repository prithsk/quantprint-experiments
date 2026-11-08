"""Robustness and analysis for the ICML version, all on the published data with no GPU.
  1. Provider defenses: rounding returned log-probabilities, adding Gaussian noise to them, and returning fewer tokens.
  2. Mixed serving: a provider quantizes only a fraction of requests.
  3. Attribution with a Bonferroni-corrected threshold per fingerprint.
  4. Bootstrap 95% intervals for detection rates, and a sample-size analysis from per-prompt score statistics.
The direction score, thresholds and positives follow the paper: fingerprints from batch-1 repeat-0 runs, positives from
batch-64 runs only, thresholds from batched BF16 noise, alpha = 1%.
"""
import copy, glob, json, math, pathlib, random, statistics, sys
import numpy as np
ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT)); import analyze as A

runs = A.load(sorted(glob.glob(str(ROOT / "data-nocache" / "*.jsonl")))); ref = runs[A.REF]
CFGS = ["fp16", "fp8", "int8-gptq", "int4-gptq", "int4-awq"]
NAMES = {"fp16": "FP16", "fp8": "FP8", "int8-gptq": "INT8 GPTQ", "int4-gptq": "INT4 GPTQ", "int4-awq": "INT4 AWQ"}
FP = {c: runs[(c, 1, 0)] for c in CFGS}
NOISE = [k for k in runs if k[0] == "bf16" and k != A.REF and k[1] != 1]
POS = {c: [k for k in runs if k[0] == c and k[1] == 64] for c in CFGS}
ALPHA, TRIALS = 0.01, 2000

def corr_scores(run, fp, tmax=None):
    out = {}
    for pid, rec in run.items():
        if pid not in ref or pid not in fp: continue
        d = A.compare(rec, ref[pid])[0]; f = A.compare(fp[pid], ref[pid])[0]
        n = min(len(d), len(f), tmax or 10**9)
        if n >= 5 and np.std(d[:n]) > 0 and np.std(f[:n]) > 0:
            out[pid] = float(np.corrcoef(d[:n], f[:n])[0, 1])
    return out

def transform(run, fn, rng):
    out = {}
    for pid, rec in run.items():
        r = dict(rec); r["lp"] = [fn(x, rng) for x in rec["lp"]]; out[pid] = r
    return out

def power(pos, neg, n, rng, alpha=ALPHA):
    if len(pos) < n or len(neg) < n: return float("nan")
    tau = sorted(statistics.mean(rng.sample(neg, n)) for _ in range(TRIALS))[int((1 - alpha) * TRIALS) - 1]
    return 100 * float(np.mean([statistics.mean(rng.sample(pos, n)) > tau for _ in range(500)]))

def detection_table(fn=None, tmax=None, ns=(5, 20), seed=0):
    rng = random.Random(seed); res = {}
    tr = (lambda run: transform(run, fn, rng)) if fn else (lambda run: run)
    noise_runs = [tr(runs[k]) for k in NOISE]
    for c in CFGS:
        neg = [s for r in noise_runs for s in corr_scores(r, FP[c], tmax).values()]
        pos = [s for k in POS[c] for s in corr_scores(tr(runs[k]), FP[c], tmax).values()]
        res[c] = [power(pos, neg, n, rng) for n in ns]
    return res

out = {}
print("1a. Rounding returned log-probabilities, detection at 5 / 20 prompts")
for d in (4, 3, 2, 1):
    r = detection_table(lambda x, rng, d=d: round(x, d)); out[f"round{d}"] = r
    print(f"   {d} decimals: " + "  ".join(f"{NAMES[c]} {r[c][0]:.0f}/{r[c][1]:.0f}" for c in CFGS))
print("1b. Gaussian noise added to returned log-probabilities")
for sd in (0.001, 0.005, 0.01, 0.02, 0.05):
    r = detection_table(lambda x, rng, sd=sd: x + rng.gauss(0, sd)); out[f"noise{sd}"] = r
    print(f"   sigma {sd}: " + "  ".join(f"{NAMES[c]} {r[c][0]:.0f}/{r[c][1]:.0f}" for c in CFGS))
print("1c. Fewer tokens per prompt")
for t in (8, 16, 32, 64):
    r = detection_table(tmax=t); out[f"tokens{t}"] = r
    print(f"   {t} tokens: " + "  ".join(f"{NAMES[c]} {r[c][0]:.0f}/{r[c][1]:.0f}" for c in CFGS))

print("2. Mixed serving: each prompt quantized with probability f, detection at 20 / 50 prompts")
rng = random.Random(7); out["mixed"] = {}
for c in CFGS:
    neg = [s for k in NOISE for s in corr_scores(runs[k], FP[c]).values()]
    pos = [s for k in POS[c] for s in corr_scores(runs[k], FP[c]).values()]
    row = {}
    for f in (0.1, 0.25, 0.5):
        cells = []
        for n in (20, 50):
            tau = sorted(statistics.mean(rng.sample(neg, n)) for _ in range(TRIALS))[int((1 - ALPHA) * TRIALS) - 1]
            hits = [statistics.mean([rng.choice(pos) if rng.random() < f else rng.choice(neg) for _ in range(n)]) > tau for _ in range(500)]
            cells.append(100 * float(np.mean(hits)))
        row[f] = cells
    out["mixed"][c] = row
    print(f"   {NAMES[c]:10s} " + "  ".join(f"f={f}: {v[0]:.0f}/{v[1]:.0f}" for f, v in row.items()))

print("3. Attribution with a Bonferroni threshold, alpha = 1%/5 per fingerprint")
score = {}
hw = {("h100",) + k[1:]: v for k, v in A.load([str(ROOT / "data-h100" / "bf16.jsonl")]).items() if k[0] == "bf16"}
allruns = {**runs, **hw}
tests = [(k, k[0]) for k in runs if k[0] in CFGS and k[1] == 64] + [(k, "bf16") for k in NOISE] + [(k, "bf16-h100") for k in hw]
for k, _ in tests:
    for c in CFGS: score[(k, c)] = corr_scores(allruns[k], FP[c])
for k in NOISE:
    for c in CFGS: score[(k, c)] = corr_scores(runs[k], FP[c])
out["bonferroni"] = {}
for alpha_each in (ALPHA, ALPHA / 5):
    accs = {}
    for n in (5, 20, 50):
        rng = random.Random(n); conf = {}
        def tau(c, keys):
            pool = [s for kk in keys for s in score[(kk, c)].values()]
            return sorted(statistics.mean(rng.sample(pool, n)) for _ in range(TRIALS))[int((1 - alpha_each) * TRIALS) - 1]
        base = {c: tau(c, NOISE) for c in CFGS}
        for k, truth in tests:
            tk = {c: tau(c, [x for x in NOISE if x != k]) for c in CFGS} if truth == "bf16" else base
            pids = [p for p in score[(k, CFGS[0])] if all(p in score[(k, c)] for c in CFGS)]
            want = "bf16" if truth.startswith("bf16") else truth; ok = 0
            for _ in range(200):
                pick = rng.sample(pids, n); m = {c: statistics.mean(score[(k, c)][p] for p in pick) for c in CFGS}
                over = [c for c in CFGS if m[c] > tk[c]]; got = max(over, key=lambda c: m[c]) if over else "bf16"
                ok += got == want
            conf.setdefault(truth, []).append(ok / 200)
        accs[n] = {t: 100 * statistics.mean(v) for t, v in conf.items()}
    out["bonferroni"][alpha_each] = accs
    print(f"   alpha per fingerprint {alpha_each:.3f}:")
    for n, a in accs.items():
        print(f"      n={n}: " + "  ".join(f"{t} {v:.1f}%" for t, v in a.items()))

print("4a. Bootstrap 95% intervals for detection at 5 prompts, resampling prompts")
rng = random.Random(11); out["ci"] = {}
for c in CFGS:
    neg = [s for k in NOISE for s in corr_scores(runs[k], FP[c]).values()]
    pos = [s for k in POS[c] for s in corr_scores(runs[k], FP[c]).values()]
    vals = []
    for _ in range(200):
        nb = [rng.choice(neg) for _ in neg]; pb = [rng.choice(pos) for _ in pos]
        tau = sorted(statistics.mean(rng.sample(nb, 5)) for _ in range(400))[int(0.99 * 400) - 1]
        vals.append(100 * float(np.mean([statistics.mean(rng.sample(pb, 5)) > tau for _ in range(200)])))
    vals.sort(); out["ci"][c] = (vals[4], vals[194])
    print(f"   {NAMES[c]:10s} 95% interval {vals[4]:.0f}% to {vals[194]:.0f}%")

print("4b. Sample-size analysis: per-prompt direction scores, Gaussian approximation")
out["n_star"] = {}
for c in CFGS:
    neg = [s for k in NOISE for s in corr_scores(runs[k], FP[c]).values()]
    pos = [s for k in POS[c] for s in corr_scores(runs[k], FP[c]).values()]
    mp, sp, mn, sn = statistics.mean(pos), statistics.stdev(pos), statistics.mean(neg), statistics.stdev(neg)
    z_a, z_b = 2.326, 2.326                     # 1% false positives, 99% detection
    nstar = math.ceil(((z_a * sn + z_b * sp) / (mp - mn)) ** 2) if mp > mn else float("inf")
    out["n_star"][c] = dict(mean_pos=mp, sd_pos=sp, mean_neg=mn, sd_neg=sn, n_star=nstar)
    print(f"   {NAMES[c]:10s} same {mp:.3f} sd {sp:.3f}   noise {mn:.3f} sd {sn:.3f}   predicted prompts for 99% at 1%: {nstar}")
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results" / "robustness.json").write_text(json.dumps(out, indent=1, default=str))
