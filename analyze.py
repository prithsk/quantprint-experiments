"""Turn collected runs into the paper's numbers.

  python analyze.py                       RQ1 and RQ2 from data/*.jsonl
  python analyze.py --audit data/api_*.jsonl   RQ3: name the precision each API run serves

Writes results/report.md, results/rq1_table.tex (rows for Table 3), results/rq3_table.tex (rows for Table 4) and
results/accuracy_vs_queries.png. The reference is BF16 at batch 1, repeat 0; every other BF16 run is noise.
"""
import argparse, glob, json, pathlib, random, statistics
import numpy as np

ROOT = pathlib.Path(__file__).parent
REF = ("bf16", 1, 0)
NAMES = {"bf16": "BF16", "fp16": "FP16", "fp8": "FP8", "int8-gptq": "INT8 GPTQ", "int4-gptq": "INT4 GPTQ", "int4-awq": "INT4 AWQ"}

def load(paths):
    runs = {}
    for p in paths:
        for line in open(p):
            r = json.loads(line); runs.setdefault((r["config"], r["batch"], r["rep"]), {})[r["pid"]] = r
    return runs

def align_key(rec):
    return rec["tokens"] if rec.get("tokens") else rec["text"]

def compare(a, ref):
    """Per-token shift on the matching prefix, plus top-k overlap and the reference margin there."""
    ka, kr = align_key(a), (ref["tokens"] if a.get("tokens") else ref["text"])
    n = 0
    while n < min(len(ka), len(kr)) and ka[n] == kr[n]:
        n += 1
    d = np.array([a["lp"][t] - ref["lp"][t] for t in range(n)])
    def topset(rec, t):
        return {str(x[0]) for x in rec["topk"][t]} if rec.get("tokens") else {str(x[0]) for x in rec["topk"][t]}
    k = max(1, len(ref["topk"][0]) if ref["topk"] else 1)
    ov = np.array([len(topset(a, t) & topset(ref, t)) / k for t in range(n)]) if a.get("tokens") else np.ones(n)
    m = np.array([ref["topk"][t][0][1] - ref["topk"][t][1][1] if len(ref["topk"][t]) > 1 else 10.0 for t in range(n)])
    return d, ov, m, n / max(1, len(kr))

def features(a, ref):
    d, ov, m, frac = compare(a, ref)
    if len(d) == 0:
        return None
    ad, low = np.abs(d), m < 0.5
    return np.array([ad.mean(), d.mean(), d.std(), ov.mean(), ad[low].mean() if low.any() else 0.0,
                     ad[~low].mean() if (~low).any() else 0.0, frac])

def per_prompt_S(run, ref):
    out = {}
    for pid, rec in run.items():
        if pid in ref:
            d, _, _, _ = compare(rec, ref[pid])
            if len(d): out[pid] = float(np.abs(d).mean())
    return out

def rq1(runs, ref, lines):
    from sklearn.metrics import roc_auc_score
    noise_keys = [k for k in runs if k[0] == "bf16" and k != REF]
    noise_S = [s for k in noise_keys for s in per_prompt_S(runs[k], ref).values()]
    def pooled_mean(keys):
        vals = [abs(x) for k in keys for pid, rec in runs[k].items() if pid in ref for x in compare(rec, ref[pid])[0]]
        return float(np.mean(vals)) if vals else float("nan")
    rows, tex = [], []
    for b in sorted({k[1] for k in noise_keys}):
        keys = [k for k in noise_keys if k[1] == b]
        label = "Same precision, repeat at batch 1" if b == 1 else f"Same precision, batch 1 vs {b}"
        rows.append((label, pooled_mean(keys), None))
    for cfg in [c for c in NAMES if c != "bf16"]:
        keys = [k for k in runs if k[0] == cfg]
        if not keys: continue
        S = [s for k in keys for s in per_prompt_S(runs[k], ref).values()]
        auc = roc_auc_score([0] * len(noise_S) + [1] * len(S), noise_S + S) if noise_S and S else float("nan")
        rows.append((f"BF16 vs {NAMES[cfg]}" + (" (control)" if cfg == "fp16" else ""), pooled_mean(keys), auc))
    lines += ["## RQ1: shift from each precision against the noise floor", "",
              "| Comparison | Mean abs shift, nats | AUC against noise |", "|---|---|---|"]
    for label, mean, auc in rows:
        lines.append(f"| {label} | {mean:.5f} | {'---' if auc is None else f'{auc:.3f}'} |")
        tex.append(f"{label} & {mean:.5f} & {'---' if auc is None else f'{auc:.3f}'} \\\\ \\hline")
    (ROOT / "results" / "rq1_table.tex").write_text("\n".join(tex) + "\n")
    return noise_keys, noise_S

def systematic(runs, ref, noise_keys, lines):
    """Does a precision shift each token the same way in two independent runs? Batch noise should not."""
    def pair_stats(k1, k2):
        agree, xs, ys = [], [], []
        for pid in set(runs[k1]) & set(runs[k2]) & set(ref):
            d1 = compare(runs[k1][pid], ref[pid])[0]; d2 = compare(runs[k2][pid], ref[pid])[0]
            n = min(len(d1), len(d2))
            for x, y in zip(d1[:n], d2[:n]):
                if abs(x) > 1e-6 and abs(y) > 1e-6:
                    agree.append(np.sign(x) == np.sign(y)); xs.append(x); ys.append(y)
        if len(xs) < 3: return None
        return float(np.mean(agree)), float(np.corrcoef(xs, ys)[0, 1]), len(xs)
    lines += ["", "## Systematic or random: agreement of per-token shifts across two independent runs", "",
              "| Precision | Same sign | Correlation | Tokens compared |", "|---|---|---|---|"]
    groups = {"bf16 (noise)": noise_keys} | {c: [k for k in runs if k[0] == c] for c in NAMES if c != "bf16"}
    for name, keys in groups.items():
        keys = sorted(keys, key=lambda k: (k[1] != 64, k))         # prefer two runs at the same large batch
        if len(keys) < 2: continue
        st = pair_stats(keys[0], keys[1])
        if st: lines.append(f"| {NAMES.get(name, name)} | {100*st[0]:.1f}% | {st[1]:.3f} | {st[2]:,} |")

def power(runs, ref, noise_S, lines, alpha=0.01, trials=2000, seed=0):
    rng = random.Random(seed)
    ns = [n for n in (5, 10, 20, 50, 100, 200) if n <= len(noise_S)]
    lines += ["", f"## Detection power at a {alpha:.0%} false positive rate, by number of probe prompts", "",
              "| Precision | " + " | ".join(f"{n} prompts" for n in ns) + " |", "|---|" + "---|" * len(ns)]
    tau = {n: sorted(statistics.median(rng.sample(noise_S, n)) for _ in range(trials))[int((1 - alpha) * trials) - 1] for n in ns}
    for cfg in [c for c in NAMES if c != "bf16"]:
        S = [s for k in runs if k[0] == cfg for s in per_prompt_S(runs[k], ref).values()]
        if len(S) < max(ns, default=0): continue
        cells = [f"{100*np.mean([statistics.median(rng.sample(S, n)) > tau[n] for _ in range(trials // 4)]):.0f}%" for n in ns]
        lines.append(f"| {NAMES[cfg]} | " + " | ".join(cells) + " |")

def attribution(runs, ref, lines, ns=(1, 5, 10, 20, 50), per_run=150, seed=0):
    """Leave one batch size out: train on the others, test on the held-out batch, so the classifier is tested on
    batch noise it never saw."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import confusion_matrix
    rng = random.Random(seed)
    feats = {}
    for k, run in runs.items():
        if k == REF: continue
        fs = [f for pid, rec in run.items() if pid in ref and (f := features(rec, ref[pid])) is not None]
        if fs: feats[k] = np.array(fs)
    labels = sorted({k[0] for k in feats})
    def sample(keys, n):
        X, y = [], []
        for k in keys:
            F = feats[k]
            for _ in range(per_run):
                idx = [rng.randrange(len(F)) for _ in range(n)]; X.append(F[idx].mean(0)); y.append(k[0])
        return np.array(X), np.array(y)
    batches = sorted({k[1] for k in feats}); acc = {}; cm = None
    for n in ns:
        correct = total = 0
        for b in batches:
            train = [k for k in feats if k[1] != b]; test = [k for k in feats if k[1] == b]
            if not test or len({k[0] for k in train}) < len(labels): continue
            Xtr, ytr = sample(train, n); Xte, yte = sample(test, n)
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(Xtr, ytr)
            pred = clf.predict(Xte); correct += int((pred == yte).sum()); total += len(yte)
            if n == ns[-1]:
                c = confusion_matrix(yte, pred, labels=labels); cm = c if cm is None else cm + c
        acc[n] = correct / total if total else float("nan")
    lines += ["", "## RQ2: attribution accuracy on a held-out batch size", "",
              "| Probe prompts per decision | " + " | ".join(str(n) for n in ns) + " |", "|---|" + "---|" * len(ns),
              "| Accuracy | " + " | ".join(f"{100*acc[n]:.1f}%" for n in ns) + " |"]
    if cm is not None:
        lines += ["", f"Confusion matrix at {ns[-1]} prompts per decision, rows true and columns predicted:", "",
                  "| | " + " | ".join(NAMES.get(l, l) for l in labels) + " |", "|---|" + "---|" * len(labels)]
        for l, row in zip(labels, cm):
            lines.append(f"| {NAMES.get(l, l)} | " + " | ".join(str(int(x)) for x in row) + " |")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        plt.figure(figsize=(3.4, 2.2)); plt.plot(list(acc), [100 * acc[n] for n in acc], marker="o")
        plt.xscale("log"); plt.xlabel("Probe prompts per decision"); plt.ylabel("Accuracy (%)"); plt.tight_layout()
        plt.savefig(ROOT / "results" / "accuracy_vs_queries.png", dpi=200)
    except Exception as e:
        lines.append(f"(figure skipped: {e})")
    return feats, labels

def audit(runs, ref, feats, labels, paths, lines, n_per=None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    api = load(paths); rng = random.Random(1)
    probe_n = min(len(r) for r in api.values())
    X, y = [], []
    for k, F in feats.items():
        for _ in range(200):
            idx = [rng.randrange(len(F)) for _ in range(probe_n)]; X.append(F[idx].mean(0)); y.append(k[0])
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(np.array(X), np.array(y))
    lines += ["", "## RQ3: audit of API providers", "", "| Provider | Declared | Detected | Confidence |", "|---|---|---|---|"]
    tex = []
    for k, run in api.items():
        fs = [f for pid, rec in run.items() if pid in ref and (f := features(rec, ref[pid])) is not None]
        if not fs: continue
        p = clf.predict_proba(np.array(fs).mean(0, keepdims=True))[0]; i = int(p.argmax())
        declared = next(iter(run.values())).get("declared", "unknown")
        row = (k[0].removeprefix("api:"), declared, NAMES.get(clf.classes_[i], clf.classes_[i]), f"{100*p[i]:.0f}%")
        lines.append("| " + " | ".join(row) + " |"); tex.append(" & ".join(row) + " \\\\ \\hline")
    (ROOT / "results" / "rq3_table.tex").write_text("\n".join(tex) + "\n")

def main(a):
    (ROOT / "results").mkdir(exist_ok=True)
    runs = load(sorted(p for p in glob.glob(str(ROOT / "data" / "*.jsonl")) if not pathlib.Path(p).name.startswith("api_")))
    if REF not in runs:
        raise SystemExit("missing the reference run: collect bf16 with batch 1 first")
    ref = runs[REF]
    lines = [f"# QuantPrint results: {len(runs)} runs, {len(ref)} probe prompts", ""]
    noise_keys, noise_S = rq1(runs, ref, lines)
    systematic(runs, ref, noise_keys, lines)
    power(runs, ref, noise_S, lines)
    feats, labels = attribution(runs, ref, lines)
    if a.audit:
        audit(runs, ref, feats, labels, a.audit, lines)
    (ROOT / "results" / "report.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--audit", nargs="*")
    main(ap.parse_args())
