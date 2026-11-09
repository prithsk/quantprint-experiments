"""Score runs from another setup, meaning another GPU or another serving engine, against the reference fingerprints.
Quantized runs answer whether detection still works there, and BF16 runs answer whether honest serving gets flagged.
  python other_setup.py --ref data-nocache --other data-h100q
  python other_setup.py --ref data-nocache --other data-sglang
Fingerprints come from the reference folder's batch-1 runs and thresholds from its batched BF16 noise, as in the paper.
"""
import argparse, glob, json, pathlib, random, statistics, sys
import numpy as np
ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT)); import analyze as A, families

def corr_scores(run, fp, ref):
    out = {}
    for pid, rec in run.items():
        if pid not in ref or pid not in fp: continue
        d = A.compare(rec, ref[pid])[0]; f = A.compare(fp[pid], ref[pid])[0]; n = min(len(d), len(f))
        if n >= 5 and np.std(d[:n]) > 0 and np.std(f[:n]) > 0:
            out[pid] = float(np.corrcoef(d[:n], f[:n])[0, 1])
    return out

def tau(pool, n, alpha, rng, trials=2000):
    return sorted(statistics.mean(rng.sample(pool, n)) for _ in range(trials))[int((1 - alpha) * trials) - 1]

def main(a):
    ns = [int(x) for x in a.ns.split(",")]; rng = random.Random(0)
    runs = A.load(sorted(glob.glob(str(ROOT / a.ref / "*.jsonl")))); ref = runs[A.REF]
    other = A.load(sorted(glob.glob(str(ROOT / a.other / "*.jsonl"))))
    cfgs = [c for c in families.ORDER if c != "bf16" and (c, 1, 0) in runs]
    fps = {c: runs[(c, 1, 0)] for c in cfgs}
    noise = [k for k in runs if k[0] == "bf16" and k != A.REF and k[1] != 1]
    neg = {c: [s for k in noise for s in corr_scores(runs[k], fps[c], ref).values()] for c in cfgs}
    sc = {(k, c): corr_scores(other[k], fps[c], ref) for k in other for c in cfgs}
    lines = [f"# Other setup: {a.other} scored against {a.ref}", "", f"Fingerprints: {', '.join(families.NAMES[c] for c in cfgs)}. "
             f"False positive rate {a.alpha:.0%} per test.", ""]
    res = {"detection": {}, "false_alarms": {}, "attribution": {}}
    present = sorted({k[0] for k in other}, key=lambda c: families.ORDER.index(c) if c in families.ORDER else 99)
    lines += ["## Detection of each quantized setup by its own fingerprint", "", "| Setup | runs | mean score | " + " | ".join(f"{n} prompts" for n in ns) + " |", "|---|---|---|" + "---|" * len(ns)]
    for c in [c for c in present if c in cfgs]:
        pos = [s for k in other if k[0] == c for s in sc[(k, c)].values()]
        cells = []
        for n in ns:
            if len(pos) < n or len(neg[c]) < n: cells.append("n/a"); continue
            t = tau(neg[c], n, a.alpha, rng)
            cells.append(f"{100 * np.mean([statistics.mean(rng.sample(pos, n)) > t for _ in range(500)]):.0f}%")
        res["detection"][c] = cells
        lines.append(f"| {families.NAMES[c]} | {sum(1 for k in other if k[0] == c)} | {statistics.mean(pos) if pos else float('nan'):.3f} | " + " | ".join(cells) + " |")
    if "bf16" in present:
        lines += ["", "## False alarms on honest BF16 from the other setup", "", "| Fingerprint | " + " | ".join(f"{n} prompts" for n in ns) + " |", "|---|" + "---|" * len(ns)]
        for c in cfgs:
            test = [s for k in other if k[0] == "bf16" for s in sc[(k, c)].values()]
            cells = []
            for n in ns:
                if len(test) < n or len(neg[c]) < n: cells.append("n/a"); continue
                t = tau(neg[c], n, a.alpha, rng)
                cells.append(f"{100 * np.mean([statistics.mean(rng.sample(test, n)) > t for _ in range(500)]):.0f}%")
            res["false_alarms"][c] = cells
            lines.append(f"| {families.NAMES[c]} | " + " | ".join(cells) + " |")
    lines += ["", f"## Attribution with a Bonferroni threshold, {a.alpha / len(cfgs):.2%} per fingerprint", "",
              "| True setup | " + " | ".join(f"{n} prompts" for n in ns) + " |", "|---|" + "---|" * len(ns)]
    taus = {n: {c: tau(neg[c], n, a.alpha / len(cfgs), rng) for c in cfgs if len(neg[c]) >= n} for n in ns}
    for c in present:
        keys = [k for k in other if k[0] == c]; cells = []
        for n in ns:
            ok = tot = 0
            for k in keys:
                pids = [p for p in sc[(k, cfgs[0])] if all(p in sc[(k, x)] for x in cfgs)]
                if len(pids) < n or len(taus[n]) < len(cfgs): continue
                for _ in range(200):
                    pick = rng.sample(pids, n); m = {x: statistics.mean(sc[(k, x)][p] for p in pick) for x in cfgs}
                    over = [x for x in cfgs if m[x] > taus[n][x]]; got = max(over, key=lambda x: m[x]) if over else "bf16"
                    ok += got == c; tot += 1
            cells.append(f"{100 * ok / tot:.1f}%" if tot else "n/a")
        res["attribution"][c] = cells
        lines.append(f"| {families.NAMES.get(c, c)} | " + " | ".join(cells) + " |")
    report = "\n".join(lines); print(report)
    out = ROOT / "results"; out.mkdir(exist_ok=True)
    name = pathlib.Path(a.other).name
    (out / f"other-{name}.md").write_text(report); (out / f"other-{name}.json").write_text(json.dumps(res, indent=1))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="data-nocache"); ap.add_argument("--other", required=True)
    ap.add_argument("--alpha", type=float, default=0.01); ap.add_argument("--ns", default="5,20,50")
    main(ap.parse_args())
