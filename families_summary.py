"""One row per model family for the paper: FP16 magnitude AUC against batched BF16 noise, direction scores at batch 64
against the batch-1 fingerprint, detection from 5 prompts at 1%, and the BF16 / FP16 batch-effect ratio.
  python families_summary.py"""
import glob, json, pathlib, random, statistics, sys
import numpy as np
from sklearn.metrics import roc_auc_score
ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT)); import analyze as A
from other_setup import corr_scores, tau
out = {}
for name, d in (("Qwen2.5-7B", "data-nocache"), ("Muse Glimmer 30B", "data-muse-glimmer-30b"), ("Mistral 7B", "data-mistral-7b")):
    runs = A.load(sorted(glob.glob(str(ROOT / d / "*.jsonl")))); ref = runs[A.REF]
    noise = [k for k in runs if k[0] == "bf16" and k != A.REF and k[1] != 1]
    negS = [s for k in noise for s in A.per_prompt_S(runs[k], ref).values()]
    posS = [s for k in runs if k[0] == "fp16" for s in A.per_prompt_S(runs[k], ref).values()]
    auc = roc_auc_score([0] * len(negS) + [1] * len(posS), negS + posS)
    fp = runs[("fp16", 1, 0)]
    neg = [s for k in noise for s in corr_scores(runs[k], fp, ref).values()]
    pos = [s for k in runs if k[0] == "fp16" and k[1] == 64 for s in corr_scores(runs[k], fp, ref).values()]
    rng = random.Random(0); t = tau(neg, 5, 0.01, rng)
    det = 100 * float(np.mean([statistics.mean(rng.sample(pos, 5)) > t for _ in range(500)]))
    def batch_effect(c):
        base = runs[(c, 1, 0)]
        return float(np.mean([abs(x) for k in runs if k[0] == c and k[1] == 64 for p, r in runs[k].items() if p in base for x in A.compare(r, base[p])[0]]))
    be16, be32 = batch_effect("fp16"), batch_effect("bf16")
    reps = len({k[2] for k in runs if k[0] == "bf16"})
    out[name] = dict(auc=auc, same=statistics.mean(pos), noise=statistics.mean(neg), detect5=det, bf16_effect=be32, fp16_effect=be16, ratio=be32 / be16, repeats=reps)
    print(f"{name:17s} AUC {auc:.3f}  same {statistics.mean(pos):.3f}  noise {statistics.mean(neg):.3f}  detected@5 {det:.0f}%  "
          f"batch effect BF16 {be32:.5f} FP16 {be16:.5f} ratio {be32 / be16:.1f}x  repeats {reps}")
(ROOT / "results").mkdir(exist_ok=True); (ROOT / "results" / "families.json").write_text(json.dumps(out, indent=1))
