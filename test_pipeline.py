"""Plant known effects in synthetic data and check that analyze.py recovers them: batch noise is random per run,
each precision adds a fixed per-token shift that repeats across runs, and larger shifts make generations diverge
earlier. A fake provider declares BF16 but serves INT4 AWQ and returns text tokens only, like a real API."""
import json, pathlib, random, shutil, sys
import numpy as np
ROOT = pathlib.Path(__file__).parent
sys.argv = [sys.argv[0]]
import analyze

P, T, K = 80, 40, 5
SIGMA = {"fp16": 0.004, "fp8": 0.02, "int8-gptq": 0.01, "int4-gptq": 0.06, "int4-awq": 0.05}
NOISE = 0.003
rng = np.random.default_rng(0)
base_tok = rng.integers(0, 32000, (P, T)); base_lp = -rng.uniform(0, 3, (P, T))
bias = {c: rng.normal(0, s, (P, T)) for c, s in SIGMA.items()}
bias["int4-awq"] += 0.02                                       # AWQ also drifts in one direction
alt = rng.integers(0, 32000, (P, T, K))

def record(cfg, b, rep, pid, text_only=False):
    noise = np.zeros(T) if b == 1 and rep == 1 and cfg == "bf16" else rng.normal(0, NOISE, T) if not (b == 1 and rep == 0) else np.zeros(T)
    shift = bias.get(cfg, np.zeros((P, T)))[pid] + noise
    lp = base_lp[pid] + shift
    toks = base_tok[pid].copy()
    big = np.abs(shift) > 0.08                                  # a large shift flips the greedy token there
    if big.any():
        cut = int(np.argmax(big)); toks[cut:] = rng.integers(0, 32000, T - cut)
    topk = [sorted([[int(toks[t]), float(lp[t])]] + [[int(x), float(lp[t] - 0.5 - j)] for j, x in enumerate(alt[pid, t, :K - 1])],
                   key=lambda z: -z[1]) for t in range(T)]
    rec = {"config": cfg, "batch": b, "rep": rep, "pid": pid, "tokens": [int(x) for x in toks],
           "text": [f"t{x}" for x in toks], "lp": [float(x) for x in lp], "topk": topk}
    if text_only:
        rec["tokens"] = []; rec["topk"] = [[[f"t{x[0]}", x[1]] for x in tk] for tk in topk]
    return rec

data = ROOT / "data"; shutil.rmtree(data, ignore_errors=True); data.mkdir()
for cfg in ["bf16"] + list(SIGMA):
    with open(data / f"{cfg}.jsonl", "w") as fh:
        for b in (1, 8, 64):
            for rep in (0, 1):
                if cfg != "bf16" and b == 1 and rep == 0:
                    pass
                for pid in range(P):
                    fh.write(json.dumps(record(cfg, b, rep, pid)) + "\n")
with open(data / "api_fake.jsonl", "w") as fh:
    for pid in range(40):
        r = record("int4-awq", 64, 0, pid, text_only=True); r["config"] = "api:FakeProvider"; r["declared"] = "bf16"
        fh.write(json.dumps(r) + "\n")

class A: audit = [str(data / "api_fake.jsonl")]
analyze.main(A)
report = (ROOT / "results" / "report.md").read_text()
rows = {l.split("|")[1].strip(): l for l in report.splitlines() if l.startswith("| ")}
auc = lambda name: float(rows[name].split("|")[3])
assert auc("BF16 vs INT4 GPTQ") > 0.95 and auc("BF16 vs FP8") > 0.9, "large shifts should separate cleanly from noise"
assert auc("BF16 vs FP16 (control)") < auc("BF16 vs INT4 GPTQ"), "the FP16 control should sit below INT4"
sys_rows = report.split("## Systematic")[1].split("##")[0]
noise_same = float([l for l in sys_rows.splitlines() if l.startswith("| BF16 (noise)") or l.startswith("| bf16 (noise)")][0].split("|")[2].strip("% "))
int4_same = float([l for l in sys_rows.splitlines() if l.startswith("| INT4 GPTQ")][0].split("|")[2].strip("% "))
assert int4_same > 85 and 35 < noise_same < 65, ("systematic shifts should agree in sign; noise should not", int4_same, noise_same)
acc_line = [l for l in report.splitlines() if l.startswith("| Accuracy")][0]
assert float(acc_line.split("|")[-2].strip(" %")) > 80, acc_line
audit_row = [l for l in report.splitlines() if l.startswith("| FakeProvider")][0]
assert "INT4 AWQ" in audit_row and "bf16" in audit_row, audit_row
shutil.rmtree(data); shutil.rmtree(ROOT / "results")
print("all tests passed")
