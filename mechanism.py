"""Mechanism check: how large is the batch-size effect inside each precision? Compares each precision's batch-64 runs
with its own batch-1 repeat-0 run, token by token on matching prefixes. If batch effects come from rounding differences,
setups that compute in FP16 (10 mantissa bits) should show far smaller batch effects than BF16 (7 mantissa bits)."""
import glob, json, pathlib, statistics, sys
import numpy as np
ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT)); import analyze as A
runs = A.load(sorted(glob.glob(str(ROOT / "data-nocache" / "*.jsonl"))))
compute = {"bf16": "BF16", "fp16": "FP16", "fp8": "FP8 dynamic, BF16 activations", "int8-gptq": "FP16", "int4-gptq": "FP16", "int4-awq": "FP16"}
out = {}
for c in ["bf16", "fp16", "fp8", "int8-gptq", "int4-gptq", "int4-awq"]:
    base = runs[(c, 1, 0)]; shifts = []
    for k in [k for k in runs if k[0] == c and k[1] == 64]:
        for pid, rec in runs[k].items():
            if pid in base:
                shifts += [abs(x) for x in A.compare(rec, base[pid])[0]]
    out[c] = float(np.mean(shifts))
    print(f"{c:10s} arithmetic {compute[c]:32s} batch-64 vs own batch-1, mean |shift| {out[c]:.5f} nats over {len(shifts):,} tokens")
print(f"\nratio BF16 / FP16 batch effect: {out['bf16'] / out['fp16']:.1f}x")
pathlib.Path(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results" / "mechanism.json").write_text(json.dumps(out, indent=1))
