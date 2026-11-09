"""Mechanism check: how large is the batch-size effect inside each precision? Compares each precision's batch-64 runs
with its own batch-1 repeat-0 run, token by token on matching prefixes. If batch effects come from rounding differences,
setups that compute in FP16 (10 mantissa bits) should show far smaller batch effects than BF16 (7 mantissa bits)."""
import glob, json, pathlib, statistics, sys
import numpy as np
ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT)); import analyze as A
import families
DATA = sys.argv[sys.argv.index("--data") + 1] if "--data" in sys.argv else "data-nocache"
HW = sys.argv[sys.argv.index("--hw") + 1] if "--hw" in sys.argv else "data-h100"
runs = A.load(sorted(glob.glob(str(ROOT / DATA / "*.jsonl"))))
compute = {"bf16": "BF16", "fp16": "FP16", "fp8": "FP8 dynamic, BF16 activations", "int8-gptq": "FP16", "int8-w8a16": "FP16",
           "int4-gptq": "FP16", "int4-awq": "FP16", "int4-bnb": "BF16, 4-bit weights",
           "int8-wo": "BF16, INT8 weights", "int4-llmc": "BF16, INT4 weights", "kv-fp8": "BF16, FP8 KV cache"}
out = {}
for c in [c for c in families.ORDER if (c, 1, 0) in runs]:
    base = runs[(c, 1, 0)]; shifts = []
    for k in [k for k in runs if k[0] == c and k[1] == 64]:
        for pid, rec in runs[k].items():
            if pid in base:
                shifts += [abs(x) for x in A.compare(rec, base[pid])[0]]
    out[c] = float(np.mean(shifts))
    print(f"{c:10s} arithmetic {compute[c]:32s} batch-64 vs own batch-1, mean |shift| {out[c]:.5f} nats over {len(shifts):,} tokens")
if 'fp16' in out: print(f"\nratio BF16 / FP16 batch effect: {out['bf16'] / out['fp16']:.1f}x")
pathlib.Path(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results" / f"mechanism-{DATA}.json").write_text(json.dumps(out, indent=1))
