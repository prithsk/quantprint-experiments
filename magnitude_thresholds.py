"""Table 6 magnitude rows and the RQ3 discussion: the magnitude test with a threshold from all BF16 noise runs against
one from batched noise only, batch 8 and 64. Reports detection power for each precision and false alarms on H100 BF16."""
import sys, random, statistics, glob
import pathlib
ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT)); import analyze as A
import numpy as np
runs = A.load(sorted(glob.glob(str(ROOT / "data-nocache" / "*.jsonl")))); ref = runs[A.REF]
hw = {k: v for k, v in A.load([str(ROOT / "data-h100" / "bf16.jsonl")]).items() if k[0] == "bf16"}
def S_of(keys, src): return [s for k in keys for s in A.per_prompt_S(src[k], ref).values()]
noise_all = [k for k in runs if k[0] == "bf16" and k != A.REF]
noise_batched = [k for k in noise_all if k[1] != 1]
ns = (5, 10, 20, 50); alpha, trials = 0.01, 2000
def table(noise_keys, label):
    rng = random.Random(0); neg = S_of(noise_keys, runs)
    tau = {n: sorted(statistics.median(rng.sample(neg, n)) for _ in range(trials))[int((1 - alpha) * trials) - 1] for n in ns}
    print(f"\nMagnitude test, threshold from {label} ({len(neg)} per-prompt values)")
    for cfg in ["fp16", "fp8", "int8-gptq", "int4-gptq", "int4-awq"]:
        S = S_of([k for k in runs if k[0] == cfg], runs)
        print(f"  {cfg:10s} power " + "  ".join(f"{n}:{100*np.mean([statistics.median(rng.sample(S, n)) > tau[n] for _ in range(500)]):.0f}%" for n in ns))
    H = S_of(list(hw), hw)
    print(f"  {'H100 BF16':10s} FALSE alarms " + "  ".join(f"{n}:{100*np.mean([statistics.median(rng.sample(H, n)) > tau[n] for _ in range(500)]):.0f}%" for n in ns))
table(noise_all, "all BF16 noise runs, as in the report")
table(noise_batched, "batched noise only, batch 8 and 64")
# How do the H100 shifts compare with L4 batched noise per prompt?
H = S_of(list(hw), hw); Nb = S_of(noise_batched, runs)
print(f"\nmedian per-prompt shift: L4 batched noise {statistics.median(Nb):.5f}, H100 BF16 {statistics.median(H):.5f}")
print("H100 runs:", sorted(hw))
