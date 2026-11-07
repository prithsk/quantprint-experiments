"""The hardware test must stay near its false positive rate when other hardware adds no shift, and must flag runs when
other hardware adds its own systematic shift, like FP16 does. Reuses the synthetic generator from test_pipeline."""
import json, pathlib, shutil, sys, io, contextlib
import numpy as np
ROOT = pathlib.Path(__file__).parent
src = (ROOT / "test_pipeline.py").read_text()
gen = src[:src.index("class A:")]                              # synthetic data generator only, no assertions
ns = {"__file__": str(ROOT / "test_pipeline.py"), "__name__": "gen"}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(gen.replace("import analyze", "import analyze as _a"), "gen", "exec"), ns)
import analyze

def make_hw(folder, hw_bias):
    d = ROOT / folder; shutil.rmtree(d, ignore_errors=True); d.mkdir()
    with open(d / "bf16.jsonl", "w") as fh:
        for b in (1, 64):
            for rep in (0, 1):
                for pid in range(ns["P"]):
                    r = ns["record"]("bf16", b, rep, pid)
                    if hw_bias is not None:
                        r["lp"] = [x + float(y) for x, y in zip(r["lp"], hw_bias[pid])]
                    fh.write(json.dumps(r) + "\n")

def report_for(folder):
    class A: audit = None; data = "data"; hardware = folder
    with contextlib.redirect_stdout(io.StringIO()):
        analyze.main(A)
    rep = (ROOT / "results" / "report.md").read_text()
    sec = rep.split("## Hardware")[1].split("## ")[0]
    row = lambda name: [float(x.strip(" %")) for x in [l for l in sec.splitlines() if l.startswith("| " + name)][0].split("|")[2:-1]]
    return row("Magnitude"), row("Direction, FP16")

make_hw("data-hw-clean", None)
mag, fp16 = report_for("data-hw-clean")
assert max(mag) <= 10 and max(fp16) <= 10, ("no hardware effect should stay near the 1% rate", mag, fp16)
make_hw("data-hw-biased", ns["bias"]["fp16"] * 1.0)              # hardware that shifts like FP16
mag_b, fp16_b = report_for("data-hw-biased")
assert fp16_b[0] >= 90, ("a systematic hardware shift like FP16 must be flagged", fp16_b)
print(f"clean hardware: magnitude {mag}, FP16 fingerprint {fp16}")
print(f"FP16-like hardware: magnitude {mag_b}, FP16 fingerprint {fp16_b}")
for f in ("data", "data-hw-clean", "data-hw-biased", "results"):
    shutil.rmtree(ROOT / f, ignore_errors=True)
print("all tests passed")
