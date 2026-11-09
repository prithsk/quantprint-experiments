"""Tests for the scripts behind the remaining ICML experiments. Every test uses synthetic data with a known answer."""
import contextlib, io, json, pathlib, random, shutil, sys, types
import numpy as np
ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT))
src = (ROOT / "test_pipeline.py").read_text()
ns = {"__file__": str(ROOT / "test_pipeline.py"), "__name__": "gen"}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(src[:src.index("class A:")], "gen", "exec"), ns)           # writes synthetic reference runs to data/
record, P = ns["record"], ns["P"]
import families, other_setup, audit_score, rut, sglang_collect
QUANT = ["fp16", "fp8", "int8-gptq", "int4-gptq", "int4-awq"]

def write(folder, recs):
    d = ROOT / folder; shutil.rmtree(d, ignore_errors=True); d.mkdir()
    by = {}
    for r in recs: by.setdefault(r["config"], []).append(r)
    for c, rs in by.items():
        with open(d / f"{c.replace(':', '_')}.jsonl", "w") as fh:
            for r in rs: fh.write(json.dumps(r) + "\n")

def run(fn, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(types.SimpleNamespace(**kw))
    return buf.getvalue()

# 1. Another GPU: quantized runs with fresh noise must be detected, honest BF16 must stay near the 1% rate.
write("data-test-other", [record(c, b, rep, pid) for c in ["bf16"] + QUANT for b in (1, 64) for rep in (5, 6) for pid in range(P)])
run(other_setup.main, ref="data", other="data-test-other", alpha=0.01, ns="5,20")
res = json.loads((ROOT / "results" / "other-data-test-other.json").read_text())
pct = lambda s: float(s.strip("%"))
assert pct(res["detection"]["int4-awq"][1]) >= 90, res["detection"]
assert max(pct(v[1]) for v in res["false_alarms"].values()) <= 10, res["false_alarms"]
# Attribution scores only prompts that every fingerprint can score, as in the paper, and the synthetic INT4 GPTQ
# fingerprint diverges early, so check 5 prompts; the real data supports 50.
assert pct(res["attribution"]["int4-awq"][0]) >= 80 and pct(res["attribution"]["bf16"][0]) >= 80, res["attribution"]
print("other_setup: detection", res["detection"]["int4-awq"], "| worst BF16 false alarm", max(v[1] for v in res["false_alarms"].values()))

# 2. Audit: a provider serving INT4 AWQ while declaring BF16 is caught, an honest one is cleared, garbled tokens get no verdict.
def api(cfg, name, declared, garble=False):
    out = []
    for pid in range(P):
        r = record(cfg, 64, 9, pid, text_only=True); r["config"] = f"api:{name}"; r["declared"] = declared; r["batch"] = 0; r["rep"] = 0
        if garble: r["text"] = ["zz" + t for t in r["text"]]
        out.append(r)
    return out
d = ROOT / "data-test-audit"; shutil.rmtree(d, ignore_errors=True); d.mkdir()
for name, recs in (("FakeAWQ", api("int4-awq", "FakeAWQ", "bf16")), ("Honest", api("bf16", "Honest", "bf16")), ("Garbled", api("int4-awq", "Garbled", "bf16", True))):
    with open(d / f"api_{name}.jsonl", "w") as fh:
        for r in recs: fh.write(json.dumps(r) + "\n")
run(audit_score.main, ref="data", audit="data-test-audit", alpha=0.01, min_prompts=10)
aud = json.loads((ROOT / "results" / "audit.json").read_text())
assert aud["FakeAWQ"]["detected"] == "int4-awq", aud["FakeAWQ"]
assert aud["Honest"]["detected"] == "bf16", aud["Honest"]
assert aud["Garbled"]["detected"] is None, aud["Garbled"]
print("audit_score:", {k: v["detected"] for k, v in aud.items()}, "| FakeAWQ confidence", aud["FakeAWQ"]["confidence"])

# 3. RUT: identical distributions reject near alpha, shifted ones nearly always.
rng = np.random.default_rng(0); d = ROOT / "data-rut-test"; shutil.rmtree(d, ignore_errors=True); d.mkdir()
with open(d / "scores.jsonl", "w") as fh:
    for pid in range(300):
        for idx in range(100):
            fh.write(json.dumps({"config": "bf16", "role": "reference", "pid": pid, "idx": idx, "logrank": float(rng.normal())}) + "\n")
        fh.write(json.dumps({"config": "bf16", "role": "target", "pid": pid, "idx": 0, "logrank": float(rng.normal())}) + "\n")
        fh.write(json.dumps({"config": "int4-awq", "role": "target", "pid": pid, "idx": 0, "logrank": float(rng.normal(0.8, 1))}) + "\n")
run(rut.analyze, family="qwen2.5-7b", out="data-rut-test", quantprint=None, ns="20,50", prompts=300, gpu_mem=0.85)
r = json.loads((ROOT / "results" / "rut-qwen2.5-7b.json").read_text())
assert r["rut/bf16/0.05"][1] <= 12, r["rut/bf16/0.05"]
assert r["rut/int4-awq/0.05"][1] >= 90, r["rut/int4-awq/0.05"]
print("rut: null rejection", r["rut/bf16/0.05"], "| shifted power", r["rut/int4-awq/0.05"])

# 4. SGLang parser: the documented format parses, anything else stops with a message to paste.
item = {"text": " Hi there", "meta_info": {"output_token_logprobs": [[-0.1, 13, " Hi"], [-0.5, 99, " there"]],
        "output_top_logprobs": [[[-0.1, 13, " Hi"], [-2.3, 7, " Hello"]], [[-0.5, 99, " there"], [-1.0, 5, "!"]]]}}
toks, text, lp, topk = sglang_collect.parse(item, 5)
assert toks == [13, 99] and text == [" Hi", " there"] and lp == [-0.1, -0.5] and topk[0][0] == [13, -0.1], (toks, text, lp, topk)
try:
    sglang_collect.parse({"meta_info": {}}, 5); raise AssertionError("bad format should stop")
except SystemExit as e:
    assert "Paste this to Claude" in str(e)
print("sglang parser: ok")

assert families.data_dir("qwen2.5-7b") == "data-nocache" and families.data_dir("llama3.1-8b") == "data-llama3.1-8b"
for f in ("data", "data-test-other", "data-test-audit", "data-rut-test", "results"):
    shutil.rmtree(ROOT / f, ignore_errors=True)
print("all tests passed")
