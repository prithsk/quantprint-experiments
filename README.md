# quantprint-experiments

The experiments behind "Above the Noise Floor: Identifying Quantized LLM Serving from Log-Probabilities."

Model: Qwen2.5-7B-Instruct, chosen because Qwen publishes official quantized versions of the same model, so every
configuration has known ground truth, and because its Apache license needs no access approval.

| Configuration | Checkpoint | Arithmetic |
|---|---|---|
| bf16, the reference | Qwen/Qwen2.5-7B-Instruct | BF16 |
| fp16, a control | Qwen/Qwen2.5-7B-Instruct | FP16 |
| fp8 | Qwen/Qwen2.5-7B-Instruct with vLLM FP8 | FP8 |
| int8-gptq | Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8 | FP16 |
| int4-gptq | Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4 | FP16 |
| int4-awq | Qwen/Qwen2.5-7B-Instruct-AWQ | FP16 |

The FP16 control matters: the quantized checkpoints compute in FP16 while the reference computes in BF16, so without
it part of every measured "quantization shift" could just be FP16 arithmetic.

## What each script produces

| Script | Paper section | Output |
|---|---|---|
| `collect.py` | Data for RQ1 and RQ2 | `data/<config>.jsonl`: greedy top-5 log-probabilities for 200 probe prompts per run |
| `analyze.py` | RQ1, RQ2 | `results/report.md`, `results/rq1_table.tex` for Table 3, `results/accuracy_vs_queries.png` |
| `audit_openrouter.py` | RQ3 | `data/api_<provider>.jsonl`, then `analyze.py --audit` writes `results/rq3_table.tex` for Table 4 |

`analyze.py` also reports two things the paper needs: whether each precision shifts tokens the same way in
independent runs, which batch noise should not, and detection power against the number of probe prompts.

## Run it on a GPU

Needs a GPU with at least 24 GB and BF16 support: an A100 or L4 on Colab, or a rented one. A T4 is too small.

```
git clone https://github.com/prithsk/quantprint-experiments && cd quantprint-experiments
pip install -r requirements-gpu.txt
python collect.py --config bf16 --batches 1,8,64 --repeats 2
python collect.py --config fp16 --batches 1,64 --repeats 2
python collect.py --config fp8 --batches 1,64 --repeats 2
python collect.py --config int8-gptq --batches 1,64 --repeats 2
python collect.py --config int4-gptq --batches 1,64 --repeats 2
python collect.py --config int4-awq --batches 1,64 --repeats 2
python analyze.py
```

Expect about 10 to 20 minutes per configuration after the first model download. On Colab, copy `data/` to Google
Drive when done, since the Colab disk does not persist.

## RQ3: audit real providers

```
cp .env.example .env            # add OPENROUTER_API_KEY
python audit_openrouter.py --model qwen/qwen-2.5-7b-instruct --probes 50
python analyze.py --audit data/api_*.jsonl
```

The audit uses the same probe prompts, routes each request to one provider with fallbacks off, and compares the
precision each provider declares on OpenRouter with the precision QuantPrint detects. If no provider of this model
returns log-probabilities, the audit needs a different model, and that model's local runs must be collected first.

## Hardware test

Honest BF16 served on a different GPU should not be flagged. Collect BF16 on another GPU into its own folder and
score it against the reference GPU's fingerprints:

```
python collect.py --config bf16 --batches 1,64 --repeats 2 --probes 120 --out data-a100
python analyze.py --data data-nocache --hardware data-a100
```

The report gives how often each test falsely flags the other GPU. About 1% means hardware does not matter; far more
means each GPU leaves its own fingerprint. `python test_hardware.py` checks that the scoring catches a planted
hardware shift and stays near 1% without one.

## Not covered yet

The paper's vLLM against SGLang row needs a second collector for SGLang, which installs in a separate environment.
Until then, the engine comparison stays a placeholder.

## Tests

`python test_pipeline.py` runs without a GPU. It plants known effects in synthetic data and checks the analysis
recovers them: random batch noise agrees in sign about half the time, systematic shifts agree far more often,
attribution holds on a held-out batch size, and a fake provider that declares BF16 but serves INT4 AWQ is caught.

## License

MIT for the code. Probe prompts come from Databricks Dolly 15k, CC BY-SA 3.0.
