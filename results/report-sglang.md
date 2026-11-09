# QuantPrint results: 30 runs, 120 probe prompts

## RQ1: shift from each precision against the noise floor

| Comparison | Mean abs shift, nats | AUC against noise |
|---|---|---|
| Same precision, repeat at batch 1 | 0.00000 | --- |
| Same precision, batch 1 vs 64 | 0.00990 | --- |
| BF16 vs FP16 (control) | 0.00964 | 0.685 |
| BF16 vs INT8 GPTQ | 0.01107 | 0.743 |
| BF16 vs INT4 GPTQ | 0.05820 | 0.961 |
| BF16 vs INT4 AWQ | 0.06314 | 0.967 |

## Systematic or random: agreement of per-token shifts across two runs at different batch sizes, the fair test

| Precision | Runs compared | Same sign | Correlation | Tokens compared |
|---|---|---|---|---|
| bf16 (noise) | batch 1 rep 2 vs batch 64 rep 0 | no shift | no shift | 0 |
| FP16 | batch 1 rep 2 vs batch 64 rep 0 | 96.2% | 0.989 | 4,556 |
| INT8 GPTQ | batch 1 rep 2 vs batch 64 rep 0 | 97.0% | 0.993 | 4,508 |
| INT4 GPTQ | batch 1 rep 2 vs batch 64 rep 0 | 99.6% | 1.000 | 1,383 |
| INT4 AWQ | batch 1 rep 2 vs batch 64 rep 0 | 99.9% | 1.000 | 1,228 |

## Systematic or random: agreement of per-token shifts across two runs at the same batch size

| Precision | Runs compared | Same sign | Correlation | Tokens compared |
|---|---|---|---|---|
| bf16 (noise) | batch 64 rep 0 vs batch 64 rep 1 | 83.2% | 0.762 | 4,368 |
| FP16 | batch 64 rep 0 vs batch 64 rep 1 | 96.6% | 0.990 | 4,534 |
| INT8 GPTQ | batch 64 rep 0 vs batch 64 rep 1 | 96.9% | 0.992 | 4,516 |
| INT4 GPTQ | batch 64 rep 0 vs batch 64 rep 1 | 99.5% | 1.000 | 1,390 |
| INT4 AWQ | batch 64 rep 0 vs batch 64 rep 1 | 99.8% | 1.000 | 1,244 |

## Detection power at a 1% false positive rate, by number of probe prompts

| Precision | 5 prompts | 10 prompts | 20 prompts | 50 prompts | 100 prompts | 200 prompts |
|---|---|---|---|---|---|---|
| FP16 | 3% | 9% | 15% | 75% | 100% | 100% |
| INT8 GPTQ | 15% | 38% | 73% | 100% | 100% | 100% |
| INT4 GPTQ | 100% | 100% | 100% | 100% | 100% | 100% |
| INT4 AWQ | 100% | 100% | 100% | 100% | 100% | 100% |

## Direction detector: correlation with each precision's fingerprint, power at 1% false positives

| Precision | Mean score, same precision | Mean score, BF16 noise | 5 prompts | 10 prompts | 20 prompts | 50 prompts |
|---|---|---|---|---|---|---|
| FP16 | 0.987 | 0.523 | 100% | 100% | 100% | 100% |
| INT8 GPTQ | 0.990 | 0.416 | 100% | 100% | 100% | 100% |
| INT4 GPTQ | 1.000 | -0.012 | 100% | 100% | 100% | 100% |
| INT4 AWQ | 1.000 | 0.008 | 100% | 100% | 100% | 100% |

## RQ2: attribution accuracy on a held-out batch size

| Probe prompts per decision | 1 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|
| Accuracy | 43.8% | 52.2% | 65.7% | 66.4% | 80.8% |

Confusion matrix at 50 prompts per decision, rows true and columns predicted:

| | BF16 | FP16 | INT4 AWQ | INT4 GPTQ | INT8 GPTQ |
|---|---|---|---|---|---|
| BF16 | 300 | 266 | 0 | 0 | 184 |
| FP16 | 40 | 793 | 0 | 0 | 67 |
| INT4 AWQ | 0 | 0 | 821 | 79 | 0 |
| INT4 GPTQ | 0 | 0 | 99 | 801 | 0 |
| INT8 GPTQ | 33 | 68 | 0 | 0 | 799 |
