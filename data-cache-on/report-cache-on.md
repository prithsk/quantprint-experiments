# QuantPrint results: 26 runs, 200 probe prompts

## RQ1: shift from each precision against the noise floor

| Comparison | Mean abs shift, nats | AUC against noise |
|---|---|---|
| Same precision, repeat at batch 1 | 0.00226 | --- |
| Same precision, batch 1 vs 8 | 0.00904 | --- |
| Same precision, batch 1 vs 64 | 0.00927 | --- |
| BF16 vs FP16 (control) | 0.00904 | 0.569 |
| BF16 vs FP8 | 0.03344 | 0.932 |
| BF16 vs INT8 GPTQ | 0.01080 | 0.684 |
| BF16 vs INT4 GPTQ | 0.05707 | 0.928 |
| BF16 vs INT4 AWQ | 0.06149 | 0.952 |

## Systematic or random: agreement of per-token shifts across two runs at different batch sizes, the fair test

| Precision | Runs compared | Same sign | Correlation | Tokens compared |
|---|---|---|---|---|
| bf16 (noise) | batch 1 rep 1 vs batch 64 rep 0 | 67.0% | 0.496 | 1,844 |
| FP16 | batch 1 rep 1 vs batch 64 rep 0 | 96.9% | 0.988 | 7,935 |
| FP8 | batch 1 rep 1 vs batch 64 rep 0 | 71.6% | 0.582 | 3,678 |
| INT8 GPTQ | batch 1 rep 1 vs batch 64 rep 0 | 97.2% | 0.993 | 7,549 |
| INT4 GPTQ | batch 1 rep 1 vs batch 64 rep 0 | 99.9% | 1.000 | 1,334 |
| INT4 AWQ | batch 1 rep 1 vs batch 64 rep 0 | 99.8% | 1.000 | 1,243 |

## Systematic or random: agreement of per-token shifts across two runs at the same batch size

| Precision | Runs compared | Same sign | Correlation | Tokens compared |
|---|---|---|---|---|
| bf16 (noise) | batch 64 rep 0 vs batch 64 rep 1 | 71.5% | 0.563 | 6,905 |
| FP16 | batch 64 rep 0 vs batch 64 rep 1 | 97.2% | 0.989 | 7,912 |
| FP8 | batch 64 rep 0 vs batch 64 rep 1 | 85.5% | 0.814 | 4,154 |
| INT8 GPTQ | batch 64 rep 0 vs batch 64 rep 1 | 97.8% | 0.994 | 7,488 |
| INT4 GPTQ | batch 64 rep 0 vs batch 64 rep 1 | 99.7% | 1.000 | 1,337 |
| INT4 AWQ | batch 64 rep 0 vs batch 64 rep 1 | 99.9% | 1.000 | 1,244 |

## Detection power at a 1% false positive rate, by number of probe prompts

| Precision | 5 prompts | 10 prompts | 20 prompts | 50 prompts | 100 prompts | 200 prompts |
|---|---|---|---|---|---|---|
| FP16 | 0% | 1% | 2% | 5% | 9% | 21% |
| FP8 | 100% | 100% | 100% | 100% | 100% | 100% |
| INT8 GPTQ | 15% | 32% | 65% | 96% | 100% | 100% |
| INT4 GPTQ | 100% | 100% | 100% | 100% | 100% | 100% |
| INT4 AWQ | 100% | 100% | 100% | 100% | 100% | 100% |

## RQ2: attribution accuracy on a held-out batch size

| Probe prompts per decision | 1 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|
| Accuracy | 49.1% | 69.0% | 76.0% | 84.2% | 90.3% |

Confusion matrix at 50 prompts per decision, rows true and columns predicted:

| | BF16 | FP16 | FP8 | INT4 AWQ | INT4 GPTQ | INT8 GPTQ |
|---|---|---|---|---|---|---|
| BF16 | 659 | 89 | 0 | 0 | 0 | 2 |
| FP16 | 112 | 468 | 0 | 0 | 0 | 20 |
| FP8 | 0 | 0 | 597 | 0 | 0 | 3 |
| INT4 AWQ | 0 | 0 | 0 | 540 | 60 | 0 |
| INT4 GPTQ | 0 | 0 | 0 | 57 | 543 | 0 |
| INT8 GPTQ | 5 | 14 | 0 | 0 | 0 | 581 |
