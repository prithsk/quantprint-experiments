# QuantPrint results: 15 runs, 120 probe prompts

## RQ1: shift from each precision against the noise floor

| Comparison | Mean abs shift, nats | AUC against noise |
|---|---|---|
| Same precision, repeat at batch 1 | 0.00000 | --- |
| Same precision, batch 1 vs 8 | 0.00676 | --- |
| Same precision, batch 1 vs 64 | 0.00672 | --- |
| BF16 vs FP16 (control) | 0.00667 | 0.617 |

## Systematic or random: agreement of per-token shifts across two runs at different batch sizes, the fair test

| Precision | Runs compared | Same sign | Correlation | Tokens compared |
|---|---|---|---|---|
| bf16 (noise) | batch 1 rep 2 vs batch 64 rep 0 | no shift | no shift | 0 |
| FP16 | batch 1 rep 2 vs batch 64 rep 0 | 96.6% | 0.988 | 5,900 |

## Systematic or random: agreement of per-token shifts across two runs at the same batch size

| Precision | Runs compared | Same sign | Correlation | Tokens compared |
|---|---|---|---|---|
| bf16 (noise) | batch 64 rep 0 vs batch 64 rep 1 | 82.1% | 0.746 | 5,223 |
| FP16 | batch 64 rep 0 vs batch 64 rep 1 | 98.8% | 0.995 | 5,905 |

## Detection power at a 1% false positive rate, by number of probe prompts

| Precision | 5 prompts | 10 prompts | 20 prompts | 50 prompts | 100 prompts | 200 prompts |
|---|---|---|---|---|---|---|
| FP16 | 0% | 2% | 2% | 17% | 38% | 83% |

## Direction detector: correlation with each precision's fingerprint, power at 1% false positives

| Precision | Mean score, same precision | Mean score, BF16 noise | 5 prompts | 10 prompts | 20 prompts | 50 prompts |
|---|---|---|---|---|---|---|
| FP16 | 0.987 | 0.624 | 100% | 100% | 100% | 100% |

## RQ2: attribution accuracy on a held-out batch size

| Probe prompts per decision | 1 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|
| Accuracy | 52.8% | 71.7% | 77.5% | 81.6% | 83.7% |

Confusion matrix at 50 prompts per decision, rows true and columns predicted:

| | BF16 | FP16 |
|---|---|---|
| BF16 | 860 | 340 |
| FP16 | 3 | 897 |
