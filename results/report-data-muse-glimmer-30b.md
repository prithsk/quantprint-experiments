# QuantPrint results: 15 runs, 120 probe prompts

## RQ1: shift from each precision against the noise floor

| Comparison | Mean abs shift, nats | AUC against noise |
|---|---|---|
| Same precision, repeat at batch 1 | 0.00000 | --- |
| Same precision, batch 1 vs 8 | 0.00592 | --- |
| Same precision, batch 1 vs 64 | 0.00612 | --- |
| BF16 vs FP16 (control) | 0.00575 | 0.604 |

## Systematic or random: agreement of per-token shifts across two runs at different batch sizes, the fair test

| Precision | Runs compared | Same sign | Correlation | Tokens compared |
|---|---|---|---|---|
| bf16 (noise) | batch 1 rep 2 vs batch 64 rep 0 | no shift | no shift | 0 |
| FP16 | batch 1 rep 2 vs batch 64 rep 0 | 96.6% | 0.989 | 3,723 |

## Systematic or random: agreement of per-token shifts across two runs at the same batch size

| Precision | Runs compared | Same sign | Correlation | Tokens compared |
|---|---|---|---|---|
| bf16 (noise) | batch 64 rep 0 vs batch 64 rep 1 | 74.5% | 0.580 | 3,229 |
| FP16 | batch 64 rep 0 vs batch 64 rep 1 | 97.8% | 0.992 | 3,801 |

## Detection power at a 1% false positive rate, by number of probe prompts

| Precision | 5 prompts | 10 prompts | 20 prompts | 50 prompts | 100 prompts | 200 prompts |
|---|---|---|---|---|---|---|
| FP16 | 1% | 1% | 2% | 18% | 43% | 94% |

## Direction detector: correlation with each precision's fingerprint, power at 1% false positives

| Precision | Mean score, same precision | Mean score, BF16 noise | 5 prompts | 10 prompts | 20 prompts | 50 prompts |
|---|---|---|---|---|---|---|
| FP16 | 0.986 | 0.540 | 100% | 100% | 100% | 100% |

## RQ2: attribution accuracy on a held-out batch size

| Probe prompts per decision | 1 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|
| Accuracy | 36.2% | 55.2% | 63.6% | 71.8% | 75.9% |

Confusion matrix at 50 prompts per decision, rows true and columns predicted:

| | BF16 | FP16 |
|---|---|---|
| BF16 | 744 | 456 |
| FP16 | 50 | 850 |
