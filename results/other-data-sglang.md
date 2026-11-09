# Other setup: data-sglang scored against data-nocache

Fingerprints: FP16, FP8, INT8 GPTQ, INT4 GPTQ, INT4 AWQ. False positive rate 1% per test.

## Detection of each quantized setup by its own fingerprint

| Setup | runs | mean score | 5 prompts | 20 prompts | 50 prompts |
|---|---|---|---|---|---|
| FP16 | 6 | 0.984 | 100% | 100% | 100% |
| INT8 GPTQ | 6 | 0.989 | 100% | 100% | 100% |
| INT4 GPTQ | 6 | 1.000 | 100% | 100% | 100% |
| INT4 AWQ | 6 | 0.999 | 100% | 100% | 100% |

## False alarms on honest BF16 from the other setup

| Fingerprint | 5 prompts | 20 prompts | 50 prompts |
|---|---|---|---|
| FP16 | 2% | 5% | 3% |
| FP8 | 1% | 0% | 0% |
| INT8 GPTQ | 1% | 2% | 1% |
| INT4 GPTQ | 3% | 4% | 4% |
| INT4 AWQ | 4% | 8% | 24% |

## Attribution with a Bonferroni threshold, 0.20% per fingerprint

| True setup | 5 prompts | 20 prompts | 50 prompts |
|---|---|---|---|
| BF16 | 96.8% | 96.2% | 98.9% |
| FP16 | 100.0% | 100.0% | 100.0% |
| INT8 GPTQ | 100.0% | 100.0% | 100.0% |
| INT4 GPTQ | 100.0% | 100.0% | 100.0% |
| INT4 AWQ | 100.0% | 100.0% | 100.0% |