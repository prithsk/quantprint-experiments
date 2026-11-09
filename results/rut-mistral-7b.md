# RUT and QuantPrint on mistral-7b

RUT: 300 prompts with 100 reference responses each. A BF16 target row is a false positive rate, and every other row is detection power.

## alpha = 5%

| Setup | Test | 5 prompts | 10 prompts | 20 prompts | 50 prompts | 100 prompts |
|---|---|---|---|---|---|---|
| BF16 | RUT | 4% | 5% | 4% | 3% | 2% |
| FP16 | RUT | 6% | 7% | 6% | 3% | 3% |
| FP16 | QuantPrint direction | 100% | 100% | 100% | 100% | 100% |

## alpha = 1%

| Setup | Test | 5 prompts | 10 prompts | 20 prompts | 50 prompts | 100 prompts |
|---|---|---|---|---|---|---|
| BF16 | RUT | 1% | 2% | 1% | 1% | 0% |
| FP16 | RUT | 1% | 1% | 2% | 0% | 0% |
| FP16 | QuantPrint direction | 100% | 100% | 100% | 100% | 100% |
