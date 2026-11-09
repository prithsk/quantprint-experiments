# RUT and QuantPrint on qwen2.5-7b

RUT: 300 prompts with 100 reference responses each. A BF16 target row is a false positive rate, and every other row is detection power.

## alpha = 5%

| Setup | Test | 5 prompts | 10 prompts | 20 prompts | 50 prompts | 100 prompts |
|---|---|---|---|---|---|---|
| BF16 | RUT | 5% | 5% | 8% | 13% | 22% |
| FP16 | RUT | 5% | 6% | 7% | 12% | 15% |
| FP16 | QuantPrint direction | 100% | 100% | 100% | 100% | 100% |
| INT8 GPTQ | RUT | 7% | 3% | 5% | 4% | 4% |
| INT8 GPTQ | QuantPrint direction | 100% | 100% | 100% | 100% | 100% |
| INT4 GPTQ | RUT | 30% | 51% | 83% | 100% | 100% |
| INT4 GPTQ | QuantPrint direction | 100% | 100% | 100% | 100% | 100% |
| INT4 AWQ | RUT | 36% | 59% | 85% | 100% | 100% |
| INT4 AWQ | QuantPrint direction | 100% | 100% | 100% | 100% | 100% |

## alpha = 1%

| Setup | Test | 5 prompts | 10 prompts | 20 prompts | 50 prompts | 100 prompts |
|---|---|---|---|---|---|---|
| BF16 | RUT | 1% | 1% | 1% | 3% | 5% |
| FP16 | RUT | 1% | 1% | 1% | 3% | 5% |
| FP16 | QuantPrint direction | 100% | 100% | 100% | 100% | 100% |
| INT8 GPTQ | RUT | 1% | 0% | 1% | 0% | 1% |
| INT8 GPTQ | QuantPrint direction | 100% | 100% | 100% | 100% | 100% |
| INT4 GPTQ | RUT | 12% | 31% | 62% | 98% | 100% |
| INT4 GPTQ | QuantPrint direction | 100% | 100% | 100% | 100% | 100% |
| INT4 AWQ | RUT | 14% | 34% | 73% | 100% | 100% |
| INT4 AWQ | QuantPrint direction | 100% | 100% | 100% | 100% | 100% |
