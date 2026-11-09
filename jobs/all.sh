#!/usr/bin/env bash
# Every GPU experiment, one after another, on one 80 GB H100 or A100. About 5 to 6 hours. A failed step is logged and
# the rest still run. Model files are deleted between families so the disk never fills.
#   export HF_TOKEN=hf_...   then   bash run_gpu.sh jobs/all.sh
clean() { rm -rf ~/.cache/huggingface/hub/models--* 2>/dev/null; echo "cleared model cache"; }
step() { echo; echo "=================== $* ==================="; if bash "$@"; then echo "OK: $*"; else echo "FAILED: $* (continuing)"; fi; }
step jobs/h100_quant.sh
step jobs/rut.sh qwen2.5-7b;   clean
step jobs/family.sh llama3.1-8b; clean
step jobs/family.sh mistral-7b
step jobs/rut.sh mistral-7b;   clean
step jobs/family.sh qwen2.5-32b; clean
step jobs/sglang.sh
echo; echo "Results in /workspace:"; ls -la /workspace/results-*.tar.gz 2>/dev/null
