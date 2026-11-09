#!/usr/bin/env bash
# Llama 3.1 8B, Mistral and Muse Glimmer on an H100: BF16, FP16, FP8, INT8, INT4 and an FP8 KV cache, all on the same
# GPU in folders that end in -h100, plus RUT on Mistral's new setups. About 5 hours. Use an H100, since FP8 needs it.
# Llama is gated: accept its license on Hugging Face first, or the job skips it and runs the rest.
#   bash run_gpu.sh jobs/h100_families.sh
set -e
nvidia-smi --query-gpu=name --format=csv,noheader | grep -q H100 || { echo "STOP: this job needs an H100 for native FP8"; exit 1; }
export DATA_SUFFIX=-h100
[ -L ~/.cache/huggingface ] || { [ -d /workspace/.cache/huggingface ] && { rm -rf ~/.cache/huggingface; mkdir -p ~/.cache; ln -s /workspace/.cache/huggingface ~/.cache/huggingface; }; } || true
clean() { rm -rf ~/.cache/huggingface/hub/models--* 2>/dev/null; echo "cleared model cache"; }
step() { echo; echo "=================== $* ==================="; if bash "$@"; then echo "OK: $*"; else echo "FAILED: $* (continuing)"; fi; }
for F in llama3.1-8b mistral-7b muse-glimmer-30b; do
  python -c "import families,sys; sys.exit(0 if 'int4-llmc' in families.configs('$F') else 1)" && step jobs/quantize.sh $F
  step jobs/family.sh $F
  [ $F = mistral-7b ] && step jobs/rut.sh $F
  clean
done
echo; echo "Results in /workspace:"; ls -la /workspace/results-*.tar.gz 2>/dev/null
