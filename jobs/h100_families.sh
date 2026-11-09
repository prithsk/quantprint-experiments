#!/usr/bin/env bash
# Mistral 7B, Llama 3.1 8B and Qwen2.5-14B with quantized setups, on one GPU with native FP8, best an L40S. Each family
# gets BF16, FP16, FP8, INT8, INT4 and an FP8 KV cache, plus RUT on Mistral. About 5 hours on an L40S. A 24 GB GPU
# such as an RTX 4090 runs the 7B and 8B families and skips the 14B one.
# Data lands in folders named after the GPU, for example data-mistral-7b-4090.
#   bash run_gpu.sh jobs/h100_families.sh
# Muse Glimmer needs an 80 GB GPU. To include it on an H100, run: export FAMILIES="mistral-7b llama3.1-8b muse-glimmer-30b"
# Llama is gated: accept its license on Hugging Face first, or the job skips it and runs the rest.
set -e
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
TAG=$(echo "$GPU" | grep -oE "H100|H200|B200|L40S|L40|4090|5090|L4|6000 Ada|PRO 6000" | head -1 | tr -d ' ' | tr 'A-Z' 'a-z')
[ -n "$TAG" ] || { echo "STOP: $GPU has no native FP8. Use an L40S, L40, RTX 6000 Ada, RTX PRO 6000, RTX 4090, RTX 5090 or H100."; exit 1; }
export DATA_SUFFIX=-$TAG
echo "GPU $GPU: data folders end in $DATA_SUFFIX"
[ -L ~/.cache/huggingface ] || { [ -d /workspace/.cache/huggingface ] && { rm -rf ~/.cache/huggingface; mkdir -p ~/.cache; ln -s /workspace/.cache/huggingface ~/.cache/huggingface; }; } || true
clean() { rm -rf ~/.cache/huggingface/hub/models--* 2>/dev/null; echo "cleared model cache"; }
step() { echo; echo "=================== $* ==================="; if bash "$@"; then echo "OK: $*"; else echo "FAILED: $* (continuing)"; fi; }
MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
for F in ${FAMILIES:-mistral-7b llama3.1-8b qwen2.5-14b}; do
  if [ $F = qwen2.5-14b ] && [ "$MEM" -lt 40000 ]; then echo "SKIP: Qwen2.5-14B needs a GPU with 40 GB or more"; continue; fi
  if [ $F = muse-glimmer-30b ] && ! nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk '{exit !($1 > 70000)}'; then
    echo "SKIP: Muse Glimmer needs an 80 GB GPU"; continue
  fi
  python -c "import families,sys; sys.exit(0 if 'int4-llmc' in families.configs('$F') else 1)" && step jobs/quantize.sh $F
  step jobs/family.sh $F
  [ $F = mistral-7b ] && step jobs/rut.sh $F
  clean
done
echo; echo "Results in /workspace:"; ls -la /workspace/results-*.tar.gz 2>/dev/null
