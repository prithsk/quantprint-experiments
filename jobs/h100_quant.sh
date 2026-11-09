#!/usr/bin/env bash
# Quantized Qwen 7B setups on this GPU, scored against the L4 fingerprints. About 25 minutes on an H100.
set -e
TAG=$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -oE "H100|A100|H200|B200|L40S|A6000|A40" | head -1 | tr 'A-Z' 'a-z')
[ -n "$TAG" ] || { echo "STOP: use an H100 or A100, not $(nvidia-smi --query-gpu=name --format=csv,noheader)"; exit 1; }
OUT=data-${TAG}q; mkdir -p $OUT
CONFIGS="fp16 fp8 int8-gptq int4-gptq int4-awq"
if [ "$TAG" = h100 ] && [ -s data-h100/bf16.jsonl ]; then cp data-h100/bf16.jsonl $OUT/; touch $OUT/bf16.done; else CONFIGS="bf16 $CONFIGS"; fi
for c in $CONFIGS; do
  [ -f $OUT/$c.done ] && { echo "$c already collected"; continue; }; rm -f $OUT/$c.jsonl
  python collect.py --family qwen2.5-7b --config $c --batches 1,64 --repeats 2 --probes 120 --out $OUT 2>&1 | grep -E "probes in|wrote|Error|error" || true
  [ -s $OUT/$c.jsonl ] && touch $OUT/$c.done || { echo "STOP: $c produced no data"; exit 1; }
done
python other_setup.py --ref data-nocache --other $OUT
tar czf /workspace/results-$OUT.tar.gz $OUT results/other-$OUT.* && echo "Saved /workspace/results-$OUT.tar.gz"
