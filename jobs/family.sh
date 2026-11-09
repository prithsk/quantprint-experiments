#!/usr/bin/env bash
# Every setup of one model family, then the paper's analyses on it.
#   bash run_gpu.sh jobs/family.sh llama3.1-8b      (also mistral-7b, or qwen2.5-32b on an 80 GB GPU)
set -e
F=${1:?give a family: llama3.1-8b, mistral-7b or qwen2.5-32b}
D=$(python -c "import families; print(families.data_dir('$F'))")
python check_models.py --family $F | tee logs/check-$F.txt
READY=$(grep '^READY:' logs/check-$F.txt | cut -d' ' -f2)
mkdir -p $D results
for c in ${READY//,/ }; do
  [ -f $D/$c.done ] && { echo "$c already collected"; continue; }; rm -f $D/$c.jsonl
  B=1,64; [ "$c" = bf16 ] && B=1,8,64
  python collect.py --family $F --config $c --batches $B --repeats 2 --probes 120 2>&1 | grep -E "probes in|wrote|Error|error" || true
  [ -s $D/$c.jsonl ] && touch $D/$c.done || { echo "STOP: $c produced no data"; exit 1; }
done
python analyze.py --data $D > results/report-$D.md 2>&1; sed -n '/## RQ1/,/## Systematic/p;/## Direction detector/,/## RQ2/p' results/report-$D.md
python mechanism.py --data $D
python robustness.py --data $D | tail -25
tar czf /workspace/results-$F.tar.gz $D results/*$D* && echo "Saved /workspace/results-$F.tar.gz"
