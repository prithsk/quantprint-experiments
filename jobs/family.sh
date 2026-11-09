#!/usr/bin/env bash
# Every setup of one model family, then the paper's analyses on it.
#   bash run_gpu.sh jobs/family.sh muse-glimmer-30b   (80 GB GPU; also mistral-7b, llama3.1-8b or qwen2.5-32b)
set -e
F=${1:?give a family: llama3.1-8b, mistral-7b or qwen2.5-32b}
D=$(python -c "import families; print(families.data_dir('$F'))")
mkdir -p logs; python check_models.py --family $F | tee logs/check-$F.txt
READY=$(grep '^READY:' logs/check-$F.txt | cut -d' ' -f2)
mkdir -p $D results
for c in ${READY//,/ }; do
  [ -f $D/$c.done ] && { echo "$c already collected"; continue; }; rm -f $D/$c.jsonl
  B=1,64; [ "$c" = bf16 ] && B=1,8,64
  python collect.py --family $F --config $c --batches $B --repeats ${REPEATS:-2} --probes 120 2>&1 | grep -E "probes in|wrote|Error|error" || true
  if [ -s $D/$c.jsonl ]; then touch $D/$c.done
  elif [ "$c" = bf16 ]; then echo "STOP: the BF16 reference produced no data, and every other setup depends on it"; exit 1
  else echo "SKIP: $c produced no data, so the analysis leaves it out; its error is in the lines above"; fi
done
python analyze.py --data $D > results/report-$D.md 2>&1; sed -n '/## RQ1/,/## Systematic/p;/## Direction detector/,/## RQ2/p' results/report-$D.md
python mechanism.py --data $D
python robustness.py --data $D | tail -25
tar czf /workspace/results-$F${DATA_SUFFIX:-}.tar.gz $D results/*$D* && echo "Saved /workspace/results-$F.tar.gz"
