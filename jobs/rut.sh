#!/usr/bin/env bash
# The RUT baseline of Zhu et al. next to QuantPrint on one family. About 45 minutes on an H100 for Qwen 7B.
#   bash run_gpu.sh jobs/rut.sh qwen2.5-7b
set -e
F=${1:-qwen2.5-7b}; D=data-rut-$F${DATA_SUFFIX:-}; mkdir -p $D results
python check_models.py --family $F | tee logs/check-rut-$F.txt
READY=$(grep '^READY:' logs/check-rut-$F.txt | cut -d' ' -f2)
for c in ${READY//,/ }; do
  [ -s $D/target-$c.jsonl ] && { echo "target $c already sampled"; continue; }
  python rut.py sample --family $F --config $c --role target --out $D 2>&1 | grep -E "wrote|Error|error" || true
done
[ -s $D/reference-bf16.jsonl ] || python rut.py sample --family $F --config bf16 --role reference --out $D 2>&1 | grep -E "wrote|Error|error" || true
python rut.py score --family $F --out $D 2>&1 | grep -E "scoring|of|wrote|Error|error|Paste" || true
QP=$(python -c "import families; print(families.data_dir('$F'))")
python rut.py analyze --family $F --out $D $( [ -d "$QP" ] && echo --quantprint $QP )
tar czf /workspace/results-rut-$F${DATA_SUFFIX:-}.tar.gz $D results/rut-$F.* && echo "Saved /workspace/results-rut-$F${DATA_SUFFIX:-}.tar.gz"
