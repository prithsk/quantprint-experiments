#!/usr/bin/env bash
cd ~/quantprint-experiments
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "L4 data files: $(ls data-nocache/*.jsonl 2>/dev/null | wc -l) (must be 6)"
[ "$(ls data-nocache/*.jsonl 2>/dev/null | wc -l)" = 6 ] || { echo "STOP: L4 data missing from the repo"; exit 1; }
echo "Installing vLLM, about 5 minutes..."
python3 -m pip install -q -r requirements-gpu.txt || { echo "STOP: install failed"; exit 1; }
echo "Collecting BF16 on this GPU, about 10 minutes..."
python3 collect.py --config bf16 --batches 1,64 --repeats 2 --probes 120 --out data-hw 2>&1 | grep -E "probes in|wrote|Error|error" || true
[ -s data-hw/bf16.jsonl ] || { echo "STOP: collection produced no data"; exit 1; }
python3 analyze.py --data data-nocache --hardware data-hw > report.txt 2>&1
sed -n '/## Hardware/,/## RQ2/p' report.txt
echo "DONE"
