#!/usr/bin/env bash
# Build an INT4 GPTQ checkpoint for one family with llm-compressor, in its own Python environment.
#   bash jobs/quantize.sh mistral-7b
set -e
F=${1:?give a family}; OUT=ckpt/$F-w4a16
[ -f $OUT/config.json ] && { echo "INT4 checkpoint for $F already built"; exit 0; }
[ -x ~/venv-llmc/bin/python ] || uv venv -q --seed ~/venv-llmc
echo "Installing llm-compressor, about 3 minutes the first time..."
uv pip install -q --python ~/venv-llmc/bin/python llmcompressor datasets accelerate
echo "Quantizing $F to INT4 GPTQ, about 15 minutes for 7B and an hour for 30B..."
~/venv-llmc/bin/python quantize.py --family $F --out $OUT
