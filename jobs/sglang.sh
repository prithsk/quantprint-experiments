#!/usr/bin/env bash
# QuantPrint on a second engine. Each Qwen 7B setup gets its own SGLang server with prefix caching off, in a separate
# Python environment so SGLang and vLLM never fight over package versions. About 60 minutes on an H100.
set -e
[ -x ~/venv-sglang/bin/python ] || uv venv -q --seed ~/venv-sglang
echo "Installing SGLang, about 5 minutes the first time..."
uv pip install -q --python ~/venv-sglang/bin/python "sglang[all]" datasets numpy scikit-learn
SG=~/venv-sglang/bin/python; OUT=data-sglang; mkdir -p $OUT logs results
for c in bf16 fp16 fp8 int8-gptq int4-gptq int4-awq; do
  [ -f $OUT/$c.done ] && { echo "$c already collected"; continue; }; rm -f $OUT/$c.jsonl
  read MODEL DTYPE Q < <(python -c "import families; c=families.configs('qwen2.5-7b')['$c']; print(c['model'], c['dtype'], c['quantization'] or '-')")
  ARGS="--model-path $MODEL --dtype $DTYPE --port 30000 --disable-radix-cache --mem-fraction-static 0.8"
  [ "$Q" != "-" ] && ARGS="$ARGS --quantization $Q"
  $SG -m sglang.launch_server $ARGS > logs/sglang-$c.log 2>&1 & PID=$!
  for i in $(seq 1 180); do curl -sf http://127.0.0.1:30000/health >/dev/null && break; sleep 5; done
  if ! curl -sf http://127.0.0.1:30000/health >/dev/null; then
    echo "SKIP: SGLang did not start for $c; its log ends:"; tail -15 logs/sglang-$c.log; kill $PID 2>/dev/null; sleep 5
    [ "$c" = bf16 ] && { echo "STOP: the BF16 reference is required"; exit 1; }; continue
  fi
  $SG sglang_collect.py --config $c --out $OUT --repeats ${REPEATS:-2} || echo "SKIP: collection failed for $c"
  kill $PID; wait $PID 2>/dev/null || true; sleep 5
  if [ -s $OUT/$c.jsonl ]; then touch $OUT/$c.done
  elif [ "$c" = bf16 ]; then echo "STOP: the BF16 reference produced no data"; exit 1
  else echo "SKIP: $c produced no data, so the analysis leaves it out"; fi
done
python other_setup.py --ref data-nocache --other $OUT
python analyze.py --data $OUT > results/report-sglang.md 2>&1; sed -n '/## Direction detector/,/## RQ2/p' results/report-sglang.md
tar czf /workspace/results-sglang.tar.gz $OUT results/other-$OUT.* results/report-sglang.md && echo "Saved /workspace/results-sglang.tar.gz"
