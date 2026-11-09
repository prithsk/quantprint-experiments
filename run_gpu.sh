#!/usr/bin/env bash
# One command per experiment on a rented GPU. It builds a clean Python environment the first time, installs
# everything and runs the job in the background, so a closed browser tab never stops it.
#   bash run_gpu.sh jobs/h100_quant.sh
#   bash run_gpu.sh jobs/family.sh llama3.1-8b
cd "$(dirname "$0")"; REPO=$(pwd)
job=$(basename "$1" .sh); [ -n "${2:-}" ] && job="$job-$2"
mkdir -p logs; LOG="$REPO/logs/$job.log"
nohup bash -c "
  cd '$REPO'
  trap 'echo; echo FAILED: the last lines above show why' ERR
  set -eE
  if [ ! -x ~/venv/bin/python ]; then python3 -m pip install -q uv && python3 -m uv venv -q --seed ~/venv; fi
  source ~/venv/bin/activate
  echo 'Installing, about 3 minutes the first time...'
  uv pip install -q -r requirements-gpu.txt
  echo \"GPU: \$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)\"
  bash $*
  echo; echo DONE
" > "$LOG" 2>&1 &
sleep 2
echo "Running in the background. If the tab disconnects, reconnect and run: tail -F $LOG"
tail -F "$LOG"
