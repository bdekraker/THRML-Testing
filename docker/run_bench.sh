#!/usr/bin/env bash
set -euo pipefail

if [[ ! -d "/workspace/pipelines" ]]; then
  echo "Mount your repo into /workspace (e.g. -v \$PWD:/workspace) before running."
  exit 1
fi

export PYTHONPATH=/workspace:${PYTHONPATH:-}

: "${BLOCKING:=checkerboard,stripes,supercell,random}"
: "${STEPS_PER_SAMPLE:=6}"
: "${NUM_CHAINS:=4}"
: "${J_VALUES:=0.30}"
: "${SEED:=0}"
: "${OUT:=outputs/bench_gpu_run.json}"

python pipelines/bench.py \
  --blocking ${BLOCKING//,/ } \
  --J ${J_VALUES//,/ } \
  --sps ${STEPS_PER_SAMPLE//,/ } \
  --num_chains ${NUM_CHAINS//,/ } \
  --seed "${SEED}" \
  --out "${OUT}"
