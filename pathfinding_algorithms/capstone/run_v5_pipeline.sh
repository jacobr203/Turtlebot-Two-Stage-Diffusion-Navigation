#!/usr/bin/env bash
# Sequential v5 pipeline: wait for the running evaluation matrix, then
# train single_stage_v5, then evaluate it on the same axes.
#
# Strictly sequential on purpose. Sampling latency is measured during
# evaluation trials, and CPU contention inflates it — an over-sharded run
# logged 347 ms against a 250 ms tick budget when the model actually
# samples in 45 ms. Training alongside evaluation would corrupt the
# latency column and degrade the closed-loop numbers with it.
set -euo pipefail
cd "$(dirname "$0")"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# 1. wait for the evaluation matrix to finish
while pgrep -f "run_matrix.py" >/dev/null; do sleep 60; done
log "evaluation matrix finished"

# let gazebo processes drain before claiming the cores
pkill -9 -f gzserver 2>/dev/null || true
sleep 10

# 2. train the single-stage counterpart
log "training single_stage_v5"
.venv/bin/python -m diffusion.train \
    --config diffusion/configs/single_stage_v5.yaml
log "training finished"

CKPT="diffusion/runs/single_stage_v5/best.pt"
[ -f "$CKPT" ] || { log "ERROR: $CKPT missing"; exit 1; }

# 3. evaluate it on the same worlds, rates and trial count
log "evaluating single_stage_v5"
export SINGLE_STAGE_CKPT="$PWD/$CKPT"
for k in 0 1 2; do
    taskset -c $((5*k))-$((5*k+3)) \
        .venv/bin/python evaluation/run_matrix.py \
        --config evaluation/configs/eval_full_v5_single_stage.yaml \
        --shard "$k/3" --worker "$k" > "/tmp/v5_single_w$k.log" 2>&1 &
done
wait
log "single-stage evaluation finished"
log "aggregate with: .venv/bin/python evaluation/aggregate.py --run-id full_v5_single_stage"
