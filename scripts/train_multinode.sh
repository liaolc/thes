#!/usr/bin/env bash
# Multi-node training script. Intended to be launched via srun in run_train_multinode.sl.
# Do not call directly — MACHINE_RANK and MASTER_ADDR must be set by the caller.
#
# Usage (via run_train_multinode.sl):
#   sbatch scripts/run_train_multinode.sl
#
# Environment variables (all set by run_train_multinode.sl):
#   NUM_GPUS        GPUs per node (default: 4)
#   NUM_NODES       Total nodes   (default: 2)
#   MACHINE_RANK    This node's rank (0 = master, 1 = worker, ...)
#   MASTER_ADDR     IP of rank-0 node
#   PORT            NCCL rendezvous port (default: 29500)

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate scd
git checkout native

CONFIG="${1:-options/scd_minecraft.yml}"
NUM_GPUS="${NUM_GPUS:-4}"
NUM_NODES="${NUM_NODES:-2}"
MACHINE_RANK="${MACHINE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
PORT="${PORT:-29500}"
RESUME_FROM="${RESUME_FROM:-checkpoint-90000}"

TOTAL_PROCESSES=$(( NUM_GPUS * NUM_NODES ))

echo "[node ${MACHINE_RANK}/${NUM_NODES}] master=${MASTER_ADDR}:${PORT} total_gpus=${TOTAL_PROCESSES} config=${CONFIG} resume=${RESUME_FROM}"

accelerate launch \
    --num_processes "$TOTAL_PROCESSES" \
    --num_machines "$NUM_NODES" \
    --machine_rank "$MACHINE_RANK" \
    --main_process_ip "$MASTER_ADDR" \
    --main_process_port "$PORT" \
    train.py \
    -opt "$CONFIG" \
    --resume_from_checkpoint "$RESUME_FROM"
