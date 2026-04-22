#!/usr/bin/env bash
# Rolling Sink inference — training-free long-horizon evaluation.
# Runs run_decoupled_inference.py with Rolling Sink enabled.
# Auto-detects the latest ema.pth from the experiment directory unless
# CHECKPOINT is explicitly set.
#
# Usage:
#   bash scripts/inference_rolling_sink.sh [extra args...]
#
# Key env-var overrides (all optional):
#   CONFIG          YAML config path         (default: options/scd_minecraft.yml)
#   CHECKPOINT      Path to ema.pth          (default: auto-detect latest)
#   NUM_GPUS        accelerate processes     (default: 4)
#   PORT            main_process_port        (default: 29501)
#   RS_MAX_FRAMES   Rolling Sink K           (default: 144)
#   RS_SINK_FRAMES  Rolling Sink S           (default: 119)
#   RS_BLOCK_FRAMES Rolling Sink R           (default: 3)
#   CONTEXT_LENGTHS space-separated list     (default: 144)
#   GUIDANCE_SCALES space-separated list     (default: 1.5 1.0)
set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate scd

CONFIG="${CONFIG:-options/scd_minecraft.yml}"
NUM_GPUS="${NUM_GPUS:-4}"
PORT="${PORT:-29501}"

RS_MAX_FRAMES="${RS_MAX_FRAMES:-300}"
RS_SINK_FRAMES="${RS_SINK_FRAMES:-249}"
RS_BLOCK_FRAMES="${RS_BLOCK_FRAMES:-3}"

read -ra CONTEXT_LENGTHS <<< "${CONTEXT_LENGTHS:-144}"
read -ra GUIDANCE_SCALES  <<< "${GUIDANCE_SCALES:-1.5 1.0}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
NAME_SUFFIX="_rolling_sink_${TIMESTAMP}"

# Auto-detect latest checkpoint if CHECKPOINT not set
if [ -z "${CHECKPOINT:-}" ]; then
    # Derive experiment name from the YAML 'name:' field
    EXP_NAME=$(python -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('$CONFIG')
print(cfg.name)
")
    MODELS_DIR="experiments/${EXP_NAME}/models"
    if [ ! -d "$MODELS_DIR" ]; then
        echo "Error: models directory not found at $MODELS_DIR"
        echo "Set CHECKPOINT=<path/to/ema.pth> to override."
        exit 1
    fi
    LATEST=$(ls -d "$MODELS_DIR"/checkpoint-* 2>/dev/null \
        | sort -t- -k2 -n | tail -n 1)
    if [ -z "$LATEST" ]; then
        echo "Error: no checkpoint-* directories found under $MODELS_DIR"
        exit 1
    fi
    CHECKPOINT="${LATEST}/ema.pth"
    echo "Auto-detected checkpoint: $CHECKPOINT"
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: checkpoint not found at $CHECKPOINT"
    exit 1
fi

echo "Config:     $CONFIG"
echo "Checkpoint: $CHECKPOINT"
echo "GPUs:       $NUM_GPUS"
echo "RS K/S/R:   $RS_MAX_FRAMES / $RS_SINK_FRAMES / $RS_BLOCK_FRAMES"
echo "Output:     results/scd_minecraft_decoupled_eval${NAME_SUFFIX}/"

accelerate launch \
    --num_processes "$NUM_GPUS" \
    --num_machines 1 \
    --main_process_port "$PORT" \
    inference/run_decoupled_inference.py \
    --opt "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --rolling-sink \
    --rolling-sink-max-frames  "$RS_MAX_FRAMES" \
    --rolling-sink-sink-frames "$RS_SINK_FRAMES" \
    --rolling-sink-block-frames "$RS_BLOCK_FRAMES" \
    --context-lengths "${CONTEXT_LENGTHS[@]}" \
    --guidance-scales "${GUIDANCE_SCALES[@]}" \
    --name-suffix "$NAME_SUFFIX" \
    "$@"
