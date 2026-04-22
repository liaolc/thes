#!/bin/bash
#SBATCH --job-name=SCD_rolling_K300_f300              # Job name
#SBATCH --output=SCD_rolling_K300_f300.out            # Standard output
#SBATCH --error=SCD_rolling_K300_f300.err             # Standard error
#SBATCH --nodes=1                                # Number of nodes
#SBATCH --ntasks=1                               # Number of tasks
#SBATCH --cpus-per-task=16                       # Number of CPU cores per task
#SBATCH --gres=gpu:4                             # 4 GPUs
#SBATCH --mem=128G                               # Total memory
#SBATCH --partition=short                        # Partition
#SBATCH --time=12:00:00                          # Wall time (hh:mm:ss)
#SBATCH --mail-type=BEGIN,END,FAIL               # Email on start, end, fail
#SBATCH --mail-user=liaolc@bc.edu               # Your email
#SBATCH --nodelist=g[003-019]
source ~/miniconda3/etc/profile.d/conda.sh

cd /home/liaolc/sparse-causal-diffusion
git checkout RollingSink

# CHECKPOINT auto-detects the latest ema.pth from experiments/<name>/models/.
# Override by setting: export CHECKPOINT=experiments/.../ema.pth
export NUM_GPUS=4
export CONFIG="options/scd_minecraft.yml"

# Optional Rolling Sink overrides (defaults: K=144, S=119, R=3)
# export RS_MAX_FRAMES=144
# export RS_SINK_FRAMES=119
# export RS_BLOCK_FRAMES=3

# Optional sweep overrides
# export CONTEXT_LENGTHS="144"
# export GUIDANCE_SCALES="1.5 1.0"

bash scripts/inference_rolling_sink.sh
