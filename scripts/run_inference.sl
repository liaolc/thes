#!/bin/bash
#SBATCH --job-name=SCD_inference_600          # Job name
#SBATCH --output=SCD_inference_600.out                       # Standard output
#SBATCH --error=SCD_inference_600.err                        # Standard error
#SBATCH --nodes=1                                   # Number of nodes
#SBATCH --ntasks=1                                  # Number of tasks
#SBATCH --cpus-per-task=16                           # Number of CPU cores per task
#SBATCH --gres=gpu:4                                # Request 1 GPU
#SBATCH --mem=128G                                   # Total memory
#SBATCH --partition=short                        # Partition
#SBATCH --time=12:00:00                             # Wall time (hh:mm:ss)
#SBATCH --mail-type=BEGIN,END,FAIL                  # Email on start, end, fail
#SBATCH --mail-user=liaolc@bc.edu                 # Your email
#SBATCH --nodelist=g[003-019]
source ~/miniconda3/etc/profile.d/conda.sh

cd /home/liaolc/sparse-causal-diffusion

# Set CHECKPOINT to the ema.pth from a completed training run, e.g.:
# export CHECKPOINT=experiments/scd_minecraft/models/checkpoint-100000/ema.pth
export CHECKPOINT=/scratch/liaolc/scd/scd_minecraft/checkpoint-130000/ema.pth
if [ -z "${CHECKPOINT:-}" ]; then
    echo "Error: CHECKPOINT env var must be set to a trained transformer checkpoint (ema.pth)"
    exit 1
fi

NUM_GPUS=4 bash scripts/inference.sh "$CHECKPOINT"
