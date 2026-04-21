#!/bin/bash
#SBATCH --job-name=SCD_train_single           # Job name
#SBATCH --output=SCD_train_single.out                       # Standard output
#SBATCH --error=SCD_train_single.err                        # Standard error
#SBATCH --nodes=1                                   # Number of nodes
#SBATCH --ntasks=1                                  # Number of tasks
#SBATCH --cpus-per-task=16                           # Number of CPU cores per task
#SBATCH --gres=gpu:4                                # Request 1 GPU
#SBATCH --mem=128G                                   # Total memory
#SBATCH --partition=medium                        # Partition
#SBATCH --time=48:00:00                             # Wall time (hh:mm:ss)
#SBATCH --mail-type=BEGIN,END,FAIL                  # Email on start, end, fail
#SBATCH --mail-user=liaolc@bc.edu                 # Your email
#SBATCH --nodelist=g[010-018]

export NUM_GPUS=4
export RESUME_FROM="checkpoint-10000"

cd /home/liaolc/sparse-causal-diffusion
bash scripts/train.sh