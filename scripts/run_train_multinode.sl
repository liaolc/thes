#!/bin/bash
#SBATCH --job-name=SCD_train_multinodemed3
#SBATCH --output=SCD_train_multinodemed3.out
#SBATCH --error=SCD_train_multinodemed3.err
#SBATCH --nodes=2                                   # Two nodes
#SBATCH --ntasks-per-node=1                         # One launcher task per node (accelerate handles the 4 GPUs)
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4                                # 4 GPUs per node (8 total)
#SBATCH --mem=128G
#SBATCH --partition=medium
#SBATCH --time=48:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=liaolc@bc.edu
#SBATCH --nodelist=g[003-019]
#SBATCH --dependency=afterany:2377952

# Resolve master node (rank 0) hostname
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR
export NUM_GPUS=4
export NUM_NODES=2
export PORT=29500
export CONFIG="options/scd_minecraft.yml"

# Auto-detect latest checkpoint from scratch
SCRATCH_DIR="/scratch/liaolc/scd/scd_minecraft"
LATEST=$(ls -d "$SCRATCH_DIR"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -n 1)
if [ -z "$LATEST" ]; then
    echo "Error: no checkpoint-* found in $SCRATCH_DIR"
    exit 1
fi
export RESUME_FROM="$LATEST"
echo "Resuming from: $RESUME_FROM"

echo "Master node: $MASTER_ADDR"
echo "Node list:   $SLURM_JOB_NODELIST"

# srun launches one task per node; SLURM_NODEID (0 or 1) becomes the machine rank.
# Double-quoted so $MASTER_ADDR/$NUM_GPUS/$NUM_NODES expand from this shell,
# but \$SLURM_NODEID is escaped so it expands inside each per-node subshell.
srun bash -c "
    source ~/miniconda3/etc/profile.d/conda.sh
    cd /home/liaolc/sparse-causal-diffusion
    MACHINE_RANK=\$SLURM_NODEID \
    NUM_GPUS=$NUM_GPUS \
    NUM_NODES=$NUM_NODES \
    MASTER_ADDR=$MASTER_ADDR \
    PORT=$PORT \
    RESUME_FROM=$RESUME_FROM \
    bash scripts/train_multinode.sh $CONFIG
"
