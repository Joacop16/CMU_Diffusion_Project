#!/bin/bash -l
#SBATCH --job-name=ddpm_baseline
#SBATCH --clusters=gpu
#SBATCH --partition=jmendoza-arenas
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --qos=short
#SBATCH --mail-user=jop204@pitt.edu
#SBATCH --mail-type=FAIL,END
#SBATCH --output=/ix1/jmendoza-arenas/jop204/CMU_Diffusion_Project/logs/%x-%j.out

# Usage (from the project folder, on the login node):
#   mkdir -p logs
#   sbatch train.sh
# Extra arguments are passed to train.py and override the config, e.g.:
#   sbatch train.sh --num_epochs 20 --batch_size 32

PROJ=/ix1/jmendoza-arenas/jop204/CMU_Diffusion_Project
ENV=/ix1/jmendoza-arenas/jop204/envs/ddpm

module purge
module load python/3.12.8
source $ENV/bin/activate

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1          # log lines show up immediately
unset WORLD_SIZE                   # train.py must run as a single process

# Weights & Biases: online if the server is reachable, otherwise offline
export WANDB_DIR=$PROJ/wandb
mkdir -p $WANDB_DIR
# if curl -s -m 10 -o /dev/null https://api.forge.coreweave.com; then
export WANDB_MODE=online
# else
#     echo "W&B server not reachable, using offline mode"
#     export WANDB_MODE=offline
# fi
export WANDB_ENTITY=jpenuela-organization

cd $PROJ

echo "Host: $(hostname) | Job: $SLURM_JOB_ID | Start: $(date)"
echo "Python: $(which python)"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "SLURM_NTASKS=$SLURM_NTASKS  WANDB_MODE=$WANDB_MODE"

python train.py --config configs/ddpm.yaml --run_name ddpm_baseline "$@"

echo "End: $(date)"
