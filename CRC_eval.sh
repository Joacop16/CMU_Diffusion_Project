#!/bin/bash -l
#SBATCH --job-name=ddpm_eval
#SBATCH --clusters=gpu
#SBATCH --partition=jmendoza-arenas
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mail-user=jop204@pitt.edu
#SBATCH --mail-type=FAIL,END
#SBATCH --output=/ix1/jmendoza-arenas/jop204/CMU_Diffusion_Project/logs/%x-%j.out
 
# Usage: sbatch CRC_eval_simple.sh <checkpoint> <ddim|ddpm> <steps> [extra args]
# Example: sbatch CRC_eval_simple.sh keep/checkpoint_epoch_49.pth ddim 50
 
CKPT=$1
SAMPLER=$2
STEPS=$3
shift 3
 
module purge
module load python/3.12.8
source /ix1/jmendoza-arenas/jop204/envs/ddpm/bin/activate
 
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
unset WORLD_SIZE
 
cd /ix1/jmendoza-arenas/jop204/CMU_Diffusion_Project
 
if [ "$SAMPLER" = "ddim" ]; then USE_DDIM=true; else USE_DDIM=false; fi
 
python inference.py --config configs/ddpm.yaml --ckpt "$CKPT" \
    --use_ddim $USE_DDIM --num_inference_steps $STEPS --batch_size 100 "$@"
 