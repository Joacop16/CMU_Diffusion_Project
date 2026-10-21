#!/bin/bash
BASE=/ix1/jmendoza-arenas/jop204
mkdir -p $BASE/envs $BASE/pip_cache $BASE/tmp

module purge
module load python/3.12.8          # add any prerequisite modules before this, if spider listed some

export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR=$BASE/pip_cache
export TMPDIR=$BASE/tmp

python --version
python -m venv $BASE/envs/ddpm
source $BASE/envs/ddpm/bin/activate

# Stop if the wrong Python is active
case "$(which python)" in
  $BASE/envs/ddpm/*) echo "Env OK: $(which python)" ;;
  *) echo "ERROR: wrong python: $(which python)"; exit 1 ;;
esac

python -m pip install --upgrade pip
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126

cd $BASE/CMU_Diffusion_Project     # folder containing requirements.txt
pip install -r requirements.txt

pip list 2>/dev/null | grep -Ei "^(torch|torchvision|torchmetrics|torch-fidelity) "
python -c "import torch; print(torch.__version__, torch.version.cuda)"