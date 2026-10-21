# 0. Find prerequisites, then load Python
module spider python/ondemand-jupyter-python3.10   # note any "must load first" modules
# module load <prereqs> python/ondemand-jupyter-python3.10

# 1. Folders, conda package cache, pip cache, temp dir (all on /ix1)
BASE=/ix1/jmendoza-arenas/jop204
mkdir -p $BASE/envs $BASE/conda_pkgs $BASE/pip_cache $BASE/tmp
printf "pkgs_dirs:\n- $BASE/conda_pkgs\n" > ~/.condarc     # overwrites an existing ~/.condarc
export PIP_CACHE_DIR=$BASE/pip_cache
export TMPDIR=$BASE/tmp
export PYTHONNOUSERSITE=1

# 2. Create and activate the environment
conda create --prefix=$BASE/envs/ddpm python=3.12 -y
source activate $BASE/envs/ddpm
which python pip          # both must point inside $BASE/envs/ddpm
python --version          # should say 3.12.x

# 3. Install PyTorch with CUDA 12.6, then the project requirements
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126
cd $BASE/ddpm_project     # wherever requirements.txt lives
pip install -r requirements.txt

# 4. Confirm torch was not replaced by a CPU build
pip list 2>/dev/null | grep -Ei "^(torch|torchvision|torchmetrics|torch-fidelity) "
python -c "import torch; print(torch.__version__, torch.version.cuda)"