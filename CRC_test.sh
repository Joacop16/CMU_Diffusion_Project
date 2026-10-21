#!/bin/bash
BASE=/ix1/jmendoza-arenas/jop204

module purge
module load python/3.12.8
source $BASE/envs/ddpm/bin/activate
export PYTHONNOUSERSITE=1

echo "== Python =="
which python
python --version

echo "== GPU =="
nvidia-smi || { echo "No GPU visible on this node"; exit 1; }

python - <<'EOF'
import torch
print("torch:", torch.__version__, "| cuda build:", torch.version.cuda, "| available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA not available"
print("device:", torch.cuda.get_device_name(0))
print("memory (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))

x = torch.randn(4096, 4096, device="cuda")
print("fp32 matmul:", (x @ x).sum().item())                      # forces a real kernel launch

xb = x.bfloat16()
print("bf16 matmul:", (xb @ xb).float().sum().item())            # confirms bf16 works for mixed precision

# cuDNN conv, which the U-Net depends on
conv = torch.nn.Conv2d(3, 8, 3, padding=1).cuda()
print("conv output:", tuple(conv(torch.randn(2, 3, 32, 32, device="cuda")).shape))
print("ALL GPU CHECKS PASSED")
EOF