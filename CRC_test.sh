module load <same modules as above>
source activate /ix1/jmendoza-arenas/jop204/envs/ddpm
nvidia-smi                                   # note the driver version and GPU name
python - <<'EOF'
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
x = torch.randn(4096, 4096, device="cuda")
print((x @ x).sum().item())                  # forces a real kernel launch
EOF