# IDL Fall 2026 - Guided Project: Diffusion

**Midterm:** October 23, 2026  
**Final:** TBD

## Current Progress (October 23, 2026)

We implemented and trained an unconditional pixel-space DDPM on ImageNet-100 (128 × 128), and implemented a DDIM scheduler to reduce inference steps. The midterm evaluation uses the same checkpoint after 45 training epochs and 5,000 generated images per sampler.

| Sampler | Steps | FID ↓ | IS ↑ | Time / 5k images (s) |
|---|---:|---:|---:|---:|
| DDPM | 200 | 67.72 | 7.87 | 8,898 |
| DDIM | 10 | 84.18 | 8.12 | 446 |
| DDIM | 50 | 69.60 | 8.25 | 2,227 |
| DDIM | 100 | 68.97 | 8.19 | 4,441 |

The next steps are latent diffusion using the provided pre-trained VAE, classifier-free guidance, and experiments with improved noise schedules. These are **not** included in the midterm results.

## Setup

Use Python 3.12 from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Download `imagenet100_128x128.tar.gz` from the [course dataset folder](https://drive.google.com/drive/u/0/folders/1Hr8LU7HHPEad8ALmMo5cvisazsm6zE8Z), then extract it:

```bash
mkdir -p data
tar -xzf imagenet100_128x128.tar.gz -C data
```

The expected folders are `data/imagenet100_128x128/train/` and `data/imagenet100_128x128/validation/` (100 classes; 130,000 training and 5,000 validation images). The configuration is in `configs/ddpm.yaml`.

## Training

Training requires a suitable GPU. To train the pixel-space baseline:

```bash
python train.py --config configs/ddpm.yaml
```

The default configuration specifies 50 epochs; the midterm report evaluates the checkpoint saved after **45 completed epochs**. On the Pittsburgh CRC cluster, the project also provides `CRC_train.sh` (with cluster-specific paths and environment settings).

## Inference and Evaluation

Specify a trained checkpoint (not included in this repository). The following commands generate 5,000 samples and compute FID/IS against the validation split:

```bash
# DDPM: 200 steps
python inference.py --config configs/ddpm.yaml --ckpt /path/to/checkpoint_epoch_44.pth --use_ddim false --num_inference_steps 200 --batch_size 100

# DDIM: 50 steps
python inference.py --config configs/ddpm.yaml --ckpt /path/to/checkpoint_epoch_44.pth --use_ddim true --num_inference_steps 50 --batch_size 100
```

Change `--num_inference_steps` to `10` or `100` for the other reported DDIM settings. The available `CRC_eval.sh` script is configured for the Pittsburgh CRC environment and needs its local paths adjusted before use elsewhere. Set `WANDB_MODE=disabled` to run without W&B logging where applicable.

## Project Components

- **DDPM:** `schedulers/scheduling_ddpm.py`, `pipelines/ddpm.py`, `models/unet.py`, `train.py`
- **DDIM:** `schedulers/scheduling_ddim.py` (select with `--use_ddim true`)
- **Evaluation:** `inference.py` (FID, IS, and sample grids)
- **Final-stage work:** `models/vae.py`, `models/class_embedder.py`, and latent/CFG integration
