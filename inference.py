import os 
import sys 
import time
import argparse
import numpy as np
import ruamel.yaml as yaml
import torch
import wandb 
import logging 
from logging import getLogger as get_logger
from tqdm import tqdm 
from PIL import Image
import torch.nn.functional as F

from torchvision import datasets, transforms
from torchvision.utils  import make_grid

from models import UNet, VAE, ClassEmbedder
from schedulers import DDPMScheduler, DDIMScheduler
from pipelines import DDPMPipeline
from utils import seed_everything, load_checkpoint

from train import parse_args

logger = get_logger(__name__)


def resolve_val_dir(args):
    """Use --val_dir if given, else a 'validation' or 'val' folder next to the train folder."""
    if args.val_dir is not None:
        return args.val_dir
    parent = os.path.dirname(os.path.normpath(args.data_dir))
    for name in ('validation', 'val'):
        candidate = os.path.join(parent, name)
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(f"No validation folder found next to {args.data_dir}; pass --val_dir.")


def image_folder_loader(folder, image_size, batch_size, num_workers, num_images=None, seed=0):
    """Load an ImageFolder as uint8 tensors in [0, 255], optionally a random subset."""
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.PILToTensor(),
    ])
    dataset = datasets.ImageFolder(folder, transform=transform)
    if num_images is not None and num_images < len(dataset):
        g = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(dataset), generator=g)[:num_images].tolist()
        dataset = torch.utils.data.Subset(dataset, indices)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def pil_to_uint8(images):
    """List of PIL images -> uint8 tensor (N, 3, H, W)."""
    return torch.stack([transforms.functional.pil_to_tensor(img.convert('RGB')) for img in images])


def compute_fid_is(gen_batches, ref_loader, device):
    """
    FID of generated images against the reference set, and IS of the generated images.
    gen_batches: iterable of uint8 tensors (N, 3, H, W).
    """
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore
    
    fid = FrechetInceptionDistance(feature=2048).to(device)
    inception = InceptionScore().to(device)
    
    for ref_images, _ in tqdm(ref_loader, desc="reference features"):
        fid.update(ref_images.to(device), real=True)
    
    num_gen = 0
    for gen_images in tqdm(gen_batches, desc="generated features"):
        gen_images = gen_images.to(device)
        fid.update(gen_images, real=False)
        inception.update(gen_images)
        num_gen += gen_images.size(0)
    
    fid_score = fid.compute().item()
    is_mean, is_std = inception.compute()
    return fid_score, is_mean.item(), is_std.item(), num_gen


def main():
    # parse arguments
    args = parse_args()
    
    # seed everything
    seed_everything(args.seed)
    device = torch.device(args.device)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    
    # setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    val_dir = None
    if args.eval_real:
        # sanity check: real train images vs validation. FID should be low and IS high.
        val_dir = resolve_val_dir(args)
        logger.info(f"Scoring {args.num_eval_images} real images from {args.data_dir} against {val_dir}")
        gen_loader = image_folder_loader(args.data_dir, args.image_size, args.batch_size,
                                         args.num_workers, args.num_eval_images, args.seed)
        ref_loader = image_folder_loader(val_dir, args.image_size, args.batch_size, args.num_workers)
        fid_score, is_mean, is_std, num_gen = compute_fid_is(
            (images for images, _ in gen_loader), ref_loader, device)
        logger.info(f"[real train vs val] N={num_gen} FID: {fid_score:.3f} IS: {is_mean:.3f} +- {is_std:.3f}")
        return

    if args.ckpt is None:
        raise ValueError("Provide a trained checkpoint with --ckpt /path/to/checkpoint.")
    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    scheduler_class = DDIMScheduler if args.use_ddim else DDPMScheduler
    scheduler = scheduler_class(
        num_train_timesteps=args.num_train_timesteps,
        num_inference_steps=args.num_inference_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        beta_schedule=args.beta_schedule,
        variance_type=args.variance_type,
        prediction_type=args.prediction_type,
        clip_sample=args.clip_sample,
        clip_sample_range=args.clip_sample_range,
    )
    if scheduler is None:
        raise NotImplementedError("Complete the scheduler setup TODO in inference.py before sampling.")
    scheduler = scheduler.to(device)
    
    # setup model
    logger.info("Creating model")
    # unet
    unet = UNet(input_size=args.unet_in_size, input_ch=args.unet_in_ch, T=args.num_train_timesteps, ch=args.unet_ch, ch_mult=args.unet_ch_mult, attn=args.unet_attn, num_res_blocks=args.unet_num_res_blocks, dropout=args.unet_dropout, conditional=args.use_cfg, c_dim=args.unet_ch)
    # preint number of parameters
    num_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    logger.info(f"Number of parameters: {num_params / 10 ** 6:.2f}M")
    
    # vae 
    vae = None
    if args.latent_ddpm:        
        vae = VAE()
        vae.init_from_ckpt('pretrained/model.ckpt')
        vae.eval()
    # cfg
    class_embedder = None
    if args.use_cfg:
        # TODO: class embeder
        class_embedder = ClassEmbedder(None)
        
    # send to device
    unet = unet.to(device)
    if vae:
        vae = vae.to(device)
    if class_embedder:
        class_embedder = class_embedder.to(device)
        
    # load checkpoint
    load_checkpoint(unet, scheduler, vae=vae, class_embedder=class_embedder, checkpoint_path=args.ckpt)
    unet.eval()
    if class_embedder is not None:
        class_embedder.eval()
    
    pipeline = DDPMPipeline(unet=unet, scheduler=scheduler, vae=vae, class_embedder=class_embedder)

    
    logger.info("***** Running Infrence *****")
    
    # we run inference to generate 5000 images; with cfg, 50 images per class 
    sampler_name = 'ddim' if args.use_ddim else 'ddpm'
    all_images = []
    batch_size = args.batch_size
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start_time = time.time()
    if args.use_cfg:
        # generate 50 images per class
        for i in tqdm(range(args.num_classes)):
            logger.info(f"Generating 50 images for class {i}")
            batch_size = 50
            classes = torch.full((batch_size,), i, dtype=torch.long, device=device)
            gen_images = pipeline(
                batch_size=batch_size, num_inference_steps=args.num_inference_steps, classes=classes,
                guidance_scale=args.cfg_guidance_scale, generator=generator, device=device)
            all_images.append(pil_to_uint8(gen_images))
    else:
        # generate 5000 images
        for start in tqdm(range(0, args.num_eval_images, args.batch_size)):
            batch_size = min(args.batch_size, args.num_eval_images - start)
            gen_images = pipeline(
                batch_size=batch_size, num_inference_steps=args.num_inference_steps,
                generator=generator, device=device)
            all_images.append(pil_to_uint8(gen_images))
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.time() - start_time
    num_gen = sum(images.size(0) for images in all_images)
    logger.info(f"Sampling with {sampler_name} ({args.num_inference_steps} steps): {num_gen} images in "
                f"{elapsed:.1f}s ({elapsed / num_gen:.3f}s/image)")
    
    # save a sample grid next to the checkpoint for the report
    out_dir = os.path.join(os.path.dirname(os.path.abspath(args.ckpt)), '..', 'samples')
    os.makedirs(out_dir, exist_ok=True)
    grid = make_grid(torch.cat(all_images)[:64], nrow=8)
    grid_path = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(args.ckpt))[0]}_{sampler_name}_{args.num_inference_steps}steps.png")
    Image.fromarray(grid.permute(1, 2, 0).numpy()).save(grid_path)
    logger.info(f"Saved sample grid to {grid_path}")
    
    # load validation images as reference batch
    val_dir = resolve_val_dir(args)
    ref_loader = image_folder_loader(val_dir, args.image_size, args.batch_size, args.num_workers)
    
    # compute FID and IS with torchmetrics
    fid_score, is_mean, is_std, num_gen = compute_fid_is(all_images, ref_loader, device)
    logger.info(f"[{sampler_name}, {args.num_inference_steps} steps] N={num_gen} "
                f"FID: {fid_score:.3f} IS: {is_mean:.3f} +- {is_std:.3f}")


if __name__ == '__main__':
    main()
