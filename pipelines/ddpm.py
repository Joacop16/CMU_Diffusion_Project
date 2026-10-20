
from typing import List, Optional, Tuple, Union
from PIL import Image
from tqdm import tqdm
import torch 
import torch.nn as nn
from utils import randn_tensor

class DDPMPipeline:
    def __init__(self, unet, scheduler, vae=None, class_embedder=None):
        self.unet = unet
        self.scheduler = scheduler
        
        # NOTE: this is for latent DDPM
        self.vae = None
        if vae is not None:
            self.vae = vae
            
        # NOTE: this is for CFG
        if class_embedder is not None:
            self.class_embedder = class_embedder

    def numpy_to_pil(self, images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    def progress_bar(self, iterable=None, total=None):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )

        if iterable is not None:
            return tqdm(iterable, **self._progress_bar_config)
        elif total is not None:
            return tqdm(total=total, **self._progress_bar_config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")

    def _prepare_conditioning_inputs(self, batch_size, classes, guidance_scale, device):
        """Validate class inputs without implementing the student CFG exercise."""
        if classes is None:
            if guidance_scale is not None:
                raise ValueError("Provide classes when specifying guidance_scale.")
            return None, False
        if not hasattr(self, "class_embedder"):
            raise ValueError("class_embedder is required when classes are provided.")
        if isinstance(classes, int):
            classes = [classes] * batch_size
        classes = torch.as_tensor(classes, dtype=torch.long, device=device)
        if classes.shape != (batch_size,):
            raise ValueError("Provide one class per image in batch_size.")
        use_guidance = guidance_scale is not None and guidance_scale != 1.0
        return classes, use_guidance

    
    @torch.no_grad()
    def __call__(
        self, 
        batch_size: int = 1,
        num_inference_steps: int = 1000,
        classes: Optional[Union[int, List[int]]] = None,
        guidance_scale : Optional[float] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        device = None,
    ):
        image_shape = (batch_size, self.unet.input_ch, self.unet.input_size, self.unet.input_size)
        if device is None:
            device = next(self.unet.parameters()).device
        device = torch.device(device)
        
        # NOTE: this is for CFG
        classes, use_guidance = self._prepare_conditioning_inputs(
            batch_size, classes, guidance_scale, device
        )
        
        if classes is not None:
            # the unconditional "class" is the extra index n_classes (one past the real labels)
            # TODO (SOLVED): get uncond classes
            uncond_classes = torch.full_like(classes, self.class_embedder.num_classes)
            # TODO (SOLVED): get class embeddings from classes
            class_embeds = self.class_embedder(classes) 
            # TODO (SOLVED): get uncon class embeddings
            uncond_embeds = self.class_embedder(uncond_classes) 
        
        # TODO (SOLVED): starts with random noise
        image = randn_tensor(image_shape, generator=generator, device=device)

        # TODO (SOLVED): set step values using set_timesteps of scheduler
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        
        # TODO (SOLVED): inverse diffusion process with for loop
        for t in self.progress_bar(self.scheduler.timesteps):
            
            # NOTE: this is for CFG (classifier-free guidance): it lets you ask for a specific class (“malamute”) and steer the sampling toward it. With classes=None, the code skips all of it,
            if use_guidance:
                # TODO (SOLVED): implement cfg
                # Stack uncond and cond along the batch axis so one UNet call yields both
                # predictions. Order matters: [uncond, cond] must match chunk(2) below.
                model_input = torch.cat([image, image])
                c = torch.cat([uncond_embeds, class_embeds])
            else:
                model_input = image
                # A scale of 1 uses the conditional prediction without CFG mixing.
                c = class_embeds if classes is not None else None
            
            # TODO (SOLVED): 1. predict noise model_output
            # t is a 0-dim tensor; UNet.forward broadcasts it across the batch.
            model_output = self.unet(model_input, t, c=c)
            
            if use_guidance:
                # TODO (SOLVED): implement cfg
                uncond_model_output, cond_model_output = model_output.chunk(2)
                # Eq. 10: eps_hat = eps_uncond + w * (eps_cond - eps_uncond)
                model_output = uncond_model_output + guidance_scale * (cond_model_output - uncond_model_output)
            
            # TODO (SOLVED): 2. compute previous image: x_t -> x_t-1 using scheduler
            # Pass the same generator as the initial noise so a seed reproduces the sample.
            image = self.scheduler.step(model_output, t, image, generator=generator)
            
        # NOTE: this is for latent DDPM
        # TODO: use VQVAE to get final image
        if self.vae is not None: #The VAE is a pretrained network that compresses a 128×128 image into a small “latent” image. Latent DDPM runs the diffusion on those small images and decodes at the end
            # NOTE: remember to rescale your images
            # Undo the 0.1845 scaling applied after encoding in train.py, then decode.
            image = image / 0.1845
            image = self.vae.decode(image)
            # TODO (SOLVED): clamp your images values
            # The decoder output can slightly exceed [-1, 1].
            image = image.clamp(-1, 1)
        
        # TODO (SOLVED): return final image, re-scale to [0, 1]
        # Model works in [-1, 1]; numpy_to_pil expects [0, 1].
        image = (image / 2 + 0.5).clamp(0, 1)
        
        # convert to PIL images
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = self.numpy_to_pil(image)
        
        return image
        

