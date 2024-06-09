import os
import wandb
import logging
import inspect
import argparse
import subprocess

from pathlib import Path
from tqdm.auto import tqdm
from einops import rearrange
from omegaconf import OmegaConf
from typing import Dict, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available

from transformers import CLIPTextModel, CLIPTokenizer

from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.models.unet import UNet3DConditionModel
from animatediff.utils.util import save_videos_grid, zero_rank_print
from animatediff.utils.util import load_weights

def init_dist(launcher="slurm", backend='nccl', port=29500, **kwargs):
    """Initializes distributed environment."""
    if launcher == 'pytorch':
        rank = int(os.environ['RANK'])
        num_gpus = torch.cuda.device_count()
        local_rank = rank % num_gpus
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, **kwargs)
        
    elif launcher == 'slurm':
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        local_rank = proc_id % num_gpus
        torch.cuda.set_device(local_rank)
        addr = subprocess.getoutput(
            f'scontrol show hostname {node_list} | head -n1')
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        port = os.environ.get('PORT', port)
        os.environ['MASTER_PORT'] = str(port)
        dist.init_process_group(backend=backend)
        zero_rank_print(f"proc_id: {proc_id}; local_rank: {local_rank}; ntasks: {ntasks}; node_list: {node_list}; num_gpus: {num_gpus}; addr: {addr}; port: {port}")
        
    else:
        raise NotImplementedError(f'Not implemented launcher type: `{launcher}`!')
    
    return local_rank


def inference(
    name: str,
    use_wandb: bool,
    launcher: str,
    output_dir: str,
    pretrained_model_path: str,
    validation_data: Dict,
    unet_checkpoint_path: str = "",
    unet_additional_kwargs: Dict = {},
    ema_decay: float = 0.9999,
    noise_scheduler_kwargs = None,
    global_seed: int = 42,
    is_debug: bool = False,
    motion_module: str = "",
    motion_module_lora_configs: list = [],
    adapter_lora_path: str = "",
    adapter_lora_scale: float = 1.0,
    dreambooth_path: str = "",
    lora_model_path: str = "",
    lora_alpha: float = 0.8,
):
    check_min_version("0.10.0.dev0")

    # Initialize distributed inference
    local_rank = init_dist(launcher=launcher)
    global_rank = dist.get_rank()
    num_processes = dist.get_world_size()
    is_main_process = global_rank == 0

    seed = global_seed + global_rank
    torch.manual_seed(seed)

    # Logging folder
    folder_name = "debug" if is_debug else name
    output_dir = os.path.join(output_dir, folder_name)
    if is_debug and os.path.exists(output_dir):
        os.system(f"rm -rf {output_dir}")

    *_, config = inspect.getargvalues(inspect.currentframe())

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if is_main_process and (not is_debug) and use_wandb:
        run = wandb.init(project="animatediff", name=folder_name, config=config)

    # Handle the output folder creation
    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/sanity_check", exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDIMScheduler(**OmegaConf.to_container(noise_scheduler_kwargs))

    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")

    unet = UNet3DConditionModel.from_pretrained_2d(
        pretrained_model_path, subfolder="unet", 
        unet_additional_kwargs=OmegaConf.to_container(unet_additional_kwargs)
    )

    # Load pretrained unet weights
    if unet_checkpoint_path != "":
        zero_rank_print(f"from checkpoint: {unet_checkpoint_path}")
        unet_checkpoint_path = torch.load(unet_checkpoint_path, map_location="cpu")
        if "global_step" in unet_checkpoint_path: zero_rank_print(f"global_step: {unet_checkpoint_path['global_step']}")
        state_dict = unet_checkpoint_path["state_dict"] if "state_dict" in unet_checkpoint_path else unet_checkpoint_path
        m, u = unet.load_state_dict(state_dict, strict=False)
        zero_rank_print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

    # Move models to GPU
    vae.to(local_rank)
    text_encoder.to(local_rank)
    unet.to(local_rank)

    # Enable xformers
    if is_xformers_available():
        unet.enable_xformers_memory_efficient_attention()
    else:
        raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Validation pipeline
    validation_pipeline = AnimationPipeline(
        unet=unet, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, scheduler=noise_scheduler,
    ).to("cuda")
    controlnet = controlnet_images = None

    validation_pipeline = load_weights(
        validation_pipeline,
        # motion module
        motion_module_path=motion_module,
        motion_module_lora_configs=motion_module_lora_configs,
        # domain adapter
        adapter_lora_path=adapter_lora_path,
        adapter_lora_scale=adapter_lora_scale,
        # image layers
        dreambooth_model_path=dreambooth_path,
        lora_model_path=lora_model_path,
        lora_alpha=lora_alpha,
    ).to("cuda")

    validation_pipeline.enable_vae_slicing()

    # Set unet to evaluation mode
    unet.eval()

    generator = torch.Generator(device=local_rank)
    generator.manual_seed(global_seed)

    height = validation_data["sample_size"][0] if not isinstance(validation_data["sample_size"], int) else validation_data["sample_size"]
    width  = validation_data["sample_size"][1] if not isinstance(validation_data["sample_size"], int) else validation_data["sample_size"]

    for idx, prompt in enumerate(validation_data["prompts"]):
        sample = validation_pipeline(
            prompt,
            generator=generator,
            video_length=validation_data["sample_n_frames"],
            height=height,
            width=width,
            **validation_data,
        ).videos
        save_path = f"{output_dir}/samples/sample-{idx}.mp4"
        save_videos_grid(sample, save_path)
        if is_main_process:
          logging.info(f"Saved sample to {save_path}")

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--launcher", type=str, choices=["pytorch", "slurm"], default="pytorch")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    name = Path(args.config).stem
    config = OmegaConf.load(args.config)

    inference(name=name, launcher=args.launcher, use_wandb=args.wandb, **config)