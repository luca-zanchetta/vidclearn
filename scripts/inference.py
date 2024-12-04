import torch
import argparse
from tuneavideo.pipelines.pipeline_tuneavideo import TuneAVideoPipeline
from tuneavideo.models.unet import UNet3DConditionModel
from tuneavideo.util import save_videos_grid
from tqdm import tqdm
from omegaconf import OmegaConf

def inference(
    pretrained_model_path: str,
    model_path: str,
    prompt_file: str,
    test_n: int,
    inv_latent_path: str,
    frames_per_video: int,
    height: int,
    width: int,
    inference_steps: int,
    guidance_scale: float
):
    curr_video = 0
    
    # Setup model
    unet = UNet3DConditionModel.from_pretrained(model_path, subfolder='unet', torch_dtype=torch.float16).to('cuda')
    pipe = TuneAVideoPipeline.from_pretrained(pretrained_model_path, unet=unet, torch_dtype=torch.float16).to("cuda")
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_vae_slicing()

    # Perform inference
    ddim_inv_latent = torch.load(inv_latent_path).to(torch.float16)
    initial_noise = torch.rand_like(ddim_inv_latent)
    with open(prompt_file, 'r') as file:
        lines = file.readlines()
        
        for line in tqdm(lines, desc='Inference'):
            video_name, prompt = line.strip().split(':')
            curr_video += 1
        
            video = pipe(
                prompt, 
                latents=initial_noise, 
                video_length=frames_per_video,
                height=height,
                width=width, 
                num_inference_steps=inference_steps,
                guidance_scale=guidance_scale
            ).videos

            # Save generated video
            save_path = f'./inference_samples/inference_samples_{test_n}/{curr_video}.gif'
            save_videos_grid(video, save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/inference.yaml")
    args = parser.parse_args()
    
    inference(**OmegaConf.load(args.config))