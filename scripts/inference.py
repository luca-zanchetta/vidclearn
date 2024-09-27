import torch
from tuneavideo.pipelines.pipeline_tuneavideo import TuneAVideoPipeline
from tuneavideo.models.unet import UNet3DConditionModel
from tuneavideo.util import save_videos_grid
from tqdm import tqdm

# Hyperparameters
pretrained_model_path = "./checkpoints/stable-diffusion-v1-4"
model_path = "./final_model/model-1"
prompt_file = "./data/test_captions.txt"
curr_video = 0
test_n = 1
inv_latent_path = f"final_model/inv_latents/ddim_latent-500.pt"
frames_per_video = 24
height = 512
width = 512
inference_steps = 150
guidance_scale = 12.5

# Setup model
unet = UNet3DConditionModel.from_pretrained(model_path, subfolder='unet', torch_dtype=torch.float16).to('cuda')
pipe = TuneAVideoPipeline.from_pretrained(pretrained_model_path, unet=unet, torch_dtype=torch.float16).to("cuda")
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_vae_slicing()

# Perform inference
ddim_inv_latent = torch.load(inv_latent_path).to(torch.float16)
with open(prompt_file, 'r') as file:
    lines = file.readlines()
    
    for line in tqdm(lines, desc='Inference'):
        video_name, prompt = line.strip().split(':')
        curr_video += 1
    
        video = pipe(
            prompt, 
            latents=ddim_inv_latent, 
            video_length=frames_per_video,
            height=height,
            width=width, 
            num_inference_steps=inference_steps,
            guidance_scale=guidance_scale
        ).videos

        # Save generated video
        save_path = f'./inference_samples/inference_samples_{test_n}/{curr_video}.gif'
        save_videos_grid(video, save_path)