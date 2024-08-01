import torch
from tuneavideo.pipelines.pipeline_tuneavideo import TuneAVideoPipeline
from tuneavideo.models.unet import UNet3DConditionModel
from tuneavideo.util import save_videos_grid

# Hyperparameters
pretrained_model_path = "./checkpoints/stable-diffusion-v1-4"
model_path = "./outputs/man-skiing"
prompt = "spider man is skiing"
save_path = f"./inference_samples/1.gif"
inv_latent_path = f"{model_path}/inv_latents/ddim_latent-500.pt"
frames_per_video = 24
height = 512
width = 512
inference_steps = 50
guidance_scale = 12.5

# Setup model
unet = UNet3DConditionModel.from_pretrained(model_path, subfolder='unet', torch_dtype=torch.float16).to('cuda')
pipe = TuneAVideoPipeline.from_pretrained(pretrained_model_path, unet=unet, torch_dtype=torch.float16).to("cuda")
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_vae_slicing()

# Perform inference
ddim_inv_latent = torch.load(inv_latent_path).to(torch.float16)
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
save_videos_grid(video, save_path)