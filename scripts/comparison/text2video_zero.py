# Code adapted from https://huggingface.co/docs/diffusers/api/pipelines/text_to_video_zero
import torch
import os
from diffusers.utils import export_to_video
from diffusers import TextToVideoZeroPipeline
from tqdm import tqdm

# Hyperparameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
generated_videos_folder = './inference_samples/comparison/text2video_zero'
prompts_file = './data/test_captions.txt'
height = 512
width = 512
frames_per_video = 24
inference_steps = 150
guidance_scale = 12.5

os.makedirs(generated_videos_folder, exist_ok=True)

# Load model
model_id = "sd-legacy/stable-diffusion-v1-5"
pipe = TextToVideoZeroPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

# Optimize for GPU memory
pipe.enable_vae_slicing()

# Perform inference
with open(prompts_file, "r") as file:
    lines = file.readlines()
    i = 1
    
    for line in tqdm(lines, desc='Text2Video_Zero Inference'):
        video_name, prompt = line.strip().split(':')
        
        result = pipe(
            prompt=prompt,
            video_length=frames_per_video,
            height=height,
            width=width,
            num_inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            generator=torch.Generator(device).manual_seed(33),
        ).images
        result = [(r * 255).astype("uint8") for r in result]
        export_to_video(result, generated_videos_folder + f"/{i}.gif")
        i += 1