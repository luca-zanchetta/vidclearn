# Code adapted from https://huggingface.co/ali-vilab/text-to-video-ms-1.7b
import torch
import os
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
from tqdm import tqdm

# Hyperparameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
generated_videos_folder = './inference_samples/comparison/modelscope_t2v'
prompts_file = './data/test_captions.txt'
height = 512
width = 512
frames_per_video = 24
inference_steps = 150
guidance_scale = 12.5

os.makedirs(generated_videos_folder, exist_ok=True)

# Load model
pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# Optimize for GPU memory
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()

# generate
# Perform inference
with open(prompts_file, "r") as file:
    lines = file.readlines()
    i = 1
    
    for line in tqdm(lines, desc='ModelScope_T2V Inference'):
        video_name, prompt = line.strip().split(':')
        
        video_frames = pipe(
            prompt=prompt,
            num_inference_steps=inference_steps,
            height=height,
            width=width,
            num_frames=frames_per_video,
            guidance_scale=guidance_scale,
            generator=torch.Generator(device).manual_seed(33),
        ).frames
        export_to_video(video_frames, generated_videos_folder + f"/{i}.gif")
        i += 1
