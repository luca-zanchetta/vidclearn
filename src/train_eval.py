import torch
import os
import shutil

from tuneavideo.pipelines.pipeline_tuneavideo import TuneAVideoPipeline
from tuneavideo.models.unet import UNet3DConditionModel
from diffusers import DDIMScheduler
from tuneavideo.util import ddim_inversion, save_videos_grid
from src.clip_multiple import compute_clip_score
from src.clip import compute_clip_score_single
from src.utils import load_video
from tqdm import tqdm

def init_eval_test(model_path, prompt_file, validation_data, accelerator):
    # Setup model
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    unet = UNet3DConditionModel.from_pretrained_2d(model_path, subfolder='unet').to(accelerator.device, dtype=weight_dtype)
    pipe = TuneAVideoPipeline.from_pretrained(model_path, unet=unet, torch_dtype=torch.float16).to(accelerator.device)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_vae_slicing()
    ddim_inv_scheduler = DDIMScheduler.from_pretrained(model_path, subfolder='scheduler')
    ddim_inv_scheduler.set_timesteps(validation_data.num_inv_steps)
    
    # Perform DDIM Inversion with random latents
    random_latents = torch.randn((1, 4, validation_data.video_length, validation_data.height // 8, validation_data.width // 8)).to(weight_dtype).to(accelerator.device)
    ddim_inv_latent = None
    if validation_data.use_inv_latent:
        ddim_inv_latent = ddim_inversion(
            pipe, ddim_inv_scheduler, video_latent=random_latents,
            num_inv_steps=validation_data.num_inv_steps, prompt="")[-1].to(weight_dtype)
    
    # Perform inference    
    with open(prompt_file, 'r') as file:
        lines = file.readlines()
        
        for idx, line in tqdm(enumerate(lines), desc='Initial Inference', total=len(lines)):
            video_name, prompt = line.strip().split(':')
        
            video = pipe(
                prompt, 
                latents=ddim_inv_latent,
                video_length=validation_data.video_length,
                height=validation_data.height,
                width=validation_data.width, 
                num_inference_steps=validation_data.num_inference_steps,
                guidance_scale=validation_data.guidance_scale
            ).videos
            
            # Save generated video
            save_path = f'./tmp/{idx+1}.gif'
            save_videos_grid(video, save_path)
    
    # Compute CLIP Score
    frame_size = (validation_data.width, validation_data.height)
    videos = os.listdir('./tmp')
    videos = sorted(videos, key=lambda x: int(x.split('.')[0]))
    generated_videos = []
    for path in tqdm(videos, total=len(videos)):
        video_path = os.path.join('./tmp', path)
        generated_videos.append(load_video(video_path, validation_data.video_length, frame_size))
    
    clip_score = compute_clip_score(generated_videos, prompt_file, frame_size, accelerator)
    clip_score = round(clip_score.item()*100, 5)
    
    try:
        shutil.rmtree('./tmp')
    except Exception as err:
        pass
    
    del videos, generated_videos, unet, pipe, ddim_inv_scheduler
    return clip_score

def middle_eval_test(model_path, unet, prompt_file, validation_data, inv_latents_path, accelerator):   
    # Setup model
    pipe = TuneAVideoPipeline.from_pretrained(model_path, unet=unet, torch_dtype=torch.float16).to(accelerator.device)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_vae_slicing()
    
    # Perform inference
    ddim_inv_latent = torch.load(inv_latents_path).to(torch.float16)
    with open(prompt_file, 'r') as file:
        lines = file.readlines()
        
        for idx, line in tqdm(enumerate(lines), desc='Inference on test set', total=len(lines)):
            video_name, prompt = line.strip().split(':')
        
            video = pipe(
                prompt,
                latents=ddim_inv_latent,
                video_length=validation_data.video_length,
                height=validation_data.height,
                width=validation_data.width, 
                num_inference_steps=validation_data.num_inference_steps,
                guidance_scale=validation_data.guidance_scale
            ).videos
            
            # Save generated video
            save_path = f'./tmp/{idx+1}.gif'
            save_videos_grid(video, save_path)
    
    # Compute CLIP Score
    frame_size = (validation_data.width, validation_data.height)
    videos = os.listdir('./tmp')
    videos = sorted(videos, key=lambda x: int(x.split('.')[0]))
    generated_videos = []
    for path in tqdm(videos, total=len(videos)):
        video_path = os.path.join('./tmp', path)
        generated_videos.append(load_video(video_path, validation_data.video_length, frame_size))
    
    clip_score = compute_clip_score(generated_videos, prompt_file, frame_size, accelerator)
    clip_score = round(clip_score.item()*100, 5)
    
    try:
        shutil.rmtree('./tmp')
    except Exception as err:
        pass
    
    del generated_videos, videos, ddim_inv_latent, pipe
    return clip_score

def middle_eval_train(model_path, unet, prompt, validation_data, inv_latents_path, accelerator):
    # Setup model
    pipe = TuneAVideoPipeline.from_pretrained(model_path, unet=unet, torch_dtype=torch.float16).to(accelerator.device)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_vae_slicing()
    
    # Perform inference
    ddim_inv_latent = torch.load(inv_latents_path).to(torch.float16)
    video = pipe(
        prompt,
        latents=ddim_inv_latent,
        video_length=validation_data.video_length,
        height=validation_data.height,
        width=validation_data.width, 
        num_inference_steps=validation_data.num_inference_steps,
        guidance_scale=validation_data.guidance_scale
    ).videos
    
    # Save generated video
    save_path = f'./tmp/tmp.gif'
    save_videos_grid(video, save_path)
    del video
    
    # Compute CLIP Score
    frame_size = (validation_data.width, validation_data.height)
    video = load_video(save_path, validation_data.video_length, frame_size)
    clip_score = compute_clip_score_single(video, prompt, frame_size)
    clip_score = round(clip_score.item()*100, 5)
    
    try:
        shutil.rmtree('./tmp')
    except Exception as err:
        pass
    
    del video, ddim_inv_latent, pipe
    return clip_score
    
def end_eval_train(pretrained_model_path, last_model_path, prompts, validation_data, inv_latents_path, mixed_precision, clip_file):
    # Setup model
    weight_dtype = torch.float32
    if mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    unet = UNet3DConditionModel.from_pretrained_2d(last_model_path, subfolder='unet').to('cuda', dtype=weight_dtype)
    pipe = TuneAVideoPipeline.from_pretrained(pretrained_model_path, unet=unet, torch_dtype=torch.float16).to('cuda')
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_vae_slicing()
    
    # Perform inference
    ddim_inv_latent = torch.load(inv_latents_path).to(torch.float16)
    for idx, prompt in tqdm(enumerate(prompts), desc='Inference on train set', total=len(prompts)):
        video = pipe(
            prompt,
            latents=ddim_inv_latent,
            video_length=validation_data.video_length,
            height=validation_data.height,
            width=validation_data.width, 
            num_inference_steps=validation_data.num_inference_steps,
            guidance_scale=validation_data.guidance_scale
        ).videos
        
        # Save generated video
        save_path = f'./tmp/tmp.gif'
        save_videos_grid(video, save_path)
        del video
    
        # Compute and save CLIP Score
        frame_size = (validation_data.width, validation_data.height)
        video = load_video(save_path, validation_data.video_length, frame_size)
        clip_score = compute_clip_score_single(video, prompt, frame_size)
        clip_score = round(clip_score.item()*100, 5)
        
        with open(clip_file, "a") as file:
            file.write(f"{idx+1}:{clip_score}\n")
            file.close()
    
    try:
        shutil.rmtree('./tmp')
    except Exception as err:
        pass
    
    del ddim_inv_latent, pipe, unet, video