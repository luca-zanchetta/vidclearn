import os
import re
import torch
import argparse
from PIL import Image
from moviepy.editor import VideoFileClip
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
from omegaconf import OmegaConf
from src.utils import save_frames_to_video

def post_processing(
    video_path: str,
    prompts_file: str,
    avg_clip_score: float,
    output_folder: str,
):
    os.makedirs(output_folder, exist_ok=True)

    # Load the CLIP model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    with open(prompts_file, 'r') as file:
        prompts = file.readlines()
        
        # Extract video number
        file_name = os.path.basename(video_path)
        match = re.search(r'(\d+)\.gif$', file_name)
        if match:
            gif_number = int(match.group(1))
        else:
            gif_number = None
        if gif_number is not None:
            prompt = prompts[gif_number-1]
        
        # Load video
        video = VideoFileClip(video_path)
        
        # Extract and process frames
        frames = []
        for t in tqdm(range(0, int(video.duration * video.fps)), desc='Processing Frames...'):
            frame = video.get_frame(t / video.fps)
            frame = Image.fromarray(frame).convert('RGB')
            
            # Preprocess the image and prompt
            inputs = processor(text=[prompt], images=frame, return_tensors="pt", padding=True)
            
            # Get image and text features
            with torch.no_grad():
                outputs = model(**inputs)
                image_features = outputs.image_embeds
                text_features = outputs.text_embeds
            
            # Normalize the features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Compute cosine similarity (CLIP score)
            clip_score = (image_features * text_features).sum(dim=-1).item()
            clip_score = round((clip_score*100), 3)
            print(f"[INFO] Frame {t}, CLIP Score: {clip_score}")
            
            if clip_score >= avg_clip_score:
                frames.append(frame)
                
        save_frames_to_video(frames, output_folder + str(gif_number) + ".gif", fps=video.fps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/post_processing.yaml")
    args = parser.parse_args()
    
    post_processing(**OmegaConf.load(args.config))