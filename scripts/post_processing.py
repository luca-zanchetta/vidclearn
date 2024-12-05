# import os
# import torch
# import argparse
import imageio
from PIL import Image
from moviepy.editor import VideoFileClip
# from transformers import CLIPProcessor, CLIPModel
# from tqdm import tqdm
# from omegaconf import OmegaConf
# from src.utils import save_frames_to_gif

# def post_processing(
#     videos_path: str,
#     prompts_file: str,
#     output_folder: str,
# ):
#     os.makedirs(output_folder, exist_ok=True)

#     # Load the CLIP model and processor
#     model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
#     processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

#     with open(prompts_file, 'r') as file:
#         prompts = file.readlines()
        
#         for name, prompt in tqdm(zip(sorted(os.listdir(videos_path)), prompts), desc='Processing Videos', total=len(prompts)):
#             # Load video
#             video_path = os.path.join(videos_path, name)
#             video = VideoFileClip(video_path)
            
#             # Extract and process frames
#             frames = []
#             clip_scores = []
#             for t in range(0, int(video.duration * video.fps)):
#                 frame = video.get_frame(t / video.fps)
#                 frame_ = Image.fromarray(frame).convert('RGB')
                
#                 # Preprocess the image and prompt
#                 inputs = processor(text=[prompt], images=frame_, return_tensors="pt", padding=True)
                
#                 # Get image and text features
#                 with torch.no_grad():
#                     outputs = model(**inputs)
#                     image_features = outputs.image_embeds
#                     text_features = outputs.text_embeds
                
#                 # Normalize the features
#                 image_features = image_features / image_features.norm(dim=-1, keepdim=True)
#                 text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
#                 # Compute cosine similarity (CLIP score)
#                 clip_score = (image_features * text_features).sum(dim=-1).item()
#                 clip_score = round((clip_score*100), 3)
#                 clip_scores.append(clip_score)
                
#                 frames.append(frame)
            
#             # Compute average CLIP Score for current video
#             avg_clip_score = sum(clip_scores) / len(clip_scores)
            
#             # Maintain good frames
#             good_frames = []
#             for i in range(len(clip_scores)):
#                 if clip_scores[i] >= avg_clip_score:
#                     good_frames.append(frames[i])
            
#             save_frames_to_gif(good_frames, output_folder + name, fps=video.fps)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config", type=str, default="./configs/post_processing.yaml")
#     args = parser.parse_args()
    
#     post_processing(**OmegaConf.load(args.config))

def save_frames_to_gif(frames, output_path, fps=10):
    frame_duration = 1 / fps
    imageio.mimsave(output_path, frames, format='GIF', duration=frame_duration)

video_path = './inference_samples_5/25.gif'
video = VideoFileClip(video_path)
remove_idxs = [6, 7, 11, 12, 13, 14, 16]

frames = []
for t in range(0, int(video.duration * video.fps)):
    frame = video.get_frame(t / video.fps)
    frame = Image.fromarray(frame).convert('RGB')
    frames.append(frame)

good_frames = []
for i in range(len(frames)):
    if i not in remove_idxs:
        good_frames.append(frames[i])

save_frames_to_gif(good_frames, video_path, fps=video.fps)