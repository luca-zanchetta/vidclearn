import numpy as np
import torch
import cv2

from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

# Function for extracting frames from video
def extract_frames_from_video(video, target_size=(299, 299)):
    frames = video.permute(1, 2, 3, 0).numpy()  # Convert to (T, H, W, C)
    frames = [cv2.resize(frame, target_size) for frame in frames]  # Resize each frame to target_size
    frames = [torch.tensor(frame, dtype=torch.uint8).permute(2, 0, 1).unsqueeze(0) for frame in frames]  # Convert each frame to (C, H, W) and add batch dimension
    return frames


def compute_fid(real_videos, generated_videos):
    fid = FrechetInceptionDistance(feature=64)
    
    for video in tqdm(real_videos, total=len(real_videos)):
        frames = extract_frames_from_video(video)
        for frame in frames:
            fid.update(frame, real=True)
    
    for video in tqdm(generated_videos, total=len(generated_videos)):
        frames = extract_frames_from_video(video)
        for frame in frames:
            fid.update(frame, real=False)
    
    fid_score = fid.compute()
    return fid_score