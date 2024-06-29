import torch
import numpy as np

from tqdm import tqdm
from torchmetrics.image.inception import InceptionScore

# Function for extracting frames from video
def extract_frames_from_video(video):
    # Convert the video tensor back to (T, H, W, C)
    video = video.permute(1, 2, 3, 0)
    video = video.numpy()
    
    # Convert each frame to a NumPy array and add to the frames list
    frames = [frame.astype(np.uint8) for frame in video]
    
    return frames

def compute_is(generated_videos):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load frames from the generated video folder
    generated_frames = []
    for video in tqdm(generated_videos, total=len(generated_videos)):
        frames = extract_frames_from_video(video)
        generated_frames.extend(frames)

    # Convert frames to uint8 tensor directly and scale to [0, 255] range
    # Ensure the frames are in the format (N, 3, H, W)
    generated_frames = torch.stack([
        torch.tensor(frame).permute(2, 0, 1) for frame in tqdm(generated_frames, total=len(generated_frames))
    ], dim=0).to(torch.uint8).to(device)

    # Initialize the InceptionScore metric
    inception_score = InceptionScore().to(device)
    
    # Compute IS score
    generated_is_score = inception_score(generated_frames)
    
    is_score, std_deviation = generated_is_score
    
    return is_score.item(), std_deviation.item()
