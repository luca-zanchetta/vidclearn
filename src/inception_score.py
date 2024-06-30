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

def compute_is(generated_videos, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load frames from the generated video folder
    generated_frames = []
    for video in tqdm(generated_videos, desc="Extracting frames", total=len(generated_videos)):
        frames = extract_frames_from_video(video)
        generated_frames.extend(frames)

    # Initialize the InceptionScore metric
    inception_score = InceptionScore().to(device)
    
    # Process in batches with a progress bar
    for i in tqdm(range(0, len(generated_frames), batch_size), desc="Computing Inception Score"):
        batch = generated_frames[i:i+batch_size]
        batch_tensors = torch.stack([
            torch.tensor(frame).permute(2, 0, 1) for frame in batch
        ], dim=0).to(torch.uint8).to(device)
        
        # Update the inception score with the batch
        inception_score.update(batch_tensors)
    
    # Compute the IS score
    generated_is_score = inception_score.compute()
    
    is_score, std_deviation = generated_is_score
    
    return is_score.item(), std_deviation.item()
