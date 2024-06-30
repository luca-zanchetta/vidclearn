import torch
import cv2

from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

# Function for extracting frames from video
def extract_frames_from_video(video, target_size=(299, 299)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames = video.permute(1, 2, 3, 0).numpy()
    frames = [cv2.resize(frame, target_size) for frame in frames]
    frames = [torch.tensor(frame, dtype=torch.uint8).permute(2, 0, 1).unsqueeze(0).to(device) for frame in frames]
    return frames

def compute_fid(real_videos, generated_videos, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fid = FrechetInceptionDistance(feature=64).to(device)
    
    for video in tqdm(real_videos, desc='Processing Real Videos', total=len(real_videos)):
        frames = extract_frames_from_video(video)
        for i in range(0, len(frames), batch_size):
            batch = torch.cat(frames[i:i + batch_size])
            fid.update(batch, real=True)

    for video in tqdm(generated_videos, desc='Processing Generated Videos', total=len(generated_videos)):
        frames = extract_frames_from_video(video)
        for i in range(0, len(frames), batch_size):
            batch = torch.cat(frames[i:i + batch_size])
            fid.update(batch, real=False)
    
    fid_score = fid.compute()
    return fid_score
