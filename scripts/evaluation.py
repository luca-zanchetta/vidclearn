import os
import datetime
import torch
import numpy as np
import cv2

from src.fvd import compute_fvd
from src.fid import compute_fid
from src.inception_score import compute_is
from src.clip import compute_clip_score
from tqdm import tqdm

# Function to load and preprocess video
def load_video(video_path, num_frames=16, frame_size=(256, 256)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < num_frames and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, frame_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    
    if len(frames) == 0:
        return None  # Return None if no frames were read
    
    # If video has fewer frames than num_frames, pad with the last frame
    while len(frames) < num_frames:
        frames.append(frames[-1])
    
    video = np.array(frames, dtype=np.uint8)
    video = torch.tensor(video, dtype=torch.uint8)
    return video.permute(3, 0, 1, 2)  # Convert to (C, T, H, W)


# Hyperparameters
real_videos_folder = './animatediff/data/MSRVTT/videos/test'
generated_videos_folder = './inference_samples/inference_samples_1/inference/samples'
prompts_file = './configs/inference/inference.yaml'
num_videos_tot = len(os.listdir(real_videos_folder))
frame_size = (128, 128)
frames_per_video = 16

# Load videos
print("\n[INFO] Loading real videos...")
real_videos = []
for path in tqdm(os.listdir(real_videos_folder), total=len(os.listdir(real_videos_folder))):
    video_path = os.path.join(real_videos_folder, path)
    real_videos.append(load_video(video_path, frames_per_video, frame_size))

print("\n[INFO] Loading generated videos...")
generated_videos = []
for path in tqdm(os.listdir(generated_videos_folder), total=len(os.listdir(generated_videos_folder))):
    video_path = os.path.join(generated_videos_folder, path)
    generated_videos.append(load_video(video_path, frames_per_video, frame_size))

# Compute FVD
print("\n[INFO] Computing FVD [1/4]...")
fvd = compute_fvd(real_videos, generated_videos)

# Compute FID
print("\n[INFO] Computing FID [2/4]...")
fid_score = compute_fid(real_videos, generated_videos)

# Compute IS
print("\n[INFO] Computing Inception Score [3/4]...")
is_score, std_deviation = compute_is(generated_videos)

# Compute CLIP score
print("\n[INFO] Computing CLIP Score [4/4]...")
clip_score = compute_clip_score(generated_videos, prompts_file, frame_size, frames_per_video)

# Metrics recap
print("\n******************* METRICS RESULTS ************************\n")
print(f'[RES] FVD:', round(fvd, 5))
print(f'[RES] FID: {round(fid_score.item(), 5)}')
print(f'[RES] IS: {round(is_score, 5)}')
print(f'[RES] IS Standard Deviation: {round(std_deviation, 5)}')
print(f'[RES] CLIP Score: {round(clip_score.item(), 5)}')
print("**************************************************************")

# Save results
now =  datetime.datetime.now()
formatted_datetime = now.strftime('%Y_%m_%d_%H_%M_%S')

output_strings = [
    f'[RES] FVD: {round(fvd, 5)}\n',
    f'[RES] FID: {round(fid_score.item(), 5)}\n',
    f'[RES] IS: {round(is_score, 5)}\n',
    f'[RES] IS Standard Deviation: {round(std_deviation, 5)}\n',
    f'[RES] CLIP Score: {round(clip_score.item(), 5)}\n'
]

with open(f'./evaluation_results/metrics_{formatted_datetime}.txt', 'w') as file:
    file.writelines(output_strings)