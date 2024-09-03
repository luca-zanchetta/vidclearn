import os
import datetime

from src.fvd import compute_fvd
from src.fid import compute_fid
from src.inception_score import compute_is
from src.clip_multiple import compute_clip_score
from src.utils import load_video
from tqdm import tqdm

# Hyperparameters
test_n = 1
real_videos_folder = './data/test_videos'
generated_videos_folder = f'./inference_samples/inference_samples_{test_n}/'
prompts_file = './data/test_captions.txt'
num_videos_tot = len(os.listdir(real_videos_folder))
frame_size = (512, 512)
frames_per_video = 24

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
clip_score = compute_clip_score(generated_videos, prompts_file, frame_size)

# Metrics recap
print("\n******************* METRICS RESULTS ************************\n")
print(f'[RES] FVD:', round(fvd, 5))
print(f'[RES] FID: {round(fid_score.item(), 5)}')
print(f'[RES] IS: {round(is_score, 5)}')
print(f'[RES] CLIP Score: {round(clip_score.item()*100, 5)}')
print("**************************************************************")