import os
from tqdm import tqdm

from src.utils import load_video
from src.ssim import compute_ssim
from src.psnr import compute_psnr
from src.clip_multiple import compute_clip_score
from src.fwe import compute_fwe
from src.of_consistency import compute_video_optical_flow, compute_optical_flow_consistency


# Hyperparameters
test_n = 1
train_videos_folder = './data/train_videos'
generated_videos_folder = f'./inference_samples/inference_samples_{test_n}/'
prompts_file = './data/ChatGPT_test_captions.txt'
frame_size = (512, 512)
frames_per_video = 20


# Load videos
print("\n[INFO] Loading real videos...")
real_videos = []
for path in tqdm(os.listdir(train_videos_folder), total=len(os.listdir(train_videos_folder))):
    video_path = os.path.join(train_videos_folder, path)
    real_videos.append(load_video(video_path, frames_per_video, frame_size))

print("\n[INFO] Loading generated videos...")
generated_videos = []
for path in tqdm(os.listdir(generated_videos_folder), total=len(os.listdir(generated_videos_folder))):
    video_path = os.path.join(generated_videos_folder, path)
    generated_videos.append(load_video(video_path, frames_per_video, frame_size))


# Compute CLIP score
print("\n[INFO] Computing CLIP Score [1/5]...")
clip_score = compute_clip_score(generated_videos, prompts_file, frame_size)


# Compute Average SSIM Score:
print("\n[INFO] Computing Average SSIM Score [2/5]...")
scores = []
for generated_video, real_video in tqdm(zip(generated_videos, real_videos)):
    ssim_score = compute_ssim(generated_video, real_video, frame_size)
    scores.append(ssim_score)
avg_ssim = sum(scores)/len(scores)


# Compute PSNR:
print("\n[INFO] Computing Average PSNR [3/5]...")
scores = []
for generated_video, real_video in tqdm(zip(generated_videos, real_videos)):
    psnr = compute_psnr(generated_video, real_video)
    scores.append(psnr)
avg_psnr = sum(scores) / len(scores)


# Compute Flow Warping Error
print("\n[INFO] Computing Average Flow Warping Error [4/5]...")
scores = []
for generated_video, real_video in tqdm(zip(generated_videos, real_videos)):
    fwe = compute_fwe(generated_video, real_video, frame_size)
    scores.append(fwe)
avg_fwe = sum(scores) / len(scores)


# Compute Optical Flow Consistency
print("\n[INFO] Computing Average Optical Flow Consistency [5/5]...")
scores = []
for generated_video, real_video in tqdm(zip(generated_videos, real_videos)):
    generated_flows = compute_video_optical_flow(generated_video)
    reference_flows = compute_video_optical_flow(real_video)
    consistency_metric = compute_optical_flow_consistency(generated_flows, reference_flows)
    scores.append(consistency_metric)
avg_ofc = sum(scores) / len(scores)


# Metrics recap
print("\n******************* METRICS RESULTS ************************\n")
print(f'[RES] CLIP Score: {round(clip_score.item() * 100, 5)}')
print(f'[RES] Average SSIM: {round(avg_ssim, 5)}')
print(f'[RES] Average PSNR: {round(avg_psnr.item(), 5)} dB')
print(f'[RES] Average Flow Warping Error: {round(avg_fwe, 5)}')
print(f'[RES] Average Optical Flow Consistency: {round(avg_ofc, 5)}')
print("**************************************************************")