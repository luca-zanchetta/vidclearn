import os

from src.fvd import compute_fvd
from src.fid import compute_fid
from src.inception_score import compute_is
from src.clip import compute_clip_score

# Hyperparameters
real_videos_folder = './animatediff/data/MSRVTT/videos/test'
generated_videos_folder = './inference_samples/inference_samples_4/inference/samples'
prompts_file = './configs/inference/inference.yaml'
num_videos_tot = len(os.listdir(real_videos_folder))
frame_size = (128, 128)
frames_per_video = 16

# Compute FVD
print("\n[INFO] Computing FVD [1/4]...")
fvd = compute_fvd(real_videos_folder, generated_videos_folder, num_videos_tot, frame_size, frames_per_video)

# Compute FID
print("\n[INFO] Computing FID [2/4]...")
fid_score = compute_fid(real_videos_folder, generated_videos_folder, frame_size=frame_size)

# Compute IS
print("\n[INFO] Computing Inception Score [3/4]...")
is_score, std_deviation = compute_is(generated_videos_folder, frame_size)

# Compute CLIP score
print("\n[INFO] Computing CLIP Score [4/4]...")
clip_score = compute_clip_score(generated_videos_folder, prompts_file, frame_size, frames_per_video)

# Metrics recap
print("\n******************* METRICS RESULTS ************************\n")
print(f'[RES] FVD:', round(fvd, 5))
print(f'[RES] FID: {round(fid_score.item(), 5)}')
print(f'[RES] IS: {round(is_score.item(), 5)}')
print(f'[RES] IS Standard Deviation: {round(std_deviation.item(), 5)}')
print(f'[RES] CLIP Score: {round(clip_score.item(), 5)}')
print("**************************************************************")