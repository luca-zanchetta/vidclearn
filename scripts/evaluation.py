import os
import argparse
from src.fvd import compute_fvd
from src.fid import compute_fid
from src.inception_score import compute_is
from src.clip_multiple import compute_clip_score
from src.fwt import compute_fwt
from src.bwt import compute_bwt
from src.evolution_rate import compute_evolution_rate
from src.utils import load_video
from tqdm import tqdm
from omegaconf import OmegaConf

def evaluation(
    generated_videos_folder: str,
    real_videos_folder: str,
    train_videos_folder: str,
    clip_file_test: str,
    clip_file_train_middle: str,
    clip_file_train_end: str,
    prompts_file: str,
    height: int,
    width: int,
    frames_per_video: int
):
    frame_size = (width, height)
    tot_train_videos = len(os.listdir(train_videos_folder))

    # Load videos
    print("\n[INFO] Loading real videos...")
    videos = os.listdir(real_videos_folder)
    videos = sorted(videos)
    real_videos = []
    for path in tqdm(videos, total=len(videos)):
        video_path = os.path.join(real_videos_folder, path)
        real_videos.append(load_video(video_path, frames_per_video, frame_size))

    print("\n[INFO] Loading generated videos...")
    videos = os.listdir(generated_videos_folder)
    videos = sorted(videos, key=lambda x: int(x.split('.')[0]))
    generated_videos = []
    for path in tqdm(videos, total=len(videos)):
        video_path = os.path.join(generated_videos_folder, path)
        generated_videos.append(load_video(video_path, frames_per_video, frame_size))

    # Compute FVD
    print("\n[INFO] Computing FVD [1/7]...")
    fvd = compute_fvd(real_videos, generated_videos)

    # Compute FID
    print("\n[INFO] Computing FID [2/7]...")
    fid_score = compute_fid(real_videos, generated_videos)

    # Compute IS
    print("\n[INFO] Computing Inception Score [3/7]...")
    is_score, std_deviation = compute_is(generated_videos)

    # Compute CLIP score
    print("\n[INFO] Computing CLIP Score [4/7]...")
    clip_score = compute_clip_score(generated_videos, prompts_file, frame_size)

    # Compute FWT
    print("\n[INFO] Computing FWT [5/7]...")
    fwt = compute_fwt(tot_train_videos, clip_file_test)

    # Compute BWT
    print("\n[INFO] Computing BWT [6/7]...")
    bwt = compute_bwt(tot_train_videos, clip_file_train_middle, clip_file_train_end)

    # Compute Evolution Rate
    print("\n[INFO] Computing Evolution Rate [7/7]...")
    compute_evolution_rate(clip_file_test)
    print("[WARNING] The evolution rates can be plotted by executing the command \'python -m plots.plot_evo_rates\'")

    # Metrics recap
    print("\n******************* METRICS RESULTS ************************\n")
    print(f'[RES] FVD:', round(fvd, 3))
    print(f'[RES] FID: {round(fid_score.item(), 3)}')
    print(f'[RES] IS: {round(is_score, 3)}')
    print(f'[RES] CLIP Score: {round(clip_score.item()*100, 3)}')
    print(f'[RES] FWT: {round(fwt, 3)}')
    print(f'[RES] BWT: {round(bwt, 3)}')
    print("**************************************************************")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/evaluation.yaml")
    args = parser.parse_args()
    
    evaluation(**OmegaConf.load(args.config))