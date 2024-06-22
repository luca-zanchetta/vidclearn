import os
import torch
import torchvision

from torchvision.io import read_video
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm


def load_video_frames(video_path, frame_size=(128, 128)):
    """Load video and return frames as a list of tensors."""
    video_frames, _, _ = read_video(video_path, output_format="TCHW", pts_unit='sec')
    video_frames = torchvision.transforms.functional.resize(video_frames, frame_size)
    return video_frames


def load_videos_from_folder(folder_path, frame_size=(128, 128)):
    """Load all videos from a folder and return them as a list of tensors."""    
    video_tensors = []
    num_videos_tot = len(os.listdir(folder_path))
    for filename in tqdm(os.listdir(folder_path), total=num_videos_tot):
        if filename.endswith((".mp4", ".avi", ".mov")):
            video_path = os.path.join(folder_path, filename)
            video_frames = load_video_frames(video_path, frame_size)
            video_tensors.append(video_frames)
    return video_tensors


def compute_fid(video_folder1, video_folder2, frame_size=(128, 128)):
    """Compute FID score between two folders of videos."""
    
    print("[INFO] Loading real videos...")
    video_tensors1 = load_videos_from_folder(video_folder1, frame_size)
    print("[INFO] Loading generated videos...")
    video_tensors2 = load_videos_from_folder(video_folder2, frame_size)
    
    print("[INFO] Computing FID scores...")
    fid = FrechetInceptionDistance(feature=64)
    
    for video in tqdm(video_tensors1, total=len(video_tensors1)):
        fid.update(video, real=True)
    
    for video in tqdm(video_tensors2, total=len(video_tensors2)):
        fid.update(video, real=False)
    
    fid_score = fid.compute()
    return fid_score

if __name__ == "__main__":
    # Paths to the folders containing the videos
    real_videos_folder = './provatest'
    generated_videos_folder = './provainference'
    frame_size = (128,128)

    # Compute FID score
    fid_score = compute_fid(real_videos_folder, generated_videos_folder, frame_size=frame_size)
    print(f"FID score: {round(fid_score.item(), 5)}")