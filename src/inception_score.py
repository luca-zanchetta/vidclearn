import torch
import torchvision.transforms as transforms
import os
import glob
from torchvision.io import read_video
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from torchmetrics.image.inception import InceptionScore

# Define a function to extract frames from videos
def extract_frames(video_path, frame_size=(256, 256)):
    video_frames, _, _ = read_video(video_path, pts_unit='sec')

    transform = transforms.Compose([
        transforms.Resize(frame_size),
        transforms.CenterCrop(frame_size),
        transforms.ToTensor(),
    ])

    # Permute dimensions to match (H, W, C) for each frame
    frames = [transform(to_pil_image(frame)) for frame in video_frames.permute(0, 3, 1, 2)]
    return torch.stack(frames)

# Define a function to load all frames from a folder of videos
def load_video_frames_from_folder(folder_path, frame_size):
    print("[INFO] Loading video frames...")
    all_frames = []
    video_paths = glob.glob(os.path.join(folder_path, '*.mp4'))
    for video_path in tqdm(video_paths, total=len(video_paths)):
        frames = extract_frames(video_path, frame_size)
        all_frames.extend(frames)
    return torch.stack(all_frames)


def compute_is(generated_folder, frame_size):
    # Load frames from the generated video folder
    generated_frames = load_video_frames_from_folder(generated_folder, frame_size)

    # Convert frames to uint8
    generated_frames = (generated_frames * 255).byte()

    # Initialize the InceptionScore metric
    inception_score = InceptionScore()

    # Compute IS score
    generated_is_score = inception_score(generated_frames)
    is_score, std_deviation = generated_is_score
    
    return is_score, std_deviation

if __name__ == "__main__":
    # Paths to the folder containing generated videos
    generated_folder = './provainference'
    frame_size = (128,128)

    is_score, std_deviation = compute_is(generated_folder, frame_size)
    print(f'[INFO] IS Score: {round(is_score.item(), 5)}')
    print(f'[INFO] Standard deviation: {round(std_deviation.item(), 5)}')

