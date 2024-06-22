import cv2
import torch
import torchvision.models as models
import numpy as np
import os
import itertools

from scipy.linalg import sqrtm
from tqdm import tqdm

# Define function to load and preprocess video
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
    
    video = np.array(frames, dtype=np.float32)
    video = torch.tensor(video, dtype=torch.float32)
    return video.permute(3, 0, 1, 2)  # Convert to (C, T, H, W)

# Function to calculate Fréchet Distance
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return np.sum(diff**2) + np.trace(sigma1 + sigma2 - 2 * covmean)

def extract_features(video_path, model, num_frames=16, frame_size=(256, 256), mean=None, std=None, device='cpu'):
    video = load_video(video_path, num_frames=num_frames, frame_size=frame_size)
    if video is None:
        return None  # Return None if video couldn't be loaded
    
    video = video / 255.0  # Normalize pixel values to [0, 1]

    if mean is not None and std is not None:
        video = (video - mean[:, None, None, None]) / std[:, None, None, None]

    video = video.unsqueeze(0).to(device)  # Add batch dimension and move to device

    with torch.no_grad():
        features = model(video).cpu().numpy()
    return features


def compute_fvd(real_videos_folder, generated_videos_folder, num_videos_tot, image_size, frames_per_video):
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    real_features = []
    generated_features = []

    print("[INFO] Extracting features...\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    i3d_model = models.video.r3d_18(pretrained=True).to(device)
    i3d_model.eval()

    for real_video_name, generated_video_name in tqdm(itertools.zip_longest(os.listdir(real_videos_folder), os.listdir(generated_videos_folder), fillvalue=None), total=num_videos_tot):
        if real_video_name is None or generated_video_name is None:
            continue

        real_video_path = os.path.join(real_videos_folder, real_video_name)
        generated_video_path = os.path.join(generated_videos_folder, generated_video_name)

        real_feature = extract_features(real_video_path, i3d_model, num_frames=frames_per_video, frame_size=image_size, mean=mean, std=std, device=device)
        generated_feature = extract_features(generated_video_path, i3d_model, num_frames=frames_per_video, frame_size=image_size, mean=mean, std=std, device=device)

        if real_feature is not None and generated_feature is not None:
            real_features.append(real_feature)
            generated_features.append(generated_feature)

    real_features = np.array(real_features).reshape(len(real_features), -1)
    generated_features = np.array(generated_features).reshape(len(generated_features), -1)

    if real_features.shape[0] == 1:
        real_features = np.vstack([real_features, real_features])
    if generated_features.shape[0] == 1:
        generated_features = np.vstack([generated_features, generated_features])

    mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu_generated, sigma_generated = np.mean(generated_features, axis=0), np.cov(generated_features, rowvar=False)

    sigma_real = (sigma_real + sigma_real.T) / 2
    sigma_generated = (sigma_generated + sigma_generated.T) / 2

    # Adding small value to covariance matrices for numerical stability
    eps = 1e-6
    sigma_real += np.eye(sigma_real.shape[0]) * eps
    sigma_generated += np.eye(sigma_generated.shape[0]) * eps

    fvd = calculate_frechet_distance(mu_real, sigma_real, mu_generated, sigma_generated)
    return fvd

if __name__ == "__main__":
    real_videos_folder = './provatest'
    generated_videos_folder = './provainference'
    num_videos_tot = len(os.listdir(real_videos_folder))
    image_size = (128, 128)
    frames_per_video = 16

    fvd = compute_fvd(real_videos_folder, generated_videos_folder, num_videos_tot, image_size, frames_per_video)
    print('[INFO] FVD:', round(fvd, 5))