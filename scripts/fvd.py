import cv2
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import os
import itertools
from scipy.linalg import sqrtm


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
    
    # If video has fewer frames than num_frames, pad with the last frame
    while len(frames) < num_frames:
        frames.append(frames[-1])
        
    video = np.array(frames, dtype=np.float32)
    video = torch.tensor(video, dtype=torch.float32)
    return video.permute(3, 0, 1, 2)  # Convert to (C, T, H, W)


# Function to calculate Fréchet Distance
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)


real_videos_folder = './animatediff/data/MSRVTT/videos/test'
generated_videos_folder = './inference_samples/inference_samples_1/inference/samples'
num_videos = 0
num_videos_tot = len(os.listdir(real_videos_folder))
sum_fvd = 0
image_size = (128, 128)
frames_per_video = 16

# Define mean and std for normalization
mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])

for real_video_name, generated_video_name in itertools.zip_longest(os.listdir(real_videos_folder), os.listdir(generated_videos_folder), fillvalue=None):
    if real_video_name is None or generated_video_name is None:
        continue
    
    num_videos += 1
    
    real_video_path = os.path.join(real_videos_folder, real_video_name)
    generated_video_path = os.path.join(generated_videos_folder, generated_video_name)

    # Load real and generated videos
    real_video = load_video(real_video_path, num_frames=frames_per_video, frame_size=image_size)
    generated_video = load_video(generated_video_path, num_frames=frames_per_video, frame_size=image_size)

    # Normalize videos
    real_video = real_video / 255.0  # Normalize pixel values to [0, 1]
    generated_video = generated_video / 255.0  # Normalize pixel values to [0, 1]

    # Manually normalize each frame
    real_video = (real_video - mean[:, None, None, None]) / std[:, None, None, None]
    generated_video = (generated_video - mean[:, None, None, None]) / std[:, None, None, None]

    # Add batch dimension
    real_video = real_video.unsqueeze(0)  # (1, C, T, H, W)
    generated_video = generated_video.unsqueeze(0)  # (1, C, T, H, W)

    # Load pre-trained I3D model
    i3d_model = models.video.r3d_18(pretrained=True)
    i3d_model.eval()

    # Extract features
    with torch.no_grad():
        real_features = i3d_model(real_video).cpu().numpy()
        generated_features = i3d_model(generated_video).cpu().numpy()

    # Reshape features to (samples, features)
    real_features = real_features.reshape(real_features.shape[0], -1)
    generated_features = generated_features.reshape(generated_features.shape[0], -1)

    # If there's only one sample, expand the dimensions to avoid issues with covariance calculation
    if real_features.shape[0] == 1:
        real_features = np.vstack([real_features, real_features])
    if generated_features.shape[0] == 1:
        generated_features = np.vstack([generated_features, generated_features])

    # Calculate mean and covariance of features
    mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu_generated, sigma_generated = np.mean(generated_features, axis=0), np.cov(generated_features, rowvar=False)

    # Check if the covariances are at least 2D
    if sigma_real.ndim < 2 or sigma_generated.ndim < 2:
        print(f'[ERROR] Covariance matrices must be at least 2D, got shapes {sigma_real.shape} and {sigma_generated.shape}')
        continue

    # Compute FVD
    fvd = calculate_frechet_distance(mu_real, sigma_real, mu_generated, sigma_generated)
    print(f'[INFO] FVD for video {num_videos}/{num_videos_tot}:', round(fvd, 5))
    sum_fvd += fvd
    
if num_videos > 0:
    print(f"\n[INFO] Average FVD score: {round((sum_fvd // num_videos), 5)}\n")
else:
    print("\n[INFO] No valid videos processed.\n")
