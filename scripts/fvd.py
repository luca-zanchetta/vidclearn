import cv2
import torch
import torchvision.transforms as transforms
import torchvision.models.video as models
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


real_videos_folder = 'real_videos_folder'
generated_videos_folder = 'generated_videos_folder'
num_videos = 0
sum_fvd = 0
image_size = (256,256)
frames_per_video = 16

for real_video_name, generated_video_name in itertools.zip_longest(os.listdir(real_videos_folder), os.listdir(generated_videos_folder), fillvalue=None):
    num_videos += 1
    
    real_video_path = os.path.join(real_videos_folder, real_video_name)
    generated_video_path = os.path.join(generated_videos_folder, generated_video_name)

    # Load real and generated videos
    real_video = load_video(real_video_path, num_frames=frames_per_video, frame_size=image_size)
    generated_video = load_video(generated_video_path, num_frames=frames_per_video, frame_size=image_size)

    # Preprocess videos for I3D model
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    real_video = transform(real_video)
    generated_video = transform(generated_video)

    # Add batch dimension
    real_video = real_video.unsqueeze(0)  # (1, C, T, H, W)
    generated_video = generated_video.unsqueeze(0)  # (1, C, T, H, W)

    # Load pre-trained I3D model
    i3d_model = models.video.r3d_18(pretrained=True)
    i3d_model.eval()

    # Extract features
    with torch.no_grad():
        real_features = i3d_model(real_video).cpu().numpy().flatten()
        generated_features = i3d_model(generated_video).cpu().numpy().flatten()

    # Calculate mean and covariance of features
    mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu_generated, sigma_generated = np.mean(generated_features, axis=0), np.cov(generated_features, rowvar=False)

    # Compute FVD
    fvd = calculate_frechet_distance(mu_real, sigma_real, mu_generated, sigma_generated)
    print(f'[INFO] FVD for video {num_videos}:', fvd)
    sum_fvd += fvd
    
print(f"\n[INFO] Average FVD score: {sum_fvd//num_videos}\n")