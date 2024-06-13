import torch
import numpy as np
import cv2
import os
import itertools

from torchvision.models import inception_v3
from torch.nn.functional import adaptive_avg_pool2d
from scipy.linalg import sqrtm

def get_inception_activations(images, model, batch_size=32):
    model.eval()
    activations = []

    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            if batch.shape[1] == 1:  # Convert grayscale to RGB
                batch = batch.repeat(1, 3, 1, 1)
            batch = adaptive_avg_pool2d(model(batch), (1, 1)).squeeze(3).squeeze(2)
            activations.append(batch)

    activations = torch.cat(activations, dim=0)
    return activations.cpu().numpy()


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset) @ (sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    mean_diff = mu1 - mu2
    mean_norm = mean_diff @ mean_diff

    trace_covmean = np.trace(covmean)

    return mean_norm + np.trace(sigma1) + np.trace(sigma2) - 2 * trace_covmean


def extract_frames_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()
    return frames


def preprocess_frames(frames, image_size=(256, 256)):
    tensor_frames = []
    for frame in frames:
        frame = cv2.resize(frame, image_size)
        frame = torch.tensor(frame).permute(2, 0, 1).float() / 255.0  # Normalize to [0, 1]
        tensor_frames.append(frame)

    tensor_frames = torch.stack(tensor_frames)
    return tensor_frames


def compute_fid(real_video_path, generated_video_path, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inception = inception_v3(pretrained=True, transform_input=False).to(device)
    inception.fc = torch.nn.Identity()  # Removing the final classification layer

    real_frames = extract_frames_from_video(real_video_path)
    gen_frames = extract_frames_from_video(generated_video_path)

    real_tensor_frames = preprocess_frames(real_frames).to(device)
    gen_tensor_frames = preprocess_frames(gen_frames).to(device)

    real_activations = get_inception_activations(real_tensor_frames, inception, batch_size)
    gen_activations = get_inception_activations(gen_tensor_frames, inception, batch_size)

    mu_real = np.mean(real_activations, axis=0)
    sigma_real = np.cov(real_activations, rowvar=False)
    mu_gen = np.mean(gen_activations, axis=0)
    sigma_gen = np.cov(gen_activations, rowvar=False)

    fid_score = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
    return fid_score


real_videos_folder = 'real_videos_folder'
generated_videos_folder = 'generated_videos_folder'
num_videos = 0
sum_fid = 0
image_size = (256,256)
frames_per_video = 16

for real_video_name, generated_video_name in itertools.zip_longest(os.listdir(real_videos_folder), os.listdir(generated_videos_folder), fillvalue=None):
    num_videos += 1
    
    real_video_path = os.path.join(real_videos_folder, real_video_name)
    generated_video_path = os.path.join(generated_videos_folder, generated_video_name)

    fid_score = compute_fid(real_video_path, generated_video_path)
    print(f'[INFO] FID Score for video {num_videos}: {fid_score}')
    sum_fid += fid_score
    
print(f"\n[INFO] Average FVD score: {sum_fid//num_videos}\n")