import torch
import torchvision.models as models
import numpy as np
import os
import itertools

from scipy.linalg import sqrtm
from tqdm import tqdm

# Function to calculate Fréchet Distance
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return np.sum(diff**2) + np.trace(sigma1 + sigma2 - 2 * covmean)

def extract_features(video, model, mean=None, std=None, device='cpu'):
    video = video / 255.0  # Normalize pixel values to [0, 1]

    if mean is not None and std is not None:
        video = (video - mean[:, None, None, None]) / std[:, None, None, None]

    video = video.unsqueeze(0).to(device)  # Add batch dimension and move to device

    with torch.no_grad():
        features = model(video).cpu().numpy()
    return features


def compute_fvd(real_videos, generated_videos):
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    real_features = []
    generated_features = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    i3d_model = models.video.r3d_18(pretrained=True).to(device)
    i3d_model.eval()

    for real_video, generated_video in tqdm(itertools.zip_longest(real_videos, generated_videos, fillvalue=None), total=len(generated_videos)):
        real_feature = extract_features(real_video, i3d_model, device=device)
        generated_feature = extract_features(generated_video, i3d_model, mean=mean, std=std, device=device)

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