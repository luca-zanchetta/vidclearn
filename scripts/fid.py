import torch
import numpy as np
import cv2
import os
import itertools
import scipy.linalg

from torchvision.models import inception_v3
from torch.nn.functional import adaptive_avg_pool2d
from tqdm import tqdm


def get_inception_activations(images, model, batch_size=32):
    model.eval()
    activations = []

    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            if batch.shape[1] == 1:  # Convert grayscale to RGB
                batch = batch.repeat(1, 3, 1, 1)
            batch = model(batch)
            if batch.dim() == 4:
                batch = adaptive_avg_pool2d(batch, (1, 1)).squeeze(3).squeeze(2)
            activations.append(batch)

    activations = torch.cat(activations, dim=0)
    return activations.cpu().numpy()


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-5):
    # Ensure covariance matrices are positive semi-definite
    eigenvalues1, _ = np.linalg.eig(sigma1)
    eigenvalues2, _ = np.linalg.eig(sigma2)
    if np.any(eigenvalues1 < 0) or np.any(eigenvalues2 < 0):
        print("Covariance matrices are not positive semi-definite. Adding epsilon to diagonal.")
        sigma1 += eps * np.eye(sigma1.shape[0])
        sigma2 += eps * np.eye(sigma2.shape[0])

    # Compute the square root of the matrix product
    covmean = scipy.linalg.sqrtm(sigma1 @ sigma2)

    # Compute the FID score
    mean_diff = mu1 - mu2
    mean_norm = mean_diff @ mean_diff
    trace_covmean = np.trace(covmean)

    # Ensure the FID score is real-valued
    fid_score = mean_norm + np.trace(sigma1) + np.trace(sigma2) - 2 * np.real(trace_covmean)
    return fid_score
    

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


def compute_fid(real_videos_folder, generated_videos_folder, batch_size=32):
    real_features = []
    generated_features = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inception = inception_v3(pretrained=True, transform_input=False).to(device)
    inception.fc = torch.nn.Identity()  # Removing the final classification layer
    
    print("[INFO] Extracting features...\n")
    
    for real_video_name, generated_video_name in tqdm(itertools.zip_longest(os.listdir(real_videos_folder), os.listdir(generated_videos_folder), fillvalue=None), total=num_videos_tot):
        if real_video_name is None or generated_video_name is None:
            continue
        
        real_video_path = os.path.join(real_videos_folder, real_video_name)
        generated_video_path = os.path.join(generated_videos_folder, generated_video_name)

        real_frames = extract_frames_from_video(real_video_path)
        gen_frames = extract_frames_from_video(generated_video_path)

        real_tensor_frames = preprocess_frames(real_frames).to(device)
        gen_tensor_frames = preprocess_frames(gen_frames).to(device)

        real_activations = get_inception_activations(real_tensor_frames, inception, batch_size)
        real_features.append(real_activations)
        
        gen_activations = get_inception_activations(gen_tensor_frames, inception, batch_size)
        generated_features.append(gen_activations)

    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    mu_gen = np.mean(generated_features, axis=0)
    sigma_gen = np.cov(generated_features, rowvar=False)

    fid_score = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
    return fid_score


real_videos_folder = './animatediff/data/MSRVTT/videos/test'
generated_videos_folder = './inference_samples/inference_samples_1/inference/samples'
num_videos_tot = len(os.listdir(real_videos_folder))
image_size = (128,128)
frames_per_video = 16

fid_score = compute_fid(real_videos_folder, generated_videos_folder)
print(f'[INFO] FID Score: {round(fid_score, 5)}')