import cv2
import torch

from torchmetrics.functional import structural_similarity_index_measure as ssim

# Function to extract frames from a video
def extract_frames_from_video(video, target_size=(299, 299)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames = video.permute(1, 2, 3, 0).cpu().numpy()
    frames = [cv2.resize(frame, target_size) for frame in frames]
    frames = [torch.tensor(frame, dtype=torch.uint8).permute(2, 0, 1).unsqueeze(0).to(device) for frame in frames]
    return frames

# Function to compute the average SSIM score between a generated video and its corresponding reference real video
def compute_ssim(generated_video, real_video, frame_size):
    generated_frames = extract_frames_from_video(generated_video, frame_size)
    real_frames = extract_frames_from_video(real_video, frame_size)

    if len(generated_frames) != len(real_frames):
        raise ValueError("[ERROR] The two videos have different number of frames!")

    ssim_values = []
    for gen_frame, ref_frame in zip(generated_frames, real_frames):
        gen_frame = gen_frame.float() / 255.0
        ref_frame = ref_frame.float() / 255.0
        ssim_value = ssim(gen_frame, ref_frame, data_range=1.0)
        ssim_values.append(ssim_value.item())

    average_ssim = sum(ssim_values) / len(ssim_values)
    return average_ssim