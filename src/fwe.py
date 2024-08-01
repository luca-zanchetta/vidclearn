import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

# Function to extract frames from a video
def extract_frames_from_video(video, target_size=(299, 299)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames = video.permute(1, 2, 3, 0).cpu().numpy()
    frames = [cv2.resize(frame, target_size) for frame in frames]
    frames = [torch.tensor(frame, dtype=torch.uint8).permute(2, 0, 1).unsqueeze(0).to(device) for frame in frames]
    return frames

# Function to load an image as a tensor
def load_image(img):
    transform = transforms.ToTensor()
    img = transform(img).unsqueeze(0)
    return img

# Function to compute the optical flow between two subsequent frames
def compute_optical_flow(prev_img, next_img):
    prev_img = prev_img.squeeze().permute(1, 2, 0).cpu().numpy()
    next_img = next_img.squeeze().permute(1, 2, 0).cpu().numpy()
    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_RGB2GRAY)
    next_gray = cv2.cvtColor(next_img, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

# Function to warp a single frame
def warp_frame(frame, flow, device):
    h, w = flow.shape[:2]
    flow_map = np.column_stack((np.tile(np.arange(w), h), np.repeat(np.arange(h), w)))
    warp_map = (flow_map + flow.reshape(-1, 2)).astype(np.float32)
    warp_map[:, 0] = 2.0 * warp_map[:, 0] / (w - 1) - 1.0
    warp_map[:, 1] = 2.0 * warp_map[:, 1] / (h - 1) - 1.0
    warp_map = warp_map.reshape(h, w, 2)
    warp_map = torch.tensor(warp_map).unsqueeze(0).to(device)
    frame = frame.float()
    warped_frame = F.grid_sample(frame, warp_map, align_corners=False)
    return warped_frame

# Function to compute the Flow Warping Error
def compute_fwe(generated_video, real_video, frame_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generated_frames = extract_frames_from_video(generated_video, frame_size)
    real_frames = extract_frames_from_video(real_video, frame_size)

    total_error = 0.0
    n_frames = len(generated_frames)

    for i in range(n_frames - 1):
        gen_frame1 = generated_frames[i]
        gen_frame2 = generated_frames[i + 1]
        ref_frame1 = real_frames[i]
        ref_frame2 = real_frames[i + 1]

        flow = compute_optical_flow(gen_frame1, gen_frame2)
        warped_gen_frame2 = warp_frame(gen_frame2, flow, device)

        ref_frame2_tensor = ref_frame2.float()
        error = F.l1_loss(warped_gen_frame2, ref_frame2_tensor)
        total_error += error.item()

    fwe = total_error / (n_frames - 1)
    return fwe