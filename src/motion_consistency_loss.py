import torch
import cv2
import torch.nn.functional as F
import numpy as np

# Initialize ORB detector
orb = cv2.ORB_create()

# Function to compute the optical flow between two subsequent frames
def compute_optical_flow(prev_img, next_img):
    prev_img = prev_img.squeeze().permute(1, 0, 2).cpu().numpy()
    next_img = next_img.squeeze().permute(1, 0, 2).cpu().numpy()
    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_RGB2GRAY)
    next_gray = cv2.cvtColor(next_img, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return torch.from_numpy(flow).permute(2, 0, 1)  # Convert to tensor format for consistency

# Function to compute the optical flow of a video
def compute_video_optical_flow(video_tensor):
    video_frames = video_tensor.permute(1, 2, 3, 0)  # Convert to (T, H, W, C)
    flows = []
    for i in range(video_frames.shape[0] - 1):
        flow = compute_optical_flow(video_frames[i], video_frames[i + 1])
        flows.append(flow)
    return torch.stack(flows).unsqueeze(0)

# Function to extract the ORB keypoints' descriptors
def extract_orb_keypoints_descriptors(frame):
    # Convert tensor frame to numpy and grayscale
    frame_np = frame.squeeze().permute(1, 2, 0).cpu().numpy().astype('uint8')
    frame_gray = cv2.cvtColor(frame_np, cv2.COLOR_RGB2GRAY)
    # Detect and compute keypoints and descriptors
    keypoints, descriptors = orb.detectAndCompute(frame_gray, None)
    return keypoints, descriptors

# Function to compute the feature trajectory error
def compute_feature_trajectory_error(frames, optical_flow):
    frames = torch.stack(frames, dim=1).float()
    B, T, C, H, W = frames.shape
    trajectory_error = 0.0
    for t in range(T - 1):
        # Extract keypoints and descriptors for frames at time t and t+1
        keypoints_t, descriptors_t = extract_orb_keypoints_descriptors(frames[:, t])
        keypoints_t1, descriptors_t1 = extract_orb_keypoints_descriptors(frames[:, t + 1])

        if descriptors_t is None or descriptors_t1 is None:
            continue  # Skip if no descriptors were found

        # Estimate keypoints' positions in the next frame
        keypoints_estimated = []
        for kp in keypoints_t:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            flow_vec = optical_flow[:, t, :, y, x].mean(dim=0)[:2].cpu().numpy()
            keypoints_estimated.append([x + flow_vec[0], y + flow_vec[1]])

        # Convert keypoints to numpy arrays for computing error
        keypoints_estimated = np.array(keypoints_estimated)
        keypoints_t1_coords = np.array([kp.pt for kp in keypoints_t1])

        # Calculate trajectory error as L2 norm between estimated and actual keypoints in t+1
        if keypoints_estimated.size and keypoints_t1_coords.size:
            trajectory_error += F.mse_loss(
                torch.tensor(keypoints_estimated, dtype=torch.float32),
                torch.tensor(keypoints_t1_coords, dtype=torch.float32)
            ).item()

    return trajectory_error / (T - 1)

# Function to compute the Flow Warping Error
def compute_flow_warp_error(frames, optical_flow):
    frames = torch.stack(frames, dim=1).float()  # Stack frames along the time dimension
    B, T, C, H, W = frames.shape  # Extract dimensions
    flow_warp_error = 0.0
    
    # Ensure optical_flow is on the same device as frames
    optical_flow = optical_flow.to(frames.device)

    # Check and adjust optical_flow dimensions
    if optical_flow.dim() == 4:  # If shape is (B, T, H, W), add a flow component dimension
        optical_flow = optical_flow.unsqueeze(2)  # Reshape to (B, T, 1, H, W)
    elif optical_flow.size(2) != 2:
        raise ValueError("optical_flow should have 2 flow components (x and y) in the third dimension")

    # Create a base grid for warping
    grid = torch.stack(torch.meshgrid(
        torch.linspace(-1, 1, H, device=frames.device),
        torch.linspace(-1, 1, W, device=frames.device)
    ), dim=-1).unsqueeze(0)  # Shape: (1, H, W, 2)

    # Expand the grid to match the batch size
    grid = grid.expand(B, H, W, 2)  # Shape: (B, H, W, 2)
    
    for t in range(T - 1):
        # Ensure optical_flow has sufficient time steps
        if optical_flow.size(1) <= t:
            raise IndexError("optical_flow does not have enough time steps to match frames")

        # Warp the next frame using the optical flow and compute the error
        warped_frame = F.grid_sample(
            frames[:, t + 1], 
            grid + optical_flow[:, t].permute(0, 2, 3, 1), 
            mode='bilinear', 
            align_corners=True
        )
        flow_warp_error += F.mse_loss(warped_frame, frames[:, t])

    return flow_warp_error / (T - 1)

# Function to compute the motion consistency loss
def motion_consistency_loss(generated_frames, test_frames, gen_optical_flow, test_optical_flow, weights=(1.0, 1.0, 1.0)):
    warp_weight, flow_weight, traj_weight = weights

    # 1. Flow Warping Error
    gen_flow_warp_error = compute_flow_warp_error(generated_frames, gen_optical_flow)
    test_flow_warp_error = compute_flow_warp_error(test_frames, test_optical_flow)
    flow_warp_loss = warp_weight * torch.abs(gen_flow_warp_error - test_flow_warp_error)

    # 2. Optical Flow Consistency Loss
    flow_consistency_loss = flow_weight * F.mse_loss(gen_optical_flow, test_optical_flow)

    # 3. Short-Term Feature Trajectory Error with ORB
    gen_traj_error = compute_feature_trajectory_error(generated_frames, gen_optical_flow)
    test_traj_error = compute_feature_trajectory_error(test_frames, test_optical_flow)
    traj_loss = traj_weight * torch.abs(torch.tensor(gen_traj_error - test_traj_error))

    # Combine all losses
    total_loss = flow_warp_loss + flow_consistency_loss + traj_loss
    return total_loss