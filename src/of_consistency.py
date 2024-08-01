import cv2
import numpy as np

def compute_optical_flow(prev_img, next_img):
    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_RGB2GRAY)
    next_gray = cv2.cvtColor(next_img, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

def compute_video_optical_flow(video_tensor):
    video_frames = video_tensor.permute(1, 2, 3, 0).numpy()  # Convert to (T, H, W, C)
    flows = []
    for i in range(video_frames.shape[0] - 1):
        flow = compute_optical_flow(video_frames[i], video_frames[i + 1])
        flows.append(flow)
    return flows

def compute_optical_flow_consistency(generated_flows, reference_flows):
    total_consistency = 0
    for gen_flow, ref_flow in zip(generated_flows, reference_flows):
        consistency = np.linalg.norm(gen_flow - ref_flow)
        total_consistency += consistency
    return total_consistency / len(generated_flows)
