import torch
import numpy as np
import cv2

# Function to load and preprocess video
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
    
    video = np.array(frames, dtype=np.uint8)
    video = torch.tensor(video, dtype=torch.uint8)
    return video.permute(3, 0, 1, 2)  # Convert to (C, T, H, W)