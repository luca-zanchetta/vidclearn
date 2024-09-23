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

def extract_frames_from_video(video, target_size=(299, 299)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames = video.permute(1, 2, 3, 0).numpy()
    frames = [cv2.resize(frame, target_size) for frame in frames]
    frames = [torch.tensor(frame, dtype=torch.uint8).permute(2, 0, 1).unsqueeze(0).to(device) for frame in frames]
    return frames

def combine_batches(batch, replay_samples):
    # Extract pixel_values and prompt_ids from replay_samples
    replay_pixel_values = [sample["pixel_values"] for sample in replay_samples]
    replay_prompt_ids = [sample["prompt_ids"] for sample in replay_samples]
    
    # Concatenate both batch and replay_samples
    combined_pixel_values = torch.cat([batch["pixel_values"]] + replay_pixel_values)
    combined_prompt_ids = torch.cat([batch["prompt_ids"]] + replay_prompt_ids)

    # print("Original pixel_values shape:", batch["pixel_values"].shape)
    # print("Replay samples pixel_values shape:", [sample["pixel_values"].shape for sample in replay_samples])
    # print("Combined pixel_values shape:", combined_pixel_values.shape)

    replay_batch = {
        "pixel_values": combined_pixel_values,
        "prompt_ids": combined_prompt_ids
    }
    return replay_batch
