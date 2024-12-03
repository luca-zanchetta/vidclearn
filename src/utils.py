import torch
import numpy as np
import cv2
from tuneavideo.pipelines.pipeline_tuneavideo import TuneAVideoPipeline
from tuneavideo.util import save_videos_grid

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

def generate_video(pretrained_model_path, unet, ddim_inv_latent, prompt, validation_data):
    pipe = TuneAVideoPipeline.from_pretrained(pretrained_model_path, unet=unet, torch_dtype=torch.float16).to("cuda")
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_vae_slicing()
    
    video = pipe(
        prompt, 
        latents=ddim_inv_latent, 
        video_length=validation_data.video_length,
        height=validation_data.height,
        width=validation_data.width, 
        num_inference_steps=validation_data.num_inference_steps,
        guidance_scale=validation_data.guidance_scale
    ).videos
    
    # Save generated video
    save_path = f'./tmp/video.mp4'
    save_videos_grid(video, save_path)
    del video, pipe
    return save_path

def save_frames_to_video(frames, output_path, fps=8):
    if not frames:
        raise ValueError("No frames to save!")

    # Get the frame size (assumes all frames have the same dimensions)
    frame_height, frame_width, channels = frames[0].shape
    frame_size = (frame_width, frame_height)

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    for frame in frames:
        out.write(frame)  # Write each frame to the video

    out.release()