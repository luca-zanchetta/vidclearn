import os
import torch
import torchvision.transforms as transforms
import yaml

from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from torchvision.io import read_video
from torch.nn.functional import cosine_similarity
from tqdm import tqdm

# Initialize the CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Function to extract frames from a video file
def extract_frames(video_path, num_frames=1):
    video, _, _ = read_video(video_path, pts_unit='sec')
    frames = video.permute(0, 3, 1, 2)  # Change to (N, C, H, W) format
    if len(frames) < num_frames:
        raise ValueError(f"Video {video_path} has less than {num_frames} frames.")
    return frames[:num_frames]

# Function to preprocess frames
def preprocess_frames(frames, frame_size):
    transform = transforms.Compose([
        transforms.Resize(frame_size),
        transforms.ToTensor()
    ])
    return [transform(Image.fromarray(frame.permute(1, 2, 0).numpy())) for frame in frames]

# Function to load the YAML file and extract prompts
def extract_prompts(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
        prompts = data.get('validation_data', {}).get('prompts', [])
        return prompts
    

def compute_clip_score(generated_videos_folder, prompts_file, frame_size, frames_per_video):    
    # Extract prompts
    prompts = extract_prompts(prompts_file)

    # Store all frames and prompts for batch processing
    all_frames = []
    all_prompts = []

    for video_file, prompt in tqdm(zip(sorted(os.listdir(generated_videos_folder)), prompts), total=len(os.listdir(generated_videos_folder))):
        video_path = os.path.join(generated_videos_folder, video_file)
        
        # Extract and preprocess frames
        frames = extract_frames(video_path, frames_per_video)
        preprocessed_frames = preprocess_frames(frames, frame_size)
        
        all_frames.append(preprocessed_frames)
        all_prompts.append(prompt)

    # Process each video's frames and compute their features
    video_image_features = []

    for frames in tqdm(all_frames, total=len(all_frames)):
        frames = torch.stack(frames)  # Convert list of tensors to a single 4D tensor (B, C, H, W)
        pil_images = [transforms.ToPILImage()(frame) for frame in frames]
        image_inputs = clip_processor(images=pil_images, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            image_features = clip_model.get_image_features(**image_inputs)
        
        # Average the features of the frames for the current video
        avg_image_features = image_features.mean(dim=0)
        video_image_features.append(avg_image_features)

    video_image_features = torch.stack(video_image_features)

    # Ensure prompts are in the correct format
    if isinstance(all_prompts, list) and isinstance(all_prompts[0], str):
        text_inputs = clip_processor(text=all_prompts, return_tensors="pt", padding=True, truncation=True)
    else:
        raise ValueError("Prompts should be a list of strings.")

    # Encode text with padding and truncation
    with torch.no_grad():
        text_features = clip_model.get_text_features(input_ids=text_inputs['input_ids'], attention_mask=text_inputs['attention_mask'])

    # Compute cosine similarity (CLIP score)
    clip_scores = cosine_similarity(video_image_features, text_features).cpu().numpy()

    # Return average CLIP score
    average_clip_score = clip_scores.mean()
    return average_clip_score


if __name__ == '__main__':
    # Folder containing generated videos and corresponding prompts
    generated_videos_folder = "./provainference"
    prompts_file = './configs/inference/inference.yaml'
    frame_size = (128, 128)
    frames_per_video = 16

    clip_score = compute_clip_score(generated_videos_folder, prompts_file, frame_size, frames_per_video)
    print(f"Average CLIP Score: {round(clip_score.item(), 5)}")