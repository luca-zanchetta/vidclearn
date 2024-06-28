import torch
import torchvision.transforms as transforms
import yaml
import numpy as np

from PIL import Image
from torch.nn.functional import cosine_similarity
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel


# Function to extract frames from a video file
def extract_frames_from_video(video):
    # Convert the video tensor back to (T, H, W, C)
    video = video.permute(1, 2, 3, 0)
    video = video.numpy()
    
    # Convert each frame to a NumPy array and add to the frames list
    frames = [frame.astype(np.uint8) for frame in video]
    
    return frames

# Function to preprocess frames
def preprocess_frames(frames, frame_size):
    transform = transforms.Compose([
        transforms.Resize(frame_size),
        transforms.ToTensor()
    ])
    return [transform(Image.fromarray(frame)) for frame in frames]

# Function to load the YAML file and extract prompts
def extract_prompts(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
        prompts = data.get('validation_data', {}).get('prompts', [])
        return prompts
    

def compute_clip_score(generated_videos, prompts_file, frame_size, frames_per_video):
    # Initialize the CLIP model and processor
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Extract prompts
    prompts = extract_prompts(prompts_file)

    # Store all frames and prompts for batch processing
    all_frames = []
    all_prompts = []
    
    for video, prompt in tqdm(zip(generated_videos, prompts), total=len(generated_videos)):
        # Extract and preprocess frames
        frames = extract_frames_from_video(video)
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