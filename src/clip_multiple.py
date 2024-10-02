import torch
import torchvision.transforms as transforms
import numpy as np

from PIL import Image
from torch.nn.functional import cosine_similarity
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

# Function to extract frames from a video file
def extract_frames_from_video(video):
    video = video.permute(1, 2, 3, 0)
    video = video.numpy()
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
    prompts = []
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
        for line in tqdm(lines, desc='Extracting prompts'):
            video_name, prompt = line.strip().split(':')
            prompts.append(prompt)
        
        return prompts

def compute_clip_score(generated_videos, prompts_file, frame_size, accelerator = None):
    device = None
    if accelerator is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = accelerator.device

    # Initialize the CLIP model and processor
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Extract prompts
    prompts = extract_prompts(prompts_file)

    all_frames = []
    all_prompts = []

    for video, prompt in tqdm(zip(generated_videos, prompts), desc='Extracting Frames', total=len(generated_videos)):
        frames = extract_frames_from_video(video)
        preprocessed_frames = preprocess_frames(frames, frame_size)
        all_frames.append(preprocessed_frames)
        all_prompts.append(prompt)

    video_image_features = []

    for frames in tqdm(all_frames, desc='Processing Extracted Frames', total=len(all_frames)):
        frames = torch.stack(frames).to(device)  # Move frames to GPU
        pil_images = [transforms.ToPILImage()(frame.cpu()) for frame in frames]  # Convert to PIL on CPU
        image_inputs = clip_processor(images=pil_images, return_tensors="pt", padding=True, truncation=True).to(device)
        
        with torch.no_grad():
            image_features = clip_model.get_image_features(**image_inputs)
        
        avg_image_features = image_features.mean(dim=0)
        video_image_features.append(avg_image_features)

    video_image_features = torch.stack(video_image_features).to(device)

    if isinstance(all_prompts, list) and isinstance(all_prompts[0], str):
        text_inputs = clip_processor(text=all_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    else:
        raise ValueError("Prompts should be a list of strings.")

    with torch.no_grad():
        text_features = clip_model.get_text_features(input_ids=text_inputs['input_ids'], attention_mask=text_inputs['attention_mask'])

    clip_scores = cosine_similarity(video_image_features, text_features).cpu().numpy()

    average_clip_score = clip_scores.mean()
    return average_clip_score
