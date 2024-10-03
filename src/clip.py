import torch
import torchvision.transforms as transforms
import numpy as np

from PIL import Image
from torch.nn.functional import cosine_similarity
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

# Function to compute the CLIP Score between a single generated video and its corresponding prompt
def compute_clip_score_single(generated_video, prompt, frame_size, accelerator=None):
    if accelerator is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = accelerator.device

    # Initialize the CLIP model and processor
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Extract frames from the video
    frames = extract_frames_from_video(generated_video)
    preprocessed_frames = preprocess_frames(frames, frame_size)

    # Process frames and extract image features
    frames = torch.stack(preprocessed_frames).to(device)  # Move frames to GPU
    pil_images = [transforms.ToPILImage()(frame.cpu()) for frame in frames]  # Convert to PIL on CPU
    image_inputs = clip_processor(images=pil_images, return_tensors="pt", padding=True, truncation=True).to(device)
    
    with torch.no_grad():
        image_features = clip_model.get_image_features(**image_inputs)
    
    avg_image_features = image_features.mean(dim=0)

    # Process the prompt and extract text features
    text_inputs = clip_processor(text=[prompt], return_tensors="pt", padding=True, truncation=True).to(device)
    
    with torch.no_grad():
        text_features = clip_model.get_text_features(input_ids=text_inputs['input_ids'], attention_mask=text_inputs['attention_mask'])

    # Compute cosine similarity between video features and text features
    clip_score = cosine_similarity(avg_image_features.unsqueeze(0), text_features).cpu().numpy().mean()
    
    return clip_score