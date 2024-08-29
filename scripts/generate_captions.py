import os
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from tqdm import tqdm
from PIL import Image

# Initialize the BLIP-2 model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl").to(device)

# Function to generate caption for an image
def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

# Function to process all images in a folder and generate captions
def process_folder(folder_path):
    captions = {}
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                caption = generate_caption(image_path)
                captions[image_path] = caption
                print(f"Generated caption for {image_path}: {caption}")
    return captions

# Function to save captions to a text file
def save_captions(captions, output_file):
    with open(output_file, 'w') as f:
        for image_path, caption in captions.items():
            f.write(f"{image_path}: {caption}\n")

# Main function
def main():
    folder_path = "./data/DAVIS_images"
    output_file = "./data/captions.txt"

    # Process the folder and generate captions
    captions = process_folder(folder_path)

    # Save the captions to a file
    save_captions(captions, output_file)
    print(f"All captions have been saved to {output_file}")

if __name__ == "__main__":
    main()
