import cv2
import os

def images_to_video(image_folder, video_name, fps=30):
    images = [img for img in sorted(os.listdir(image_folder)) if img.endswith((".png", ".jpg", ".jpeg"))]
    
    if not images:
        print(f"No images found in {image_folder}. Skipping.")
        return
    
    # Read the first image to determine the width and height
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify codec, here it's for mp4 format
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)
    
    video.release()
    print(f"Video saved as {video_name}")

def process_folders(root_folder, output_folder, fps=30):
    # Make sure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Loop through all subfolders
    for subdir, _, _ in os.walk(root_folder):
        if subdir == root_folder:  # Skip the root folder itself
            continue

        # Create output video path
        subfolder_name = os.path.basename(subdir)
        video_name = os.path.join(output_folder, f"{subfolder_name}.mp4")

        # Convert images to video
        images_to_video(subdir, video_name, fps)

if __name__ == "__main__":
    root_folder = "./data/test_images"
    output_folder = "./data/test_videos"
    fps = 30  # Frames per second for the output videos
    
    process_folders(root_folder, output_folder, fps)
