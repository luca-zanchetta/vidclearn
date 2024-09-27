import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_frames(video_path, num_frames=5):
    """
    Extracts frames from a video.
    
    :param video_path: Path to the video file
    :param num_frames: Number of frames to extract from the video
    :return: A list of frames (as numpy arrays)
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Select `num_frames` equally spaced indices
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)  # Move to the frame index
        ret, frame = cap.read()  # Read the frame
        if ret:
            # Convert the frame from BGR (OpenCV format) to RGB (Matplotlib format)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    
    cap.release()
    return frames

def plot_frames(frames1, frames2, frames3, num_frames=5):
    """
    Plot selected frames from two videos side by side in a single figure.
    
    :param frames1: List of frames from the first video
    :param frames2: List of frames from the second video
    :param num_frames: Number of frames to plot from each video
    """
    # Create a figure with 3 rows (one for each video) and `num_frames` columns
    fig, axs = plt.subplots(3, num_frames, figsize=(15, 5))

    # Select 5 equally spaced frames from all the videos
    indices1 = np.linspace(0, len(frames1) - 1, num_frames, dtype=int)
    indices2 = np.linspace(0, len(frames2) - 1, num_frames, dtype=int)
    indices3 = np.linspace(0, len(frames3) - 1, num_frames, dtype=int)

    # Plot frames from the first video in the first row
    for i, idx in enumerate(indices1):
        axs[0, i].imshow(frames1[idx])
        axs[0, i].axis('off')  # Hide the axes

    # Plot frames from the second video in the second row
    for i, idx in enumerate(indices2):
        axs[1, i].imshow(frames2[idx])
        axs[1, i].axis('off')  # Hide the axes
        
    # Plot frames from the third video in the third row
    for i, idx in enumerate(indices3):
        axs[2, i].imshow(frames3[idx])
        axs[2, i].axis('off')  # Hide the axes

    plt.show()


last_video_seen = './data/train_videos/walking.mp4'
original_video = './data/test_videos/golf.mp4'
generated_video = './inference_samples_7/9.gif'

# Extract 5 frames from each video
frames1 = extract_frames(last_video_seen, num_frames=5)
frames2 = extract_frames(original_video, num_frames=5)
frames3 = extract_frames(generated_video, num_frames=5)

# Plot the frames
plot_frames(frames1, frames2, frames3, num_frames=5)
