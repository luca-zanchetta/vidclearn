import matplotlib.pyplot as plt
import math
from moviepy.editor import VideoFileClip
from PIL import Image

video_path = './inference_samples_5/25.gif'
video = VideoFileClip(video_path)

frames = []
for t in range(0, int(video.duration * video.fps)):
    frame = video.get_frame(t / video.fps)
    frame = Image.fromarray(frame).convert('RGB')
    frames.append(frame)

# Number of frames to display
num_frames = len(frames)

# Determine grid size for subplots
cols = 5  # Number of columns (you can change this)
rows = math.ceil(num_frames / cols)  # Calculate the number of rows needed

# Create a figure
fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))  # Adjust the figsize as needed
axes = axes.flatten()  # Flatten the 2D grid to iterate easily

# Plot each frame with its index
for i, ax in enumerate(axes):
    if i < num_frames:
        ax.imshow(frames[i])
        ax.set_title(f"Frame {i}")
        ax.axis('off')  # Turn off axis for better visualization
    else:
        ax.axis('off')  # Hide unused subplots

# Adjust layout
plt.tight_layout()
plt.show()
