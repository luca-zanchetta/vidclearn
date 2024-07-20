import os
import csv
import ast
import math
import argparse

from tqdm import tqdm
from pytube import YouTube
from pytube.exceptions import PytubeError
from datetime import datetime
from moviepy.video.io.VideoFileClip import VideoFileClip
from pytube. innertube import _default_clients

_default_clients[ "ANDROID"][ "context"]["client"]["clientVersion"] = "19.08.35"
_default_clients["IOS"]["context"]["client"]["clientVersion"] = "19.08.35"
_default_clients[ "ANDROID_EMBED"][ "context"][ "client"]["clientVersion"] = "19.08.35"
_default_clients[ "IOS_EMBED"][ "context"]["client"]["clientVersion"] = "19.08.35"
_default_clients["IOS_MUSIC"][ "context"]["client"]["clientVersion"] = "6.41"
_default_clients[ "ANDROID_MUSIC"] = _default_clients[ "ANDROID_CREATOR" ]

def on_progress(stream, chunk, bytes_remaining):
    total_size = stream.filesize
    bytes_downloaded = total_size - bytes_remaining
    progress = int(bytes_downloaded / total_size * 100)
    tqdm_bar.update(progress - tqdm_bar.n)


def timestamp_to_seconds(timestamp):
    # Convert timestamp to datetime object
    dt = datetime.strptime(timestamp, '%H:%M:%S.%f')
    
    # Calculate total seconds
    total_seconds = dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6
    return total_seconds


def get_folder_size(folder_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.isfile(filepath):
                total_size += os.path.getsize(filepath)
    return total_size


def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)

    size = s
    unit_of_measurement = size_name[i]

    return size, unit_of_measurement


def validate_args(args):
    if args.max_folder_size <= 0:
        raise ValueError("max_folder_size must be greater than 0.")
    if args.end_row <= 0:
        raise ValueError("end_row must be greater than 0.")
    if args.start_row < 0:
        raise ValueError("start_row cannot be negative.")
    if args.current_row < 0:
        raise ValueError("current_row cannot be negative.")
    if args.clip_id < 0:
        raise ValueError("clip_id cannot be negative.")
    if not os.path.isfile(args.csv_filename):
        raise FileNotFoundError(f"The CSV file '{args.csv_filename}' does not exist.")
    if args.end_row <= args.start_row:
        raise ValueError("end_row must be > than start_row.")
    if args.end_row <= args.current_row:
        raise ValueError("end_row must be > than current_row.")
    if args.current_row > args.start_row:
        raise ValueError("current_row must be <= than start_row.")
    if args.directory != 'train' and args.directory != 'validation' and args.directory != 'test':
        raise ValueError("directory must be one among \'train\', \'validation\' and \'test\'.")


# Parse command-line arguments
parser = argparse.ArgumentParser(description='Download and process YouTube videos.')
parser.add_argument('--max_folder_size', type=float, default=50.0, help='Maximum folder size in GB.')
parser.add_argument('--end_row', type=int, default=10000, help='End row of the CSV to process.')
parser.add_argument('--start_row', type=int, default=0, help='Start row of the CSV to process.')
parser.add_argument('--current_row', type=int, default=0, help='Current row counter.')
parser.add_argument('--clip_id', type=int, default=0, help='Clip ID counter.')
parser.add_argument('--directory', type=str, required=True, help='Directory to save videos and captions. Options available: \'train\', \'validation\' and \'test\'.')
parser.add_argument('--csv_filename', type=str, required=True, help='CSV filename containing video data.')
parser.add_argument('--captions_filename', type=str, required=True, help='CSV filename to save captions.')

args = parser.parse_args()

validate_args(args)

# Assign variables from arguments
max_folder_size = args.max_folder_size
end_row = args.end_row
start_row = args.start_row
current_row = args.current_row
clip_id = args.clip_id
directory = args.directory
csv_filename = args.csv_filename
captions_filename = args.captions_filename

try:
    os.mkdir(directory)
except Exception as e:
    print(f"Directory '{directory}' already exists!")

# Open the CSV file
with open(csv_filename, mode='r', newline='', encoding='utf-8') as file:
    reader = csv.reader(file)
    
    # Iterate over the rows in the CSV file
    for row in reader:
        # Check current folder size, if train
        if directory == 'train' and current_row >= start_row:
            folder_path = os.path.join(os.getcwd(), 'train')
            size_in_bytes = get_folder_size(folder_path)
            size, unit_of_measurement = convert_size(size_in_bytes)
            print(f"[INFO] train folder size: {size} {unit_of_measurement}")
            if unit_of_measurement == "GB" and size >= max_folder_size:
                break
        
        if start_row == 0:
            with open(captions_filename, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(['clip_id', 'caption'])

        # Skip header row
        if current_row != 0 and current_row >= start_row:
            try:
                # Extract useful info
                video_id = row[0]
                url = row[1]
                video_name = video_id + ".mp4"

                # Set progress bar
                tqdm_bar = tqdm(total=100, desc="Downloading video", unit='%')

                # Download video
                yt = YouTube(url=url, on_progress_callback=on_progress)
                
                stream = yt.streams.get_highest_resolution()
                output_path = stream.download()
                tqdm_bar.close()

                # Rename video
                download_dir, original_filename = os.path.split(output_path)
                new_filepath = os.path.join(download_dir, directory, video_name)
                os.rename(output_path, new_filepath)

                # Generate video clips & captions
                timestamps = ast.literal_eval(row[2])
                captions = ast.literal_eval(row[3])

                for index, pair in enumerate(timestamps):
                    start_time = timestamp_to_seconds(pair[0])
                    end_time = timestamp_to_seconds(pair[1])
                    
                    # Load the video file
                    video = VideoFileClip(new_filepath)

                    # Cut the video
                    cut_video = video.subclip(start_time, end_time)

                    # Save the cut video
                    cut_video_filename = f"clip_{clip_id}.mp4"
                    cut_video_filepath = os.path.join(download_dir, directory, cut_video_filename)
                    cut_video.write_videofile(cut_video_filepath, codec="libx264")

                    # Close original video
                    video.close()

                    # Extract and save corresponding caption
                    caption = captions[index]
                    with open(captions_filename, mode='a', newline='', encoding='utf-8') as file:
                        writer = csv.writer(file)
                        writer.writerow([cut_video_filename, caption])

                    clip_id += 1

                    if index == 2:
                        print("\n")

                # Remove original video
                os.unlink(new_filepath)

            except PytubeError as e:
                print(f"[ERROR] Expected error: {e}")
            except Exception as e:
                print(f"[ERROR] Unexpected error: {e}")

        if current_row == end_row:
            break
        else:
            current_row += 1
            if start_row == 0:
                start_row += 1
            print(f"[INFO] Current row: {current_row}")

print("*********************DONE**************************")
print(f"[INFO] Next start row: {current_row + 1}")
print(f"[INFO] Next clip_id: {clip_id}")