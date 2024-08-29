import os
from src.utils import load_video
from src.clip import compute_clip_score
from tqdm import tqdm

# Hyperparameters
real_video_dir = './data/train_videos'
prompt_file = './data/train_prompts.txt'
output_file = './data/train_captions.txt'
frame_size = (512, 512)
frames_per_video = 24
prompts = []
i = 1

with open(prompt_file, 'r') as file:
    lines = file.readlines()
    for line in lines:
        curr_video_name = line.split('/')[3]
        succ_video_name = lines[i].split('/')[3]

        if i+1 <= len(lines):
            i += 1

        if curr_video_name == succ_video_name:
            prompt = line.split(': ')[1][:-1]
            prompts.append(prompt)
        else:
            # Load video
            video_path = os.path.join(real_video_dir, curr_video_name+".mp4")
            video = load_video(video_path, frames_per_video, frame_size)
            if video is None:
                print("[ERROR] Loading generated video was not successful.")
            
            # Compute CLIP Score
            index_max_score = 0
            max_score = 0
            for prompt in tqdm(prompts, desc='Computing CLIP Scores...'):
                clip_score = compute_clip_score(video, prompt, frame_size)
                if clip_score > max_score:
                    max_score = clip_score
                    index_max_score = prompts.index(prompt)
            
            # Save prompt having the maximum CLIP Score
            final_prompt = prompts[index_max_score]
            with open(output_file, 'a') as out:
                out.write(f"{video_path}:{final_prompt}\n")
            
            prompts = []