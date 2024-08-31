import os
import decord
decord.bridge.set_bridge('torch')

from torch.utils.data import Dataset
from einops import rearrange
from transformers import CLIPTokenizer

class TuneAVideoDataset(Dataset):
    def __init__(
            self,
            pretrained_model_path: str,
            video_dir: str,
            prompt_file: str,
            width: int = 512,
            height: int = 512,
            n_sample_frames: int = 8,
            sample_start_idx: int = 0,
            sample_frame_rate: int = 1,
    ):
        self.video_dir = video_dir
        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
        self.width = width
        self.height = height
        self.n_sample_frames = n_sample_frames
        self.sample_start_idx = sample_start_idx
        self.sample_frame_rate = sample_frame_rate

        # Load the prompts from the file
        with open(prompt_file, 'r') as f:
            lines = f.readlines()

        self.video_prompt_pairs = []
        for line in lines:
            video_name, prompt = line.strip().split(':')
            video_path = os.path.join(video_dir, video_name)
            prompt_ids = self.tokenizer(
                prompt, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids[0]
            self.video_prompt_pairs.append((video_path, prompt_ids))

    def __len__(self):
        return len(self.video_prompt_pairs)

    def __getitem__(self, index):
        video_path, prompt_ids = self.video_prompt_pairs[index]

        # Load and sample video frames
        vr = decord.VideoReader(video_path, width=self.width, height=self.height)
        sample_index = list(range(self.sample_start_idx, len(vr), self.sample_frame_rate))[:self.n_sample_frames]
        video = vr.get_batch(sample_index)
        video = rearrange(video, "f h w c -> f c h w")

        example = {
            "pixel_values": (video / 127.5 - 1.0),
            "prompt_ids": prompt_ids
        }

        return example
