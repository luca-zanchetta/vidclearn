import torch
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm

class EWC:
    def __init__(self, model, dataloader, criterion, noise_scheduler, vae, text_encoder, accelerator, n_sample_frames, device='cuda'):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.noise_scheduler = noise_scheduler
        self.vae = vae
        self.text_encoder = text_encoder
        self.accelerator = accelerator
        self.n_sample_frames = n_sample_frames
        self.device = device
        
        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16
            
        # Ensure VAE and model parameters match the precision
        self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)
        
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._calculate_fisher()

        # Store the current parameters' values
        for n, p in self.params.items():
            self._means[n] = p.clone().detach()

    def _calculate_fisher(self):
        precision_matrices = {}
        for n, p in self.params.items():
            precision_matrices[n] = torch.zeros_like(p)

        self.model.eval()
        for batch in tqdm(self.dataloader, desc='Computing Fisher Information'):
            with self.accelerator.accumulate(self.model):
                # Convert videos to latent space
                pixel_values = batch["pixel_values"].to(self.weight_dtype)
                video_length = pixel_values.shape[1]
                pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                
                # with torch.cuda.amp.autocast(enabled=(self.weight_dtype == torch.float16)):
                latents = self.vae.encode(pixel_values).latent_dist.sample()
                
                latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                latents = latents * 0.18215

                if latents.shape[2] < self.n_sample_frames:
                    padding_value = self.n_sample_frames - latents.shape[2]
                    latents = F.pad(latents, (0, 0, 0, 0, padding_value, 0), "constant", 0)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each video
                timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = self.text_encoder(batch["prompt_ids"])[0]

                # Get the target for loss depending on the prediction type
                if self.noise_scheduler.prediction_type == "epsilon":
                    target = noise
                elif self.noise_scheduler.prediction_type == "v_prediction":
                    target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {self.noise_scheduler.prediction_type}")

                # Predict the noise residual and compute loss
                model_pred = self.model(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                loss.backward()

                for n, p in self.model.named_parameters():
                    if p.grad is not None:
                        precision_matrices[n] += p.grad.detach() ** 2

        # Normalize by the number of samples
        for n, p in precision_matrices.items():
            precision_matrices[n] /= len(self.dataloader)

        return precision_matrices

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            if n in self._precision_matrices:
                _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
                loss += _loss.sum()
        return loss
