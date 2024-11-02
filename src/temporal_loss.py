import torch.nn.functional as F

def temporal_loss(latents):
    temporal_loss = 0.0
    if latents.shape[2] > 1:    # Ensure there are multiple frames to compare
        # Shift the latents by 1 along the frame dimension
        latents_next_frame = latents[:, :, 1:, :, :]        # Frame 2 onwards
        latents_current_frame = latents[:, :, :-1, :, :]    # Frame 1 to second last
        
        # Compute temporal consistency loss
        temporal_loss = F.l1_loss(latents_current_frame, latents_next_frame)
    return temporal_loss