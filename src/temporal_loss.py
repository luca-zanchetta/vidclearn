import torch.nn.functional as F

def temporal_loss(model_pred, noisy_latents):
    noisy_latents_diff = (noisy_latents[:, :, 1:] - noisy_latents[:, :, :-1]).float()
    model_pred_diff = (model_pred[:, :, 1:] - model_pred[:, :, :-1]).float()
    
    return F.mse_loss(model_pred_diff, noisy_latents_diff)