import torch

def compute_mse(generated_video, real_video):
    assert generated_video.shape == real_video.shape, "[ERROR] Videos must have the same shape!"
    generated_video = generated_video.float()
    real_video = real_video.float()
    mse = torch.mean((generated_video - real_video) ** 2)
    return mse

def compute_psnr(generated_video, real_video, max_pixel_value=255.0, epsilon=1e-10):
    mse = compute_mse(generated_video, real_video)
    psnr = 10 * torch.log10(max_pixel_value ** 2 / (mse + epsilon))
    return psnr