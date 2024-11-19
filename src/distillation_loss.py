import torch.nn.functional as F

def distillation_loss(student_output, teacher_output, target, temperature, alpha, model_n):
    """
    Compute distillation loss using KL divergence between the student and teacher model outputs.
    """
    
    # Reconstruction loss:
    mse_loss = F.mse_loss(student_output.float(), target.float(), reduction="mean")
    if model_n == 1:
        return mse_loss
    
    # KL Divergence loss:
    teacher_output = F.softmax(teacher_output / temperature, dim=-1)
    student_output = F.log_softmax(student_output / temperature, dim=-1)
    kl_div_loss = F.kl_div(student_output, teacher_output, reduction="batchmean") * (temperature ** 2)
    
    # Distillation loss:
    distill_loss = alpha*kl_div_loss + (1-alpha)*mse_loss
    return distill_loss