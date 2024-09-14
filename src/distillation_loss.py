import torch.nn.functional as F

def distillation_loss(student_output, teacher_output, temperature):
    """
    Compute distillation loss using KL divergence between the student and teacher model outputs.
    """
    teacher_output = F.softmax(teacher_output / temperature, dim=-1)
    student_output = F.log_softmax(student_output / temperature, dim=-1)
    
    distill_loss = F.kl_div(student_output, teacher_output, reduction="batchmean") * (temperature ** 2)
    return distill_loss