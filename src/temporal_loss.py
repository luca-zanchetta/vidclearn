import torch.nn.functional as F

def temporal_loss(student_output, target):
    target_diff = (target[:, :, 1:] - target[:, :, :-1]).float()
    student_output_diff = (student_output[:, :, 1:] - student_output[:, :, :-1]).float()
    
    return F.mse_loss(target_diff, student_output_diff, reduction='mean')