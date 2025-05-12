import torch
import torch.nn as nn
from pytorch3d.transforms import matrix_to_axis_angle, rotation_6d_to_matrix

def js_div_loss(p, q):
    """
    Jensen-Shannon Divergence Loss
    """
    KLDivLoss = nn.KLDivLoss(reduction='batchmean')
    eps = 1e-8 # to prevent exploding gradient
    log_mean = ((p + q) / 2 + eps).log()
    return (KLDivLoss(log_mean, p + eps) + KLDivLoss(log_mean, q + eps)) / 2
    
def zipf_loss(freq, zipf_dist):
    """
    regularization term under Zipf's law
    """
    sorted_freq = torch.sort(freq, descending=True)[0].float()
    loss = js_div_loss(sorted_freq, zipf_dist)
    return loss
    
def repr2smpl(repr):
    """
    convert 271-dimensional motion representation to SMPL
    """
    B = repr.shape[0]; device = repr.device
    root_aa = matrix_to_axis_angle(rotation_6d_to_matrix(repr[:, 6:12]))
    local_joint_aa = matrix_to_axis_angle(rotation_6d_to_matrix(repr[:, 15:141].view(-1, 21, 6))).view(-1, 21*3)
    
    trans = torch.cumsum(repr[:-1, 3:6], dim=0)
    trans = torch.vstack([torch.zeros(1, 3).to(device), trans])
    smpl_pose = torch.cat([trans[:, :3], root_aa, local_joint_aa, torch.zeros(B, 6).to(device)], dim=1)

    return smpl_pose