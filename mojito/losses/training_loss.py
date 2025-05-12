import torch
import torch.nn as nn

from .base import BaseLosses
from .utils import js_div_loss

class CommitLoss(nn.Module):
    """
    Useless Wrapper
    """
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, commit, commit2, **kwargs):
        return commit
    
class MojitoLosses(BaseLosses):
    def __init__(self, cfg, num_joints, **kwargs):
        self.cfg = cfg
        recons_loss = cfg.LOSS.ABLATION.RECONS_LOSS

        # Define losses
        losses = []; params = {}
        losses.append("recons_feature")
        params['recons_feature'] = cfg.LOSS.LAMBDA_FEATURE

        losses.append("foot_contact")
        params["foot_contact"] = cfg.LOSS.LAMBDA_CONTACT

        losses.append("foot_contact_velocity")
        params["foot_contact_velocity"] = cfg.LOSS.LAMBDA_CONTACT

        losses.append("vq_commit")
        params['vq_commit'] = cfg.LOSS.LAMBDA_COMMIT
            
        losses.append("vq_zipf")
        params['vq_zipf'] = cfg.LOSS.LAMBDA_ZIPF
            
        losses.append("matching")
        params['matching'] = cfg.LOSS.LAMBDA_MATCHING

        losses.append("vq_dist")
        params['vq_dist'] = cfg.LOSS.LAMBDA_DIST

        # Define loss functions & weights
        losses_func = {}
        for loss in losses:
            if loss.split('_')[0] == "recons" or loss == "foot_contact_velocity":
                if recons_loss == "l1":
                    losses_func[loss] = nn.L1Loss
                elif recons_loss == "l2":
                    losses_func[loss] = nn.MSELoss
                elif recons_loss == "l1_smooth":
                    losses_func[loss] = nn.SmoothL1Loss
            elif loss == "matching":
                losses_func[loss] = nn.MSELoss
            elif loss == "foot_contact":
                losses_func[loss] = nn.BCELoss
            elif loss.split("_")[1] in ["commit", "zipf", "dist"]:
                losses_func[loss] = CommitLoss
            else:
                raise NotImplementedError(f"Loss {loss} not implemented.")

        super().__init__(cfg, losses, params, losses_func, num_joints, **kwargs)
        
        # Define non-linear mapping
        self.sigmoid = nn.Sigmoid()

    def update(self, rs_set, epoch):
        '''Update the losses'''
        total: float = 0.0
        verbose = {}

        if epoch < self.cfg.TRAIN.SWITCH_EPOCH:
            # reconstruction loss
            m_recon_loss = self._update_loss("recons_feature", rs_set["m_recon"][..., :-4], rs_set["m_ref"][..., :-4])
            total += m_recon_loss

            # contact loss
            foot_contact_loss = self._update_loss("foot_contact", self.sigmoid(rs_set["m_recon"][..., -4:]), rs_set["m_ref"][..., -4:])
            total += foot_contact_loss
            # self._update_loss("foot_contact_velocity", self.foot_contact_velocity_loss(rs_set["m_rst"]), 0)
            # total += foot_contact_loss

            # vq commit loss
            m_commit_loss = self._update_loss("vq_commit", rs_set["m_commit_loss"], rs_set["m_commit_loss"])
            total += m_commit_loss
            verbose.update({
                "m_recon_loss": m_recon_loss.item(),
                "m_commit_loss": m_commit_loss.item(),
                "foot_contact_loss": foot_contact_loss.item(),
            })

            # vq zipf loss
            if self.cfg.MODEL.params.vae_model.params.use_zipf:
                m_zipf_loss = self._update_loss("vq_zipf", rs_set["m_zipf_loss"], rs_set["m_zipf_loss"])
                total += m_zipf_loss
                verbose.update({"m_zipf_loss": m_zipf_loss.item()})
        else:
            # matching loss
            matching_loss = self._update_loss("matching", rs_set["i_quantized"], rs_set["m_quantized"])
            total += matching_loss
            verbose.update({"matching_loss": matching_loss.item()})

            # vq zipf loss
            if self.cfg.MODEL.params.vae_model.params.use_zipf:
                i_zipf_loss = self._update_loss("vq_zipf", rs_set["i_zipf_loss"], rs_set["i_zipf_loss"])
                total += i_zipf_loss
                verbose.update({"i_zipf_loss": i_zipf_loss.item()})

            sorted_m_freq = torch.sort(rs_set["m_freq"], descending=True)[0].float()
            sorted_i_freq = torch.sort(rs_set["i_freq"], descending=True)[0].float()
            _dist_loss = js_div_loss(sorted_m_freq, sorted_i_freq)
            dist_loss = self._update_loss("vq_dist", _dist_loss, _dist_loss)
            total += dist_loss
            verbose.update({"dist_loss": dist_loss.item()})
            
        verbose.update({"total": total.item()})

        # Update the total loss
        self.total += total.detach()
        self.count += 1

        return total, verbose
    
    def foot_contact_velocity_loss(self, rst):
        foot_joints = torch.tensor([7, 8, 10, 11])
        B, T, _ = rst.shape
        velocity = rst[..., 204:267].reshape(B, T, -1, 3)[..., foot_joints - 1, :] # (B, T, 4, 3)
        velocity_norm = torch.norm(velocity, dim=-1) # (B, T, 4)
        contact = self.sigmoid(rst[..., -4:]) # (B, T, 4)
        value = torch.einsum("btn,btn->btn", contact, velocity_norm)
        loss = torch.mean(value)
        
        return loss
