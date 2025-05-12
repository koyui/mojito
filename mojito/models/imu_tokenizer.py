import torch
import torch.nn as nn
from copy import deepcopy

from mojito.config import instantiate_from_config
from mojito.losses.training_loss import MojitoLosses

class MojitoTokenizer(nn.Module):
    def __init__(
        self,
        cfg,
        vae_model,
        smoother,
        nfeats_motion=75,
        nfeats_imu=72,
        motion_codebook_size=1024,
        imu_codebook_size=2048,
        phase='train',
        **kwargs
    ):
        super().__init__()

        self.phase = phase
        self.njoints = 22
        self.rt_config = cfg.TRAIN if phase == "train" else cfg.TEST

        # instantiate model components
        vae_model["params"]["nfeats"] = nfeats_motion
        imu_vae = deepcopy(vae_model)
        imu_vae["params"]["nfeats"] = nfeats_imu
        vae_model["params"]["code_num"] = motion_codebook_size
        imu_vae["params"]["code_num"] = imu_codebook_size
        self.motion_vae = instantiate_from_config(vae_model)
        self.imu_vae = instantiate_from_config(imu_vae)
        self.smoother = instantiate_from_config(smoother)

        # set train or test mode of quantizer
        training = phase == 'train'
        self.motion_vae.training = training
        self.imu_vae.training = training
        self.motion_vae.quantizer.training = training
        self.imu_vae.quantizer.training = training

        # disable reuires_grad of smoother
        for p in self.smoother.parameters():
            p.requires_grad_(False)

        # instantiate the losses
        self._losses = torch.nn.ModuleDict({"losses_train": MojitoLosses(cfg, self.njoints)})

        # count codebook frequency
        self.codePred = []
        self.codeFrequency = torch.zeros((motion_codebook_size, ))

    def train_motion_vae_forward(self, batch):
        """
        forward pass for traning motion vqvae
        """
        motion_feats_ref = batch["motion"]
        motion_feats_rst, _, motion_loss_commit, _, additional_loss, motion_freq = self.motion_vae(motion_feats_ref)
        rs_set = {
            "m_ref": motion_feats_ref,
            "m_recon": motion_feats_rst,
            "m_commit_loss": motion_loss_commit,
            "m_freq": motion_freq
        }  
        if self.motion_vae.use_zipf:
            rs_set["m_zipf_loss"] = additional_loss["zipf_loss"]
        
        return rs_set
    
    def train_imu_tokenizer_forward(self, batch):
        """
        forward pass for training imu tokenizer
        """
        imu_feats_ref = batch["imu"]
        imu_quantized, imu_freq = self.imu_vae.quantize_only(imu_feats_ref)
        rs_set = {
            "i_ref": imu_feats_ref,
            "i_freq": imu_freq,
            "i_quantized": imu_quantized
        }

        return rs_set

    def train_forward(self, batch, epoch):
        if epoch < self.rt_config.SWITCH_EPOCH:
            rs_set = self.train_motion_vae_forward(batch)
        else:
            rs_set = self.train_imu_tokenizer_forward(batch)
        
        return rs_set

    def smooth_postprocess(self, before_smooth):
        """
        postprocess for smoothed motion
        """
        smoothed_pose = self.smoother.smooth_smpl(before_smooth[:, 3:])
        after_smooth = torch.hstack([before_smooth[:, :3], smoothed_pose])[:, :75]
        
        return after_smooth

    @torch.no_grad()
    def val_imu_vae_forward(self, batch):
        gt_imu = batch["imu"].unsqueeze(0)
        imu_quantized, _ = self.imu_vae.quantize_only(gt_imu)

        return imu_quantized

    @torch.no_grad()
    def get_motion_code(self, batch):
        """
        get quantized motion code (token)
        NOTE: only useful for training
        """
        gt_motion = batch["motion"]["pose"].unsqueeze(0)
        return self.motion_vae.encode(gt_motion)[0].squeeze(0)

    @torch.no_grad()
    def get_imu_code(self, batch):
        """
        get quantized IMU code (token)
        """
        gt_imu = batch["imu"].unsqueeze(0)
        return self.imu_vae.encode(gt_imu)[0].squeeze(0)